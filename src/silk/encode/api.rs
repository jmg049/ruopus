//! Public SILK encoder driver (RFC 6716 §5.2; normative `silk/enc_API.c`).
//!
//! [`SilkEncoder`] (mono) and [`SilkStereoEncoder`] (mid/side) wrap the
//! per-frame [`SilkChannelEncoder`] with the SILK payload framing: the
//! per-frame VAD flags and the LBRR flag(s) precede the coded frames. Both
//! handle 10/20 ms (one frame) and 40/60 ms (two/three 20 ms frames, the
//! later ones conditionally coded) and produce a range-coded SILK payload
//! that [`crate::silk::SilkDecoder`] (and libopus) decode. The stereo path
//! runs the LR→MS analysis, codes the predictor weights and per-frame
//! mid-only flag, and conditionally codes the side channel (with the
//! mid-only→side transition reset). Frames are always coded active (no DTX)
//! and without in-band FEC.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::range::RangeEncoder;

use super::super::indices::CondCoding;
use super::frame::SilkChannelEncoder;
use super::stereo::{StereoEncState, lr_to_ms, stereo_encode_mid_only, stereo_encode_pred};

/// A SILK encoder for one mono stream.
#[derive(Clone)]
pub struct SilkEncoder {
    ch: SilkChannelEncoder,
    final_range: u32,
}

impl SilkEncoder {
    /// A new encoder at the given internal rate (`fs_khz` ∈ {8, 12, 16}) and
    /// subframe count (`nb_subfr` = 4 for 20 ms, 2 for 10 ms).
    #[must_use]
    pub fn new(fs_khz: i32, nb_subfr: usize) -> Self {
        SilkEncoder {
            ch: SilkChannelEncoder::new(fs_khz, nb_subfr),
            final_range: 0,
        }
    }

    /// Sets the target bitrate (bps), which maps to the per-frame coding SNR.
    pub fn set_bitrate(&mut self, bps: i32) {
        self.ch.set_bitrate(bps);
    }

    /// Encodes `input` to a SILK payload of at most `max_payload` bytes,
    /// lowering the coding SNR (bitrate) and re-encoding from a snapshot if
    /// the first attempt overshoots - a coarse closed-loop rate control.
    /// Returns `None` if even the minimum bitrate cannot fit. The encoder
    /// state advances exactly once (the accepted attempt).
    pub fn encode_capped(&mut self, input: &[i16], max_payload: usize) -> Option<Vec<u8>> {
        let snapshot = self.clone();
        let mut bps = self.ch.target_rate_bps;
        for _ in 0..6 {
            let bytes = self.encode(input);
            if bytes.len() <= max_payload {
                return Some(bytes);
            }
            // Overshot: restore the pre-encode state and try a lower rate.
            if bps <= 6_000 {
                return None;
            }
            bps = (bps * 3 / 4).max(6_000);
            *self = snapshot.clone();
            self.set_bitrate(bps);
        }
        None
    }

    /// The range coder state after the last [`encode`](Self::encode)
    /// (`OPUS_GET_FINAL_RANGE`).
    #[must_use]
    pub const fn final_range(&self) -> u32 {
        self.final_range
    }

    /// Encodes `input` (i16 PCM at the internal rate) into a SILK payload.
    /// The number of SILK frames in the packet is inferred from the length:
    /// one frame is `nb_subfr * 5 * fs_khz` samples, and a 40/60 ms packet is
    /// 2/3 such (20 ms) frames.
    ///
    /// # Panics
    ///
    /// Panics if `input` is not a whole number of frames, or if the coded
    /// packet does not fit the range coder (it always does for valid inputs).
    #[must_use]
    pub fn encode(&mut self, input: &[i16]) -> Vec<u8> {
        let frame_length = self.ch.nb_subfr * 5 * self.ch.fs_khz as usize;
        assert!(
            !input.is_empty() && input.len() % frame_length == 0,
            "input must be a whole number of frames"
        );
        let mut enc = RangeEncoder::new(1275);
        self.encode_into(&mut enc, input, None);
        self.final_range = enc.range_size();
        // `finalize` returns the full allocated buffer; shrink to the bytes the
        // coder actually used (SILK is purely range-coded, no raw-bit tail) so
        // the payload is the real frame size.
        let bits = (enc.tell_frac() as usize + 7) >> 3;
        let nbytes = bits.div_ceil(8).max(2);
        enc.shrink(nbytes);
        enc.finalize().expect("SILK packet fits the range coder")
    }

    /// Writes the SILK header and frames for `input` into the shared range
    /// coder `enc`, without finalising it (for hybrid packets, where CELT
    /// continues in the same coder). Does not record `final_range`.
    ///
    /// `max_bits`, when set, is a hard cap on the cumulative coded size (in
    /// bits, as `enc.tell()` measures it): each frame scales its gains coarser
    /// until the running total fits, reserving room for the CELT high band in
    /// hybrid packets.
    ///
    /// # Panics
    ///
    /// Panics if `input` is not a whole number of frames.
    pub fn encode_into(&mut self, enc: &mut RangeEncoder, input: &[i16], max_bits: Option<i32>) {
        let frame_length = self.ch.nb_subfr * 5 * self.ch.fs_khz as usize;
        assert!(
            !input.is_empty() && input.len() % frame_length == 0,
            "input must be a whole number of frames"
        );
        let n_frames = input.len() / frame_length;

        // Header: per-frame VAD flags (all active) then the LBRR flag (no FEC).
        for _ in 0..n_frames {
            enc.encode_bit_logp(true, 1);
        }
        enc.encode_bit_logp(false, 1);

        for i in 0..n_frames {
            // The first frame of a packet is coded independently; later frames
            // condition their gains/lag on the previous frame.
            let cond = if i == 0 {
                CondCoding::Independently
            } else {
                CondCoding::Conditionally
            };
            self.ch
                .encode_frame(enc, &input[i * frame_length..(i + 1) * frame_length], cond, max_bits);
        }
    }
}

/// A SILK encoder for one stereo stream (mid/side coding).
pub struct SilkStereoEncoder {
    stereo: StereoEncState,
    mid: SilkChannelEncoder,
    side: SilkChannelEncoder,
    hist_l: [i16; 2],
    hist_r: [i16; 2],
    prev_mid_only: bool,
    total_rate_bps: i32,
    final_range: u32,
    fs_khz: i32,
    nb_subfr: usize,
}

impl SilkStereoEncoder {
    /// A new stereo encoder at the given internal rate and subframe count.
    #[must_use]
    pub fn new(fs_khz: i32, nb_subfr: usize) -> Self {
        SilkStereoEncoder {
            stereo: StereoEncState::default(),
            mid: SilkChannelEncoder::new(fs_khz, nb_subfr),
            side: SilkChannelEncoder::new(fs_khz, nb_subfr),
            hist_l: [0; 2],
            hist_r: [0; 2],
            prev_mid_only: false,
            total_rate_bps: 36_000,
            final_range: 0,
            fs_khz,
            nb_subfr,
        }
    }

    /// Sets the total (both channels) target bitrate (bps).
    pub fn set_bitrate(&mut self, bps: i32) {
        self.total_rate_bps = bps;
    }

    /// The range coder state after the last [`encode`](Self::encode).
    #[must_use]
    pub const fn final_range(&self) -> u32 {
        self.final_range
    }

    /// Whether the most recently coded frame included the side channel (i.e.
    /// was not mid-only).
    #[must_use]
    pub const fn side_active(&self) -> bool {
        !self.prev_mid_only
    }

    /// Encodes one packet of interleaved-by-channel `left`/`right` PCM (i16 at
    /// the internal rate, a whole number of frames each) into a stereo SILK
    /// payload.
    ///
    /// # Panics
    ///
    /// Panics if the channels differ in length or are not a whole number of
    /// frames, or if the packet does not fit the range coder.
    #[must_use]
    pub fn encode(&mut self, left: &[i16], right: &[i16]) -> Vec<u8> {
        let mut enc = RangeEncoder::new(1275);
        self.encode_into(&mut enc, left, right);
        self.final_range = enc.range_size();
        let bits = (enc.tell_frac() as usize + 7) >> 3;
        let nbytes = bits.div_ceil(8).max(2);
        enc.shrink(nbytes);
        enc.finalize().expect("SILK stereo packet fits the range coder")
    }

    /// Writes the stereo SILK header and frames into the shared range coder
    /// `enc`, without finalising it (for hybrid packets). Does not record
    /// `final_range`.
    ///
    /// # Panics
    ///
    /// Panics if the channels differ in length or are not a whole number of
    /// frames.
    pub fn encode_into(&mut self, enc: &mut RangeEncoder, left: &[i16], right: &[i16]) {
        let fl = self.nb_subfr * 5 * self.fs_khz as usize;
        assert_eq!(left.len(), right.len(), "channel length mismatch");
        assert!(!left.is_empty() && left.len() % fl == 0, "whole frames");
        let n_frames = left.len() / fl;

        // Pass 1: LR→MS per frame (advances the stereo state), collecting the
        // mid/side frames, predictor indices, per-channel rates and mid-only.
        struct Fd {
            mid: Vec<i16>,
            side: Vec<i16>,
            ix: [[i8; 3]; 2],
            rates: [i32; 2],
            mid_only: bool,
        }
        let mut frames: Vec<Fd> = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let lf = &left[f * fl..(f + 1) * fl];
            let rf = &right[f * fl..(f + 1) * fl];
            let mut x1 = vec![0i16; fl + 2];
            let mut x2 = vec![0i16; fl + 2];
            x1[0..2].copy_from_slice(&self.hist_l);
            x2[0..2].copy_from_slice(&self.hist_r);
            x1[2..].copy_from_slice(lf);
            x2[2..].copy_from_slice(rf);
            self.hist_l = [lf[fl - 2], lf[fl - 1]];
            self.hist_r = [rf[fl - 2], rf[fl - 1]];
            let (ix, mid_only, rates) = lr_to_ms(
                &mut self.stereo,
                &mut x1,
                &mut x2,
                self.total_rate_bps,
                128,
                false,
                self.fs_khz,
                fl,
            );
            frames.push(Fd {
                mid: x1[2..fl + 2].to_vec(),
                side: x2[1..fl + 1].to_vec(),
                ix,
                rates,
                mid_only: mid_only == 1,
            });
        }

        // Header: ch0 (mid) VAD flags (all active) + LBRR, then ch1 (side)
        // VAD flags (active iff the side is coded) + LBRR.
        for _ in 0..n_frames {
            enc.encode_bit_logp(true, 1);
        }
        enc.encode_bit_logp(false, 1);
        for fd in &frames {
            enc.encode_bit_logp(!fd.mid_only, 1);
        }
        enc.encode_bit_logp(false, 1);

        for (i, fd) in frames.iter().enumerate() {
            stereo_encode_pred(&mut *enc, &fd.ix);
            if fd.mid_only {
                stereo_encode_mid_only(&mut *enc, 1);
            }
            let mid_cond = if i == 0 {
                CondCoding::Independently
            } else {
                CondCoding::Conditionally
            };
            self.mid.set_bitrate(fd.rates[0]);
            self.mid.encode_frame(&mut *enc, &fd.mid, mid_cond, None);
            if !fd.mid_only {
                if self.prev_mid_only {
                    self.side.reset_side_prediction();
                }
                let side_cond = if i == 0 {
                    CondCoding::Independently
                } else if self.prev_mid_only {
                    CondCoding::IndependentlyNoLtpScaling
                } else {
                    CondCoding::Conditionally
                };
                self.side.set_bitrate(fd.rates[1]);
                self.side.encode_frame(&mut *enc, &fd.side, side_cond, None);
            }
            self.prev_mid_only = fd.mid_only;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::RangeDecoder;
    use crate::silk::api::{DecControl, SilkDecoder};
    use alloc::vec;

    /// A mono SILK payload decodes through the full `SilkDecoder` API and
    /// reproduces the encoder's reconstruction. With the internal rate equal
    /// to the API rate the output resampler is a pure delay, so `out` equals
    /// the encoder's NSQ output `xq` shifted by that (small) delay.
    #[test]
    fn mono_payload_round_trips_through_the_silk_decoder() {
        let (fs_khz, nb_subfr) = (16i32, 4usize);
        let frame_length = nb_subfr * 5 * fs_khz as usize;
        let ltp_mem = 20 * fs_khz as usize;

        let mut seed = 0x7331_u32;
        let input: Vec<i16> = (0..frame_length)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let n = ((seed >> 20) as i32 - 2048) / 4;
                let tone = ((i as f32 * 0.13).sin() * 2000.0) as i32;
                (n + tone).clamp(-30000, 30000) as i16
            })
            .collect();

        let mut e = SilkEncoder::new(fs_khz, nb_subfr);
        e.set_bitrate(24000);
        let bytes = e.encode(&input);
        assert!(!bytes.is_empty());
        let xq_enc: Vec<i16> = e.ch.nsq.xq[ltp_mem..ltp_mem + frame_length].to_vec();

        let ctl = DecControl {
            channels_internal: 1,
            channels_api: 1,
            internal_sample_rate: 16000,
            api_sample_rate: 16000,
            payload_size_ms: 20,
        };
        let mut d = SilkDecoder::new();
        let mut dec = RangeDecoder::new(&bytes);
        let mut out: Vec<i16> = vec![];
        d.decode(&mut dec, &ctl, true, &mut out);

        assert_eq!(out.len(), frame_length, "one frame of output");
        // The output resampler imposes a pure delay; find it and confirm the
        // decoded signal equals the encoder's reconstruction beyond it.
        let delay = (0..=16usize)
            .find(|&d| out[d..] == xq_enc[..frame_length - d])
            .expect("decoded output matches the encoder reconstruction at some small delay");
        assert!(delay <= 16, "unexpected resampler delay {delay}");
        assert!(out[..delay].iter().all(|&v| v == 0), "pre-delay samples are zero");
    }

    /// A 40 ms (two-frame) packet exercises the conditional-coding path: the
    /// second frame conditions its gains/lag on the first. The whole packet
    /// decodes coherently through the full `SilkDecoder` API (a desync would
    /// destroy the correlation with the input).
    #[test]
    fn two_frame_packet_round_trips_through_the_silk_decoder() {
        let (fs_khz, nb_subfr) = (16i32, 4usize);
        let frame_length = nb_subfr * 5 * fs_khz as usize;
        let total = 2 * frame_length;

        // A continuous periodic tone spanning both frames.
        let input: Vec<i16> = (0..total)
            .map(|i| {
                let mut s = 2400.0 * (core::f32::consts::TAU * i as f32 / 100.0).sin();
                s += 800.0 * (core::f32::consts::TAU * i as f32 / 50.0).sin();
                s += ((i as i32 * 1733 + 3) % 173 - 86) as f32 * 1.0;
                s.clamp(-30000.0, 30000.0) as i16
            })
            .collect();

        let mut e = SilkEncoder::new(fs_khz, nb_subfr);
        e.set_bitrate(24000);
        let bytes = e.encode(&input);
        assert!(!bytes.is_empty());

        let ctl = DecControl {
            channels_internal: 1,
            channels_api: 1,
            internal_sample_rate: 16000,
            api_sample_rate: 16000,
            payload_size_ms: 40,
        };
        let mut d = SilkDecoder::new();
        let mut dec = RangeDecoder::new(&bytes);
        let mut out: Vec<i16> = vec![];
        d.decode(&mut dec, &ctl, true, &mut out);
        d.decode(&mut dec, &ctl, false, &mut out);
        assert_eq!(out.len(), total, "two frames of output");

        // Correlate (delay-aligned) with the input; a conditional-coding
        // desync would wreck this.
        let delay = 13usize;
        let (mut sig, mut dot, mut eo) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..total - delay {
            let a = f64::from(input[i]);
            let b = f64::from(out[i + delay]);
            sig += a * a;
            dot += a * b;
            eo += b * b;
        }
        let corr = dot / (sig.sqrt() * eo.sqrt()).max(1.0);
        assert!(corr > 0.9, "two-frame reconstruction correlation {corr:.3} too low");
    }

    /// A stereo SILK stream round-trips through the full `SilkDecoder` API:
    /// the decoder finishes each packet on the encoder's exact range state
    /// (bit-exact through the stereo predictor, mid-only flag, side coding and
    /// the mid-only→side transition), and the output tracks the input.
    #[test]
    fn stereo_round_trips_through_the_silk_decoder() {
        let (fs_khz, nb_subfr) = (16i32, 4usize);
        let fl = nb_subfr * 5 * fs_khz as usize;

        let mut e = SilkStereoEncoder::new(fs_khz, nb_subfr);
        let mut d = SilkDecoder::new();
        let ctl = DecControl {
            channels_internal: 2,
            channels_api: 2,
            internal_sample_rate: 16000,
            api_sample_rate: 16000,
            payload_size_ms: 20,
        };

        let sample = |n: i32| -> (i16, i16) {
            let t = core::f32::consts::TAU * n as f32;
            let l = 6000.0 * (t / 90.0).sin();
            let r = 3000.0 * (t / 90.0 + 0.3).sin() + 5000.0 * (t / 53.0).sin();
            (l as i16, r as i16)
        };

        let mut saw_side = false;
        let mut last = (vec![0i16; fl], vec![0i16; fl], vec![0i16; 2 * fl]);
        for f in 0..60i32 {
            let mut l = vec![0i16; fl];
            let mut r = vec![0i16; fl];
            for i in 0..fl {
                let (a, b) = sample(f * fl as i32 + i as i32);
                l[i] = a;
                r[i] = b;
            }
            let bytes = e.encode(&l, &r);
            // The side channel becomes active once the width builds up.
            if !bytes.is_empty() && e.side_active() {
                saw_side = true;
            }
            let mut dec = RangeDecoder::new(&bytes);
            let mut out: Vec<i16> = vec![];
            d.decode(&mut dec, &ctl, true, &mut out);
            assert_eq!(out.len(), 2 * fl, "stereo frame output length");
            assert_eq!(dec.range_size(), e.final_range(), "range mismatch at frame {f}");
            last = (l, r, out);
        }

        assert!(saw_side, "side channel should activate within 60 frames");
        // Delay-aligned correlation of the decoded left channel with the input.
        let (l, _r, out) = last;
        let dec_l: Vec<i16> = out.iter().step_by(2).copied().collect();
        let corr = (0..32usize)
            .map(|delay| {
                let (mut s, mut dot, mut eo) = (0.0f64, 0.0f64, 0.0f64);
                for i in 0..fl - delay {
                    let a = f64::from(l[i]);
                    let b = f64::from(dec_l[i + delay]);
                    s += a * a;
                    dot += a * b;
                    eo += b * b;
                }
                dot / (s.sqrt() * eo.sqrt()).max(1.0)
            })
            .fold(0.0f64, f64::max);
        assert!(corr > 0.8, "stereo left-channel correlation {corr:.3} too low");
    }
}

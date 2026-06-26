//! Public SILK encoder driver (RFC 6716 §5.2).
//!
//! [`SilkEncoder`] (mono) and [`SilkStereoEncoder`] (mid/side) wrap the
//! per-frame [`SilkChannelEncoder`] with the SILK payload framing: the
//! per-frame VAD flags and the LBRR flag(s) precede the coded frames. Both
//! handle 10/20 ms (one frame) and 40/60 ms (two/three 20 ms frames, the
//! later ones conditionally coded) and produce a range-coded SILK payload
//! that [`crate::silk::SilkDecoder`] decodes. The stereo path
//! runs the LR→MS analysis, codes the predictor weights and per-frame
//! mid-only flag, and conditionally codes the side channel (with the
//! mid-only→side transition reset). Frames are always coded active (no DTX).
//! Both paths support in-band FEC (LBRR): each packet carries a reduced-rate
//! redundant copy of the *previous* packet's frame(s) so a lost packet can be
//! recovered from its successor via `decode_fec`.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::super::indices::CondCoding;
use super::super::tables::{LBRR_FLAGS_2_ICDF, LBRR_FLAGS_3_ICDF};
use super::frame::SilkChannelEncoder;
use super::stereo::{StereoEncState, lr_to_ms, stereo_encode_mid_only, stereo_encode_pred};
use crate::range::RangeEncoder;

/// A SILK encoder for one mono stream.
#[derive(Clone)]
pub struct SilkEncoder {
    ch: SilkChannelEncoder,
    final_range: u32,
    /// In-band FEC (LBRR) generation enabled (`OPUS_SET_INBAND_FEC`).
    use_inband_fec: bool,
    /// Expected packet-loss percentage 0-100 (`PacketLoss_perc`): drives the
    /// loss-robust LTP scaling and the LBRR gain increase.
    packet_loss_perc: i32,
    /// Whether the *previous* packet carried LBRR (`LBRR_in_previous_packet`).
    /// Selects the LBRR gain increase: 7 on the first LBRR packet, otherwise a
    /// `packet_loss_perc`-driven value.
    lbrr_in_previous_packet: bool,
    /// The previous packet's coded frames (`indices`, `pulses`) captured for
    /// LBRR. In-band FEC carries a redundant copy of the *previous* packet's
    /// frame(s) in the current packet, so a lost packet can be recovered from
    /// its successor (matching libopus, whose `indices_LBRR`/`pulses_LBRR` are
    /// filled by one packet and emitted in the next). Empty when no LBRR is
    /// pending (FEC just enabled, or the previous packet's frame count differs).
    lbrr_prev: Vec<(super::super::indices::SideInfoIndices, Vec<i8>)>,
}

impl SilkEncoder {
    /// A new encoder at the given internal rate (`fs_khz` ∈ {8, 12, 16}) and
    /// subframe count (`nb_subfr` = 4 for 20 ms, 2 for 10 ms).
    #[must_use]
    pub fn new(fs_khz: i32, nb_subfr: usize) -> Self {
        SilkEncoder {
            ch: SilkChannelEncoder::new(fs_khz, nb_subfr),
            final_range: 0,
            use_inband_fec: false,
            packet_loss_perc: 0,
            lbrr_in_previous_packet: false,
            lbrr_prev: Vec::new(),
        }
    }

    /// Sets the target bitrate (bps), which maps to the per-frame coding SNR.
    pub fn set_bitrate(&mut self, bps: i32) {
        self.ch.set_bitrate(bps);
    }

    /// Sets the expected packet-loss percentage 0-100
    /// (`OPUS_SET_PACKET_LOSS_PERC`). When > 0, independently coded voiced
    /// frames raise their LTP scaling index for loss robustness, and any LBRR
    /// copy is coded at a reduced rate (a larger gain increase).
    pub fn set_packet_loss_perc(&mut self, perc: i32) {
        self.packet_loss_perc = perc.clamp(0, 100);
        self.ch.set_packet_loss_perc(perc);
    }

    /// Enables or disables in-band FEC (LBRR) generation. When enabled, each
    /// packet carries a redundant copy of its SILK frame(s) so the decoder can
    /// reconstruct a lost frame from the next packet via `decode_fec`.
    pub fn set_inband_fec(&mut self, on: bool) {
        self.use_inband_fec = on;
    }

    /// Whether in-band FEC is enabled.
    #[must_use]
    pub const fn inband_fec(&self) -> bool {
        self.use_inband_fec
    }

    /// Sets the encode complexity 0-10 (the pitch-search depth).
    pub fn set_complexity(&mut self, complexity: u8) {
        self.ch.set_complexity(complexity);
    }

    /// Encodes `input` to a SILK payload of at most `max_payload` bytes. Each
    /// attempt applies the per-frame hard bit cap (the gain-multiplier rate
    /// control), which scales the gains coarser until the frame fits; if the
    /// cap's gain ceiling is not enough, the coding SNR (bitrate) is lowered
    /// and the encode retried from a snapshot. Returns `None` if even the
    /// minimum bitrate cannot fit. The encoder state advances exactly once.
    ///
    /// # Panics
    ///
    /// Panics if `input` is not a whole number of frames.
    pub fn encode_capped(&mut self, input: &[i16], max_payload: usize) -> Option<Vec<u8>> {
        let max_bits = (max_payload * 8) as i32;
        let snapshot = self.clone();
        let mut bps = self.ch.target_rate_bps;
        for _ in 0..6 {
            let mut enc = RangeEncoder::new(1275);
            self.encode_into(&mut enc, input, Some(max_bits));
            let bits = (enc.tell_frac() as usize + 7) >> 3;
            let nbytes = bits.div_ceil(8).max(2);
            if nbytes <= max_payload {
                self.final_range = enc.range_size();
                enc.shrink(nbytes);
                return Some(enc.finalize().expect("capped SILK packet fits the range coder"));
            }
            // The hard cap's gain ceiling was not enough; lower the rate and retry.
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
        self.ch.set_n_frames_per_packet(n_frames as i32);

        if !self.use_inband_fec {
            // Header: per-frame VAD flags (all active) then the LBRR flag (off).
            for _ in 0..n_frames {
                enc.encode_bit_logp(true, 1);
            }
            enc.encode_bit_logp(false, 1);
            for i in 0..n_frames {
                // The first frame of a packet is coded independently; later
                // frames condition their gains/lag on the previous frame.
                let cond = if i == 0 {
                    CondCoding::Independently
                } else {
                    CondCoding::Conditionally
                };
                self.ch
                    .encode_frame(enc, &input[i * frame_length..(i + 1) * frame_length], cond, max_bits);
            }
            // No LBRR emitted this packet; the next FEC packet (if any) starts
            // fresh at the full gain increase.
            self.lbrr_in_previous_packet = false;
            self.lbrr_prev.clear();
            return;
        }

        // In-band FEC. The current packet carries the LBRR copy of the *previous*
        // packet's frames (if that packet had the same frame count), then the
        // current frames coded normally. We capture the current frames' coded
        // indices/pulses into `lbrr_prev` for the *next* packet to emit as LBRR.
        //
        // In a hybrid packet (`max_bits` set), the LBRR copy must share the SILK
        // byte budget with the regular frames so the CELT high band still fits:
        // measure the LBRR cost on a trial clone first and only carry it if it
        // leaves the regular frames at least half the budget. Otherwise this
        // packet skips its LBRR (the previous frame loses FEC), but the current
        // frames are still captured for the *next* packet to try.
        let mut have_lbrr = self.lbrr_prev.len() == n_frames;
        if have_lbrr && let Some(cap) = max_bits {
            let lbrr_bits = {
                let mut trial = enc.clone();
                let mut ch_trial = self.ch.clone();
                for (i, (ind, pulses)) in self.lbrr_prev.iter().enumerate() {
                    let cond = if i == 0 {
                        CondCoding::Independently
                    } else {
                        CondCoding::Conditionally
                    };
                    ch_trial.emit_frame(&mut trial, ind, pulses, cond, true);
                }
                trial.tell() as i32 - enc.tell() as i32
            };
            // Reserve at least half the SILK budget for the regular frames.
            if lbrr_bits > cap / 2 {
                have_lbrr = false;
            }
        }

        // Header: VAD flags (all active), the LBRR flag, then for multi-frame
        // packets the per-frame LBRR symbol (all frames carry LBRR here).
        for _ in 0..n_frames {
            enc.encode_bit_logp(true, 1);
        }
        enc.encode_bit_logp(have_lbrr, 1);
        if have_lbrr && n_frames > 1 {
            let table: &[u8] = if n_frames == 2 {
                &LBRR_FLAGS_2_ICDF
            } else {
                &LBRR_FLAGS_3_ICDF
            };
            let symbol = (1usize << n_frames) - 1; // all frames flagged
            enc.encode_icdf(symbol - 1, table, 8);
        }

        // Emit the previous packet's LBRR frames first (decoder reads all LBRR
        // before the regular frames, advancing the same entropy history).
        if have_lbrr {
            let prev = core::mem::take(&mut self.lbrr_prev);
            for (i, (ind, pulses)) in prev.iter().enumerate() {
                let cond = if i == 0 {
                    CondCoding::Independently
                } else {
                    CondCoding::Conditionally
                };
                self.ch.emit_frame(enc, ind, pulses, cond, true);
            }
        } else {
            self.lbrr_prev.clear();
        }

        // LBRR gain increase for *this* packet's redundant copies
        // (`silk_control_codec`): 7 when the previous packet carried no LBRR
        // (it was coded at a higher bitrate), otherwise reduced as the expected
        // packet loss rises: max(7 - floor(perc * 0.4), 2). The current frames
        // are encoded with this reduced-rate LBRR copy stashed for the *next*
        // packet to emit. (With FEC off the copy is full-rate; here it is
        // always on, so the increase is always applied.)
        let lbrr_gain_increases = if self.lbrr_in_previous_packet {
            (7 - ((self.packet_loss_perc * 26214) >> 16)).max(2) // SMULWB(perc, 0.4_Q16)
        } else {
            7
        };
        self.ch.set_lbrr_gain_increases(lbrr_gain_increases);
        self.lbrr_in_previous_packet = true;

        // Emit the current frames, capturing each for the next packet's LBRR.
        // `encode_frame_capture` codes the regular frame into `enc` (honouring
        // the cumulative `max_bits` cap for hybrid, so the CELT high band keeps
        // its room) and returns the regular indices+pulses plus a reduced-rate
        // LBRR copy. The LBRR copy (or the regular frame when the second NSQ
        // pass is disabled) is stashed for the next packet to emit.
        let mut current: Vec<(super::super::indices::SideInfoIndices, Vec<i8>)> = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let cond = if i == 0 {
                CondCoding::Independently
            } else {
                CondCoding::Conditionally
            };
            let f = &input[i * frame_length..(i + 1) * frame_length];
            let ((ind, pulses), lbrr) = self.ch.encode_frame_capture(enc, f, cond, max_bits);
            current.push(lbrr.unwrap_or((ind, pulses)));
        }
        self.lbrr_prev = current;
    }
}

/// One stereo packet's captured LBRR data (for emission in the next packet).
/// Mirrors `SilkEncoder::lbrr_prev` but carries the stereo predictor indices,
/// the mid-only flag, and both channels' coded frames.
#[derive(Clone)]
struct StereoLbrrFrame {
    /// Stereo predictor indices coded by `stereo_encode_pred`.
    ix: [[i8; 3]; 2],
    /// Whether this frame was mid-only (no side channel coded).
    mid_only: bool,
    /// Captured mid (channel 0) indices and pulses.
    mid: (super::super::indices::SideInfoIndices, Vec<i8>),
    /// Captured side (channel 1) indices and pulses, present iff `!mid_only`.
    side: Option<(super::super::indices::SideInfoIndices, Vec<i8>)>,
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
    /// In-band FEC (LBRR) generation enabled.
    use_inband_fec: bool,
    /// Expected packet-loss percentage 0-100 (`PacketLoss_perc`): drives the
    /// loss-robust LTP scaling and the LBRR gain increase on both channels.
    packet_loss_perc: i32,
    /// Whether the *previous* packet carried LBRR (`LBRR_in_previous_packet`):
    /// selects the LBRR gain increase (7 on the first LBRR packet, otherwise a
    /// `packet_loss_perc`-driven value).
    lbrr_in_previous_packet: bool,
    /// The previous packet's captured LBRR frames, emitted as the LBRR copy in
    /// the current packet (one-packet delay, matching libopus). Empty when no
    /// LBRR is pending (FEC just enabled, or the previous packet's frame count
    /// differed).
    lbrr_prev: Vec<StereoLbrrFrame>,
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
            use_inband_fec: false,
            packet_loss_perc: 0,
            lbrr_in_previous_packet: false,
            lbrr_prev: Vec::new(),
        }
    }

    /// Sets the total (both channels) target bitrate (bps).
    pub fn set_bitrate(&mut self, bps: i32) {
        self.total_rate_bps = bps;
    }

    /// Enables or disables in-band FEC (LBRR) generation. When enabled, each
    /// stereo packet carries a redundant copy of the previous packet's mid (and
    /// side, when coded) frames so the decoder can reconstruct a lost frame.
    pub fn set_inband_fec(&mut self, on: bool) {
        self.use_inband_fec = on;
    }

    /// Whether in-band FEC is enabled.
    #[must_use]
    pub const fn inband_fec(&self) -> bool {
        self.use_inband_fec
    }

    /// Sets the expected packet-loss percentage 0-100
    /// (`OPUS_SET_PACKET_LOSS_PERC`). When > 0, independently coded voiced
    /// frames raise their LTP scaling index for loss robustness, and any LBRR
    /// copy is coded at a reduced rate (a larger gain increase).
    pub fn set_packet_loss_perc(&mut self, perc: i32) {
        self.packet_loss_perc = perc.clamp(0, 100);
        self.mid.set_packet_loss_perc(perc);
        self.side.set_packet_loss_perc(perc);
    }

    /// Sets the encode complexity 0-10 for both channels.
    pub fn set_complexity(&mut self, complexity: u8) {
        self.mid.set_complexity(complexity);
        self.side.set_complexity(complexity);
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
        self.encode_into(&mut enc, left, right, None);
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
    /// `max_bits`, when set, hard-caps the cumulative coded size (in bits, as
    /// `enc.tell()` measures it): the mid frame is capped to `max_bits`, then
    /// the side frame to the same cumulative budget, so the combined SILK low
    /// band leaves the CELT high band room in a hybrid packet.
    ///
    /// # Panics
    ///
    /// Panics if the channels differ in length or are not a whole number of
    /// frames.
    pub fn encode_into(&mut self, enc: &mut RangeEncoder, left: &[i16], right: &[i16], max_bits: Option<i32>) {
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

        if !self.use_inband_fec {
            // Header: ch0 (mid) VAD flags (all active) + LBRR (off), then ch1
            // (side) VAD flags (active iff the side is coded) + LBRR (off).
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
                self.mid.encode_frame(&mut *enc, &fd.mid, mid_cond, max_bits);
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
                    self.side.encode_frame(&mut *enc, &fd.side, side_cond, max_bits);
                }
                self.prev_mid_only = fd.mid_only;
            }
            // No LBRR emitted this packet; the next FEC packet starts fresh.
            self.lbrr_in_previous_packet = false;
            self.lbrr_prev.clear();
            return;
        }

        // In-band FEC. The current packet carries the LBRR copy of the
        // *previous* packet's frames (if that packet had the same frame count),
        // then the current frames coded normally. We capture each current frame
        // into `lbrr_prev` for the *next* packet's LBRR copy.
        //
        // In a hybrid packet (`max_bits` set), the combined mid+side LBRR copy
        // shares the SILK byte budget with the regular frames: measure its cost
        // on a trial clone first and only carry it if it leaves the regular
        // frames at least half the budget. Otherwise skip this packet's LBRR.
        let mut have_lbrr = self.lbrr_prev.len() == n_frames;
        if have_lbrr && let Some(cap) = max_bits {
            let lbrr_bits = {
                let mut trial = enc.clone();
                let mut mid_t = self.mid.clone();
                let mut side_t = self.side.clone();
                for (i, f) in self.lbrr_prev.iter().enumerate() {
                    stereo_encode_pred(&mut trial, &f.ix);
                    if f.mid_only {
                        stereo_encode_mid_only(&mut trial, 1);
                    }
                    let cond = if i == 0 {
                        CondCoding::Independently
                    } else {
                        CondCoding::Conditionally
                    };
                    mid_t.emit_frame(&mut trial, &f.mid.0, &f.mid.1, cond, true);
                    if let Some((sind, spulses)) = &f.side {
                        let prev_side = i > 0 && !self.lbrr_prev[i - 1].mid_only;
                        let side_cond = if prev_side {
                            CondCoding::Conditionally
                        } else {
                            CondCoding::Independently
                        };
                        side_t.emit_frame(&mut trial, sind, spulses, side_cond, true);
                    }
                }
                trial.tell() as i32 - enc.tell() as i32
            };
            if lbrr_bits > cap / 2 {
                have_lbrr = false;
            }
        }

        // Header: ch0 (mid) VAD flags (all active) + LBRR flag (and, for
        // multi-frame packets, the per-frame LBRR symbol - all frames carry
        // LBRR), then ch1 (side) VAD flags + LBRR flag. The side's per-frame
        // LBRR symbol mirrors which frames coded a side channel.
        for _ in 0..n_frames {
            enc.encode_bit_logp(true, 1);
        }
        enc.encode_bit_logp(have_lbrr, 1);
        if have_lbrr && n_frames > 1 {
            let table: &[u8] = if n_frames == 2 {
                &LBRR_FLAGS_2_ICDF
            } else {
                &LBRR_FLAGS_3_ICDF
            };
            let symbol = (1usize << n_frames) - 1; // mid: all frames flagged
            enc.encode_icdf(symbol - 1, table, 8);
        }
        for fd in &frames {
            enc.encode_bit_logp(!fd.mid_only, 1);
        }
        // Side LBRR: a frame carries a side LBRR copy iff that previous-packet
        // frame coded its side channel (i.e. was not mid-only).
        let side_lbrr: Vec<bool> = if have_lbrr {
            self.lbrr_prev.iter().map(|f| !f.mid_only).collect()
        } else {
            Vec::new()
        };
        let side_has_lbrr = side_lbrr.iter().any(|&b| b);
        enc.encode_bit_logp(have_lbrr && side_has_lbrr, 1);
        if have_lbrr && side_has_lbrr && n_frames > 1 {
            let table: &[u8] = if n_frames == 2 {
                &LBRR_FLAGS_2_ICDF
            } else {
                &LBRR_FLAGS_3_ICDF
            };
            let mut symbol = 0usize;
            for (i, &b) in side_lbrr.iter().enumerate() {
                if b {
                    symbol |= 1 << i;
                }
            }
            enc.encode_icdf(symbol - 1, table, 8);
        }

        // Emit the previous packet's LBRR frames first. For each LBRR frame, the
        // decoder reads (when the mid is flagged): the stereo predictor, then -
        // only when the side has *no* LBRR for that frame - the mid-only flag,
        // then the mid frame, then (when flagged) the side frame.
        if have_lbrr {
            let prev = core::mem::take(&mut self.lbrr_prev);
            for (i, f) in prev.iter().enumerate() {
                stereo_encode_pred(&mut *enc, &f.ix);
                let side_lbrr_i = !f.mid_only;
                if !side_lbrr_i {
                    stereo_encode_mid_only(&mut *enc, 1);
                }
                let cond = if i == 0 {
                    CondCoding::Independently
                } else {
                    CondCoding::Conditionally
                };
                self.mid.emit_frame(&mut *enc, &f.mid.0, &f.mid.1, cond, true);
                if let Some((sind, spulses)) = &f.side {
                    // Side cond mirrors the decoder's LBRR rule: conditional only
                    // when the *previous* LBRR frame also coded a side channel
                    // (side `lbrr_flags[i-1]` set), else independent.
                    let prev_side = i > 0 && !prev[i - 1].mid_only;
                    let side_cond = if prev_side {
                        CondCoding::Conditionally
                    } else {
                        CondCoding::Independently
                    };
                    self.side.emit_frame(&mut *enc, sind, spulses, side_cond, true);
                }
            }
        } else {
            self.lbrr_prev.clear();
        }

        // LBRR gain increase for *this* packet's redundant copies
        // (`silk_control_codec`): 7 when the previous packet carried no LBRR,
        // otherwise reduced as the expected packet loss rises:
        // max(7 - floor(perc * 0.4), 2). Applied to both channels so the
        // captured copy is coded at the reduced rate.
        let lbrr_gain_increases = if self.lbrr_in_previous_packet {
            (7 - ((self.packet_loss_perc * 26214) >> 16)).max(2) // SMULWB(perc, 0.4_Q16)
        } else {
            7
        };
        self.mid.set_lbrr_gain_increases(lbrr_gain_increases);
        self.side.set_lbrr_gain_increases(lbrr_gain_increases);
        self.lbrr_in_previous_packet = true;

        // Emit the current frames, capturing each for the next packet's LBRR.
        // `encode_frame_capture` codes the regular frame into `enc` honouring
        // the cumulative `max_bits` cap (hybrid) and returns the regular
        // indices+pulses plus a reduced-rate LBRR copy. The LBRR copy (or the
        // regular frame when the second NSQ pass is disabled) is stashed.
        let mut current: Vec<StereoLbrrFrame> = Vec::with_capacity(n_frames);
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
            let ((mid_ind, mid_pulses), mid_lbrr) =
                self.mid.encode_frame_capture(&mut *enc, &fd.mid, mid_cond, max_bits);
            let mid_cap = mid_lbrr.unwrap_or((mid_ind, mid_pulses));
            let mut side_cap = None;
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
                let ((sind, spulses), side_lbrr_copy) =
                    self.side.encode_frame_capture(&mut *enc, &fd.side, side_cond, max_bits);
                side_cap = Some(side_lbrr_copy.unwrap_or((sind, spulses)));
            }
            self.prev_mid_only = fd.mid_only;
            current.push(StereoLbrrFrame {
                ix: fd.ix,
                mid_only: fd.mid_only,
                mid: mid_cap,
                side: side_cap,
            });
        }
        self.lbrr_prev = current;
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::range::RangeDecoder;
    use crate::silk::api::{DecControl, SilkDecoder};

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

//! The public Opus encoder (RFC 6716 §5).
//!
//! [`OpusEncoder::encode`] produces **CELT-only** packets at 48 kHz - a
//! single CELT frame (TOC code 0) per call, across the four CELT frame sizes
//! (2.5/5/10/20 ms), mono or stereo, any audio bandwidth. [`OpusEncoder::
//! encode_silk`] produces **SILK-mode** packets (mono, 10/20/40/60 ms,
//! narrowband/mediumband/wideband): the 48 kHz input is resampled to the
//! SILK internal rate and coded by [`crate::silk::SilkEncoder`]. The heavy
//! lifting is in [`crate::celt::encoder::CeltEncoder`] and the SILK encode
//! modules; this layer chooses the TOC byte and frames a conformant Opus
//! packet. [`OpusEncoder::encode_hybrid`] produces **hybrid** packets (SILK
//! wideband low band + CELT high band in one shared coder, super-wideband or
//! fullband), and [`OpusEncoder::encode_auto`] picks SILK or CELT per frame
//! from the frame size and target bitrate.

use alloc::vec::Vec;

use crate::celt::encoder::CeltEncoder;
use crate::packet::Bandwidth;
use crate::range::RangeEncoder;

/// Errors returned by [`OpusEncoder::encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// The frame had an unsupported number of samples per channel. At
    /// 48 kHz the encoder accepts 120, 240, 480 or 960 (2.5/5/10/20 ms).
    InvalidFrameSize,
    /// The output budget is outside the usable range: at least 3 bytes
    /// (1 TOC + 2 payload) and at most 1275 (the Opus per-frame limit).
    InvalidBudget,
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            EncodeError::InvalidFrameSize => "frame size must be 120/240/480/960 samples per channel at 48 kHz",
            EncodeError::InvalidBudget => "output budget must be between 3 and 1275 bytes",
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EncodeError {}

/// A pure-Rust Opus encoder producing CELT-only fullband packets at 48 kHz.
///
/// ```
/// use opus_native::{OpusEncoder, OpusDecoder};
/// let mut enc = OpusEncoder::new(1);
/// let mut dec = OpusDecoder::new(1);
/// let frame = vec![0.0f32; 960]; // 20 ms of mono silence
/// let packet = enc.encode(&frame, 64).unwrap();
/// let pcm = dec.decode_packet(&packet).unwrap();
/// assert_eq!(pcm.len(), 960);
/// // The decoder finishes on the encoder's range state - bit exactness.
/// assert_eq!(dec.final_range(), enc.final_range());
/// ```
pub struct OpusEncoder {
    celt: CeltEncoder,
    channels: usize,
    bandwidth: Bandwidth,
    /// SILK encoder + input resampler (mono SILK-mode path), created lazily
    /// for the current internal rate / subframe configuration.
    silk: Option<(
        crate::silk::encode::api::SilkEncoder,
        crate::silk::resampler::Resampler,
        i32,
        usize,
    )>,
    /// Stereo SILK encoder + per-channel input resamplers (created lazily).
    silk_stereo: Option<(
        crate::silk::encode::api::SilkStereoEncoder,
        crate::silk::resampler::Resampler,
        crate::silk::resampler::Resampler,
        i32,
        usize,
    )>,
    /// Target bitrate (bps) for the SILK path; `None` uses a default.
    target_bitrate: Option<u32>,
    /// Range state after the last packet, from whichever coder produced it.
    last_final_range: u32,
}

impl OpusEncoder {
    /// Creates an encoder for `channels` (1 or 2) at 48 kHz, fullband.
    ///
    /// # Panics
    ///
    /// Panics unless `channels` is 1 or 2.
    #[must_use]
    pub fn new(channels: usize) -> Self {
        assert!(channels == 1 || channels == 2, "channels must be 1 or 2");
        OpusEncoder {
            celt: CeltEncoder::with_channels(channels),
            channels,
            bandwidth: Bandwidth::FullBand,
            silk: None,
            silk_stereo: None,
            target_bitrate: None,
            last_final_range: 0,
        }
    }

    /// Restricts the coded audio bandwidth (`OPUS_SET_BANDWIDTH`). The CELT
    /// modes support narrowband, wideband, super-wideband and fullband;
    /// mediumband is treated as wideband (CELT has no 6 kHz mode).
    pub const fn set_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.bandwidth = bandwidth;
    }

    /// Selects variable bitrate at `bitrate` bits/s (`OPUS_SET_BITRATE` with
    /// VBR). Each call to [`encode`](Self::encode) then treats `max_bytes`
    /// as a ceiling and shrinks the packet to the per-frame target. Passing
    /// `None` restores constant bitrate (fill `max_bytes`).
    pub const fn set_bitrate(&mut self, bitrate: Option<u32>) {
        self.celt.set_target_bitrate(bitrate);
        self.target_bitrate = bitrate;
    }

    /// Encodes one frame, automatically choosing SILK (speech / lower rate)
    /// or CELT (music / higher rate). This is a simplified mode decision (not
    /// libopus's full hysteresis): the SILK-only frame sizes 40/60 ms always
    /// use SILK, the CELT-only sizes 2.5/5 ms always use CELT, and 10/20 ms
    /// pick SILK (wideband-or-below) or hybrid (super-wideband/fullband) at a
    /// modest target bitrate, otherwise CELT. `pcm` is interleaved 48 kHz f32;
    /// see [`encode`](Self::encode), [`encode_silk`](Self::encode_silk) and
    /// [`encode_hybrid`](Self::encode_hybrid) for the per-mode details.
    ///
    /// # Errors
    ///
    /// As [`encode`](Self::encode) / [`encode_silk`](Self::encode_silk).
    pub fn encode_auto(&mut self, pcm: &[f32], max_bytes: usize) -> Result<Vec<u8>, EncodeError> {
        if self.channels == 0 || pcm.len() % self.channels != 0 {
            return Err(EncodeError::InvalidFrameSize);
        }
        let per_ch = pcm.len() / self.channels;
        match per_ch {
            120 | 240 => self.encode(pcm, max_bytes),        // 2.5/5 ms: CELT only
            1920 | 2880 => self.encode_silk(pcm, max_bytes), // 40/60 ms: SILK only
            480 | 960 => {
                // 10/20 ms: SILK for speech up to wideband at a modest rate;
                // hybrid for super-wideband/fullband speech rates; else CELT.
                let wb_or_below = matches!(
                    self.bandwidth,
                    Bandwidth::NarrowBand | Bandwidth::MediumBand | Bandwidth::WideBand
                );
                let swb_or_fb = matches!(self.bandwidth, Bandwidth::SuperWideBand | Bandwidth::FullBand);
                if self.channels == 1 && wb_or_below && self.target_bitrate.is_some_and(|b| b <= 24_000) {
                    self.encode_silk(pcm, max_bytes)
                } else if self.channels == 1 && swb_or_fb && self.target_bitrate.is_some_and(|b| b <= 40_000) {
                    self.encode_hybrid(pcm, max_bytes)
                } else if wb_or_below && self.target_bitrate.is_some_and(|b| b <= 24_000) {
                    // Stereo speech (no stereo hybrid yet): SILK stereo.
                    self.encode_silk(pcm, max_bytes)
                } else {
                    self.encode(pcm, max_bytes)
                }
            },
            _ => Err(EncodeError::InvalidFrameSize),
        }
    }

    /// The range state after the last encoded packet (`OPUS_GET_FINAL_RANGE`).
    /// A conformant decoder finishes the packet with this exact value.
    #[must_use]
    pub const fn final_range(&self) -> u32 {
        self.last_final_range
    }

    /// Encodes one frame of interleaved 48 kHz f32 PCM in `[-1, 1]` into an
    /// Opus packet of at most `max_bytes` bytes (including the TOC).
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidFrameSize`] if `pcm` is not 120/240/480/960
    /// samples per channel; [`EncodeError::InvalidBudget`] if `max_bytes`
    /// is outside `3..=1275`.
    pub fn encode(&mut self, pcm: &[f32], max_bytes: usize) -> Result<Vec<u8>, EncodeError> {
        if self.channels == 0 || pcm.len() % self.channels != 0 {
            return Err(EncodeError::InvalidFrameSize);
        }
        let n = pcm.len() / self.channels;
        let lm = match n {
            120 => 0u8,
            240 => 1,
            480 => 2,
            960 => 3,
            _ => return Err(EncodeError::InvalidFrameSize),
        };
        if !(3..=1275).contains(&max_bytes) {
            return Err(EncodeError::InvalidBudget);
        }

        // TOC: CELT-only configs 16..=31 by bandwidth, stereo flag, code 0
        // (one frame per packet). `end` is the number of coded CELT bands.
        let (config_base, end) = match self.bandwidth {
            Bandwidth::NarrowBand => (16u8, 13usize),
            Bandwidth::MediumBand | Bandwidth::WideBand => (20, 17),
            Bandwidth::SuperWideBand => (24, 19),
            Bandwidth::FullBand => (28, 21),
        };
        let config = config_base + lm;
        let toc = (config << 3) | (u8::from(self.channels == 2) << 2);

        let payload = self.celt.encode_frame_bw(pcm, max_bytes - 1, end);
        self.last_final_range = self.celt.final_range();
        let mut packet = Vec::with_capacity(payload.len() + 1);
        packet.push(toc);
        packet.extend_from_slice(&payload);
        Ok(packet)
    }

    /// Encodes one frame of interleaved 48 kHz f32 PCM in `[-1, 1]` as a
    /// SILK-mode Opus packet (mono or stereo). The frame must be 10, 20, 40 or
    /// 60 ms (480/960/1920/2880 samples per channel). The audio bandwidth
    /// selects the SILK internal rate: narrowband (8 kHz), mediumband
    /// (12 kHz), or wideband (16 kHz; super-wideband/fullband are treated as
    /// wideband, since pure SILK tops out there). The 48 kHz input is
    /// resampled per channel to the internal rate, coded (mid/side for
    /// stereo), and wrapped with the SILK TOC.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidFrameSize`] if `pcm` is not a SILK frame size
    /// (per channel); [`EncodeError::InvalidBudget`] if the packet would
    /// exceed `max_bytes` (`3..=1275`).
    ///
    /// # Panics
    ///
    /// Panics if the internal SILK encoder fails to produce a valid frame for
    /// the given input (it does not for in-range PCM).
    pub fn encode_silk(&mut self, pcm: &[f32], max_bytes: usize) -> Result<Vec<u8>, EncodeError> {
        if self.channels == 0 || pcm.len() % self.channels != 0 {
            return Err(EncodeError::InvalidFrameSize);
        }
        let per_ch = pcm.len() / self.channels;
        let (frame_ms, lm) = match per_ch {
            480 => (10usize, 0u8),
            960 => (20, 1),
            1920 => (40, 2),
            2880 => (60, 3),
            _ => return Err(EncodeError::InvalidFrameSize),
        };
        if !(3..=1275).contains(&max_bytes) {
            return Err(EncodeError::InvalidBudget);
        }

        let (config_base, internal_khz) = match self.bandwidth {
            Bandwidth::NarrowBand => (0u8, 8i32),
            Bandwidth::MediumBand => (4, 12),
            _ => (8, 16), // wideband (and SWB/FB fall back to WB)
        };
        let nb_subfr = if frame_ms == 10 { 2 } else { 4 };
        let bitrate = self.target_bitrate.map_or(20_000, |b| b as i32);

        // 48 kHz f32 → i16.
        let to_i16 = |v: f32| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16;

        let payload = if self.channels == 1 {
            let need_new = self
                .silk
                .as_ref()
                .is_none_or(|(_, _, khz, nbs)| *khz != internal_khz || *nbs != nb_subfr);
            if need_new {
                self.silk = Some((
                    crate::silk::encode::api::SilkEncoder::new(internal_khz, nb_subfr),
                    crate::silk::resampler::Resampler::new_enc(48_000, internal_khz * 1000),
                    internal_khz,
                    nb_subfr,
                ));
            }
            let (silk, resampler, _, _) = self.silk.as_mut().expect("configured");
            silk.set_bitrate(bitrate.clamp(5000, 80_000));
            let in16: Vec<i16> = pcm.iter().map(|&v| to_i16(v)).collect();
            let out_len = per_ch * internal_khz as usize / 48;
            let mut internal = vec![0i16; out_len];
            resampler.process(&mut internal, &in16);
            let p = silk.encode(&internal);
            self.last_final_range = silk.final_range();
            p
        } else {
            // Stereo: deinterleave, resample each channel, mid/side encode.
            let need_new = self
                .silk_stereo
                .as_ref()
                .is_none_or(|(_, _, _, khz, nbs)| *khz != internal_khz || *nbs != nb_subfr);
            if need_new {
                self.silk_stereo = Some((
                    crate::silk::encode::api::SilkStereoEncoder::new(internal_khz, nb_subfr),
                    crate::silk::resampler::Resampler::new_enc(48_000, internal_khz * 1000),
                    crate::silk::resampler::Resampler::new_enc(48_000, internal_khz * 1000),
                    internal_khz,
                    nb_subfr,
                ));
            }
            let (silk, rl, rr, _, _) = self.silk_stereo.as_mut().expect("configured");
            silk.set_bitrate(bitrate.clamp(5000, 100_000));
            let l16: Vec<i16> = pcm.iter().step_by(2).map(|&v| to_i16(v)).collect();
            let r16: Vec<i16> = pcm.iter().skip(1).step_by(2).map(|&v| to_i16(v)).collect();
            let out_len = per_ch * internal_khz as usize / 48;
            let (mut li, mut ri) = (vec![0i16; out_len], vec![0i16; out_len]);
            rl.process(&mut li, &l16);
            rr.process(&mut ri, &r16);
            let p = silk.encode(&li, &ri);
            self.last_final_range = silk.final_range();
            p
        };

        if payload.len() + 1 > max_bytes {
            return Err(EncodeError::InvalidBudget);
        }

        let config = config_base + lm;
        let toc = (config << 3) | (u8::from(self.channels == 2) << 2); // code 0
        let mut packet = Vec::with_capacity(payload.len() + 1);
        packet.push(toc);
        packet.extend_from_slice(&payload);
        Ok(packet)
    }

    /// Encodes one frame as a **hybrid** Opus packet (mono, 10/20 ms,
    /// super-wideband or fullband): SILK codes the wideband low band and CELT
    /// the high band (bands 17..end) in a single shared range coder. `pcm` is
    /// interleaved 48 kHz f32.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidFrameSize`] unless mono, a 10/20 ms frame, and
    /// the bandwidth is super-wideband or fullband; [`EncodeError::
    /// InvalidBudget`] if `max_bytes` is outside `3..=1275`.
    ///
    /// # Panics
    ///
    /// Panics if the coded packet does not fit the chosen byte budget (it does
    /// not for in-range input).
    pub fn encode_hybrid(&mut self, pcm: &[f32], max_bytes: usize) -> Result<Vec<u8>, EncodeError> {
        if self.channels != 1 {
            return Err(EncodeError::InvalidFrameSize);
        }
        let (frame_ms, lm) = match pcm.len() {
            480 => (10usize, 0u8),
            960 => (20, 1),
            _ => return Err(EncodeError::InvalidFrameSize),
        };
        let (config_base, celt_end) = match self.bandwidth {
            Bandwidth::SuperWideBand => (12u8, 19usize),
            Bandwidth::FullBand => (14, 21),
            _ => return Err(EncodeError::InvalidFrameSize),
        };
        if !(3..=1275).contains(&max_bytes) {
            return Err(EncodeError::InvalidBudget);
        }

        let nb_subfr = if frame_ms == 10 { 2 } else { 4 };
        // Total packet budget from the target bitrate (CBR-filled), and a
        // modest SILK share that leaves room for the CELT high band.
        let target = self.target_bitrate.map_or(32_000, |b| b as i32);
        let nb_bytes = ((target * frame_ms as i32 / 8000) as usize).clamp(20, max_bytes);
        let silk_bps = (target / 2).clamp(8_000, 20_000);

        // SILK low band: WB (16 kHz) resample, then write into the coder.
        let need_new = self
            .silk
            .as_ref()
            .is_none_or(|(_, _, khz, nbs)| *khz != 16 || *nbs != nb_subfr);
        if need_new {
            self.silk = Some((
                crate::silk::encode::api::SilkEncoder::new(16, nb_subfr),
                crate::silk::resampler::Resampler::new_enc(48_000, 16_000),
                16,
                nb_subfr,
            ));
        }
        let mut enc = RangeEncoder::new(nb_bytes);
        {
            let (silk, resampler, _, _) = self.silk.as_mut().expect("configured");
            silk.set_bitrate(silk_bps);
            let in16: Vec<i16> = pcm
                .iter()
                .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
                .collect();
            let mut internal = vec![0i16; pcm.len() / 3];
            resampler.process(&mut internal, &in16);
            silk.encode_into(&mut enc, &internal);
        }

        // Redundancy flag (no redundant CELT frame), coded when there is room.
        let total_bits = (nb_bytes * 8) as u32;
        if enc.tell() + 37 <= total_bits {
            enc.encode_bit_logp(false, 12);
        }

        // CELT high band into the same coder.
        self.celt.encode_hybrid_into(&mut enc, pcm, nb_bytes, celt_end);
        self.last_final_range = enc.range_size();
        let payload = enc.finalize().expect("hybrid packet fits");

        let config = config_base + lm;
        let toc = config << 3; // mono, code 0
        let mut packet = Vec::with_capacity(payload.len() + 1);
        packet.push(toc);
        packet.extend_from_slice(&payload);
        Ok(packet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpusDecoder;
    use alloc::vec::Vec;

    /// A SILK-mode Opus packet decodes through `OpusDecoder` with the final
    /// range matching (no redundancy misfire) and the output tracking the
    /// input, across narrowband/mediumband/wideband and several frame sizes.
    #[test]
    fn silk_packet_round_trips_through_the_opus_decoder() {
        for &(bw, spf) in &[
            (Bandwidth::WideBand, 960usize),
            (Bandwidth::WideBand, 1920),
            (Bandwidth::MediumBand, 960),
            (Bandwidth::NarrowBand, 960),
        ] {
            let mut enc = OpusEncoder::new(1);
            enc.set_bandwidth(bw);
            let mut dec = OpusDecoder::new(1);
            // A couple of frames so cross-frame state engages.
            let mut last_corr = 0.0f64;
            for f in 0..3 {
                let pcm: Vec<f32> = (0..spf)
                    .map(|i| {
                        let t = (f * spf + i) as f32 / 48_000.0;
                        0.4 * (2.0 * core::f32::consts::PI * 220.0 * t).sin()
                            + 0.2 * (2.0 * core::f32::consts::PI * 440.0 * t).sin()
                    })
                    .collect();
                let packet = enc.encode_silk(&pcm, 1275).expect("silk encode");
                let out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(out.len(), spf, "output length for {bw:?}");
                assert_eq!(dec.final_range(), enc.final_range(), "range mismatch {bw:?} frame {f}");
                // Delay-aligned correlation over a small offset search.
                last_corr = (0..200usize)
                    .map(|d| {
                        let (mut s, mut dot, mut e) = (0.0f64, 0.0f64, 0.0f64);
                        for i in 0..spf - d {
                            let a = f64::from(pcm[i]);
                            let b = f64::from(out[i + d]);
                            s += a * a;
                            dot += a * b;
                            e += b * b;
                        }
                        dot / (s.sqrt() * e.sqrt()).max(1e-9)
                    })
                    .fold(0.0f64, f64::max);
            }
            assert!(
                last_corr > 0.85,
                "{bw:?} reconstruction correlation {last_corr:.3} too low"
            );
        }
    }

    /// A stereo SILK-mode Opus stream decodes through `OpusDecoder` with the
    /// final range matching across the mid-only→side transition.
    #[test]
    fn silk_stereo_packet_round_trips_through_the_opus_decoder() {
        let spf = 960usize; // 20 ms at 48 kHz, per channel
        let mut enc = OpusEncoder::new(2);
        enc.set_bandwidth(Bandwidth::WideBand);
        let mut dec = OpusDecoder::new(2);
        let mut saw_mismatch = false;
        let mut last_corr = 0.0f64;
        for f in 0..60 {
            let mut pcm = Vec::with_capacity(spf * 2);
            for i in 0..spf {
                let t = (f * spf + i) as f32 / 48_000.0;
                let l = 0.4 * (2.0 * core::f32::consts::PI * 200.0 * t).sin();
                let r = 0.2 * (2.0 * core::f32::consts::PI * 200.0 * t + 0.3).sin()
                    + 0.3 * (2.0 * core::f32::consts::PI * 360.0 * t).sin();
                pcm.push(l);
                pcm.push(r);
            }
            let packet = enc.encode_silk(&pcm, 1275).expect("stereo silk encode");
            let out = dec.decode_packet(&packet).expect("decode");
            assert_eq!(out.len(), spf * 2, "stereo output length");
            if dec.final_range() != enc.final_range() {
                saw_mismatch = true;
            }
            // Left-channel correlation, delay-aligned.
            let dec_l: Vec<f32> = out.iter().step_by(2).copied().collect();
            let inp_l: Vec<f32> = pcm.iter().step_by(2).copied().collect();
            last_corr = (0..700usize)
                .map(|d| {
                    let (mut s, mut dot, mut e) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..spf - d {
                        let a = f64::from(inp_l[i]);
                        let b = f64::from(dec_l[i + d]);
                        s += a * a;
                        dot += a * b;
                        e += b * b;
                    }
                    dot / (s.sqrt() * e.sqrt()).max(1e-9)
                })
                .fold(0.0f64, f64::max);
        }
        // The bit-exact range match is the oracle; the correlation only
        // sanity-checks that audio comes out (the mid-dominant rate decision
        // collapses some stereo width, so it is well below 1).
        assert!(!saw_mismatch, "stereo SILK range mismatch through OpusDecoder");
        assert!(last_corr > 0.5, "stereo correlation {last_corr:.3} too low");
    }

    /// `encode_auto` dispatches to a valid mode for each frame size / bitrate
    /// and the packets decode through `OpusDecoder` with matching final range.
    #[test]
    fn encode_auto_dispatches_and_round_trips() {
        // (per-channel samples, bandwidth, target bitrate, expected mode:
        // 's' SILK, 'h' hybrid, 'c' CELT).
        let cases = [
            (240usize, Bandwidth::FullBand, None, 'c'),         // 5 ms → CELT
            (1920, Bandwidth::WideBand, Some(20_000u32), 's'),  // 40 ms → SILK
            (960, Bandwidth::WideBand, Some(16_000), 's'),      // 20 ms low WB → SILK
            (960, Bandwidth::SuperWideBand, Some(32_000), 'h'), // 20 ms SWB speech → hybrid
            (960, Bandwidth::FullBand, Some(64_000), 'c'),      // 20 ms FB high rate → CELT
            (480, Bandwidth::WideBand, None, 'c'),              // 10 ms no rate → CELT
        ];
        for (spf, bw, br, want_mode) in cases {
            let mut enc = OpusEncoder::new(1);
            enc.set_bandwidth(bw);
            enc.set_bitrate(br);
            let mut dec = OpusDecoder::new(1);
            let mut last_packet = None;
            for f in 0..4 {
                let pcm: Vec<f32> = (0..spf)
                    .map(|i| {
                        let t = (f * spf + i) as f32 / 48_000.0;
                        0.3 * (2.0 * core::f32::consts::PI * 300.0 * t).sin()
                    })
                    .collect();
                let packet = enc.encode_auto(&pcm, 1275).expect("encode_auto");
                let out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(out.len(), spf);
                assert_eq!(dec.final_range(), enc.final_range(), "range mismatch spf={spf}");
                last_packet = Some(packet);
            }
            // The TOC config picks the mode: SILK 0..12, hybrid 12..16, CELT 16+.
            let config = last_packet.unwrap()[0] >> 3;
            let mode = if config < 12 {
                's'
            } else if config < 16 {
                'h'
            } else {
                'c'
            };
            assert_eq!(mode, want_mode, "mode for spf={spf} bw={bw:?} br={br:?}");
        }
    }

    /// A hybrid Opus packet (SILK low band + CELT high band, one coder)
    /// decodes through `OpusDecoder` with the final range matching, for
    /// super-wideband and fullband.
    #[test]
    fn hybrid_packet_round_trips_through_the_opus_decoder() {
        for &bw in &[Bandwidth::SuperWideBand, Bandwidth::FullBand] {
            let mut enc = OpusEncoder::new(1);
            enc.set_bandwidth(bw);
            enc.set_bitrate(Some(32_000));
            let mut dec = OpusDecoder::new(1);
            let mut last_corr = 0.0f64;
            for f in 0..6 {
                let pcm: Vec<f32> = (0..960)
                    .map(|i| {
                        let t = (f * 960 + i) as f32 / 48_000.0;
                        0.3 * (2.0 * core::f32::consts::PI * 300.0 * t).sin()
                            + 0.15 * (2.0 * core::f32::consts::PI * 9000.0 * t).sin()
                    })
                    .collect();
                let packet = enc.encode_hybrid(&pcm, 1275).expect("hybrid encode");
                assert_eq!(packet[0] >> 3, if bw == Bandwidth::SuperWideBand { 13 } else { 15 });
                let out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(out.len(), 960);
                assert_eq!(
                    dec.final_range(),
                    enc.final_range(),
                    "hybrid range mismatch {bw:?} frame {f}"
                );
                last_corr = (0..700usize)
                    .map(|d| {
                        let (mut s, mut dot, mut e) = (0.0f64, 0.0f64, 0.0f64);
                        for i in 0..960 - d {
                            let a = f64::from(pcm[i]);
                            let b = f64::from(out[i + d]);
                            s += a * a;
                            dot += a * b;
                            e += b * b;
                        }
                        dot / (s.sqrt() * e.sqrt()).max(1e-9)
                    })
                    .fold(0.0f64, f64::max);
            }
            assert!(last_corr > 0.7, "{bw:?} hybrid correlation {last_corr:.3} too low");
        }
    }

    #[test]
    fn round_trips_through_the_decoder_at_every_frame_size() {
        for &(spf, channels) in &[(120usize, 1usize), (240, 1), (480, 1), (960, 1), (960, 2), (240, 2)] {
            let mut enc = OpusEncoder::new(channels);
            let mut dec = OpusDecoder::new(channels);
            for f in 0..30 {
                let mut pcm = Vec::with_capacity(spf * channels);
                for i in 0..spf {
                    let t = (f * spf + i) as f32 / 48_000.0;
                    let s = 0.5 * (2.0 * core::f32::consts::PI * 440.0 * t).sin();
                    for _ in 0..channels {
                        pcm.push(s);
                    }
                }
                let packet = enc.encode(&pcm, 96).expect("encode");
                let pcm_out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(pcm_out.len(), spf * channels);
                assert_eq!(
                    dec.final_range(),
                    enc.final_range(),
                    "range mismatch at spf={spf} ch={channels} frame {f}"
                );
            }
        }
    }

    #[test]
    fn round_trips_at_every_celt_bandwidth() {
        use crate::Bandwidth::{FullBand, NarrowBand, SuperWideBand, WideBand};
        for bw in [NarrowBand, WideBand, SuperWideBand, FullBand] {
            for channels in [1usize, 2] {
                let mut enc = OpusEncoder::new(channels);
                enc.set_bandwidth(bw);
                let mut dec = OpusDecoder::new(channels);
                for f in 0..20 {
                    let mut pcm = Vec::with_capacity(960 * channels);
                    for i in 0..960 {
                        let t = (f * 960 + i) as f32 / 48_000.0;
                        let s = 0.4 * (2.0 * core::f32::consts::PI * 300.0 * t).sin();
                        for _ in 0..channels {
                            pcm.push(s);
                        }
                    }
                    let packet = enc.encode(&pcm, 120).expect("encode");
                    let out = dec.decode_packet(&packet).expect("decode");
                    assert_eq!(out.len(), 960 * channels);
                    assert_eq!(
                        dec.final_range(),
                        enc.final_range(),
                        "range mismatch bw={bw:?} ch={channels} frame {f}"
                    );
                }
            }
        }
    }

    #[test]
    fn vbr_round_trips_and_tracks_the_target_rate() {
        for &target in &[48_000u32, 96_000, 160_000] {
            let mut enc = OpusEncoder::new(1);
            enc.set_bitrate(Some(target));
            let mut dec = OpusDecoder::new(1);
            let mut total = 0usize;
            let frames = 200;
            for f in 0..frames {
                let pcm: Vec<f32> = (0..960)
                    .map(|i| {
                        let t = (f * 960 + i) as f32 / 48_000.0;
                        0.5 * (2.0 * core::f32::consts::PI * 440.0 * t).sin()
                            + 0.2 * (2.0 * core::f32::consts::PI * 1800.0 * t).sin()
                    })
                    .collect();
                // The ceiling is generous; VBR shrinks each frame.
                let packet = enc.encode(&pcm, 1000).expect("encode");
                total += packet.len();
                dec.decode_packet(&packet).expect("decode");
                assert_eq!(
                    dec.final_range(),
                    enc.final_range(),
                    "range mismatch at {target} bps, frame {f}"
                );
            }
            // 50 frames/s × bytes/frame × 8 = bits/s. Allow ±25% (the simple
            // VBR omits the analysis-module terms).
            let achieved = (total as f64 / frames as f64) * 50.0 * 8.0;
            let ratio = achieved / f64::from(target);
            assert!(
                (0.75..1.25).contains(&ratio),
                "target {target}, achieved {achieved:.0} (ratio {ratio:.2})"
            );
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut enc = OpusEncoder::new(1);
        assert_eq!(enc.encode(&[0.0; 100], 64), Err(EncodeError::InvalidFrameSize));
        assert_eq!(enc.encode(&[0.0; 960], 2), Err(EncodeError::InvalidBudget));
        assert_eq!(enc.encode(&[0.0; 960], 2000), Err(EncodeError::InvalidBudget));
    }
}

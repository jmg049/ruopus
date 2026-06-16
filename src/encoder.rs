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
//! packet. Hybrid mode and automatic SILK/CELT selection build on top.

use alloc::vec::Vec;

use crate::celt::encoder::CeltEncoder;
use crate::packet::Bandwidth;

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

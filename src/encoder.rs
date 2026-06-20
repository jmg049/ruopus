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

/// Errors returned by the [`OpusEncoder`] encode methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// The frame had an unsupported number of samples per channel for the
    /// chosen mode (CELT: 120/240/480/960; SILK: 480/960/1920/2880; hybrid:
    /// 480/960 - all at 48 kHz), or the channel count did not match.
    InvalidFrameSize,
    /// The output budget is outside the usable range (at least 3 bytes, at
    /// most 1275 - the Opus per-frame limit), or the coded packet could not
    /// be made to fit it.
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

/// The SILK low-band target bitrate for a hybrid packet at total `rate` bps
/// (`compute_silk_rate_for_hybrid`): the reference's per-channel rate table
/// (no-FEC column - 10 ms and 20 ms coincide there), interpolated, with the
/// +300 bps super-wideband nudge. CELT codes the rest.
fn compute_silk_rate_for_hybrid(rate: i32, swb: bool, channels: i32) -> i32 {
    // (total per-channel bps, SILK bps).
    const TABLE: [(i32, i32); 7] = [
        (0, 0),
        (12_000, 10_000),
        (16_000, 13_500),
        (20_000, 16_000),
        (24_000, 18_000),
        (32_000, 22_000),
        (64_000, 38_000),
    ];
    let per_ch = rate / channels.max(1);
    let n = TABLE.len();
    let mut i = 1;
    while i < n && TABLE[i].0 <= per_ch {
        i += 1;
    }
    let mut silk_rate = if i == n {
        // Above the table: give 50% of the extra bits to SILK.
        TABLE[n - 1].1 + (per_ch - TABLE[n - 1].0) / 2
    } else {
        let (x0, lo) = TABLE[i - 1];
        let (x1, hi) = TABLE[i];
        (lo * (x1 - per_ch) + hi * (per_ch - x0)) / (x1 - x0)
    };
    if swb {
        silk_rate += 300;
    }
    silk_rate * channels.max(1)
}

/// A pure-Rust Opus encoder at 48 kHz, producing CELT, SILK (mono/stereo) and
/// hybrid packets; [`encode_auto`](Self::encode_auto) chooses the mode.
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
    /// Per-channel DC-reject high-pass filter state (`hp_mem`), carried across
    /// frames so the 3 Hz high-pass is continuous.
    hp_mem: [f32; 2],
    /// Discontinuous transmission (`OPUS_SET_DTX`): emit a TOC-only packet for
    /// inactive frames after a run of silence.
    use_dtx: bool,
    /// Consecutive milliseconds without activity, in Q1 (`nb_no_activity_ms_Q1`).
    dtx_no_activity_q1: i32,
    /// TOC byte of the last coded packet, reused for DTX packets so they carry
    /// the stream's current mode/frame-size/channels.
    last_toc: u8,
    /// Adaptive background-noise-floor estimate (mean square) for the DTX
    /// activity detector, so pauses are detected relative to the ambient level.
    dtx_noise_floor: f32,
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
            hp_mem: [0.0; 2],
            use_dtx: false,
            dtx_no_activity_q1: 0,
            last_toc: 0,
            dtx_noise_floor: 1.0,
        }
    }

    /// Voice-activity test for DTX: a frame is active when its mean-square
    /// energy rises clearly above the tracked background-noise floor. The floor
    /// follows the minimum energy with a slow upward leak, so DTX engages
    /// during pauses even when there is audible background noise (which a fixed
    /// -60 dBFS threshold would always call active). A small absolute floor
    /// still classifies near-digital-silence as inactive.
    fn frame_active(&mut self, pcm: &[f32]) -> bool {
        const ACTIVE_RATIO: f32 = 8.0; // ~9 dB above the noise floor
        const ABS_FLOOR: f32 = 1e-7; // mean square ≈ -70 dBFS
        if pcm.is_empty() {
            return false;
        }
        let ms = (pcm.iter().map(|&v| v * v).sum::<f32>() / pcm.len() as f32).max(0.0);
        // Track the minimum, leaking up ~0.03 %/frame so the floor follows a
        // rising ambient level rather than sticking at an old silence.
        self.dtx_noise_floor = if ms < self.dtx_noise_floor {
            ms
        } else {
            self.dtx_noise_floor * 1.0003
        };
        ms > (self.dtx_noise_floor * ACTIVE_RATIO).max(ABS_FLOOR)
    }

    /// Enables or disables discontinuous transmission (`OPUS_SET_DTX`). With
    /// DTX on, once the input has been inactive for 200 ms the encoder emits a
    /// 1-byte TOC-only packet for each further inactive frame (up to 400 ms,
    /// then one refresh frame), which the decoder conceals as comfort noise -
    /// dropping the silence bitrate to ~0.4 kb/s. Activity is judged by an
    /// adaptive energy detector that tracks the background-noise floor, so DTX
    /// engages during pauses even with audible ambient noise.
    pub const fn set_dtx(&mut self, on: bool) {
        self.use_dtx = on;
    }

    /// Decides whether to send a DTX (TOC-only) packet for a frame with the
    /// given activity, advancing the no-activity run (`decide_dtx_mode`).
    /// `frame_ms_q1` is twice the frame length in ms.
    fn decide_dtx(&mut self, active: bool, frame_ms_q1: i32) -> bool {
        const BEFORE_DTX_Q1: i32 = 10 * 20 * 2; // NB_SPEECH_FRAMES_BEFORE_DTX, 200 ms
        const MAX_DTX_Q1: i32 = (10 + 20) * 20 * 2; // + MAX_CONSECUTIVE_DTX, 600 ms
        if active {
            self.dtx_no_activity_q1 = 0;
            return false;
        }
        self.dtx_no_activity_q1 += frame_ms_q1;
        if self.dtx_no_activity_q1 > BEFORE_DTX_Q1 {
            if self.dtx_no_activity_q1 <= MAX_DTX_Q1 {
                return true;
            }
            // Cap the run: send one refresh frame, then resume DTX.
            self.dtx_no_activity_q1 = BEFORE_DTX_Q1;
        }
        false
    }

    /// libopus `dc_reject`: a 3 Hz one-pole high-pass on the interleaved
    /// 48 kHz input, run before every encode to strip DC and sub-audible
    /// rumble. The per-channel state (`hp_mem`) persists across frames.
    fn dc_reject(&mut self, pcm: &[f32]) -> Vec<f32> {
        const CUTOFF_HZ: f32 = 3.0;
        const COEF: f32 = 6.3 * CUTOFF_HZ / 48_000.0;
        let coef2 = 1.0 - COEF;
        let ch = self.channels;
        let mut out = vec![0.0f32; pcm.len()];
        for c in 0..ch {
            let mut m = self.hp_mem[c];
            let mut i = c;
            while i < pcm.len() {
                let x = pcm[i];
                out[i] = x - m;
                m = COEF * x + coef2 * m;
                i += ch;
            }
            self.hp_mem[c] = m;
        }
        out
    }

    /// Restricts the coded audio bandwidth (`OPUS_SET_BANDWIDTH`). The CELT
    /// modes support narrowband, wideband, super-wideband and fullband;
    /// mediumband is treated as wideband (CELT has no 6 kHz mode).
    pub const fn set_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.bandwidth = bandwidth;
    }

    /// Sets the target bitrate in bits/s (`OPUS_SET_BITRATE`). For CELT
    /// ([`encode`](Self::encode)) this selects VBR, treating `max_bytes` as a
    /// ceiling and shrinking each packet to its per-frame target (`None`
    /// restores CBR, filling `max_bytes`). For SILK/hybrid it sets the coding
    /// SNR, and it drives the mode choice in [`encode_auto`](Self::encode_auto).
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

        // DTX: for 10 ms+ frames, once the input has been inactive long enough,
        // send a 1-byte TOC-only packet (the decoder conceals it). Decided once
        // per call so the no-activity run advances correctly.
        if self.use_dtx {
            if per_ch >= 480 {
                let active = self.frame_active(pcm);
                let frame_ms_q1 = (per_ch as i32 * 2 * 1000) / 48_000;
                if self.decide_dtx(active, frame_ms_q1) && self.last_toc != 0 {
                    self.last_final_range = 0;
                    return Ok(alloc::vec![self.last_toc]);
                }
            } else {
                self.dtx_no_activity_q1 = 0; // short CELT frames break the run
            }
        }

        let packet = match per_ch {
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
                if wb_or_below && self.target_bitrate.is_some_and(|b| b <= 24_000) {
                    self.encode_silk(pcm, max_bytes)
                } else if swb_or_fb && self.target_bitrate.is_some_and(|b| b <= 40_000) {
                    // Fall back to CELT-only for a frame whose hybrid SILK low
                    // band cannot be squeezed under its byte share (a rare loud
                    // transient), so the "just works" path never fails.
                    self.encode_hybrid(pcm, max_bytes)
                        .or_else(|_| self.encode(pcm, max_bytes))
                } else {
                    self.encode(pcm, max_bytes)
                }
            },
            _ => return Err(EncodeError::InvalidFrameSize),
        }?;
        if let Some(&toc) = packet.first() {
            self.last_toc = toc;
        }
        Ok(packet)
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

        let pcm = self.dc_reject(pcm);
        let payload = self.celt.encode_frame_bw(&pcm, max_bytes - 1, end);
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

        let pcm = self.dc_reject(pcm);
        let pcm = pcm.as_slice();

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
            // Closed-loop rate control: fit the byte budget (less the TOC).
            let p = silk
                .encode_capped(&internal, max_bytes - 1)
                .ok_or(EncodeError::InvalidBudget)?;
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

    /// Encodes one frame as a **hybrid** Opus packet (mono or stereo, 10/20 ms,
    /// super-wideband or fullband): SILK codes the wideband low band and CELT
    /// the high band (bands 17..end) in a single shared range coder. `pcm` is
    /// interleaved 48 kHz f32.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidFrameSize`] unless a 10/20 ms frame and the
    /// bandwidth is super-wideband or fullband; [`EncodeError::InvalidBudget`]
    /// if `max_bytes` is outside `3..=1275`.
    ///
    /// # Panics
    ///
    /// Panics if the coded packet does not fit the chosen byte budget (it does
    /// not for in-range input).
    pub fn encode_hybrid(&mut self, pcm: &[f32], max_bytes: usize) -> Result<Vec<u8>, EncodeError> {
        if self.channels == 0 || pcm.len() % self.channels != 0 {
            return Err(EncodeError::InvalidFrameSize);
        }
        let per_ch = pcm.len() / self.channels;
        let (frame_ms, lm) = match per_ch {
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
        // Total packet budget from the target bitrate, the SILK low-band rate
        // (libopus's hybrid SILK/CELT split table), and the byte share SILK may
        // use so the CELT high band always has room (`celt_floor`, scaled by
        // the number of high bands it codes).
        let target = self.target_bitrate.map_or(32_000, |b| b as i32);
        let nb_bytes = ((target * frame_ms as i32 / 8000) as usize).clamp(20, max_bytes);
        let swb = matches!(self.bandwidth, Bandwidth::SuperWideBand);
        let silk_bps = compute_silk_rate_for_hybrid(target, swb, self.channels as i32);
        let celt_floor = (celt_end - 17) * 3 + 3;
        let silk_cap = nb_bytes.saturating_sub(celt_floor).max(8);
        let to_i16 = |v: f32| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
        let internal_len = per_ch / 3; // 48 kHz → 16 kHz

        let pcm = self.dc_reject(pcm);
        let pcm = pcm.as_slice();

        let mut enc = RangeEncoder::new(nb_bytes);
        if self.channels == 1 {
            // SILK low band: WB (16 kHz) resample, then write into the coder,
            // capped so the CELT high band keeps at least `celt_floor` bytes.
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
            let (silk, resampler, _, _) = self.silk.as_mut().expect("configured");
            silk.set_bitrate(silk_bps);
            let in16: Vec<i16> = pcm.iter().map(|&v| to_i16(v)).collect();
            let mut internal = vec![0i16; internal_len];
            resampler.process(&mut internal, &in16);
            // Hard bit cap so the SILK low band leaves the CELT high band room.
            silk.encode_into(&mut enc, &internal, Some((silk_cap * 8) as i32));
        } else {
            // Stereo SILK low band: deinterleave, resample each channel to WB.
            let need_new = self
                .silk_stereo
                .as_ref()
                .is_none_or(|(_, _, _, khz, nbs)| *khz != 16 || *nbs != nb_subfr);
            if need_new {
                self.silk_stereo = Some((
                    crate::silk::encode::api::SilkStereoEncoder::new(16, nb_subfr),
                    crate::silk::resampler::Resampler::new_enc(48_000, 16_000),
                    crate::silk::resampler::Resampler::new_enc(48_000, 16_000),
                    16,
                    nb_subfr,
                ));
            }
            let (silk, rl, rr, _, _) = self.silk_stereo.as_mut().expect("configured");
            silk.set_bitrate(silk_bps);
            let l16: Vec<i16> = pcm.iter().step_by(2).map(|&v| to_i16(v)).collect();
            let r16: Vec<i16> = pcm.iter().skip(1).step_by(2).map(|&v| to_i16(v)).collect();
            let (mut li, mut ri) = (vec![0i16; internal_len], vec![0i16; internal_len]);
            rl.process(&mut li, &l16);
            rr.process(&mut ri, &r16);
            // Hard bit cap so the stereo SILK low band leaves the CELT high
            // band room (mid + side share the cumulative budget).
            silk.encode_into(&mut enc, &li, &ri, Some((silk_cap * 8) as i32));
        }

        // Redundancy flag (no redundant CELT frame), coded when there is room.
        let total_bits = (nb_bytes * 8) as u32;
        if enc.tell() + 37 <= total_bits {
            enc.encode_bit_logp(false, 12);
        }

        // CELT high band into the same coder.
        self.celt.encode_hybrid_into(&mut enc, pcm, nb_bytes, celt_end);
        self.last_final_range = enc.range_size();
        // A loud frame whose SILK low band overruns its share can leave the
        // CELT high band no room within `nb_bytes`; surface that as a budget
        // error rather than panicking (the hybrid SILK/CELT rate split is not
        // yet adaptive - see the encoder notes).
        let payload = enc.finalize().map_err(|_| EncodeError::InvalidBudget)?;

        let config = config_base + lm;
        let toc = (config << 3) | (u8::from(self.channels == 2) << 2); // code 0
        let mut packet = Vec::with_capacity(payload.len() + 1);
        packet.push(toc);
        packet.extend_from_slice(&payload);
        Ok(packet)
    }
}

/// Encodes interleaved 48 kHz f32 PCM into a complete **Ogg Opus** file
/// (RFC 7845): the `OpusHead`/`OpusTags` headers followed by the audio page
/// stream, fullband and 20 ms per packet via [`OpusEncoder::encode_auto`].
/// `channels` is 1 or 2; `bitrate` is the target in bits/s. The final partial
/// frame is zero-padded. This is the symmetric counterpart to
/// [`decode_ogg_opus`](crate::decode_ogg_opus) - together they let the codec
/// read and write standard `.opus` files.
///
/// `pre_skip` matches the reconstruction delay of the mode fullband
/// `encode_auto` selects for `bitrate` - 120 samples for CELT (> 40 kb/s),
/// 69 for hybrid (≤ 40 kb/s) - so the decoder trims the warm-up and the output
/// aligns with the input (verified at zero lag against ffmpeg/libopus).
///
/// # Panics
///
/// Panics if `channels` is not 1 or 2, or `pcm.len()` is not a multiple of it.
#[must_use]
pub fn encode_ogg_opus(pcm: &[f32], channels: usize, bitrate: u32) -> Vec<u8> {
    use crate::ogg::{OggOpusWriter, OpusHead, OpusTags};

    assert!(channels == 1 || channels == 2, "channels must be 1 or 2");
    assert!(pcm.len() % channels == 0, "pcm length must be a whole number of frames");

    const FRAME: usize = 960; // 20 ms at 48 kHz
    // Fullband encode_auto picks hybrid at ≤ 40 kb/s (SILK delay, 69) and CELT
    // above (MDCT overlap, 120).
    let pre_skip: u16 = if bitrate > 40_000 { 120 } else { 69 };

    let head = OpusHead::family0(channels as u8, pre_skip, 48_000);
    let tags = OpusTags {
        vendor: b"opus_native".to_vec(),
        comments: Vec::new(),
    };
    let mut writer = OggOpusWriter::new(&head, &tags, 1);

    let per_ch = pcm.len() / channels;
    if per_ch == 0 {
        return writer.finish();
    }

    let mut enc = OpusEncoder::new(channels);
    enc.set_bandwidth(Bandwidth::FullBand);
    enc.set_bitrate(Some(bitrate));

    let frame_samples = FRAME * channels;
    let n_frames = per_ch.div_ceil(FRAME);
    for f in 0..n_frames {
        let start = f * frame_samples;
        let end = (start + frame_samples).min(pcm.len());
        let mut frame = pcm[start..end].to_vec();
        frame.resize(frame_samples, 0.0); // zero-pad the final partial frame
        let packet = enc
            .encode_auto(&frame, 1275)
            .expect("encode_auto produces a packet for a 20 ms frame");
        writer.push(&packet, f + 1 == n_frames);
    }
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OpusDecoder, decode_ogg_opus};
    use alloc::vec::Vec;

    /// With DTX enabled, a stretch of silence after speech produces 1-byte
    /// TOC-only packets (after the 200 ms activity hangover), the decoder
    /// conceals them into finite output of the right length, and the same
    /// silence costs far fewer bytes than with DTX off.
    #[test]
    fn dtx_collapses_the_silence_bitrate() {
        // (use_dtx) -> (total silence bytes, DTX-packet count)
        let run = |use_dtx: bool| {
            let mut enc = OpusEncoder::new(1);
            enc.set_bandwidth(Bandwidth::WideBand);
            enc.set_bitrate(Some(16_000));
            enc.set_dtx(use_dtx);
            let mut dec = OpusDecoder::new(1);
            let (mut silence_bytes, mut dtx_packets) = (0usize, 0usize);
            for f in 0..100 {
                let active = f < 5;
                let pcm: Vec<f32> = (0..960)
                    .map(|i| {
                        if active {
                            let t = (f * 960 + i) as f32 / 48_000.0;
                            0.3 * (2.0 * core::f32::consts::PI * 300.0 * t).sin()
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let packet = enc.encode_auto(&pcm, 1275).expect("encode");
                if !active {
                    silence_bytes += packet.len();
                }
                if packet.len() == 1 {
                    dtx_packets += 1;
                }
                let out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(out.len(), 960);
                assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
            }
            (silence_bytes, dtx_packets)
        };

        let (dtx_silence, dtx_packets) = run(true);
        let (plain_silence, plain_packets) = run(false);

        assert_eq!(plain_packets, 0, "DTX off must never emit TOC-only packets");
        assert!(
            dtx_packets >= 15,
            "expected DTX packets during silence, got {dtx_packets}"
        );
        assert!(
            dtx_silence * 2 < plain_silence,
            "DTX silence {dtx_silence} B should be well under half of plain {plain_silence} B"
        );
    }

    /// The adaptive activity detector engages DTX during a pause that still has
    /// audible background noise (mean square well above the old fixed
    /// -60 dBFS threshold), which the previous fixed detector would have called
    /// active. The noise floor tracks the ambient level, so the pause reads as
    /// inactive.
    #[test]
    fn dtx_engages_during_a_noisy_pause() {
        let mut seed = 0x2468_1357u32;
        let mut noise = move |amp: f32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((seed >> 9) as f32 / f32::from(u16::MAX) - 0.5) * amp
        };
        let mut enc = OpusEncoder::new(1);
        enc.set_bandwidth(Bandwidth::WideBand);
        enc.set_bitrate(Some(16_000));
        enc.set_dtx(true);
        let mut dec = OpusDecoder::new(1);

        let mut dtx_packets = 0;
        for f in 0..80 {
            // 0..10: speech; 10..80: pause with ~-44 dBFS background noise
            // (mean square ≈ 1.3e-5, far above the old 1e-6 fixed threshold).
            let pcm: Vec<f32> = (0..960)
                .map(|i| {
                    if f < 10 {
                        let t = (f * 960 + i) as f32 / 48_000.0;
                        0.3 * (2.0 * core::f32::consts::PI * 300.0 * t).sin() + noise(0.02)
                    } else {
                        noise(0.012)
                    }
                })
                .collect();
            let packet = enc.encode_auto(&pcm, 1275).expect("encode");
            if packet.len() == 1 {
                dtx_packets += 1;
            }
            let out = dec.decode_packet(&packet).expect("decode");
            assert!(out.iter().all(|v| v.is_finite()));
        }
        assert!(
            dtx_packets >= 15,
            "expected DTX during the noisy pause, got {dtx_packets}"
        );
    }

    /// `encode_ogg_opus` produces a valid Ogg Opus file that `decode_ogg_opus`
    /// reads back: the header round-trips, the decoded length matches the input
    /// (within the pre-skip-trimmed tail), and the audio is strongly correlated
    /// with the input (delay-aligned by the 120-sample pre-skip).
    #[test]
    fn ogg_opus_file_round_trips() {
        // (bitrate, expected pre_skip) - CELT above 40 kb/s, hybrid below.
        for &(bitrate, want_pre_skip) in &[(64_000u32, 120u16), (32_000, 69)] {
            for &channels in &[1usize, 2] {
                // ~0.5 s of a tone, a whole number of 20 ms frames.
                let per_ch = 960 * 25;
                let mut pcm = Vec::with_capacity(per_ch * channels);
                for i in 0..per_ch {
                    let t = i as f32 / 48_000.0;
                    let s = 0.4 * (2.0 * core::f32::consts::PI * 440.0 * t).sin();
                    for _ in 0..channels {
                        pcm.push(s);
                    }
                }

                let file = encode_ogg_opus(&pcm, channels, bitrate);
                let (out, head) = decode_ogg_opus(&file).expect("decode the encoded file");
                assert_eq!(usize::from(head.channel_count), channels);
                assert_eq!(head.pre_skip, want_pre_skip, "br={bitrate}");

                // Output covers the input minus at most the codec tail delay.
                let out_per_ch = out.len() / channels;
                assert!(
                    out_per_ch >= per_ch - 960 && out_per_ch <= per_ch + 960,
                    "br={bitrate} ch={channels}: decoded {out_per_ch} samples/ch vs input {per_ch}"
                );

                // Correlation on the first channel (input vs delay-aligned output).
                let n = out_per_ch.min(per_ch) - 480;
                let (mut sig, mut dot, mut energy) = (0.0f64, 0.0f64, 0.0f64);
                for i in 0..n {
                    let a = f64::from(pcm[(480 + i) * channels]);
                    let b = f64::from(out[(480 + i) * channels]);
                    sig += a * a;
                    dot += a * b;
                    energy += b * b;
                }
                let corr = dot / (sig.sqrt() * energy.sqrt()).max(1e-9);
                assert!(
                    corr > 0.9,
                    "br={bitrate} ch={channels}: round-trip correlation {corr:.3} too low"
                );
            }
        }
    }

    /// The DC-reject high-pass strips a constant input: a pure-DC signal of
    /// 0.3 decodes to a near-zero mean (the 3 Hz high-pass removes it before
    /// coding), rather than reproducing the offset.
    #[test]
    fn dc_reject_removes_a_constant_offset() {
        let mut enc = OpusEncoder::new(1);
        enc.set_bandwidth(Bandwidth::WideBand);
        enc.set_bitrate(Some(20_000));
        let mut dec = OpusDecoder::new(1);
        let mut mean = 0.0f64;
        for _ in 0..12 {
            let pcm = alloc::vec![0.3f32; 960];
            let pkt = enc.encode_silk(&pcm, 1275).expect("encode");
            let out = dec.decode_packet(&pkt).expect("decode");
            mean = out.iter().map(|&v| f64::from(v)).sum::<f64>() / out.len() as f64;
            assert_eq!(dec.final_range(), enc.final_range());
        }
        assert!(mean.abs() < 0.03, "DC not rejected: residual mean {mean:.4}");
    }

    /// Exercise the full encode surface with pathological signals - silence,
    /// DC, full-scale, impulses, white-ish noise, decorrelated stereo - across
    /// every mode, both channel counts, and a range of frame sizes and rates.
    /// Every packet must encode without panicking and round-trip through
    /// `OpusDecoder` finishing on the encoder's exact range state. This guards
    /// the arithmetic-edge bug class (e.g. voiced-only index underflows that a
    /// simple sine never reaches).
    #[test]
    fn stress_pathological_signals_round_trip_in_every_mode() {
        // Deterministic LCG so the "noise" is reproducible without rand.
        let mut seed = 0x1234_5678u32;
        let mut noise = move || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (seed >> 9) as f32 / f32::from(u16::MAX) - 0.5
        };

        // (name, generator: |channel, global sample index| -> sample)
        type Gen = fn(usize, usize) -> f32;
        let kinds: [(&str, Gen); 6] = [
            ("silence", |_, _| 0.0),
            ("dc", |_, _| 0.4),
            ("full_scale", |_, i| if i % 2 == 0 { 0.999 } else { -0.999 }),
            ("impulses", |_, i| if i % 137 == 0 { 0.95 } else { 0.0 }),
            ("tone", |ch, i| {
                let t = i as f32 / 48_000.0;
                let ph = if ch == 1 { 0.7 } else { 0.0 };
                0.5 * (2.0 * core::f32::consts::PI * 220.0 * t + ph).sin()
            }),
            // placeholder; real noise injected below via the closure
            ("noise", |_, _| 0.0),
        ];

        for channels in [1usize, 2] {
            for &(spf, bw, rate) in &[
                (480usize, Bandwidth::WideBand, 16_000u32), // SILK 10 ms
                (960, Bandwidth::WideBand, 16_000),         // SILK 20 ms
                (1920, Bandwidth::NarrowBand, 16_000),      // SILK 40 ms
                (480, Bandwidth::FullBand, 32_000),         // hybrid 10 ms
                (960, Bandwidth::SuperWideBand, 32_000),    // hybrid 20 ms
                (240, Bandwidth::FullBand, 64_000),         // CELT 5 ms
            ] {
                for (name, make) in &kinds {
                    let mut enc = OpusEncoder::new(channels);
                    enc.set_bandwidth(bw);
                    enc.set_bitrate(Some(rate));
                    let mut dec = OpusDecoder::new(channels);
                    for f in 0..4 {
                        let mut pcm = Vec::with_capacity(spf * channels);
                        for i in 0..spf {
                            let gi = f * spf + i;
                            for ch in 0..channels {
                                let s = if *name == "noise" { noise() } else { make(ch, gi) };
                                pcm.push(s);
                            }
                        }
                        let packet = enc
                            .encode_auto(&pcm, 1275)
                            .unwrap_or_else(|e| panic!("{name} ch={channels} spf={spf} bw={bw:?}: {e:?}"));
                        let out = dec
                            .decode_packet(&packet)
                            .unwrap_or_else(|e| panic!("{name} decode ch={channels} spf={spf}: {e:?}"));
                        assert_eq!(out.len(), spf * channels);
                        assert_eq!(
                            dec.final_range(),
                            enc.final_range(),
                            "range mismatch {name} ch={channels} spf={spf} bw={bw:?} frame {f}"
                        );
                    }
                }
            }
        }
    }

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

    /// A loud signal in hybrid mode at a very tight budget makes the SILK low
    /// band overrun the CELT high band's share. The coarse-energy quantiser
    /// then sees the range coder already past `budget` - which must not panic
    /// (it once underflowed an unsigned `budget - tell`); the call returns a
    /// budget error or a valid packet, never crashes, and any packet produced
    /// round-trips through `OpusDecoder` with the matching range.
    #[test]
    fn hybrid_tight_budget_does_not_panic_on_overrun() {
        let mut seed = 0xC0FF_EE11u32;
        for &bw in &[Bandwidth::SuperWideBand, Bandwidth::FullBand] {
            let mut enc = OpusEncoder::new(1);
            enc.set_bandwidth(bw);
            enc.set_bitrate(Some(8_000)); // forces a tiny hybrid byte budget
            let mut dec = OpusDecoder::new(1);
            for _ in 0..6 {
                // Loud, broadband noise: large SILK low band, stresses the split.
                let pcm: Vec<f32> = (0..960)
                    .map(|_| {
                        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                        ((seed >> 9) as f32 / f32::from(u16::MAX) - 0.5) * 1.9
                    })
                    .collect();
                if let Ok(packet) = enc.encode_hybrid(&pcm, 1275) {
                    let out = dec.decode_packet(&packet).expect("decode");
                    assert_eq!(out.len(), 960);
                    assert_eq!(dec.final_range(), enc.final_range(), "range mismatch {bw:?}");
                }
            }
        }
    }

    /// `encode_auto` at hybrid settings never fails on a codeable frame: a
    /// loud transient whose SILK low band overruns its byte share falls back to
    /// CELT-only, and every packet round-trips through `OpusDecoder` with the
    /// matching range (mode may switch per frame via the TOC).
    #[test]
    fn encode_auto_hybrid_falls_back_to_celt_without_failing() {
        let mut seed = 0x5151_2323u32;
        let mut enc = OpusEncoder::new(1);
        enc.set_bandwidth(Bandwidth::FullBand);
        enc.set_bitrate(Some(32_000)); // routes 20 ms FB to hybrid
        let mut dec = OpusDecoder::new(1);
        for f in 0..20 {
            // Alternate quiet tones with loud broadband bursts (overruns SILK).
            let loud = f % 4 == 3;
            let pcm: Vec<f32> = (0..960)
                .map(|i| {
                    if loud {
                        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                        ((seed >> 9) as f32 / f32::from(u16::MAX) - 0.5) * 1.95
                    } else {
                        let t = (f * 960 + i) as f32 / 48_000.0;
                        0.25 * (2.0 * core::f32::consts::PI * 400.0 * t).sin()
                    }
                })
                .collect();
            let packet = enc.encode_auto(&pcm, 1275).expect("encode_auto never fails");
            let out = dec.decode_packet(&packet).expect("decode");
            assert_eq!(out.len(), 960);
            assert_eq!(dec.final_range(), enc.final_range(), "range mismatch frame {f}");
        }
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

    /// SILK closed-loop rate control: a tight byte budget that the default
    /// quality would overshoot is honoured by lowering the rate, and the
    /// capped packet still decodes through `OpusDecoder` with matching range.
    #[test]
    fn silk_rate_control_respects_a_tight_budget() {
        let mut enc = OpusEncoder::new(1);
        enc.set_bandwidth(Bandwidth::WideBand);
        enc.set_bitrate(Some(40_000)); // high target...
        let mut dec = OpusDecoder::new(1);
        for f in 0..8 {
            // A rich signal that codes large at 40 kbps.
            let pcm: Vec<f32> = (0..960)
                .map(|i| {
                    let t = (f * 960 + i) as f32 / 48_000.0;
                    0.4 * (2.0 * core::f32::consts::PI * 300.0 * t).sin()
                        + 0.3 * (2.0 * core::f32::consts::PI * 1700.0 * t).sin()
                        + 0.2 * (2.0 * core::f32::consts::PI * 5300.0 * t).sin()
                })
                .collect();
            // ...but cap the packet well below what 40 kbps would produce.
            let packet = enc.encode_silk(&pcm, 90).expect("rate-controlled encode");
            assert!(packet.len() <= 90, "packet {} exceeds the 90-byte cap", packet.len());
            let out = dec.decode_packet(&packet).expect("decode");
            assert_eq!(out.len(), 960);
            assert_eq!(dec.final_range(), enc.final_range(), "range mismatch frame {f}");
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

    /// A stereo hybrid Opus packet (SILK stereo low band + CELT stereo high
    /// band sharing one coder) decodes through `OpusDecoder` with the final
    /// range matching, for super-wideband and fullband.
    #[test]
    fn stereo_hybrid_packet_round_trips_through_the_opus_decoder() {
        for &bw in &[Bandwidth::SuperWideBand, Bandwidth::FullBand] {
            let mut enc = OpusEncoder::new(2);
            enc.set_bandwidth(bw);
            enc.set_bitrate(Some(48_000));
            let mut dec = OpusDecoder::new(2);
            for f in 0..6 {
                let mut pcm = Vec::with_capacity(960 * 2);
                for i in 0..960 {
                    let t = (f * 960 + i) as f32 / 48_000.0;
                    // Slightly decorrelated channels (phase-shifted) to exercise
                    // the side band.
                    let l = 0.3 * (2.0 * core::f32::consts::PI * 300.0 * t).sin()
                        + 0.15 * (2.0 * core::f32::consts::PI * 9000.0 * t).sin();
                    let r = 0.3 * (2.0 * core::f32::consts::PI * 300.0 * t + 0.5).sin()
                        + 0.15 * (2.0 * core::f32::consts::PI * 9000.0 * t).sin();
                    pcm.push(l);
                    pcm.push(r);
                }
                let packet = enc.encode_hybrid(&pcm, 1275).expect("stereo hybrid encode");
                let config = packet[0] >> 3;
                let stereo = (packet[0] >> 2) & 1;
                assert_eq!(config, if bw == Bandwidth::SuperWideBand { 13 } else { 15 });
                assert_eq!(stereo, 1, "stereo bit set");
                let out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(out.len(), 960 * 2);
                assert_eq!(
                    dec.final_range(),
                    enc.final_range(),
                    "stereo hybrid range mismatch {bw:?} frame {f}"
                );
            }
        }
    }

    /// Stereo hybrid stays within budget: the cumulative `max_bits` cap (mid
    /// then side) keeps the combined SILK low band under its share so the CELT
    /// high band always fits. On a busy speech-like stereo signal (voiced
    /// tones + noise under a syllabic envelope) every frame encodes with the
    /// cap; without it the SILK low band overruns and several frames fail. All
    /// packets round-trip through `OpusDecoder` with the matching range.
    #[test]
    fn stereo_hybrid_stays_within_budget_on_busy_speech() {
        let mut seed = 0x1234_5678u32;
        let mut enc = OpusEncoder::new(2);
        enc.set_bandwidth(Bandwidth::SuperWideBand);
        enc.set_bitrate(Some(24_000));
        let mut dec = OpusDecoder::new(2);
        let total = 60;
        for f in 0..total {
            let env = 0.4 + 0.55 * (2.0 * core::f32::consts::PI * f as f32 / 7.0).sin().abs();
            let mut pcm = Vec::with_capacity(960 * 2);
            for i in 0..960 {
                let t = (f * 960 + i) as f32 / 48_000.0;
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let n = ((seed >> 9) as f32 / f32::from(u16::MAX) - 0.5) * 0.25;
                let s = env
                    * (0.6 * (2.0 * core::f32::consts::PI * 220.0 * t).sin()
                        + 0.3 * (2.0 * core::f32::consts::PI * 1400.0 * t).sin()
                        + 0.2 * (2.0 * core::f32::consts::PI * 5200.0 * t).sin())
                    + n;
                pcm.push(s);
                pcm.push(s * 0.9 + n * 0.5);
            }
            let packet = enc
                .encode_hybrid(&pcm, 1275)
                .unwrap_or_else(|e| panic!("stereo hybrid frame {f} overran the budget: {e:?}"));
            let out = dec.decode_packet(&packet).expect("decode");
            assert_eq!(out.len(), 960 * 2);
            assert_eq!(dec.final_range(), enc.final_range(), "range mismatch frame {f}");
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

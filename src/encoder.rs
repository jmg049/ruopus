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
//! fullband), and [`OpusEncoder::encode_auto`] picks SILK / hybrid / CELT per
//! frame from a signal analysis ([`crate::encoder_analysis`]) plus the
//! [`Signal`] hint and [`Application`] profile, with cross-frame hysteresis and
//! automatic bandwidth selection.

use alloc::vec::Vec;

use crate::celt::encoder::CeltEncoder;
use crate::encoder_analysis::{FrameAnalysis, analyze_frame};
use crate::packet::Bandwidth;
use crate::range::RangeEncoder;

/// Signal-content hint for the automatic mode decision (`OPUS_SET_SIGNAL`).
///
/// Biases [`OpusEncoder::encode_auto`]'s speech-vs-music classification: the
/// analysis still runs, but a non-`Auto` hint shifts the decision threshold so
/// the encoder trusts the caller's knowledge of the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Signal {
    /// Let the analysis decide (the default, `OPUS_AUTO`).
    #[default]
    Auto,
    /// The source is speech: bias toward SILK / hybrid.
    Voice,
    /// The source is general audio / music: bias toward CELT.
    Music,
}

/// Coding application / latency profile (`OPUS_SET_APPLICATION`).
///
/// Shapes the mode decision the way libopus's three application presets do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Application {
    /// VoIP: optimise for speech intelligibility - bias toward SILK / hybrid
    /// (`OPUS_APPLICATION_VOIP`).
    Voip,
    /// General audio / music, the balanced default
    /// (`OPUS_APPLICATION_AUDIO`).
    #[default]
    Audio,
    /// Restricted low delay: never use SILK (which adds algorithmic delay), so
    /// every frame is coded CELT-only (`OPUS_APPLICATION_RESTRICTED_LOWDELAY`).
    RestrictedLowDelay,
}

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

/// The mode [`OpusEncoder::encode_auto`]'s decision selects for a 10/20 ms
/// frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChosenMode {
    /// SILK-only (narrowband through wideband speech).
    Silk,
    /// Hybrid (SILK low band + CELT high band, super-wideband/fullband speech).
    Hybrid,
    /// CELT-only (music / low delay).
    Celt,
}

/// The SILK low-band target bitrate for a hybrid packet at total `rate` bps:
/// a per-channel rate table
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
/// use opus_rs::{OpusEncoder, OpusDecoder};
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
        crate::silk::encode::resample_in::EncDownsampler,
        i32,
        usize,
    )>,
    /// Stereo SILK encoder + per-channel input resamplers (created lazily).
    silk_stereo: Option<(
        crate::silk::encode::api::SilkStereoEncoder,
        crate::silk::encode::resample_in::EncDownsampler,
        crate::silk::encode::resample_in::EncDownsampler,
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
    /// Encode complexity 0-10 (`OPUS_SET_COMPLEXITY`); trades analysis depth for
    /// speed. Drives the SILK pitch/shaping
    /// settings and the CELT prefilter/tf/spreading gates.
    complexity: u8,
    /// Variable bitrate (`OPUS_SET_VBR`). When `true` (the default) a set
    /// bitrate is a VBR target; when `false` the CELT path codes constant
    /// bitrate - a fixed byte count per frame at the target rate.
    vbr: bool,
    /// Forced coded channel count (`OPUS_SET_FORCE_CHANNELS`). `None` (the
    /// default, `OPUS_AUTO`) codes the configured channel count. `Some(1)` on a
    /// stereo encoder downmixes the stereo input to mono and codes a mono
    /// packet; forcing 2 on a mono encoder has no effect (mono can only ever
    /// code mono). Only the 2→1 downmix changes behaviour.
    force_channels: Option<usize>,
    /// Signal-content hint (`OPUS_SET_SIGNAL`): biases the mode decision toward
    /// SILK (`Voice`) or CELT (`Music`); `Auto` lets the analysis decide.
    signal: Signal,
    /// Application / latency profile (`OPUS_SET_APPLICATION`).
    application: Application,
    /// Caller-imposed ceiling on the automatic bandwidth selection
    /// (`OPUS_SET_MAX_BANDWIDTH`). Auto-selected bandwidths are clamped to this.
    max_bandwidth: Bandwidth,
    /// Whether the caller forced an explicit bandwidth via
    /// [`set_bandwidth`](Self::set_bandwidth). When `false`, `encode_auto`
    /// chooses the bandwidth from the analysis (clamped to `max_bandwidth`);
    /// when `true`, `self.bandwidth` is honoured as-is. The default is `false`
    /// (automatic).
    bandwidth_forced: bool,
    /// Smoothed music probability for mode hysteresis: the per-frame analysis
    /// probability is low-pass filtered into this so the chosen mode does not
    /// flip on a single borderline frame.
    mode_music_smooth: f32,
    /// In-band FEC (`OPUS_SET_INBAND_FEC`): generate redundant LBRR copies in
    /// SILK packets so a lost frame can be recovered from the next packet.
    use_inband_fec: bool,
    /// Expected packet loss percentage 0-100 (`OPUS_SET_PACKET_LOSS_PERC`).
    packet_loss_perc: u8,
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
            complexity: 10,
            vbr: true,
            force_channels: None,
            signal: Signal::Auto,
            application: Application::Audio,
            max_bandwidth: Bandwidth::FullBand,
            bandwidth_forced: false,
            mode_music_smooth: 0.5,
            use_inband_fec: false,
            packet_loss_perc: 0,
        }
    }

    /// Resets the encoder to its freshly-created state (`OPUS_RESET_STATE`),
    /// keeping the configured channels, bandwidth, bitrate and complexity but
    /// dropping all cross-frame history (the SILK/CELT state, the high-pass
    /// filter memory, the DTX activity run and the range oracle), so the next
    /// packet is coded as if it were the first.
    pub fn reset(&mut self) {
        self.celt = CeltEncoder::with_channels(self.coded_channels());
        self.celt.set_complexity(self.complexity);
        self.celt.set_target_bitrate(self.target_bitrate);
        self.silk = None;
        self.silk_stereo = None;
        self.last_final_range = 0;
        self.hp_mem = [0.0; 2];
        // `use_dtx`, `channels`, `bandwidth`, `target_bitrate`, `complexity`
        // are configuration and are preserved across a reset.
        self.dtx_no_activity_q1 = 0;
        self.last_toc = 0;
        self.dtx_noise_floor = 1.0;
        // `signal`, `application`, `max_bandwidth`, `bandwidth_forced` are
        // configuration and survive a reset; the hysteresis state is history.
        self.mode_music_smooth = 0.5;
    }

    /// The configured number of channels (1 or 2).
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// The number of channels actually coded into the packet. Equals
    /// [`channels`](Self::channels) unless `force_channels` requests fewer (a
    /// stereo encoder forced to mono codes 1), which is the only direction that
    /// changes behaviour - mono can never code stereo.
    #[must_use]
    const fn coded_channels(&self) -> usize {
        match self.force_channels {
            Some(fc) if fc < self.channels => fc,
            _ => self.channels,
        }
    }

    /// Forces the coded channel count (`OPUS_SET_FORCE_CHANNELS`). `None`
    /// (`OPUS_AUTO`, the default) codes the configured channels. `Some(1)` on a
    /// stereo encoder downmixes the stereo input to mono and codes mono packets
    /// (a mono TOC); the configured channel count and the input layout are
    /// unchanged - only the coded packet becomes mono. `Some(2)` on a mono
    /// encoder is a no-op (mono can only code mono). Switching the coded count
    /// rebuilds the CELT coder and drops the lazily-built SILK encoders, so the
    /// next packet is coded fresh for the new channel count.
    ///
    /// # Panics
    ///
    /// Panics if `Some(n)` with `n` not 1 or 2.
    pub fn set_force_channels(&mut self, force: Option<usize>) {
        if let Some(n) = force {
            assert!(n == 1 || n == 2, "force_channels must be 1 or 2");
        }
        let before = self.coded_channels();
        self.force_channels = force;
        let after = self.coded_channels();
        if after != before {
            // The coded channel count changed: rebuild the CELT coder for it and
            // drop the SILK encoders (they are keyed off the coded channels and
            // built lazily on the next encode).
            self.celt = CeltEncoder::with_channels(after);
            self.celt.set_complexity(self.complexity);
            self.celt.set_target_bitrate(self.target_bitrate);
            self.silk = None;
            self.silk_stereo = None;
        }
    }

    /// The forced coded channel count, or `None` for automatic
    /// (`OPUS_GET_FORCE_CHANNELS`).
    #[must_use]
    pub const fn force_channels(&self) -> Option<usize> {
        self.force_channels
    }

    /// High-pass-filters the interleaved input (keyed off the configured
    /// channels) and returns it laid out for the **coded** channel count: when
    /// forcing a stereo input to mono this averages the two channels into a
    /// single mono stream (matching libopus's `(l + r) * 0.5` downmix). The DC
    /// reject runs on the original channels first so its per-channel filter
    /// memory stays continuous.
    fn prepare_input(&mut self, pcm: &[f32]) -> Vec<f32> {
        let filtered = self.dc_reject(pcm);
        if self.coded_channels() == self.channels {
            return filtered;
        }
        // Only 2 -> 1 downmix reaches here (coded < configured, both in 1..=2).
        debug_assert_eq!((self.channels, self.coded_channels()), (2, 1));
        filtered.chunks_exact(2).map(|lr| (lr[0] + lr[1]) * 0.5).collect()
    }

    /// The encode complexity 0-10 (`OPUS_GET_COMPLEXITY`).
    #[must_use]
    pub const fn complexity(&self) -> u8 {
        self.complexity
    }

    /// The target bitrate in bits/s, or `None` for the per-mode default
    /// (`OPUS_GET_BITRATE`).
    #[must_use]
    pub const fn bitrate(&self) -> Option<u32> {
        self.target_bitrate
    }

    /// The coded audio bandwidth (`OPUS_GET_BANDWIDTH`).
    #[must_use]
    pub const fn bandwidth(&self) -> Bandwidth {
        self.bandwidth
    }

    /// Whether discontinuous transmission is enabled (`OPUS_GET_DTX`).
    #[must_use]
    pub const fn dtx(&self) -> bool {
        self.use_dtx
    }

    /// Enables or disables variable bitrate (`OPUS_SET_VBR`). With a bitrate
    /// set, VBR (the default) lets each CELT packet shrink to its per-frame
    /// target; CBR codes a fixed byte count per frame at that rate.
    pub const fn set_vbr(&mut self, vbr: bool) {
        self.vbr = vbr;
    }

    /// Whether variable bitrate is enabled (`OPUS_GET_VBR`).
    #[must_use]
    pub const fn vbr(&self) -> bool {
        self.vbr
    }

    /// Sets the encode complexity 0-10 (`OPUS_SET_COMPLEXITY`), clamped. Higher
    /// is better quality and slower; the default is 5. At complexity 0
    /// the encoder skips the deepest analysis (the
    /// CELT pre-filter pitch search, tf analysis, spreading; the deepest SILK
    /// pitch search).
    pub const fn set_complexity(&mut self, complexity: u8) {
        let c = if complexity > 10 { 10 } else { complexity };
        self.complexity = c;
        self.celt.set_complexity(c);
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

    /// Enables or disables in-band forward error correction
    /// (`OPUS_SET_INBAND_FEC`). When on, SILK-mode packets carry a redundant
    /// lower-priority copy (LBRR) of each frame, so the decoder can reconstruct
    /// a lost frame from the *next* packet via
    /// [`OpusDecoder::decode_fec`](crate::OpusDecoder::decode_fec). FEC applies
    /// to the SILK-only path; CELT-only and hybrid packets are unaffected.
    pub const fn set_inband_fec(&mut self, on: bool) {
        self.use_inband_fec = on;
    }

    /// Whether in-band FEC is enabled (`OPUS_GET_INBAND_FEC`).
    #[must_use]
    pub const fn inband_fec(&self) -> bool {
        self.use_inband_fec
    }

    /// Sets the expected packet-loss percentage 0-100
    /// (`OPUS_SET_PACKET_LOSS_PERC`), clamped. Higher values bias the SILK
    /// encoder toward loss-robust coding: an independently coded voiced frame
    /// raises its LTP scaling index (less inter-frame prediction dependency, so
    /// a lost frame damages fewer following frames), and any LBRR (FEC) copy is
    /// coded at a reduced rate via a larger gain increase. At 0 (the default)
    /// the output is unchanged.
    pub const fn set_packet_loss_perc(&mut self, perc: u8) {
        self.packet_loss_perc = if perc > 100 { 100 } else { perc };
    }

    /// The expected packet-loss percentage (`OPUS_GET_PACKET_LOSS_PERC`).
    #[must_use]
    pub const fn packet_loss_perc(&self) -> u8 {
        self.packet_loss_perc
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

    /// A 3 Hz one-pole high-pass on the interleaved
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

    /// Forces the coded audio bandwidth (`OPUS_SET_BANDWIDTH`). The CELT modes
    /// support narrowband, wideband, super-wideband and fullband; mediumband is
    /// treated as wideband (CELT has no 6 kHz mode).
    ///
    /// Calling this **pins** the bandwidth: [`encode_auto`](Self::encode_auto)
    /// then uses exactly this value instead of choosing one from the signal.
    /// Use [`set_auto_bandwidth`](Self::set_auto_bandwidth) to restore automatic
    /// selection (capped by [`set_max_bandwidth`](Self::set_max_bandwidth)).
    pub const fn set_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.bandwidth = bandwidth;
        self.bandwidth_forced = true;
    }

    /// Restores automatic bandwidth selection (`OPUS_SET_BANDWIDTH` with
    /// `OPUS_AUTO`): [`encode_auto`](Self::encode_auto) picks the bandwidth from
    /// the signal's spectral content and the target bitrate, clamped to
    /// [`max_bandwidth`](Self::max_bandwidth).
    pub const fn set_auto_bandwidth(&mut self) {
        self.bandwidth_forced = false;
    }

    /// Whether a bandwidth has been forced via
    /// [`set_bandwidth`](Self::set_bandwidth) (vs automatic selection).
    #[must_use]
    pub const fn bandwidth_forced(&self) -> bool {
        self.bandwidth_forced
    }

    /// The signal-content hint (`OPUS_GET_SIGNAL`).
    #[must_use]
    pub const fn signal(&self) -> Signal {
        self.signal
    }

    /// Sets the signal-content hint (`OPUS_SET_SIGNAL`): `Voice` biases the
    /// automatic mode decision toward SILK / hybrid, `Music` toward CELT, and
    /// `Auto` (the default) lets the analysis decide.
    pub const fn set_signal(&mut self, signal: Signal) {
        self.signal = signal;
    }

    /// The coding application / latency profile (`OPUS_GET_APPLICATION`).
    #[must_use]
    pub const fn application(&self) -> Application {
        self.application
    }

    /// Sets the coding application (`OPUS_SET_APPLICATION`). `RestrictedLowDelay`
    /// forces CELT-only coding (no SILK, minimal delay); `Voip` biases toward
    /// SILK / speech; `Audio` (the default) is the balanced general profile.
    pub const fn set_application(&mut self, application: Application) {
        self.application = application;
    }

    /// The ceiling on automatic bandwidth selection (`OPUS_GET_MAX_BANDWIDTH`).
    #[must_use]
    pub const fn max_bandwidth(&self) -> Bandwidth {
        self.max_bandwidth
    }

    /// Caps the bandwidth the automatic selection may choose
    /// (`OPUS_SET_MAX_BANDWIDTH`). Has no effect when a bandwidth is forced via
    /// [`set_bandwidth`](Self::set_bandwidth).
    pub const fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.max_bandwidth = bandwidth;
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

    /// Picks the auto bandwidth for a 10/20 ms frame from the analysis and the
    /// target bitrate, clamped to [`max_bandwidth`](Self::max_bandwidth).
    ///
    /// The signal's detected spectral extent is the upper bound (no point
    /// coding empty high bands), and the bitrate is a lower bound on quality:
    /// low rates can't afford super-wideband/fullband, so we narrow them. This
    /// mirrors libopus's rate-vs-bandwidth ladder (`opus_encoder.c`'s
    /// `rate_thresholds`), with hysteresis-free, documented thresholds.
    fn auto_bandwidth(&self, analysis: &FrameAnalysis) -> Bandwidth {
        // Rate ceiling: how wide a band the bitrate can usefully fill. `None`
        // (unset) means "use the full detected extent".
        let rate_cap = match self.target_bitrate {
            None => Bandwidth::FullBand,
            Some(b) if b < 11_000 => Bandwidth::NarrowBand,
            Some(b) if b < 14_000 => Bandwidth::MediumBand,
            Some(b) if b < 24_000 => Bandwidth::WideBand,
            Some(b) if b < 42_000 => Bandwidth::SuperWideBand,
            Some(_) => Bandwidth::FullBand,
        };
        // The chosen bandwidth is the narrowest of: what the signal occupies,
        // what the rate affords, and the caller's hard cap.
        analysis.detected_bandwidth.min(rate_cap).min(self.max_bandwidth)
    }

    /// Chooses SILK / hybrid / CELT for a 10/20 ms frame, applying the
    /// `application`, `signal` hint and a smoothed (hysteresis) music
    /// probability on top of the bandwidth/bitrate backbone. Returns the chosen
    /// mode; the caller dispatches to the matching coder.
    ///
    /// `bw` is the (possibly auto-selected) coded bandwidth.
    fn decide_mode(&mut self, analysis: &FrameAnalysis, bw: Bandwidth) -> ChosenMode {
        // Restricted-low-delay never uses SILK (which adds algorithmic delay).
        if self.application == Application::RestrictedLowDelay {
            return ChosenMode::Celt;
        }

        // Hysteresis: low-pass the per-frame music probability so a single
        // borderline frame can't flip the mode. A near-silent frame carries no
        // evidence, so we leave the smoothed value untouched.
        if analysis.energy > 1e-7 {
            const SMOOTH: f32 = 0.2; // ~5-frame time constant
            self.mode_music_smooth += SMOOTH * (analysis.music_probability - self.mode_music_smooth);
        }
        let mut music = self.mode_music_smooth;

        // The caller's signal hint and application bias the music score: a hint
        // shifts the decision threshold rather than hard-overriding it, so the
        // analysis still has a say on truly ambiguous content.
        match self.signal {
            Signal::Voice => music -= 0.35,
            Signal::Music => music += 0.35,
            Signal::Auto => {},
        }
        if self.application == Application::Voip {
            music -= 0.2; // VoIP leans on speech coding
        }
        let music = music.clamp(0.0, 1.0);

        // Bandwidth/bitrate backbone, unchanged from the original simplified
        // decision, gives the *speech-side* preference (SILK ≤ WB, hybrid for
        // SWB/FB at modest rates). Strong music evidence pulls toward CELT.
        let wb_or_below = matches!(bw, Bandwidth::NarrowBand | Bandwidth::MediumBand | Bandwidth::WideBand);
        let swb_or_fb = matches!(bw, Bandwidth::SuperWideBand | Bandwidth::FullBand);

        // The speech-side mode the backbone would pick for this bw/bitrate.
        let speech_mode = if wb_or_below && self.target_bitrate.is_some_and(|b| b <= 24_000) {
            Some(ChosenMode::Silk)
        } else if swb_or_fb && self.target_bitrate.is_some_and(|b| b <= 40_000) {
            Some(ChosenMode::Hybrid)
        } else {
            None
        };

        match speech_mode {
            // The backbone wants a speech mode (modest-rate SILK/hybrid range):
            // take it unless the evidence is strongly music.
            Some(mode) => {
                if music > 0.7 {
                    ChosenMode::Celt
                } else {
                    mode
                }
            },
            // The backbone defaults to CELT - high rate (> the speech-mode rate
            // ceiling) or no target rate set. Keep CELT unless the caller has
            // *explicitly* asked for speech (Voice hint or VoIP) and the content
            // is clearly speech, in which case route to the bandwidth-
            // appropriate SILK-family mode. (Auto/Audio keeps the original
            // high-rate-is-CELT behaviour.)
            None => {
                let speech_requested = self.signal == Signal::Voice || self.application == Application::Voip;
                if speech_requested && music < 0.25 {
                    if wb_or_below {
                        ChosenMode::Silk
                    } else {
                        ChosenMode::Hybrid
                    }
                } else {
                    ChosenMode::Celt
                }
            },
        }
    }

    /// Whether [`decide_mode`](Self::decide_mode)'s outcome can depend on the
    /// per-frame music probability, given the current (forced) bandwidth and
    /// settings. When it cannot - high-rate CELT with no speech hint, or
    /// restricted low delay - the mode is already pinned, so the analysis can be
    /// skipped. Mirrors `decide_mode`'s branch structure exactly, so skipping
    /// when this is false never changes the chosen mode.
    fn mode_decision_uses_analysis(&self) -> bool {
        if self.application == Application::RestrictedLowDelay {
            return false;
        }
        let bw = self.bandwidth;
        let wb_or_below = matches!(bw, Bandwidth::NarrowBand | Bandwidth::MediumBand | Bandwidth::WideBand);
        let swb_or_fb = matches!(bw, Bandwidth::SuperWideBand | Bandwidth::FullBand);
        let speech_mode_possible = (wb_or_below && self.target_bitrate.is_some_and(|b| b <= 24_000))
            || (swb_or_fb && self.target_bitrate.is_some_and(|b| b <= 40_000));
        let speech_requested = self.signal == Signal::Voice || self.application == Application::Voip;
        speech_mode_possible || speech_requested
    }

    /// Encodes one frame, automatically choosing SILK (speech), hybrid
    /// (SWB/FB speech) or CELT (music / low-delay) per frame from a signal
    /// analysis ([`crate::encoder_analysis`]) plus the
    /// [`signal`](Self::set_signal) hint, the [`application`](Self::set_application)
    /// profile and the target bitrate, with cross-frame hysteresis so the mode
    /// does not flip on a single borderline frame. The CELT-only sizes
    /// (2.5/5 ms) are always CELT; the SILK-only sizes (40/60 ms) are always
    /// SILK; 10/20 ms is where the analysis decides.
    ///
    /// When no bandwidth is forced (see [`set_bandwidth`](Self::set_bandwidth) /
    /// [`set_auto_bandwidth`](Self::set_auto_bandwidth)), the coded bandwidth is
    /// chosen from the signal's spectral extent and the bitrate, capped by
    /// [`max_bandwidth`](Self::max_bandwidth). `pcm` is interleaved 48 kHz f32;
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

        // Analyse the frame only when the result is actually used: it drives the
        // automatic bandwidth choice and the SILK/CELT/hybrid mode decision, both
        // only on 10/20 ms frames. When the bandwidth is forced and the bitrate/
        // application already pin the mode (e.g. high-rate CELT), the analysis
        // output would be discarded, so skip the per-frame pass entirely. The
        // neutral placeholder reproduces `decide_mode`'s pinned outcome exactly.
        let need_analysis =
            matches!(per_ch, 480 | 960) && (!self.bandwidth_forced || self.mode_decision_uses_analysis());
        let analysis = if need_analysis {
            analyze_frame(pcm, self.channels)
        } else {
            FrameAnalysis {
                music_probability: self.mode_music_smooth,
                detected_bandwidth: self.bandwidth,
                energy: 0.0,
            }
        };
        if !self.bandwidth_forced && matches!(per_ch, 480 | 960) {
            self.bandwidth = self.auto_bandwidth(&analysis);
        }

        let packet = match per_ch {
            120 | 240 => self.encode(pcm, max_bytes),        // 2.5/5 ms: CELT only
            1920 | 2880 => self.encode_silk(pcm, max_bytes), // 40/60 ms: SILK only
            480 | 960 => {
                let bw = self.bandwidth;
                match self.decide_mode(&analysis, bw) {
                    ChosenMode::Silk => {
                        // SILK tops out at wideband; if the analysis/rate left a
                        // wider bandwidth, narrow to WB for the SILK coder.
                        let wide = matches!(bw, Bandwidth::SuperWideBand | Bandwidth::FullBand);
                        if wide && !self.bandwidth_forced {
                            self.bandwidth = Bandwidth::WideBand;
                        }
                        let r = self.encode_silk(pcm, max_bytes);
                        if wide && !self.bandwidth_forced {
                            self.bandwidth = bw;
                        }
                        r
                    },
                    ChosenMode::Hybrid => {
                        // Hybrid needs SWB/FB; if the chosen bandwidth is
                        // narrower, fall back to CELT (or SILK) rather than
                        // mis-framing. Hybrid can overrun a tight budget on a
                        // loud transient - fall back to CELT so the path never
                        // fails.
                        if matches!(bw, Bandwidth::SuperWideBand | Bandwidth::FullBand) {
                            self.encode_hybrid(pcm, max_bytes)
                                .or_else(|_| self.encode(pcm, max_bytes))
                        } else {
                            self.encode(pcm, max_bytes)
                        }
                    },
                    ChosenMode::Celt => self.encode(pcm, max_bytes),
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

    /// The encoder's algorithmic delay in samples at 48 kHz
    /// (`OPUS_GET_LOOKAHEAD`) - the number of leading output samples the decoder
    /// should skip (`pre_skip`) to align its output with the input.
    ///
    /// The default mode is fullband CELT, whose delay is the MDCT overlap
    /// ([`crate::celt`] codes a 120-sample window at 48 kHz). This is not a
    /// fabricated constant: encoding a unit impulse and locating it in the
    /// decoded output puts the peak exactly 120 samples late (see the
    /// `lookahead_matches_measured_impulse_delay` test), matching the
    /// `pre_skip = 120` that [`encode_ogg_opus`] writes for the CELT path and
    /// libopus's own fullband CELT lookahead.
    #[must_use]
    pub const fn lookahead(&self) -> u32 {
        // CELT MDCT overlap at 48 kHz; the default (and highest-delay) mode.
        120
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
        let toc = (config << 3) | (u8::from(self.coded_channels() == 2) << 2);

        // High-pass and (when forcing mono) downmix to the coded layout.
        let pcm = self.prepare_input(pcm);
        let payload = if let Some(br) = self.target_bitrate
            && !self.vbr
        {
            // CBR: a fixed byte count for this frame at the target rate
            // (bits = bitrate * samples / 48000; bytes = bits / 8), coded with
            // the CELT coder in its fill (CBR) mode.
            let cbr_bytes = ((br as usize * n) / 384_000).clamp(2, max_bytes - 1);
            self.celt.set_target_bitrate(None);
            let p = self.celt.encode_frame_bw(&pcm, cbr_bytes, end);
            self.celt.set_target_bitrate(self.target_bitrate);
            p
        } else {
            self.celt.encode_frame_bw(&pcm, max_bytes - 1, end)
        };
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

        // High-pass and (when forcing mono) downmix to the coded layout.
        let pcm = self.prepare_input(pcm);
        let pcm = pcm.as_slice();

        let payload = if self.coded_channels() == 1 {
            let need_new = self
                .silk
                .as_ref()
                .is_none_or(|(_, _, khz, nbs)| *khz != internal_khz || *nbs != nb_subfr);
            if need_new {
                self.silk = Some((
                    crate::silk::encode::api::SilkEncoder::new(internal_khz, nb_subfr),
                    crate::silk::encode::resample_in::EncDownsampler::new(internal_khz as usize),
                    internal_khz,
                    nb_subfr,
                ));
            }
            let (silk, resampler, _, _) = self.silk.as_mut().expect("configured");
            silk.set_bitrate(bitrate.clamp(5000, 80_000));
            silk.set_complexity(self.complexity);
            silk.set_inband_fec(self.use_inband_fec);
            silk.set_packet_loss_perc(i32::from(self.packet_loss_perc));
            let internal = crate::silk::encode::resample_in::resample_48k(resampler, pcm);
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
                    crate::silk::encode::resample_in::EncDownsampler::new(internal_khz as usize),
                    crate::silk::encode::resample_in::EncDownsampler::new(internal_khz as usize),
                    internal_khz,
                    nb_subfr,
                ));
            }
            let (silk, rl, rr, _, _) = self.silk_stereo.as_mut().expect("configured");
            silk.set_bitrate(bitrate.clamp(5000, 100_000));
            silk.set_complexity(self.complexity);
            silk.set_inband_fec(self.use_inband_fec);
            silk.set_packet_loss_perc(i32::from(self.packet_loss_perc));
            let lf: Vec<f32> = pcm.iter().step_by(2).copied().collect();
            let rf: Vec<f32> = pcm.iter().skip(1).step_by(2).copied().collect();
            let li = crate::silk::encode::resample_in::resample_48k(rl, &lf);
            let ri = crate::silk::encode::resample_in::resample_48k(rr, &rf);
            let p = silk.encode(&li, &ri);
            self.last_final_range = silk.final_range();
            p
        };

        if payload.len() + 1 > max_bytes {
            return Err(EncodeError::InvalidBudget);
        }

        let config = config_base + lm;
        let toc = (config << 3) | (u8::from(self.coded_channels() == 2) << 2); // code 0
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
        // (the hybrid SILK/CELT split table), and the byte share SILK may
        // use so the CELT high band always has room (`celt_floor`, scaled by
        // the number of high bands it codes).
        let target = self.target_bitrate.map_or(32_000, |b| b as i32);
        let nb_bytes = ((target * frame_ms as i32 / 8000) as usize).clamp(20, max_bytes);
        let swb = matches!(self.bandwidth, Bandwidth::SuperWideBand);
        let coded_ch = self.coded_channels();
        let silk_bps = compute_silk_rate_for_hybrid(target, swb, coded_ch as i32);
        let celt_floor = (celt_end - 17) * 3 + 3;
        let silk_cap = nb_bytes.saturating_sub(celt_floor).max(8);

        // High-pass and (when forcing mono) downmix to the coded layout.
        let pcm = self.prepare_input(pcm);
        let pcm = pcm.as_slice();

        let mut enc = RangeEncoder::new(nb_bytes);
        if coded_ch == 1 {
            // SILK low band: WB (16 kHz) resample, then write into the coder,
            // capped so the CELT high band keeps at least `celt_floor` bytes.
            let need_new = self
                .silk
                .as_ref()
                .is_none_or(|(_, _, khz, nbs)| *khz != 16 || *nbs != nb_subfr);
            if need_new {
                self.silk = Some((
                    crate::silk::encode::api::SilkEncoder::new(16, nb_subfr),
                    crate::silk::encode::resample_in::EncDownsampler::new(16),
                    16,
                    nb_subfr,
                ));
            }
            let (silk, resampler, _, _) = self.silk.as_mut().expect("configured");
            silk.set_bitrate(silk_bps);
            silk.set_complexity(self.complexity);
            // Hybrid in-band FEC: the LBRR copy shares the SILK byte budget with
            // the regular low band. `encode_into` measures the LBRR cost against
            // the `max_bits` cap and skips it for a frame that would crowd out
            // the regular frames (and thus the CELT high band).
            silk.set_inband_fec(self.use_inband_fec);
            silk.set_packet_loss_perc(i32::from(self.packet_loss_perc));
            let internal = crate::silk::encode::resample_in::resample_48k(resampler, pcm);
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
                    crate::silk::encode::resample_in::EncDownsampler::new(16),
                    crate::silk::encode::resample_in::EncDownsampler::new(16),
                    16,
                    nb_subfr,
                ));
            }
            let (silk, rl, rr, _, _) = self.silk_stereo.as_mut().expect("configured");
            silk.set_bitrate(silk_bps);
            silk.set_complexity(self.complexity);
            // Hybrid stereo in-band FEC: the LBRR copy shares the SILK byte
            // budget, measured against `max_bits` and skipped when it would
            // crowd out the regular low band (and thus the CELT high band).
            silk.set_inband_fec(self.use_inband_fec);
            silk.set_packet_loss_perc(i32::from(self.packet_loss_perc));
            let lf: Vec<f32> = pcm.iter().step_by(2).copied().collect();
            let rf: Vec<f32> = pcm.iter().skip(1).step_by(2).copied().collect();
            let li = crate::silk::encode::resample_in::resample_48k(rl, &lf);
            let ri = crate::silk::encode::resample_in::resample_48k(rr, &rf);
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
        let toc = (config << 3) | (u8::from(self.coded_channels() == 2) << 2); // code 0
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
/// aligns with the input (verified at zero lag).
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
        vendor: b"opus_rs".to_vec(),
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
    use alloc::vec::Vec;

    use super::*;
    use crate::{OpusDecoder, decode_ogg_opus};

    /// `reset` drops all cross-frame state: a fresh encoder and a
    /// reset-after-use encoder must code the same frame to identical bytes and
    /// the same final range (`OPUS_RESET_STATE` semantics).
    #[test]
    fn reset_restores_first_packet_state() {
        let frame: Vec<f32> = (0..960)
            .map(|i| 0.3 * (2.0 * core::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
            .collect();

        let mut fresh = OpusEncoder::new(1);
        fresh.set_bitrate(Some(64_000));
        let want = fresh.encode_auto(&frame, 1275).unwrap();
        let want_range = fresh.final_range();

        let mut reused = OpusEncoder::new(1);
        reused.set_bitrate(Some(64_000));
        // Drive it through several different frames so it accumulates state...
        for k in 1..6 {
            let other: Vec<f32> = (0..960)
                .map(|i| 0.2 * (2.0 * core::f32::consts::PI * (200 * k) as f32 * i as f32 / 48_000.0).sin())
                .collect();
            let _ = reused.encode_auto(&other, 1275).unwrap();
        }
        // ...then reset and re-encode the original frame.
        reused.reset();
        let got = reused.encode_auto(&frame, 1275).unwrap();

        assert_eq!(got, want, "reset must reproduce the fresh-encoder bytes");
        assert_eq!(reused.final_range(), want_range, "reset must reproduce the final range");
    }

    /// `set_vbr(false)` codes constant bitrate - every CELT packet is the same
    /// size, the byte count for the target rate - while VBR varies with content.
    #[test]
    fn vbr_off_codes_constant_bitrate() {
        let frames: Vec<Vec<f32>> = (0..10)
            .map(|k| {
                let amp = if k % 3 == 0 { 0.5 } else { 0.1 };
                (0..960)
                    .map(|i| amp * (2.0 * core::f32::consts::PI * (200 * (k + 1)) as f32 * i as f32 / 48_000.0).sin())
                    .collect()
            })
            .collect();

        let mut cbr = OpusEncoder::new(1);
        cbr.set_bitrate(Some(64_000));
        cbr.set_vbr(false);
        assert!(!cbr.vbr());
        let cbr_sizes: Vec<usize> = frames.iter().map(|f| cbr.encode(f, 1275).unwrap().len()).collect();
        assert!(
            cbr_sizes.iter().all(|&s| s == cbr_sizes[0]),
            "CBR packets must be constant size, got {cbr_sizes:?}"
        );
        // 64000 bps * 960/48000 s / 8 = 160 payload bytes, + 1 TOC.
        assert_eq!(cbr_sizes[0], 64_000 * 960 / 384_000 + 1);

        let mut vbr = OpusEncoder::new(1);
        vbr.set_bitrate(Some(64_000));
        assert!(vbr.vbr());
        let vbr_sizes: Vec<usize> = frames.iter().map(|f| vbr.encode(f, 1275).unwrap().len()).collect();
        assert!(
            vbr_sizes.iter().any(|&s| s != vbr_sizes[0]),
            "VBR packets should vary with content, got {vbr_sizes:?}"
        );
    }

    /// The new getters mirror the setters.
    #[test]
    fn config_getters_mirror_setters() {
        let mut enc = OpusEncoder::new(2);
        enc.set_complexity(7);
        enc.set_bitrate(Some(48_000));
        enc.set_bandwidth(Bandwidth::WideBand);
        enc.set_dtx(true);
        assert_eq!(enc.channels(), 2);
        assert_eq!(enc.complexity(), 7);
        assert_eq!(enc.bitrate(), Some(48_000));
        assert_eq!(enc.bandwidth(), Bandwidth::WideBand);
        assert!(enc.dtx());
    }

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

    /// With in-band FEC enabled, a SILK packet still decodes normally on the
    /// `OpusDecoder` finishing on the encoder's exact range state - the
    /// redundant LBRR data is read-and-discarded by the normal decode path, so
    /// the oracle stays in sync. Covers single-frame (20 ms) and multi-frame
    /// (40/60 ms) packets, where the LBRR symbol is coded too.
    #[test]
    fn fec_packets_still_round_trip_on_normal_decode() {
        for &spf in &[960usize, 1920, 2880] {
            let mut enc = OpusEncoder::new(1);
            enc.set_bandwidth(Bandwidth::WideBand);
            enc.set_bitrate(Some(20_000));
            enc.set_inband_fec(true);
            enc.set_packet_loss_perc(30);
            assert!(enc.inband_fec());
            assert_eq!(enc.packet_loss_perc(), 30);
            let mut dec = OpusDecoder::new(1);
            for f in 0..4 {
                let pcm: Vec<f32> = (0..spf)
                    .map(|i| {
                        let t = (f * spf + i) as f32 / 48_000.0;
                        0.4 * (2.0 * core::f32::consts::PI * 220.0 * t).sin()
                    })
                    .collect();
                let packet = enc.encode_silk(&pcm, 1275).expect("silk encode");
                let out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(out.len(), spf);
                assert_eq!(
                    dec.final_range(),
                    enc.final_range(),
                    "range mismatch spf={spf} frame {f}"
                );
            }
        }
    }

    /// End-to-end FEC recovery: enable FEC, encode a sequence, "lose" a packet,
    /// and reconstruct it from the *next* packet's LBRR via
    /// `OpusDecoder::decode_fec`. The recovered frame must be non-trivial (not
    /// silence) and correlate with the original input. Also checks that with
    /// FEC off the encoder output is byte-identical to today (no LBRR), so the
    /// new behaviour is fully gated.
    #[test]
    fn fec_recovers_a_lost_frame() {
        let spf = 960usize; // 20 ms WB mono
        // Per-frame frequency steps, so concealment (which extrapolates the
        // previous frame's pitch) cannot predict the lost frame's content - the
        // FEC copy carries the real signal.
        let make = |f: usize| -> Vec<f32> {
            let freq = 180.0 + 90.0 * (f as f32);
            (0..spf)
                .map(|i| {
                    let t = (f * spf + i) as f32 / 48_000.0;
                    0.4 * (2.0 * core::f32::consts::PI * freq * t).sin()
                        + 0.15 * (2.0 * core::f32::consts::PI * 2.0 * freq * t).sin()
                })
                .collect()
        };

        // FEC off vs on must produce different packets (LBRR adds bytes), and
        // FEC off must match a no-FEC encoder byte-for-byte.
        let encode_seq = |fec: bool| -> Vec<Vec<u8>> {
            let mut enc = OpusEncoder::new(1);
            enc.set_bandwidth(Bandwidth::WideBand);
            enc.set_bitrate(Some(20_000));
            enc.set_inband_fec(fec);
            (0..6)
                .map(|f| enc.encode_silk(&make(f), 1275).expect("encode"))
                .collect()
        };
        let off = encode_seq(false);
        let on = encode_seq(true);
        // FEC-off output is unchanged: identical to a default encoder.
        let mut plain = OpusEncoder::new(1);
        plain.set_bandwidth(Bandwidth::WideBand);
        plain.set_bitrate(Some(20_000));
        for (f, off_p) in off.iter().enumerate() {
            let p = plain.encode_silk(&make(f), 1275).expect("encode");
            assert_eq!(&p, off_p, "FEC-off packet {f} differs from default encoder");
        }
        // FEC-on packets are larger (carry the LBRR copy).
        assert!(
            on.iter().zip(&off).any(|(a, b)| a.len() > b.len()),
            "FEC packets should be larger than non-FEC"
        );

        // Decode the stream with packet 3 lost, recovering it from packet 4's
        // LBRR. Prime the decoder with packets 0..3 first.
        let mut dec = OpusDecoder::new(1);
        for p in on.iter().take(3) {
            dec.decode_packet(p).expect("decode");
        }
        // Packet 3 lost: recover its 960-sample frame from packet 4's FEC.
        let recovered = dec.decode_fec(&on[4], spf).expect("decode_fec");
        // The recovered frame is the FEC'd duration at the end of the buffer.
        assert!(recovered.len() >= spf, "decode_fec output covers the frame");
        let frame = &recovered[recovered.len() - spf..];

        // Non-trivial: real energy, not silence/concealment-decay.
        let energy: f64 = frame.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        assert!(energy > 1.0, "recovered frame is silent (energy {energy:.3})");

        // Correlates with the original input (delay-aligned search).
        let orig = make(3);
        let corr_of = |sig: &[f32]| -> f64 {
            (0..240usize)
                .map(|d| {
                    let (mut s, mut dot, mut e) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..spf - d {
                        let a = f64::from(orig[i]);
                        let b = f64::from(sig[i + d]);
                        s += a * a;
                        dot += a * b;
                        e += b * b;
                    }
                    dot / (s.sqrt() * e.sqrt()).max(1e-9)
                })
                .fold(0.0f64, f64::max)
        };
        let fec_corr = corr_of(frame);
        assert!(
            fec_corr > 0.7,
            "recovered-frame correlation {fec_corr:.3} too low for FEC"
        );

        // Control: without FEC, recovering the same lost packet falls back to
        // plain concealment. FEC recovery must be at least as good - and here,
        // meaningfully better - proving the LBRR data is doing real work.
        let mut dec_off = OpusDecoder::new(1);
        for p in off.iter().take(3) {
            dec_off.decode_packet(p).expect("decode");
        }
        let concealed = dec_off.decode_fec(&off[4], spf).expect("decode_fec falls back");
        let conceal_frame = &concealed[concealed.len() - spf..];
        let conceal_corr = corr_of(conceal_frame);
        assert!(
            fec_corr > conceal_corr + 0.05,
            "FEC recovery ({fec_corr:.3}) should beat concealment ({conceal_corr:.3})"
        );
    }

    /// Multi-frame (40 ms, two SILK frames) FEC recovery: the LBRR symbol and
    /// both LBRR frames are coded, and recovering a lost 40 ms packet from its
    /// successor reconstructs audio that beats plain concealment of the same
    /// per-frame frequency-stepping signal.
    #[test]
    fn fec_recovers_a_lost_multiframe_packet() {
        let spf = 1920usize; // 40 ms WB mono, two 20 ms SILK frames
        let make = |f: usize| -> Vec<f32> {
            let freq = 180.0 + 70.0 * (f as f32);
            (0..spf)
                .map(|i| {
                    let t = (f * spf + i) as f32 / 48_000.0;
                    0.4 * (2.0 * core::f32::consts::PI * freq * t).sin()
                })
                .collect()
        };
        let encode_seq = |fec: bool| -> Vec<Vec<u8>> {
            let mut enc = OpusEncoder::new(1);
            enc.set_bandwidth(Bandwidth::WideBand);
            enc.set_bitrate(Some(24_000));
            enc.set_inband_fec(fec);
            (0..6)
                .map(|f| enc.encode_silk(&make(f), 1275).expect("encode"))
                .collect()
        };
        let on = encode_seq(true);
        let off = encode_seq(false);

        let recover = |packets: &[Vec<u8>]| -> Vec<f32> {
            let mut dec = OpusDecoder::new(1);
            for p in packets.iter().take(3) {
                dec.decode_packet(p).expect("decode");
            }
            dec.decode_fec(&packets[4], spf).expect("decode_fec")
        };
        let rec = recover(&on);
        let conc = recover(&off);
        assert!(rec.len() >= spf && conc.len() >= spf);

        // Correlate the recovered tail (the FEC'd 40 ms) with the original.
        let orig = make(3);
        let corr_of = |sig: &[f32]| -> f64 {
            let frame = &sig[sig.len() - spf..];
            (0..240usize)
                .map(|d| {
                    let (mut s, mut dot, mut e) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..spf - d {
                        let a = f64::from(orig[i]);
                        let b = f64::from(frame[i + d]);
                        s += a * a;
                        dot += a * b;
                        e += b * b;
                    }
                    dot / (s.sqrt() * e.sqrt()).max(1e-9)
                })
                .fold(0.0f64, f64::max)
        };
        let (fec_corr, conceal_corr) = (corr_of(&rec), corr_of(&conc));
        assert!(fec_corr > 0.6, "multi-frame FEC correlation {fec_corr:.3} too low");
        assert!(
            fec_corr > conceal_corr + 0.05,
            "multi-frame FEC ({fec_corr:.3}) should beat concealment ({conceal_corr:.3})"
        );
    }

    /// With in-band FEC enabled, a *stereo* SILK packet still decodes normally
    /// on the `OpusDecoder` finishing on the encoder's exact range state - the
    /// redundant LBRR data (mid + side) is read-and-discarded by the normal
    /// decode path, so the oracle stays in sync. Covers the mid-only→side
    /// transition (where the side LBRR flags vary per frame) and a multi-frame
    /// (40 ms) packet where the LBRR symbols are coded.
    #[test]
    fn stereo_fec_packets_still_round_trip_on_normal_decode() {
        for &spf in &[960usize, 1920] {
            let mut enc = OpusEncoder::new(2);
            enc.set_bandwidth(Bandwidth::WideBand);
            enc.set_bitrate(Some(32_000));
            enc.set_inband_fec(true);
            assert!(enc.inband_fec());
            let mut dec = OpusDecoder::new(2);
            for f in 0..16 {
                let mut pcm = Vec::with_capacity(spf * 2);
                for i in 0..spf {
                    let t = (f * spf + i) as f32 / 48_000.0;
                    // A width that builds up so the side channel activates,
                    // exercising the mid-only→side LBRR transition.
                    let w = (f as f32 / 16.0).min(1.0);
                    let l = 0.4 * (2.0 * core::f32::consts::PI * 210.0 * t).sin();
                    let r = (1.0 - w) * l + w * 0.35 * (2.0 * core::f32::consts::PI * 350.0 * t).sin();
                    pcm.push(l);
                    pcm.push(r);
                }
                let packet = enc.encode_silk(&pcm, 1275).expect("stereo silk encode");
                let out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(out.len(), spf * 2);
                assert_eq!(
                    dec.final_range(),
                    enc.final_range(),
                    "stereo FEC range mismatch spf={spf} frame {f}"
                );
            }
        }
    }

    /// End-to-end *stereo* FEC recovery: enable FEC, encode a stereo sequence,
    /// "lose" a packet, and reconstruct it from the next packet's LBRR via
    /// `OpusDecoder::decode_fec`. The recovered stereo frame must correlate with
    /// the original input and beat plain concealment, proving the mid (and side)
    /// LBRR data is doing real work. FEC-off output stays byte-identical.
    #[test]
    fn stereo_fec_recovers_a_lost_frame() {
        let spf = 960usize; // 20 ms WB stereo
        // Per-frame frequency steps so concealment cannot predict the lost
        // frame; a steady stereo width keeps the side channel coded.
        let make = |f: usize| -> Vec<f32> {
            let freq = 200.0 + 80.0 * (f as f32);
            let mut pcm = Vec::with_capacity(spf * 2);
            for i in 0..spf {
                let t = (f * spf + i) as f32 / 48_000.0;
                let l = 0.4 * (2.0 * core::f32::consts::PI * freq * t).sin();
                let r = 0.25 * (2.0 * core::f32::consts::PI * freq * t + 0.4).sin()
                    + 0.3 * (2.0 * core::f32::consts::PI * (freq * 1.6) * t).sin();
                pcm.push(l);
                pcm.push(r);
            }
            pcm
        };

        let encode_seq = |fec: bool| -> Vec<Vec<u8>> {
            let mut enc = OpusEncoder::new(2);
            enc.set_bandwidth(Bandwidth::WideBand);
            enc.set_bitrate(Some(40_000));
            enc.set_inband_fec(fec);
            (0..7)
                .map(|f| enc.encode_silk(&make(f), 1275).expect("encode"))
                .collect()
        };
        let off = encode_seq(false);
        let on = encode_seq(true);

        // FEC-off output is unchanged: identical to a default stereo encoder.
        let mut plain = OpusEncoder::new(2);
        plain.set_bandwidth(Bandwidth::WideBand);
        plain.set_bitrate(Some(40_000));
        for (f, off_p) in off.iter().enumerate() {
            let p = plain.encode_silk(&make(f), 1275).expect("encode");
            assert_eq!(&p, off_p, "FEC-off stereo packet {f} differs from default");
        }
        assert!(
            on.iter().zip(&off).any(|(a, b)| a.len() > b.len()),
            "stereo FEC packets should be larger than non-FEC"
        );

        // Recover packet 3 from packet 4's LBRR after priming with 0..3.
        let recover = |packets: &[Vec<u8>]| -> Vec<f32> {
            let mut dec = OpusDecoder::new(2);
            for p in packets.iter().take(3) {
                dec.decode_packet(p).expect("decode");
            }
            dec.decode_fec(&packets[4], spf).expect("decode_fec")
        };
        let rec = recover(&on);
        let conc = recover(&off);
        assert!(rec.len() >= spf * 2 && conc.len() >= spf * 2);

        // Correlate the recovered tail's left channel with the original input.
        let orig = make(3);
        let orig_l: Vec<f32> = orig.iter().step_by(2).copied().collect();
        let corr_of = |sig: &[f32]| -> f64 {
            let frame = &sig[sig.len() - spf * 2..];
            let frame_l: Vec<f32> = frame.iter().step_by(2).copied().collect();
            (0..300usize)
                .map(|d| {
                    let (mut s, mut dot, mut e) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..spf - d {
                        let a = f64::from(orig_l[i]);
                        let b = f64::from(frame_l[i + d]);
                        s += a * a;
                        dot += a * b;
                        e += b * b;
                    }
                    dot / (s.sqrt() * e.sqrt()).max(1e-9)
                })
                .fold(0.0f64, f64::max)
        };
        let fec_corr = corr_of(&rec);
        let conceal_corr = corr_of(&conc);
        let frame = &rec[rec.len() - spf * 2..];
        let energy: f64 = frame.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        assert!(energy > 1.0, "recovered stereo frame is silent (energy {energy:.3})");
        assert!(fec_corr > 0.6, "stereo FEC correlation {fec_corr:.3} too low");
        assert!(
            fec_corr > conceal_corr + 0.05,
            "stereo FEC ({fec_corr:.3}) should beat concealment ({conceal_corr:.3})"
        );
    }

    /// With in-band FEC enabled, *hybrid* packets (SILK low band + CELT high
    /// band) still decode normally on the `OpusDecoder` finishing on the
    /// encoder's exact range state - the SILK LBRR copy shares the byte budget
    /// and the CELT high band still fits. Covers mono and stereo, SWB and FB.
    #[test]
    fn hybrid_fec_packets_still_round_trip_on_normal_decode() {
        for &chans in &[1usize, 2] {
            for &bw in &[Bandwidth::SuperWideBand, Bandwidth::FullBand] {
                let spf = 960usize; // 20 ms
                let mut enc = OpusEncoder::new(chans);
                enc.set_bandwidth(bw);
                enc.set_bitrate(Some(32_000));
                enc.set_inband_fec(true);
                let mut dec = OpusDecoder::new(chans);
                for f in 0..10 {
                    let mut pcm = Vec::with_capacity(spf * chans);
                    for i in 0..spf {
                        let t = (f * spf + i) as f32 / 48_000.0;
                        let l = 0.35 * (2.0 * core::f32::consts::PI * 230.0 * t).sin()
                            + 0.15 * (2.0 * core::f32::consts::PI * 3500.0 * t).sin();
                        pcm.push(l);
                        if chans == 2 {
                            let r = 0.3 * (2.0 * core::f32::consts::PI * 230.0 * t + 0.5).sin()
                                + 0.15 * (2.0 * core::f32::consts::PI * 4200.0 * t).sin();
                            pcm.push(r);
                        }
                    }
                    let packet = enc.encode_hybrid(&pcm, 1275).expect("hybrid encode");
                    let out = dec.decode_packet(&packet).expect("decode");
                    assert_eq!(out.len(), spf * chans);
                    assert_eq!(
                        dec.final_range(),
                        enc.final_range(),
                        "hybrid FEC range mismatch chans={chans} bw={bw:?} frame {f}"
                    );
                }
            }
        }
    }

    /// End-to-end *hybrid* FEC recovery: enable FEC, encode a hybrid sequence,
    /// "lose" a packet, and reconstruct the SILK low band from the next packet's
    /// LBRR via `OpusDecoder::decode_fec`. The recovered frame correlates with
    /// the original (the low band carries the bulk of the energy) and beats
    /// plain concealment. The CELT high band has no FEC and is concealed.
    #[test]
    fn hybrid_fec_recovers_a_lost_frame() {
        let spf = 960usize; // 20 ms FB mono hybrid
        let make = |f: usize| -> Vec<f32> {
            let freq = 200.0 + 70.0 * (f as f32);
            (0..spf)
                .map(|i| {
                    let t = (f * spf + i) as f32 / 48_000.0;
                    0.4 * (2.0 * core::f32::consts::PI * freq * t).sin()
                        + 0.12 * (2.0 * core::f32::consts::PI * 3000.0 * t).sin()
                })
                .collect()
        };
        let encode_seq = |fec: bool| -> Vec<Vec<u8>> {
            let mut enc = OpusEncoder::new(1);
            enc.set_bandwidth(Bandwidth::FullBand);
            // A generous rate so the SILK low band has room for both the regular
            // frame and its LBRR copy alongside the CELT high band.
            enc.set_bitrate(Some(64_000));
            enc.set_inband_fec(fec);
            (0..7)
                .map(|f| enc.encode_hybrid(&make(f), 1275).expect("encode"))
                .collect()
        };
        let on = encode_seq(true);
        let off = encode_seq(false);
        // Unlike SILK-only, hybrid packets are budget-filled, so FEC-on packets
        // are not necessarily larger - the LBRR copy shares the fixed budget
        // with CELT. The recovery quality below is what proves the LBRR works.
        let _ = &off;

        let recover = |packets: &[Vec<u8>]| -> Vec<f32> {
            let mut dec = OpusDecoder::new(1);
            for p in packets.iter().take(3) {
                dec.decode_packet(p).expect("decode");
            }
            dec.decode_fec(&packets[4], spf).expect("decode_fec")
        };
        let rec = recover(&on);
        let conc = recover(&off);
        assert!(rec.len() >= spf && conc.len() >= spf);

        let orig = make(3);
        let corr_of = |sig: &[f32]| -> f64 {
            let frame = &sig[sig.len() - spf..];
            (0..300usize)
                .map(|d| {
                    let (mut s, mut dot, mut e) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..spf - d {
                        let a = f64::from(orig[i]);
                        let b = f64::from(frame[i + d]);
                        s += a * a;
                        dot += a * b;
                        e += b * b;
                    }
                    dot / (s.sqrt() * e.sqrt()).max(1e-9)
                })
                .fold(0.0f64, f64::max)
        };
        let fec_corr = corr_of(&rec);
        let conceal_corr = corr_of(&conc);
        assert!(fec_corr > 0.5, "hybrid FEC correlation {fec_corr:.3} too low");
        assert!(
            fec_corr > conceal_corr + 0.05,
            "hybrid FEC ({fec_corr:.3}) should beat concealment ({conceal_corr:.3})"
        );
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

    /// `OPUS_GET_LOOKAHEAD`: `lookahead()` reports the encoder's true
    /// algorithmic delay. Encoding a unit impulse (CELT, the default mode) and
    /// locating its peak in the decoded output puts it exactly `lookahead()`
    /// samples late - the number is measured, not asserted by fiat - and it
    /// equals the `pre_skip` the Ogg writer emits for the CELT path.
    #[test]
    fn lookahead_matches_measured_impulse_delay() {
        let mut enc = OpusEncoder::new(1);
        enc.set_bitrate(Some(96_000)); // CELT, fullband
        assert_eq!(enc.lookahead(), 120, "documented fullband CELT delay");

        let mut dec = OpusDecoder::new(1);
        let in_pos = 240usize; // impulse position within the first frame
        let mut out_all: Vec<f32> = Vec::new();
        for f in 0..6 {
            let mut frame = alloc::vec![0.0f32; 960];
            if f == 0 {
                frame[in_pos] = 0.9;
            }
            let packet = enc.encode(&frame, 1275).expect("encode");
            out_all.extend_from_slice(&dec.decode_packet(&packet).expect("decode"));
        }
        let peak_idx = out_all
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let measured_delay = peak_idx as i64 - in_pos as i64;
        assert_eq!(
            measured_delay,
            i64::from(enc.lookahead()),
            "impulse arrives {measured_delay} samples late; lookahead() says {}",
            enc.lookahead()
        );
    }

    /// `OPUS_SET_FORCE_CHANNELS` getter/setter round-trips, only the 2→1
    /// direction takes effect, and bad values are rejected.
    #[test]
    fn force_channels_getter_setter() {
        let mut stereo = OpusEncoder::new(2);
        assert_eq!(stereo.force_channels(), None);
        stereo.set_force_channels(Some(1));
        assert_eq!(stereo.force_channels(), Some(1));
        stereo.set_force_channels(None);
        assert_eq!(stereo.force_channels(), None);

        // Forcing stereo on a mono encoder is a no-op for the coded count.
        let mut mono = OpusEncoder::new(1);
        mono.set_force_channels(Some(2));
        assert_eq!(mono.force_channels(), Some(2));
        // It still only ever codes mono (verified by the round-trip test below).
    }

    #[test]
    #[should_panic(expected = "force_channels must be 1 or 2")]
    fn force_channels_rejects_out_of_range() {
        OpusEncoder::new(2).set_force_channels(Some(3));
    }

    /// `OPUS_SET_FORCE_CHANNELS = 1` on a stereo encoder downmixes the stereo
    /// input to mono and codes **mono** packets (mono TOC, stereo bit clear),
    /// which decode through a **mono** `OpusDecoder` with the matching final
    /// range - the bit-exact oracle. Covers CELT, SILK and hybrid, and the
    /// downmix is real: a hard-panned input (left only) decodes to roughly half
    /// amplitude (the `(l + r) / 2` average), not full scale.
    #[test]
    fn force_channels_downmix_round_trips_as_mono() {
        // Build a decorrelated stereo frame: left tone, right a different tone.
        let stereo_frame = |spf: usize, f: usize| -> Vec<f32> {
            let mut pcm = Vec::with_capacity(spf * 2);
            for i in 0..spf {
                let t = (f * spf + i) as f32 / 48_000.0;
                let l = 0.4 * (2.0 * core::f32::consts::PI * 300.0 * t).sin();
                let r = 0.4 * (2.0 * core::f32::consts::PI * 480.0 * t).sin();
                pcm.push(l);
                pcm.push(r);
            }
            pcm
        };

        // (label, spf, bandwidth, bitrate, encode fn)
        type Enc = fn(&mut OpusEncoder, &[f32], usize) -> Result<Vec<u8>, EncodeError>;
        let cases: [(&str, usize, Bandwidth, u32, Enc); 3] = [
            ("celt", 960, Bandwidth::FullBand, 96_000, OpusEncoder::encode),
            ("silk", 960, Bandwidth::WideBand, 24_000, OpusEncoder::encode_silk),
            ("hybrid", 960, Bandwidth::FullBand, 32_000, OpusEncoder::encode_hybrid),
        ];

        for (label, spf, bw, rate, encfn) in cases {
            let mut enc = OpusEncoder::new(2);
            enc.set_bandwidth(bw);
            enc.set_bitrate(Some(rate));
            enc.set_force_channels(Some(1));
            assert_eq!(enc.channels(), 2, "configured stays stereo");

            let mut dec = OpusDecoder::new(1); // a MONO decoder
            for f in 0..6 {
                let pcm = stereo_frame(spf, f);
                let packet = encfn(&mut enc, &pcm, 1275).expect("forced-mono encode");
                // Mono TOC: stereo bit (bit 2) must be clear.
                assert_eq!(packet[0] & 0b100, 0, "{label}: packet must be mono");
                let out = dec.decode_packet(&packet).expect("mono decode");
                assert_eq!(out.len(), spf, "{label}: mono output length");
                assert_eq!(
                    dec.final_range(),
                    enc.final_range(),
                    "{label}: forced-mono range mismatch frame {f}"
                );
            }
        }
    }

    /// The downmix is the genuine `(l + r) / 2` average, not a channel pick: a
    /// hard-left-panned stereo input (right silent) forced to mono reconstructs
    /// at roughly half the single-channel amplitude.
    #[test]
    fn force_channels_downmix_halves_a_panned_input() {
        let spf = 960usize;
        let mut enc = OpusEncoder::new(2);
        enc.set_bandwidth(Bandwidth::WideBand);
        enc.set_bitrate(Some(32_000));
        enc.set_force_channels(Some(1));
        let mut dec = OpusDecoder::new(1);

        // Reference: the same tone fed to a true mono encoder.
        let mut ref_enc = OpusEncoder::new(1);
        ref_enc.set_bandwidth(Bandwidth::WideBand);
        ref_enc.set_bitrate(Some(32_000));
        let mut ref_dec = OpusDecoder::new(1);

        let (mut downmix_rms, mut ref_rms) = (0.0f64, 0.0f64);
        for f in 0..8 {
            let mut stereo = Vec::with_capacity(spf * 2);
            let mut mono = Vec::with_capacity(spf);
            for i in 0..spf {
                let t = (f * spf + i) as f32 / 48_000.0;
                let s = 0.6 * (2.0 * core::f32::consts::PI * 300.0 * t).sin();
                stereo.push(s); // left
                stereo.push(0.0); // right silent (hard left pan)
                mono.push(s);
            }
            let dpkt = enc.encode_silk(&stereo, 1275).expect("downmix encode");
            let dout = dec.decode_packet(&dpkt).expect("decode");
            let rpkt = ref_enc.encode_silk(&mono, 1275).expect("ref encode");
            let rout = ref_dec.decode_packet(&rpkt).expect("decode");
            if f >= 4 {
                downmix_rms += dout.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
                ref_rms += rout.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
            }
        }
        let ratio = (downmix_rms / ref_rms).sqrt();
        // Left-only panned -> (l + 0)/2 = l/2, so ~0.5 of the full-mono level.
        assert!(
            (0.35..0.65).contains(&ratio),
            "downmix amplitude ratio {ratio:.3} not ~0.5 (mean-of-channels)"
        );
    }

    /// Toggling `force_channels` rebuilds the coder so the stream stays
    /// bit-exact: after switching a stereo encoder to forced-mono and back, the
    /// configured-stereo packets still decode through a stereo `OpusDecoder`
    /// with the matching range.
    #[test]
    fn force_channels_toggle_keeps_the_oracle() {
        let mut enc = OpusEncoder::new(2);
        enc.set_bandwidth(Bandwidth::FullBand);
        enc.set_bitrate(Some(96_000));
        let stereo_frame = |f: usize| -> Vec<f32> {
            let mut pcm = Vec::with_capacity(960 * 2);
            for i in 0..960 {
                let t = (f * 960 + i) as f32 / 48_000.0;
                pcm.push(0.4 * (2.0 * core::f32::consts::PI * 300.0 * t).sin());
                pcm.push(0.4 * (2.0 * core::f32::consts::PI * 480.0 * t).sin());
            }
            pcm
        };

        // Forced mono for a couple of frames (mono decoder).
        enc.set_force_channels(Some(1));
        let mut mdec = OpusDecoder::new(1);
        for f in 0..3 {
            let p = enc.encode(&stereo_frame(f), 1275).unwrap();
            assert_eq!(p[0] & 0b100, 0);
            assert_eq!(mdec.decode_packet(&p).unwrap().len(), 960);
            assert_eq!(mdec.final_range(), enc.final_range(), "mono phase frame {f}");
        }

        // Switch back to stereo coding; a fresh stereo decoder tracks it.
        enc.set_force_channels(None);
        let mut sdec = OpusDecoder::new(2);
        for f in 3..6 {
            let p = enc.encode(&stereo_frame(f), 1275).unwrap();
            assert_eq!(p[0] & 0b100, 0b100, "stereo bit set after switch back");
            assert_eq!(sdec.decode_packet(&p).unwrap().len(), 960 * 2);
            assert_eq!(sdec.final_range(), enc.final_range(), "stereo phase frame {f}");
        }
    }

    /// The mode of an `encode_auto` packet, derived from its TOC config
    /// (SILK 0..12, hybrid 12..16, CELT 16+).
    fn packet_mode(packet: &[u8]) -> char {
        let config = packet[0] >> 3;
        if config < 12 {
            's'
        } else if config < 16 {
            'h'
        } else {
            'c'
        }
    }

    /// A clearly-speech signal (a low voiced tone with formants, band-limited
    /// to the speech range) at a modest VoIP-ish rate is coded with SILK or
    /// hybrid (a SILK-family mode), and every packet round-trips through
    /// `OpusDecoder` with the matching final range.
    #[test]
    fn analysis_routes_speech_to_silk_family() {
        let mut enc = OpusEncoder::new(1);
        enc.set_application(Application::Voip);
        enc.set_bitrate(Some(20_000));
        // Automatic bandwidth (no set_bandwidth call): the analysis picks it.
        assert!(!enc.bandwidth_forced());
        let mut dec = OpusDecoder::new(1);
        let mut silk_family = 0;
        for f in 0..12 {
            let pcm: Vec<f32> = (0..960)
                .map(|i| {
                    let t = (f * 960 + i) as f32 / 48_000.0;
                    // Voiced speech-band content: pitch + a couple of low
                    // formants, nothing above ~2.5 kHz.
                    0.45 * (2.0 * core::f32::consts::PI * 160.0 * t).sin()
                        + 0.25 * (2.0 * core::f32::consts::PI * 800.0 * t).sin()
                        + 0.12 * (2.0 * core::f32::consts::PI * 2300.0 * t).sin()
                })
                .collect();
            let packet = enc.encode_auto(&pcm, 1275).expect("encode");
            let out = dec.decode_packet(&packet).expect("decode");
            assert_eq!(out.len(), 960);
            assert_eq!(dec.final_range(), enc.final_range(), "range mismatch frame {f}");
            if matches!(packet_mode(&packet), 's' | 'h') {
                silk_family += 1;
            }
        }
        // After the hysteresis settles, the steady state must be a SILK-family
        // mode for clearly-speech input.
        assert!(
            silk_family >= 8,
            "expected speech to use SILK/hybrid, only {silk_family}/12 frames did"
        );
    }

    /// A clearly-music signal (bright, broadband content reaching into the top
    /// octave) with the `Music` hint is coded with CELT, and every packet
    /// round-trips through `OpusDecoder` with the matching final range.
    #[test]
    fn analysis_routes_music_to_celt() {
        let mut seed = 0x9E37_79B9u32;
        let mut enc = OpusEncoder::new(1);
        enc.set_signal(Signal::Music);
        enc.set_bitrate(Some(24_000)); // a rate the backbone would give SILK
        let mut dec = OpusDecoder::new(1);
        let mut celt = 0;
        for f in 0..12 {
            let pcm: Vec<f32> = (0..960)
                .map(|i| {
                    let t = (f * 960 + i) as f32 / 48_000.0;
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    let n = ((seed >> 9) as f32 / f32::from(u16::MAX) - 0.5) * 0.15;
                    // Broadband musical content: bass + mids + bright highs into
                    // the top octave, plus a little noise.
                    0.3 * (2.0 * core::f32::consts::PI * 220.0 * t).sin()
                        + 0.3 * (2.0 * core::f32::consts::PI * 3500.0 * t).sin()
                        + 0.3 * (2.0 * core::f32::consts::PI * 9500.0 * t).sin()
                        + 0.25 * (2.0 * core::f32::consts::PI * 15000.0 * t).sin()
                        + n
                })
                .collect();
            let packet = enc.encode_auto(&pcm, 1275).expect("encode");
            let out = dec.decode_packet(&packet).expect("decode");
            assert_eq!(out.len(), 960);
            assert_eq!(dec.final_range(), enc.final_range(), "range mismatch frame {f}");
            if packet_mode(&packet) == 'c' {
                celt += 1;
            }
        }
        assert!(celt >= 8, "expected music to use CELT, only {celt}/12 frames did");
    }

    /// `RestrictedLowDelay` forces CELT-only coding: every `encode_auto` packet,
    /// regardless of frame size / bandwidth / bitrate / signal content, carries
    /// a CELT-only TOC config (16..=31), and round-trips through `OpusDecoder`.
    #[test]
    fn restricted_low_delay_is_always_celt() {
        for &(spf, bw, br) in &[
            (480usize, Bandwidth::WideBand, Some(16_000u32)), // would be SILK
            (960, Bandwidth::SuperWideBand, Some(32_000)),    // would be hybrid
            (960, Bandwidth::FullBand, Some(64_000)),         // CELT anyway
        ] {
            let mut enc = OpusEncoder::new(1);
            enc.set_application(Application::RestrictedLowDelay);
            enc.set_signal(Signal::Voice); // even with a Voice hint
            enc.set_bandwidth(bw);
            enc.set_bitrate(br);
            let mut dec = OpusDecoder::new(1);
            for f in 0..4 {
                let pcm: Vec<f32> = (0..spf)
                    .map(|i| {
                        let t = (f * spf + i) as f32 / 48_000.0;
                        0.3 * (2.0 * core::f32::consts::PI * 200.0 * t).sin()
                    })
                    .collect();
                let packet = enc.encode_auto(&pcm, 1275).expect("encode");
                let config = packet[0] >> 3;
                assert!(
                    config >= 16,
                    "RestrictedLowDelay must be CELT-only, got config {config} (spf={spf})"
                );
                let out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(out.len(), spf);
                assert_eq!(
                    dec.final_range(),
                    enc.final_range(),
                    "range mismatch spf={spf} frame {f}"
                );
            }
        }
    }

    /// Automatic bandwidth selection never exceeds the configured
    /// `max_bandwidth`: a fullband signal capped at wideband is coded at
    /// wideband or below, and the packets round-trip through `OpusDecoder`.
    #[test]
    fn auto_bandwidth_respects_max_bandwidth() {
        // CELT TOC configs map to bandwidth: NB 16-19, WB 20-23, SWB 24-27,
        // FB 28-31. SILK config: NB 0-3, MB 4-7, WB 8-11. We check the coded
        // bandwidth (from the TOC) never exceeds the cap.
        let bw_of = |packet: &[u8]| -> Bandwidth { crate::packet::Toc::new(packet[0]).bandwidth() };
        for &cap in &[Bandwidth::NarrowBand, Bandwidth::WideBand, Bandwidth::SuperWideBand] {
            let mut enc = OpusEncoder::new(1);
            enc.set_auto_bandwidth(); // explicit: automatic selection
            enc.set_max_bandwidth(cap);
            enc.set_bitrate(Some(48_000));
            assert_eq!(enc.max_bandwidth(), cap);
            let mut dec = OpusDecoder::new(1);
            for f in 0..6 {
                // A fullband signal: energy up to 18 kHz. The detected
                // bandwidth would be FB, but the cap must win.
                let pcm: Vec<f32> = (0..960)
                    .map(|i| {
                        let t = (f * 960 + i) as f32 / 48_000.0;
                        0.25 * (2.0 * core::f32::consts::PI * 300.0 * t).sin()
                            + 0.25 * (2.0 * core::f32::consts::PI * 6000.0 * t).sin()
                            + 0.25 * (2.0 * core::f32::consts::PI * 12000.0 * t).sin()
                            + 0.2 * (2.0 * core::f32::consts::PI * 18000.0 * t).sin()
                    })
                    .collect();
                let packet = enc.encode_auto(&pcm, 1275).expect("encode");
                let bw = bw_of(&packet);
                assert!(bw <= cap, "coded bandwidth {bw:?} exceeds cap {cap:?} (frame {f})");
                let out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(out.len(), 960);
                assert_eq!(
                    dec.final_range(),
                    enc.final_range(),
                    "range mismatch cap={cap:?} frame {f}"
                );
            }
        }
    }

    /// The new controls' getters mirror their setters.
    #[test]
    fn signal_application_bandwidth_controls_round_trip() {
        let mut enc = OpusEncoder::new(1);
        assert_eq!(enc.signal(), Signal::Auto);
        assert_eq!(enc.application(), Application::Audio);
        assert!(!enc.bandwidth_forced());
        enc.set_signal(Signal::Voice);
        enc.set_application(Application::RestrictedLowDelay);
        enc.set_max_bandwidth(Bandwidth::MediumBand);
        enc.set_bandwidth(Bandwidth::WideBand);
        assert_eq!(enc.signal(), Signal::Voice);
        assert_eq!(enc.application(), Application::RestrictedLowDelay);
        assert_eq!(enc.max_bandwidth(), Bandwidth::MediumBand);
        assert!(enc.bandwidth_forced());
        enc.set_auto_bandwidth();
        assert!(!enc.bandwidth_forced());
    }

    /// A long mixed stream - alternating clearly-speech and clearly-music
    /// segments - exercises the per-frame analysis, hysteresis and mode
    /// switching: every produced packet must round-trip through `OpusDecoder`
    /// with the matching final range (the bit-exact oracle), whatever mode the
    /// analysis lands on.
    #[test]
    fn auto_mixed_stream_every_packet_round_trips() {
        let mut seed = 0x1357_2468u32;
        let mut enc = OpusEncoder::new(1);
        enc.set_auto_bandwidth();
        enc.set_bitrate(Some(32_000));
        let mut dec = OpusDecoder::new(1);
        for f in 0..60 {
            let speechy = (f / 10) % 2 == 0;
            let pcm: Vec<f32> = (0..960)
                .map(|i| {
                    let t = (f * 960 + i) as f32 / 48_000.0;
                    if speechy {
                        0.45 * (2.0 * core::f32::consts::PI * 150.0 * t).sin()
                            + 0.2 * (2.0 * core::f32::consts::PI * 900.0 * t).sin()
                    } else {
                        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                        let n = ((seed >> 9) as f32 / f32::from(u16::MAX) - 0.5) * 0.2;
                        0.3 * (2.0 * core::f32::consts::PI * 400.0 * t).sin()
                            + 0.3 * (2.0 * core::f32::consts::PI * 8000.0 * t).sin()
                            + 0.25 * (2.0 * core::f32::consts::PI * 14000.0 * t).sin()
                            + n
                    }
                })
                .collect();
            let packet = enc.encode_auto(&pcm, 1275).expect("encode");
            let out = dec.decode_packet(&packet).expect("decode");
            assert_eq!(out.len(), 960);
            assert_eq!(dec.final_range(), enc.final_range(), "range mismatch frame {f}");
        }
    }

    /// A continuous, strongly periodic (voiced) wideband tone used by the
    /// packet-loss tests below. Each frame is phase-continuous with the next.
    fn voiced_silk_frame(f: usize, spf: usize) -> Vec<f32> {
        (0..spf)
            .map(|i| {
                let t = (f * spf + i) as f32 / 48_000.0;
                0.4 * (2.0 * core::f32::consts::PI * 200.0 * t).sin()
                    + 0.15 * (2.0 * core::f32::consts::PI * 400.0 * t).sin()
            })
            .collect()
    }

    /// `packet_loss_perc == 0` (the default) leaves SILK output byte-identical
    /// to an encoder that never touched the knob - the loss-robust LTP scaling
    /// is fully gated off at zero loss.
    #[test]
    fn packet_loss_perc_zero_is_byte_identical() {
        let spf = 960usize; // 20 ms WB mono, voiced (exercises LTP scaling)
        let mut plain = OpusEncoder::new(1);
        plain.set_bandwidth(Bandwidth::WideBand);
        plain.set_bitrate(Some(20_000));

        let mut zero = OpusEncoder::new(1);
        zero.set_bandwidth(Bandwidth::WideBand);
        zero.set_bitrate(Some(20_000));
        zero.set_packet_loss_perc(0);
        assert_eq!(zero.packet_loss_perc(), 0);

        for f in 0..8 {
            let pcm = voiced_silk_frame(f, spf);
            let p_plain = plain.encode_silk(&pcm, 1275).expect("encode");
            let p_zero = zero.encode_silk(&pcm, 1275).expect("encode");
            assert_eq!(p_plain, p_zero, "packet_loss_perc=0 packet {f} differs from default");
            assert_eq!(plain.final_range(), zero.final_range(), "range differs frame {f}");
        }
    }

    /// With `packet_loss_perc > 0`, an independently coded voiced frame raises
    /// its LTP scaling index (loss-robust coding), producing a *different*
    /// bitstream from the zero-loss encoder, yet every packet still decodes on
    /// the `OpusDecoder` with a matching final range (the oracle stays in sync).
    #[test]
    fn packet_loss_perc_raises_ltp_scaling_and_round_trips() {
        let spf = 960usize;
        let mut zero = OpusEncoder::new(1);
        zero.set_bandwidth(Bandwidth::WideBand);
        zero.set_bitrate(Some(20_000));

        let mut lossy = OpusEncoder::new(1);
        lossy.set_bandwidth(Bandwidth::WideBand);
        lossy.set_bitrate(Some(20_000));
        lossy.set_packet_loss_perc(50);
        assert_eq!(lossy.packet_loss_perc(), 50);

        let mut dec = OpusDecoder::new(1);
        let mut any_different = false;
        for f in 0..8 {
            let pcm = voiced_silk_frame(f, spf);
            let p_zero = zero.encode_silk(&pcm, 1275).expect("encode");
            let p_lossy = lossy.encode_silk(&pcm, 1275).expect("encode");
            if p_zero != p_lossy {
                any_different = true;
            }
            // The loss-robust packet still decodes with a matching oracle.
            let out = dec.decode_packet(&p_lossy).expect("decode");
            assert_eq!(out.len(), spf);
            assert_eq!(
                dec.final_range(),
                lossy.final_range(),
                "range mismatch on loss-robust packet {f}"
            );
        }
        assert!(
            any_different,
            "packet_loss_perc>0 should change the voiced bitstream (LTP scaling)"
        );
    }

    /// FEC recovery still works when `packet_loss_perc` is high: the LBRR copy
    /// is coded at a reduced rate (a larger gain increase), but recovering a
    /// lost packet from its successor still reconstructs audio that clearly
    /// beats plain concealment, and every FEC packet round-trips on normal
    /// decode with a matching final range.
    #[test]
    fn fec_recovery_survives_reduced_rate_lbrr() {
        let spf = 960usize;
        let make = |f: usize| -> Vec<f32> {
            let freq = 180.0 + 90.0 * (f as f32);
            (0..spf)
                .map(|i| {
                    let t = (f * spf + i) as f32 / 48_000.0;
                    0.4 * (2.0 * core::f32::consts::PI * freq * t).sin()
                        + 0.15 * (2.0 * core::f32::consts::PI * 2.0 * freq * t).sin()
                })
                .collect()
        };
        let encode_seq = |fec: bool, perc: u8| -> (Vec<Vec<u8>>, Vec<u32>) {
            let mut enc = OpusEncoder::new(1);
            enc.set_bandwidth(Bandwidth::WideBand);
            enc.set_bitrate(Some(20_000));
            enc.set_inband_fec(fec);
            enc.set_packet_loss_perc(perc);
            let packets: Vec<Vec<u8>> = (0..6)
                .map(|f| enc.encode_silk(&make(f), 1275).expect("encode"))
                .collect();
            // Re-encode to collect the per-packet final ranges for the oracle.
            let mut e2 = OpusEncoder::new(1);
            e2.set_bandwidth(Bandwidth::WideBand);
            e2.set_bitrate(Some(20_000));
            e2.set_inband_fec(fec);
            e2.set_packet_loss_perc(perc);
            let ranges: Vec<u32> = (0..6)
                .map(|f| {
                    let _ = e2.encode_silk(&make(f), 1275).expect("encode");
                    e2.final_range()
                })
                .collect();
            (packets, ranges)
        };
        let (on, ranges) = encode_seq(true, 80);
        let (off, _) = encode_seq(false, 80);

        // Every FEC packet decodes normally with a matching oracle.
        let mut dec_norm = OpusDecoder::new(1);
        for (f, p) in on.iter().enumerate() {
            let out = dec_norm.decode_packet(p).expect("normal decode");
            assert_eq!(out.len(), spf);
            assert_eq!(
                dec_norm.final_range(),
                ranges[f],
                "FEC normal-decode range mismatch {f}"
            );
        }

        let recover = |packets: &[Vec<u8>]| -> Vec<f32> {
            let mut dec = OpusDecoder::new(1);
            for p in packets.iter().take(3) {
                dec.decode_packet(p).expect("decode");
            }
            dec.decode_fec(&packets[4], spf).expect("decode_fec")
        };
        let rec = recover(&on);
        let conc = recover(&off);
        let orig = make(3);
        let corr_of = |sig: &[f32]| -> f64 {
            let frame = &sig[sig.len() - spf..];
            (0..240usize)
                .map(|d| {
                    let (mut s, mut dot, mut e) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..spf - d {
                        let a = f64::from(orig[i]);
                        let b = f64::from(frame[i + d]);
                        s += a * a;
                        dot += a * b;
                        e += b * b;
                    }
                    dot / (s.sqrt() * e.sqrt()).max(1e-9)
                })
                .fold(0.0f64, f64::max)
        };
        let (fec_corr, conceal_corr) = (corr_of(&rec), corr_of(&conc));
        assert!(fec_corr > 0.6, "reduced-rate FEC correlation {fec_corr:.3} too low");
        assert!(
            fec_corr > conceal_corr + 0.05,
            "reduced-rate FEC ({fec_corr:.3}) should still beat concealment ({conceal_corr:.3})"
        );
    }
}

//! The pre-conformance experimental codec ported from
//! `audio_samples::codecs::opus`.
//!
//! # What this is - and is not
//!
//! This module is the algorithmic scaffold the conformant implementation is
//! being built on: a working LP-based frame codec (SILK-style short-term +
//! long-term prediction with 16-bit residual quantisation), the
//! spectral-flatness mode-detection heuristic, the hybrid crossover filter,
//! and mid/side stereo helpers.
//!
//! It is **not** RFC 6716 Opus. Encoded frames are self-describing Rust
//! structs, not Opus bitstreams; nothing here interoperates with libopus. As
//! the conformant SILK (§4.2) and CELT (§4.3) layers land, the pieces here are
//! either superseded or absorbed into the real encoder's analysis stages
//! ([`crate::lpc`] already serves both).
//!
//! Known divergences from real Opus, inherited from the sketch:
//!
//! - Residuals are scalar-quantised to 16 bits rather than entropy-coded.
//! - LP coefficients travel as raw `f32`s rather than quantised LSF indices.
//! - Mode detection uses a time-domain SFM approximation rather than the
//!   short-FFT analysis of the reference encoder.
//! - The hybrid band split uses an IIR crossover; real hybrid mode runs SILK
//!   on a resampled low band instead.

use alloc::vec::Vec;
use core::fmt;

use crate::lpc::{
    LpcCoefficients, SILK_LPC_ORDER, estimate_pitch, lpc_analysis, lpc_residual, lpc_residual_stateful, lpc_synthesis,
    lpc_synthesis_stateful, ltp_residual, ltp_synthesis,
};
use crate::packet::{Bandwidth, Mode};

/// Scale factor for 16-bit residual quantisation: the normalised residual in
/// `[-1, 1]` is multiplied by this before rounding to `i16`.
const RESIDUAL_SCALE: f32 = 32_767.0;

/// Minimum gain, preventing division by zero on silent frames (≈ -160 dBFS).
const MIN_GAIN_THRESHOLD: f32 = 1e-8;

/// Crossover frequency between the SILK and CELT bands in hybrid mode
/// (RFC 6716 §2.1.2).
pub const HYBRID_CROSSOVER_HZ: f32 = 8_000.0;

/// An empty frame was passed where at least one sample is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyFrameError;

impl fmt::Display for EmptyFrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("frame must contain at least one sample")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EmptyFrameError {}

// ── Mode detection ────────────────────────────────────────────────────────────

/// Chooses an operating [`Mode`] for a frame using a spectral-flatness
/// heuristic over time-domain sub-band energies.
///
/// Bandwidth constraints are applied first: bandwidths below WB always take
/// SILK; a frame whose energy distribution is flat (noise/music-like,
/// SFM > 0.8) takes CELT; strongly peaked distributions (speech-like,
/// SFM < 0.4) take SILK; everything between is Hybrid.
///
/// > Real Opus computes spectral flatness from a short FFT; this time-domain
/// > approximation is the sketch heuristic and will be replaced.
#[must_use]
pub fn detect_mode(samples: &[f32], bandwidth: Bandwidth) -> Mode {
    /// Guard added to sub-band energies before `ln()`, so silent sub-bands
    /// cannot produce `ln(0)`.
    const LOG_GUARD_EPSILON: f64 = 1e-10;
    const N_SUB_BANDS: usize = 8;

    // Hard bandwidth constraints (RFC 6716 §2.1.2: SILK ≤ WB, CELT ≥ WB).
    if matches!(bandwidth, Bandwidth::NarrowBand | Bandwidth::MediumBand) {
        return Mode::SilkOnly;
    }

    let n = samples.len();
    let band_size = n / N_SUB_BANDS;
    if band_size == 0 {
        return Mode::CeltOnly;
    }

    let mut band_energies = [0.0f64; N_SUB_BANDS];
    for (b, energy) in band_energies.iter_mut().enumerate() {
        let start = b * band_size;
        let end = ((b + 1) * band_size).min(n);
        *energy = samples[start..end]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            + LOG_GUARD_EPSILON;
    }

    let arith_mean = band_energies.iter().sum::<f64>() / N_SUB_BANDS as f64;
    let log_sum: f64 = band_energies.iter().map(|&e| e.ln()).sum();
    let geom_mean = (log_sum / N_SUB_BANDS as f64).exp();
    let sfm = geom_mean / arith_mean;

    if sfm > 0.8 {
        Mode::CeltOnly
    } else if sfm < 0.4 {
        Mode::SilkOnly
    } else {
        Mode::Hybrid
    }
}

// ── SILK-style frame codec ────────────────────────────────────────────────────

/// Cross-frame LP filter memory for the experimental codec.
///
/// Initialise with [`SilkState::default`] at the start of a continuous signal
/// and pass the same state to every successive stateful encode/decode call.
/// Encoder history tracks the *input* samples; decoder history tracks the
/// *reconstructed* samples.
#[derive(Debug, Clone, Default)]
pub struct SilkState {
    /// Last [`SILK_LPC_ORDER`] input samples from the preceding frame.
    pub encoder_lpc_history: Vec<f32>,
    /// Last [`SILK_LPC_ORDER`] reconstructed samples from the preceding frame.
    pub decoder_lpc_history: Vec<f32>,
}

/// One encoded frame of the experimental codec: LP coefficients, a 16-bit
/// quantised prediction residual, the normalisation gain, and optional
/// long-term-prediction parameters. Self-contained - decoding needs no side
/// channel.
#[derive(Debug, Clone, PartialEq)]
pub struct SilkEncodedFrame {
    /// LP predictor coefficients from Levinson-Durbin.
    pub lpc_coeffs: LpcCoefficients,
    /// Residual normalised to `[-1, 1]` and quantised to 16 bits.
    pub residual_quantized: Vec<i16>,
    /// Peak absolute residual before normalisation; the decoder multiplies by
    /// this to restore scale.
    pub gain: f32,
    /// Detected pitch period in samples, when long-term prediction was used.
    pub pitch_lag: Option<usize>,
    /// Long-term prediction gain; zero when `pitch_lag` is `None`.
    pub ltp_gain: f32,
}

/// Encodes one frame: LP analysis → residual → peak-normalise → quantise.
///
/// # Errors
///
/// Returns [`EmptyFrameError`] for an empty input.
pub fn silk_encode_frame(samples: &[f32]) -> Result<SilkEncodedFrame, EmptyFrameError> {
    if samples.is_empty() {
        return Err(EmptyFrameError);
    }

    let lpc_coeffs = lpc_analysis(samples, SILK_LPC_ORDER);
    let residual = lpc_residual(samples, &lpc_coeffs);
    Ok(quantise(lpc_coeffs, residual, None, 0.0))
}

/// Decodes a frame from [`silk_encode_frame`]: dequantise → optional LTP
/// synthesis → LP synthesis. Exact inverse up to residual quantisation.
#[must_use]
pub fn silk_decode_frame(frame: &SilkEncodedFrame) -> Vec<f32> {
    let st_residual = dequantise(frame);
    lpc_synthesis(&st_residual, &frame.lpc_coeffs)
}

/// Encodes one frame with cross-frame LP state and single-tap long-term
/// prediction on the whitened residual.
///
/// # Errors
///
/// Returns [`EmptyFrameError`] for an empty input.
pub fn silk_encode_frame_stateful(
    samples: &[f32],
    sample_rate: u32,
    state: &mut SilkState,
) -> Result<SilkEncodedFrame, EmptyFrameError> {
    if samples.is_empty() {
        return Err(EmptyFrameError);
    }

    let lpc_coeffs = lpc_analysis(samples, SILK_LPC_ORDER);
    let st_residual = lpc_residual_stateful(samples, &lpc_coeffs, &mut state.encoder_lpc_history);

    let (pitch_lag, ltp_gain, final_residual) = match estimate_pitch(&st_residual, sample_rate) {
        Some((lag, gain)) => (Some(lag), gain, ltp_residual(&st_residual, lag, gain)),
        None => (None, 0.0, st_residual),
    };

    Ok(quantise(lpc_coeffs, final_residual, pitch_lag, ltp_gain))
}

/// Decodes a frame from [`silk_encode_frame_stateful`]; must be driven with
/// the same state sequence as the encoder.
#[must_use]
pub fn silk_decode_frame_stateful(frame: &SilkEncodedFrame, state: &mut SilkState) -> Vec<f32> {
    let st_residual = dequantise(frame);
    lpc_synthesis_stateful(&st_residual, &frame.lpc_coeffs, &mut state.decoder_lpc_history)
}

/// Peak-normalises and 16-bit-quantises a residual into a frame.
fn quantise(
    lpc_coeffs: LpcCoefficients,
    residual: Vec<f32>,
    pitch_lag: Option<usize>,
    ltp_gain: f32,
) -> SilkEncodedFrame {
    let gain = residual
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max)
        .max(MIN_GAIN_THRESHOLD);

    let residual_quantized = residual
        .iter()
        .map(|&r| {
            (r / gain * RESIDUAL_SCALE)
                .round()
                .clamp(-RESIDUAL_SCALE, RESIDUAL_SCALE) as i16
        })
        .collect();

    SilkEncodedFrame {
        lpc_coeffs,
        residual_quantized,
        gain,
        pitch_lag,
        ltp_gain,
    }
}

/// Dequantises a frame's residual and applies LTP synthesis when present.
fn dequantise(frame: &SilkEncodedFrame) -> Vec<f32> {
    let residual: Vec<f32> = frame
        .residual_quantized
        .iter()
        .map(|&q| f32::from(q) / RESIDUAL_SCALE * frame.gain)
        .collect();

    match frame.pitch_lag {
        Some(lag) => ltp_synthesis(&residual, lag, frame.ltp_gain),
        None => residual,
    }
}

// ── Hybrid crossover ──────────────────────────────────────────────────────────

/// Returns the IIR coefficient α placing the -3 dB point of
/// `y[n] = α·y[n-1] + (1-α)·x[n]` at `crossover_hz`.
///
/// Derived from `|H_LP(e^{jωc})|² = ½`:
/// `α = (2 - cos ωc) - √((2 - cos ωc)² - 1)`.
#[must_use]
pub fn crossover_alpha(crossover_hz: f32, sample_rate_hz: f32) -> f32 {
    let omega = core::f32::consts::TAU * crossover_hz / sample_rate_hz;
    let c = 2.0_f32 - omega.cos();
    (c - (c * c - 1.0_f32).sqrt()).clamp(0.0, 1.0)
}

/// Splits `samples` into `(low_band, high_band)` with the perfect
/// reconstruction property `low[n] + high[n] == samples[n]` exactly.
#[must_use]
pub fn crossover_split(samples: &[f32], alpha: f32) -> (Vec<f32>, Vec<f32>) {
    let one_minus = 1.0 - alpha;
    let mut lp = Vec::with_capacity(samples.len());
    let mut hp = Vec::with_capacity(samples.len());
    let mut prev = 0.0_f32;
    for &x in samples {
        let y = alpha * prev + one_minus * x;
        lp.push(y);
        hp.push(x - y);
        prev = y;
    }
    (lp, hp)
}

// ── Mid/side stereo ───────────────────────────────────────────────────────────

/// Mid/side matrix encode: `mid = (l + r) / 2`, `side = (l - r) / 2`.
///
/// Decorrelates typical stereo content, concentrating energy in the mid
/// channel so the side channel can be coded at a lower rate.
#[must_use]
pub fn mid_side_encode(left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mid = left.iter().zip(right).map(|(&l, &r)| (l + r) * 0.5).collect();
    let side = left.iter().zip(right).map(|(&l, &r)| (l - r) * 0.5).collect();
    (mid, side)
}

/// Mid/side matrix decode: `l = mid + side`, `r = mid - side`. Exact inverse
/// of [`mid_side_encode`].
#[must_use]
pub fn mid_side_decode(mid: &[f32], side: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let left = mid.iter().zip(side).map(|(&m, &s)| m + s).collect();
    let right = mid.iter().zip(side).map(|(&m, &s)| m - s).collect();
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn speechy(n: usize) -> Vec<f32> {
        // A decaying 220 Hz tone: periodic with a non-flat energy envelope.
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                (2.0 * core::f32::consts::PI * 220.0 * t).sin() * (-4.0 * t).exp() * 0.5
            })
            .collect()
    }

    #[test]
    fn silk_frame_round_trip_snr() {
        let samples = speechy(320);
        let frame = silk_encode_frame(&samples).expect("non-empty");
        let recovered = silk_decode_frame(&frame);
        assert_eq!(recovered.len(), samples.len());

        let signal: f64 = samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let noise: f64 = samples
            .iter()
            .zip(&recovered)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        let snr_db = 10.0 * (signal / noise.max(1e-30)).log10();
        assert!(snr_db > 40.0, "round-trip SNR {snr_db:.1} dB too low");
    }

    #[test]
    fn silk_stateful_round_trip_across_frames() {
        let all = speechy(640);
        let mut enc_state = SilkState::default();
        let mut dec_state = SilkState::default();
        let mut recovered = Vec::new();
        for chunk in all.chunks(160) {
            let frame = silk_encode_frame_stateful(chunk, 16_000, &mut enc_state).expect("frame");
            recovered.extend(silk_decode_frame_stateful(&frame, &mut dec_state));
        }
        let signal: f64 = all.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let noise: f64 = all
            .iter()
            .zip(&recovered)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        let snr_db = 10.0 * (signal / noise.max(1e-30)).log10();
        assert!(snr_db > 30.0, "stateful round-trip SNR {snr_db:.1} dB too low");
    }

    #[test]
    fn empty_frame_is_rejected() {
        assert_eq!(silk_encode_frame(&[]), Err(EmptyFrameError));
        let mut state = SilkState::default();
        assert_eq!(
            silk_encode_frame_stateful(&[], 16_000, &mut state),
            Err(EmptyFrameError)
        );
    }

    #[test]
    fn mode_detection_respects_bandwidth_constraints() {
        let noise: Vec<f32> = (0..960).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
        assert_eq!(detect_mode(&noise, Bandwidth::NarrowBand), Mode::SilkOnly);
        assert_eq!(detect_mode(&noise, Bandwidth::MediumBand), Mode::SilkOnly);
        // Uniform energy across sub-bands → flat → CELT.
        assert_eq!(detect_mode(&noise, Bandwidth::FullBand), Mode::CeltOnly);
    }

    #[test]
    fn mode_detection_peaked_energy_prefers_silk() {
        // All the energy in the first eighth of the frame.
        let mut samples = alloc::vec![0.0f32; 960];
        for (i, s) in samples.iter_mut().take(120).enumerate() {
            *s = (2.0 * core::f32::consts::PI * 200.0 * i as f32 / 48_000.0).sin();
        }
        assert_eq!(detect_mode(&samples, Bandwidth::FullBand), Mode::SilkOnly);
    }

    #[test]
    fn crossover_reconstructs_perfectly() {
        let samples = speechy(480);
        let alpha = crossover_alpha(HYBRID_CROSSOVER_HZ, 48_000.0);
        assert!(alpha > 0.0 && alpha < 1.0);
        let (low, high) = crossover_split(&samples, alpha);
        for ((&x, &l), &h) in samples.iter().zip(&low).zip(&high) {
            assert!((x - (l + h)).abs() < 1e-6, "perfect reconstruction violated");
        }
    }

    #[test]
    fn mid_side_is_exact_inverse() {
        let left = speechy(256);
        let right: Vec<f32> = left.iter().map(|&x| -0.3 * x + 0.1).collect();
        let (mid, side) = mid_side_encode(&left, &right);
        let (l2, r2) = mid_side_decode(&mid, &side);
        for (a, b) in left.iter().zip(&l2) {
            assert!((a - b).abs() < 1e-6);
        }
        for (a, b) in right.iter().zip(&r2) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

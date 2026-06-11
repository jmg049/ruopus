//! Linear-prediction primitives: Levinson-Durbin, analysis/synthesis filters,
//! and long-term (pitch) prediction.
//!
//! These are the timeless mathematical building blocks beneath SILK, Opus's
//! speech layer: the encoder-side analysis here is reusable as-is by a
//! conformant SILK encoder, and the filter conventions match the
//! analysis/synthesis split RFC 6716 §4.2 builds on.
//!
//! Ported from `audio_samples::codecs::opus::lpc` and decoupled from that
//! crate's container types: everything here operates on plain slices.
//!
//! # Round-trip correctness
//!
//! When both directions use **zero initial state** at the start of a frame,
//! the analysis/synthesis pair is a perfect inverse:
//!
//! ```text
//! Analysis:  e[n] = x[n] + Σ_{k=0}^{p-1} a[k] · x[n-1-k]
//! Synthesis: y[n] = e[n] - Σ_{k=0}^{p-1} a[k] · y[n-1-k]
//! ```
//!
//! Substituting: `y[0] = e[0] = x[0]`, `y[1] = e[1] - a[0]·y[0] = x[1]`, and so
//! on - exact up to floating-point precision. The stateful variants extend the
//! same identity across frame boundaries.

use alloc::vec;
use alloc::vec::Vec;

/// Small diagonal loading factor applied to `R[0]` for numerical stability.
///
/// Biases the autocorrelation matrix slightly away from singularity, which can
/// occur for signals with near-zero energy (0.001% perturbation of the
/// zero-lag energy, following the SILK reference convention).
const DIAGONAL_LOADING_EPSILON: f64 = 1e-5;

/// Minimum prediction error preventing numerical underflow in Levinson-Durbin.
const MIN_PREDICTION_ERROR: f64 = 1e-15;

/// Default LP predictor order used by SILK for wideband speech (RFC 6716
/// §4.2.7.5 uses order 16 for WB, 10 for NB/MB).
pub const SILK_LPC_ORDER: usize = 16;

/// LP predictor coefficients and the prediction error from Levinson-Durbin.
///
/// `coeffs[k]` is the coefficient `a[k+1]` in the all-zero analysis filter
/// `A(z) = 1 + a[1]·z⁻¹ + … + a[p]·z⁻ᵖ`, i.e. the gain on `x[n-k-1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LpcCoefficients {
    /// Predictor coefficients `a[1..=order]`, zero-indexed; length equals the
    /// order requested in [`lpc_analysis`].
    pub coeffs: Vec<f32>,
    /// Residual prediction-error energy after the final Levinson-Durbin step.
    /// Always positive; clamped at `1e-15` against underflow.
    pub prediction_error: f64,
}

/// Computes the biased autocorrelation of `samples` up to `max_lag` inclusive,
/// after applying a Hamming window.
///
/// The window reduces spectral leakage and improves the conditioning of the
/// Toeplitz system solved by Levinson-Durbin; the biased (sum-product) form
/// guarantees the matrix is positive semi-definite. Index `k` of the returned
/// vector holds `R[k]`; the length is `min(max_lag, n-1) + 1` for non-empty
/// input and `max_lag + 1` zeros for empty input.
///
/// Direct O(n·max_lag) computation: for LP orders (≤ 16 lags) this beats an
/// FFT approach on every frame size Opus uses.
#[must_use]
pub fn compute_autocorrelation(samples: &[f32], max_lag: usize) -> Vec<f64> {
    let n = samples.len();
    if n == 0 {
        return vec![0.0; max_lag + 1];
    }

    let windowed: Vec<f64> = samples
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = if n > 1 {
                let t = 2.0 * core::f64::consts::PI * i as f64 / (n - 1) as f64;
                0.54 - 0.46 * t.cos()
            } else {
                1.0
            };
            w * f64::from(x)
        })
        .collect();

    let effective_max = max_lag.min(n - 1);
    (0..=effective_max)
        .map(|lag| (0..n - lag).map(|i| windowed[i] * windowed[i + lag]).sum())
        .collect()
}

/// Solves the Yule-Walker equations `R · a = -r` via Levinson-Durbin.
///
/// Returns `None` for a silent (near-zero-energy) frame. Diagonal loading of
/// `R[0] × 10⁻⁵` is applied before the recursion for numerical stability.
///
/// `autocorr` must hold `R[0..=order]`.
#[must_use]
pub fn levinson_durbin(autocorr: &[f64], order: usize) -> Option<LpcCoefficients> {
    if autocorr.len() < order + 1 || autocorr[0] < 1e-12 {
        return None;
    }

    let mut r = autocorr.to_vec();
    r[0] *= 1.0 + DIAGONAL_LOADING_EPSILON;

    let mut a = vec![0.0f64; order];
    let mut a_prev = vec![0.0f64; order];
    let mut error = r[0];

    for m in 0..order {
        // Reflection coefficient k_m.
        let mut num = r[m + 1];
        for i in 0..m {
            num += a_prev[i] * r[m - i];
        }
        let k = -num / error;

        for i in 0..m {
            a[i] = a_prev[i] + k * a_prev[m - 1 - i];
        }
        a[m] = k;

        error = (error * (1.0 - k * k)).max(MIN_PREDICTION_ERROR);
        // Swap rather than clone: `a_prev` needs the current `a` next
        // iteration; `a` is fully overwritten then.
        core::mem::swap(&mut a, &mut a_prev);
    }

    Some(LpcCoefficients {
        coeffs: a_prev.iter().map(|&v| v as f32).collect(),
        prediction_error: error,
    })
}

/// LP analysis of one signal frame: windowed autocorrelation followed by
/// Levinson-Durbin.
///
/// The effective order is clamped to `min(order, samples.len() / 2)` (at least
/// 1) so the system stays well-posed; silent frames yield a zero predictor.
#[must_use]
pub fn lpc_analysis(samples: &[f32], order: usize) -> LpcCoefficients {
    let effective_order = order.min(samples.len() / 2).max(1);
    let autocorr = compute_autocorrelation(samples, effective_order);
    levinson_durbin(&autocorr, effective_order).unwrap_or_else(|| LpcCoefficients {
        coeffs: vec![0.0; effective_order],
        prediction_error: 1.0,
    })
}

/// LP analysis (all-zero) filter with zero initial state:
/// `e[n] = x[n] + Σ coeffs[k] · x[n-1-k]`.
#[must_use]
pub fn lpc_residual(samples: &[f32], coeffs: &LpcCoefficients) -> Vec<f32> {
    let order = coeffs.coeffs.len();
    let mut residual = Vec::with_capacity(samples.len());
    for n in 0..samples.len() {
        let mut sum = 0.0f32;
        for k in 0..order.min(n) {
            sum += coeffs.coeffs[k] * samples[n - 1 - k];
        }
        residual.push(samples[n] + sum);
    }
    residual
}

/// LP synthesis (all-pole) filter with zero initial state:
/// `y[n] = e[n] - Σ coeffs[k] · y[n-1-k]`. Exact inverse of [`lpc_residual`].
#[must_use]
pub fn lpc_synthesis(residual: &[f32], coeffs: &LpcCoefficients) -> Vec<f32> {
    let order = coeffs.coeffs.len();
    let mut output: Vec<f32> = Vec::with_capacity(residual.len());
    for n in 0..residual.len() {
        let mut sum = 0.0f32;
        for k in 0..order.min(n) {
            sum += coeffs.coeffs[k] * output[n - 1 - k];
        }
        output.push(residual[n] - sum);
    }
    output
}

/// [`lpc_residual`] with cross-frame input history.
///
/// `state` holds the last `order` input samples of the preceding frame
/// (zero-padded on the left when fewer are available) and is updated in place
/// to the tail of `samples`, eliminating boundary artefacts between
/// consecutive frames.
#[must_use]
pub fn lpc_residual_stateful(samples: &[f32], coeffs: &LpcCoefficients, state: &mut Vec<f32>) -> Vec<f32> {
    let order = coeffs.coeffs.len();
    normalise_state(state, order);

    let mut residual = Vec::with_capacity(samples.len());
    for n in 0..samples.len() {
        let mut sum = 0.0f32;
        for k in 0..order {
            let pos = n as isize - 1 - k as isize;
            let x = if pos >= 0 {
                samples[pos as usize]
            } else {
                state[(order as isize + pos) as usize]
            };
            sum += coeffs.coeffs[k] * x;
        }
        residual.push(samples[n] + sum);
    }

    update_state(state, samples, order);
    residual
}

/// [`lpc_synthesis`] with cross-frame output history; mirror of
/// [`lpc_residual_stateful`].
#[must_use]
pub fn lpc_synthesis_stateful(residual: &[f32], coeffs: &LpcCoefficients, state: &mut Vec<f32>) -> Vec<f32> {
    let order = coeffs.coeffs.len();
    normalise_state(state, order);

    let mut output: Vec<f32> = Vec::with_capacity(residual.len());
    for (n, &e) in residual.iter().enumerate() {
        let mut sum = 0.0f32;
        for k in 0..order {
            let pos = n as isize - 1 - k as isize;
            let y = if pos >= 0 {
                output[pos as usize]
            } else {
                state[(order as isize + pos) as usize]
            };
            sum += coeffs.coeffs[k] * y;
        }
        output.push(e - sum);
    }

    update_state(state, &output, order);
    output
}

/// Estimates the pitch period and long-term-prediction gain of a frame.
///
/// Searches the normalised autocorrelation over lags corresponding to
/// 50-500 Hz. Returns `Some((lag, gain))` with `gain = R[lag]/R[0]` clamped to
/// `[0, 0.9]` for filter stability, or `None` when the frame is too short, has
/// negligible energy, or shows no clear periodicity (`gain < 0.3`).
#[must_use]
pub fn estimate_pitch(samples: &[f32], sample_rate: u32) -> Option<(usize, f32)> {
    let n = samples.len();
    let fs = sample_rate as usize;

    let t_min = (fs / 500).max(2);
    let t_max = (fs / 50).min(n / 2);

    if t_max <= t_min || n <= t_max {
        return None;
    }

    let r0: f64 = samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    if r0 < 1e-10 {
        return None;
    }

    let (best_lag, best_r) = (t_min..=t_max)
        .map(|lag| {
            let r: f64 = (0..n - lag)
                .map(|i| f64::from(samples[i]) * f64::from(samples[i + lag]))
                .sum();
            (lag, r / r0)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))?;

    if best_r < 0.3 {
        return None;
    }

    Some((best_lag, (best_r as f32).clamp(0.0, 0.9)))
}

/// Single-tap LTP analysis (FIR) filter: `d[n] = e[n] - gain · e[n - lag]`,
/// zero initial state.
#[must_use]
pub fn ltp_residual(samples: &[f32], lag: usize, gain: f32) -> Vec<f32> {
    samples
        .iter()
        .enumerate()
        .map(|(n, &e)| e - gain * if n >= lag { samples[n - lag] } else { 0.0 })
        .collect()
}

/// Single-tap LTP synthesis (IIR) filter: `e[n] = d[n] + gain · e[n - lag]`.
/// Inverse of [`ltp_residual`]; stable for `gain < 1`.
#[must_use]
pub fn ltp_synthesis(residual: &[f32], lag: usize, gain: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; residual.len()];
    for n in 0..residual.len() {
        let prev = if n >= lag { output[n - lag] } else { 0.0 };
        output[n] = residual[n] + gain * prev;
    }
    output
}

/// Ensures `state` has exactly `order` elements, zero-padding on the left.
fn normalise_state(state: &mut Vec<f32>, order: usize) {
    while state.len() < order {
        state.insert(0, 0.0);
    }
    if state.len() > order {
        let excess = state.len() - order;
        state.drain(0..excess);
    }
}

/// Updates `state` to hold the last `order` elements of `samples`.
fn update_state(state: &mut Vec<f32>, samples: &[f32], order: usize) {
    let n = samples.len();
    if n >= order {
        state.clear();
        state.extend_from_slice(&samples[n - order..]);
    } else {
        let tail: Vec<f32> = state[n..].to_vec();
        state.clear();
        state.extend_from_slice(&tail);
        state.extend_from_slice(samples);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn sine(n: usize, freq: f32, rate: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * core::f32::consts::PI * freq * i as f32 / rate).sin() * 0.5)
            .collect()
    }

    #[test]
    fn round_trip_no_quantization() {
        let samples = sine(64, 440.0, 44_100.0);
        let coeffs = lpc_analysis(&samples, SILK_LPC_ORDER);
        let residual = lpc_residual(&samples, &coeffs);
        let recovered = lpc_synthesis(&residual, &coeffs);
        for (orig, rec) in samples.iter().zip(&recovered) {
            assert!((orig - rec).abs() < 1e-4, "round-trip error {:.2e}", (orig - rec).abs());
        }
    }

    #[test]
    fn levinson_returns_none_for_silence() {
        let autocorr = vec![0.0f64; SILK_LPC_ORDER + 1];
        assert!(levinson_durbin(&autocorr, SILK_LPC_ORDER).is_none());
    }

    #[test]
    fn stateful_round_trip_single_frame() {
        let samples = sine(64, 440.0, 44_100.0);
        let coeffs = lpc_analysis(&samples, SILK_LPC_ORDER);
        let mut enc_state = Vec::new();
        let mut dec_state = Vec::new();
        let residual = lpc_residual_stateful(&samples, &coeffs, &mut enc_state);
        let recovered = lpc_synthesis_stateful(&residual, &coeffs, &mut dec_state);
        for (o, r) in samples.iter().zip(&recovered) {
            assert!((o - r).abs() < 1e-4, "stateful round-trip error {:.2e}", (o - r).abs());
        }
    }

    #[test]
    fn stateful_cross_frame_continuity() {
        let n = 64usize;
        let all = sine(n * 2, 440.0, 44_100.0);
        let coeffs0 = lpc_analysis(&all[..n], SILK_LPC_ORDER);
        let coeffs1 = lpc_analysis(&all[n..], SILK_LPC_ORDER);

        let mut enc_state = Vec::new();
        let mut dec_state = Vec::new();
        let res0 = lpc_residual_stateful(&all[..n], &coeffs0, &mut enc_state);
        let res1 = lpc_residual_stateful(&all[n..], &coeffs1, &mut enc_state);
        let mut recovered = lpc_synthesis_stateful(&res0, &coeffs0, &mut dec_state);
        recovered.extend(lpc_synthesis_stateful(&res1, &coeffs1, &mut dec_state));

        for (o, r) in all.iter().zip(&recovered) {
            assert!((o - r).abs() < 1e-3, "cross-frame error {:.2e}", (o - r).abs());
        }
    }

    #[test]
    fn ltp_round_trip() {
        let samples = sine(200, 220.0, 44_100.0);
        let (lag, gain) = (100, 0.7);
        let residual = ltp_residual(&samples, lag, gain);
        let recovered = ltp_synthesis(&residual, lag, gain);
        for (o, r) in samples.iter().zip(&recovered) {
            assert!((o - r).abs() < 1e-4, "LTP round-trip error {:.2e}", (o - r).abs());
        }
    }

    #[test]
    fn pitch_detection_finds_sine_period() {
        let freq_hz = 220.0_f32;
        let sample_rate = 44_100_u32;
        let expected_lag = (sample_rate as f32 / freq_hz).round() as usize; // ~200
        let samples: Vec<f32> = (0..882)
            .map(|i| (2.0 * core::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect();
        let (lag, gain) = estimate_pitch(&samples, sample_rate).expect("pitch detected");
        assert!(
            lag.abs_diff(expected_lag) <= 2,
            "detected lag {lag}, expected ~{expected_lag}"
        );
        assert!(gain > 0.3, "LTP gain {gain:.2} too low");
    }
}

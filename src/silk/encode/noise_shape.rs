//! Noise-shaping analysis for the SILK encoder (RFC 6716 §5.2).
//!
//! [`noise_shape_analysis`] derives the per-subframe noise-shaping
//! parameters the noise-shaping quantiser uses to colour the quantisation
//! noise so it hides under the signal: the short-term AR shaping filter
//! (`AR`), the spectral tilt, the low-frequency shaper (`LF_*_shp`), the
//! harmonic shaper (voiced only), and the pre-quantisation per-subframe
//! gains (the shaping-residual energy with the SNR-driven gain tweak). It
//! also picks the sparseness-driven quantiser offset type.
//!
//! This is the float analysis half; its outputs are converted to the Q
//! formats [`super::nsq`] consumes here.
//!
//! The warped-filter path (used by the reference at complexity ≥ 5) is
//! implemented for completeness, but the
//! encode driver currently selects the unwarped configuration so the plain
//! [`super::nsq::nsq`] sees ordinary (non-warped) shaping coefficients.

extern crate alloc;

use super::super::indices::{MAX_NB_SUBFR, TYPE_VOICED};
use super::dsp::{apply_sine_window, autocorrelation, bwexpander, energy, k2a, schur};

/// `MAX_SHAPE_LPC_ORDER`.
pub(crate) const MAX_SHAPE_LPC_ORDER: usize = 24;
/// `SHAPE_LPC_WIN_MAX` (15 ms × 16 kHz).
const SHAPE_LPC_WIN_MAX: usize = 15 * 16;

// Tuning parameters.
const BANDWIDTH_EXPANSION: f32 = 0.94;
const FIND_PITCH_WHITE_NOISE_FRACTION: f32 = 1e-3;
const SHAPE_WHITE_NOISE_FRACTION: f32 = 3e-5;
const BG_SNR_DECR_DB: f32 = 2.0;
const HARM_SNR_INCR_DB: f32 = 2.0;
const ENERGY_VARIATION_THRESHOLD_QNT_OFFSET: f32 = 0.6;
const LOW_FREQ_SHAPING: f32 = 4.0;
const LOW_QUALITY_LOW_FREQ_SHAPING_DECR: f32 = 0.5;
const HP_NOISE_COEF: f32 = 0.25;
const HARM_HP_NOISE_COEF: f32 = 0.35;
const HARMONIC_SHAPING: f32 = 0.3;
const HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING: f32 = 0.2;
const SUBFR_SMTH_COEF: f32 = 0.4;
const MIN_QGAIN_DB: f32 = 2.0;
const SUB_FRAME_LENGTH_MS: i32 = 5;

/// Cross-frame noise-shaping state (`silk_shape_state_FLP`): the smoothed
/// harmonic-shaping gain and tilt carried between frames.
#[derive(Clone, Copy, Default)]
pub(crate) struct ShapeState {
    pub harm_shape_gain_smth: f32,
    pub tilt_smth: f32,
}

/// Per-frame inputs to [`noise_shape_analysis`].
#[derive(Clone, Copy)]
pub(crate) struct NoiseShapeConfig {
    pub fs_khz: i32,
    pub nb_subfr: usize,
    pub subfr_length: usize,
    /// Lookahead used for the shaping analysis windows (`la_shape`).
    pub la_shape: usize,
    /// Window length `SUB_FRAME_LENGTH_MS*fs_kHz + 2*la_shape`.
    pub shape_win_length: usize,
    pub shaping_lpc_order: usize,
    /// Warping coefficient in Q16 (0 selects the unwarped path).
    pub warping_q16: i32,
    pub signal_type: i32,
    pub snr_db_q7: i32,
    pub speech_activity_q8: i32,
    /// Quality of the lowest two VAD bands (Q15); `[32768; 2]` is clean.
    pub input_quality_bands_q15: [i32; 2],
    pub use_cbr: bool,
    /// Normalised long-term (pitch) correlation, voiced only.
    pub ltp_corr: f32,
    /// Prediction gain of the pitch analysis whitening filter.
    pub pred_gain: f32,
    /// Pitch lags per subframe (voiced low-frequency shaping only).
    pub pitch_l: [i32; MAX_NB_SUBFR],
}

/// The Q-formatted noise-shaping parameters [`super::nsq::nsq`] consumes,
/// plus the float gains and quality measures the gain/lambda stage needs.
pub(crate) struct NoiseShapeResult {
    pub ar_q13: [i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    pub tilt_q14: [i32; MAX_NB_SUBFR],
    pub lf_shp_q14: [i32; MAX_NB_SUBFR],
    pub harm_shape_gain_q14: [i32; MAX_NB_SUBFR],
    /// Pre-quantisation per-subframe gains (linear), fed to `process_gains`.
    pub gains: [f32; MAX_NB_SUBFR],
    pub quant_offset_type: i32,
    pub input_quality: f32,
    pub coding_quality: f32,
}

/// Logistic sigmoid `1 / (1 + e^-x)`.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Round to nearest integer.
fn float2int(x: f32) -> i32 {
    x.round() as i32
}

/// Warped autocorrelation through a chain
/// of first-order allpass sections (warping coefficient `warping`).
fn warped_autocorrelation(corr: &mut [f32], input: &[f32], warping: f32, order: usize) {
    debug_assert!(order & 1 == 0);
    let warping = f64::from(warping);
    let mut state = [0.0f64; MAX_SHAPE_LPC_ORDER + 1];
    let mut c = [0.0f64; MAX_SHAPE_LPC_ORDER + 1];
    for &xn in input {
        let mut tmp1 = f64::from(xn);
        let mut i = 0;
        while i < order {
            let tmp2 = state[i] + warping * state[i + 1] - warping * tmp1;
            state[i] = tmp1;
            c[i] += state[0] * tmp1;
            tmp1 = state[i + 1] + warping * state[i + 2] - warping * tmp2;
            state[i + 1] = tmp2;
            c[i + 1] += state[0] * tmp2;
            i += 2;
        }
        state[order] = tmp1;
        c[order] += state[0] * tmp1;
    }
    for i in 0..order + 1 {
        corr[i] = c[i] as f32;
    }
}

/// `warped_gain`: gain that flattens the warped filter's mean log response.
fn warped_gain(coefs: &[f32], lambda: f32, order: usize) -> f32 {
    let lambda = -lambda;
    let mut gain = coefs[order - 1];
    for i in (0..order - 1).rev() {
        gain = lambda * gain + coefs[i];
    }
    1.0 / (1.0 - lambda * gain)
}

/// `limit_coefs`: bandwidth-expand until all coefficients are below `limit`.
fn limit_coefs(coefs: &mut [f32], limit: f32, order: usize) {
    for iter in 0..10 {
        let mut maxabs = -1.0f32;
        let mut ind = 0usize;
        for (i, &c) in coefs.iter().enumerate().take(order) {
            let t = c.abs();
            if t > maxabs {
                maxabs = t;
                ind = i;
            }
        }
        if maxabs <= limit {
            return;
        }
        let chirp = 0.99 - (0.8 + 0.1 * iter as f32) * (maxabs - limit) / (maxabs * (ind as f32 + 1.0));
        bwexpander(coefs, order, chirp);
    }
}

/// `warped_true2monic_coefs`: convert warped true coefficients to monic
/// pseudo-warped coefficients, bandwidth-limiting the amplitude.
fn warped_true2monic_coefs(coefs: &mut [f32], lambda: f32, limit: f32, order: usize) {
    for i in (1..order).rev() {
        coefs[i - 1] -= lambda * coefs[i];
    }
    let mut gain = (1.0 - lambda * lambda) / (1.0 + lambda * coefs[0]);
    for c in coefs.iter_mut().take(order) {
        *c *= gain;
    }
    for iter in 0..10 {
        let mut maxabs = -1.0f32;
        let mut ind = 0usize;
        for (i, &c) in coefs.iter().enumerate().take(order) {
            let t = c.abs();
            if t > maxabs {
                maxabs = t;
                ind = i;
            }
        }
        if maxabs <= limit {
            return;
        }
        for i in 1..order {
            coefs[i - 1] += lambda * coefs[i];
        }
        gain = 1.0 / gain;
        for c in coefs.iter_mut().take(order) {
            *c *= gain;
        }
        let chirp = 0.99 - (0.8 + 0.1 * iter as f32) * (maxabs - limit) / (maxabs * (ind as f32 + 1.0));
        bwexpander(coefs, order, chirp);
        for i in (1..order).rev() {
            coefs[i - 1] -= lambda * coefs[i];
        }
        gain = (1.0 - lambda * lambda) / (1.0 + lambda * coefs[0]);
        for c in coefs.iter_mut().take(order) {
            *c *= gain;
        }
    }
}

/// Derive the per-subframe noise-shaping parameters. `x_buf` holds `la_shape` samples of
/// history, then the `frame_length` frame, then `la_shape` samples of
/// lookahead (length `frame_length + 2*la_shape`); the frame starts at
/// index `la_shape`. `pitch_res` is the pitch-analysis residual over the
/// frame (used only for the unvoiced sparseness measure).
#[allow(clippy::needless_range_loop, reason = "computed index ranges mirror the reference")]
pub(crate) fn noise_shape_analysis(
    state: &mut ShapeState,
    cfg: &NoiseShapeConfig,
    pitch_res: &[f32],
    x_buf: &[f32],
) -> NoiseShapeResult {
    let order = cfg.shaping_lpc_order;
    let frame_length = cfg.nb_subfr * cfg.subfr_length;
    debug_assert_eq!(x_buf.len(), frame_length + 2 * cfg.la_shape);

    let mut res = NoiseShapeResult {
        ar_q13: [0; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
        tilt_q14: [0; MAX_NB_SUBFR],
        lf_shp_q14: [0; MAX_NB_SUBFR],
        harm_shape_gain_q14: [0; MAX_NB_SUBFR],
        gains: [0.0; MAX_NB_SUBFR],
        quant_offset_type: 0,
        input_quality: 0.0,
        coding_quality: 0.0,
    };

    // --- Gain control ---
    let mut snr_adj_db = cfg.snr_db_q7 as f32 * (1.0 / 128.0);
    let input_quality =
        0.5 * (cfg.input_quality_bands_q15[0] + cfg.input_quality_bands_q15[1]) as f32 * (1.0 / 32768.0);
    let coding_quality = sigmoid(0.25 * (snr_adj_db - 20.0));
    res.input_quality = input_quality;
    res.coding_quality = coding_quality;

    if !cfg.use_cbr {
        let b = 1.0 - cfg.speech_activity_q8 as f32 * (1.0 / 256.0);
        snr_adj_db -= BG_SNR_DECR_DB * coding_quality * (0.5 + 0.5 * input_quality) * b * b;
    }
    if cfg.signal_type == TYPE_VOICED {
        snr_adj_db += HARM_SNR_INCR_DB * cfg.ltp_corr;
    } else {
        snr_adj_db += (-0.4 * cfg.snr_db_q7 as f32 * (1.0 / 128.0) + 6.0) * (1.0 - input_quality);
    }

    // --- Sparseness / quantiser offset ---
    if cfg.signal_type == TYPE_VOICED {
        res.quant_offset_type = 0;
    } else {
        let n_samples = 2 * cfg.fs_khz as usize;
        let n_segs = (SUB_FRAME_LENGTH_MS * cfg.nb_subfr as i32) as usize / 2;
        let mut energy_variation = 0.0f32;
        let mut log_energy_prev = 0.0f32;
        for k in 0..n_segs {
            let seg = &pitch_res[k * n_samples..(k + 1) * n_samples];
            let nrg = n_samples as f32 + energy(seg) as f32;
            let log_energy = nrg.log2();
            if k > 0 {
                energy_variation += (log_energy - log_energy_prev).abs();
            }
            log_energy_prev = log_energy;
        }
        res.quant_offset_type =
            i32::from(energy_variation <= ENERGY_VARIATION_THRESHOLD_QNT_OFFSET * (n_segs as f32 - 1.0));
    }

    // --- Bandwidth expansion / warping control ---
    let strength = FIND_PITCH_WHITE_NOISE_FRACTION * cfg.pred_gain;
    let bw_exp = BANDWIDTH_EXPANSION / (1.0 + strength * strength);
    let warping = cfg.warping_q16 as f32 / 65536.0 + 0.01 * coding_quality;

    // --- AR shaping coefficients and gains, per subframe ---
    let flat_part = cfg.fs_khz as usize * 3;
    let slope_part = (cfg.shape_win_length - flat_part) / 2;
    let mut x_windowed = [0.0f32; SHAPE_LPC_WIN_MAX];
    let mut auto_corr = [0.0f32; MAX_SHAPE_LPC_ORDER + 1];
    let mut rc = [0.0f32; MAX_SHAPE_LPC_ORDER + 1];
    // Float AR coefficients, same per-subframe layout as `ar_q13`.
    let mut ar_f = [0.0f32; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER];

    for k in 0..cfg.nb_subfr {
        // x_ptr points at this block's start in x_buf (x - la_shape + k*subfr).
        let x0 = k * cfg.subfr_length;
        let x_ptr = &x_buf[x0..x0 + cfg.shape_win_length];

        apply_sine_window(&mut x_windowed[..slope_part], &x_ptr[..slope_part], 1, slope_part);
        x_windowed[slope_part..slope_part + flat_part].copy_from_slice(&x_ptr[slope_part..slope_part + flat_part]);
        let s = slope_part + flat_part;
        apply_sine_window(
            &mut x_windowed[s..s + slope_part],
            &x_ptr[s..s + slope_part],
            2,
            slope_part,
        );

        if cfg.warping_q16 > 0 {
            warped_autocorrelation(&mut auto_corr, &x_windowed[..cfg.shape_win_length], warping, order);
        } else {
            autocorrelation(&mut auto_corr, &x_windowed[..cfg.shape_win_length], order + 1);
        }
        auto_corr[0] += auto_corr[0] * SHAPE_WHITE_NOISE_FRACTION + 1.0;

        let nrg = schur(&mut rc, &auto_corr, order);
        let ar = &mut ar_f[k * MAX_SHAPE_LPC_ORDER..k * MAX_SHAPE_LPC_ORDER + order];
        k2a(ar, &rc, order);
        let mut gain = nrg.max(0.0).sqrt();

        if cfg.warping_q16 > 0 {
            gain *= warped_gain(ar, warping, order);
        }
        bwexpander(ar, order, bw_exp);
        if cfg.warping_q16 > 0 {
            warped_true2monic_coefs(ar, warping, 3.999, order);
        } else {
            limit_coefs(ar, 3.999, order);
        }
        res.gains[k] = gain;
    }

    // --- Gain tweaking ---
    let gain_mult = 2.0f32.powf(-0.16 * snr_adj_db);
    let gain_add = 2.0f32.powf(0.16 * MIN_QGAIN_DB);
    for k in 0..cfg.nb_subfr {
        res.gains[k] = res.gains[k] * gain_mult + gain_add;
    }

    // --- Low-frequency shaping and noise tilt ---
    let mut strength = LOW_FREQ_SHAPING
        * (1.0 + LOW_QUALITY_LOW_FREQ_SHAPING_DECR * (cfg.input_quality_bands_q15[0] as f32 * (1.0 / 32768.0) - 1.0));
    strength *= cfg.speech_activity_q8 as f32 * (1.0 / 256.0);
    let (mut lf_ma, mut lf_ar) = ([0.0f32; MAX_NB_SUBFR], [0.0f32; MAX_NB_SUBFR]);
    let tilt;
    if cfg.signal_type == TYPE_VOICED {
        for k in 0..cfg.nb_subfr {
            let b = 0.2 / cfg.fs_khz as f32 + 3.0 / cfg.pitch_l[k] as f32;
            lf_ma[k] = -1.0 + b;
            lf_ar[k] = 1.0 - b - b * strength;
        }
        tilt =
            -HP_NOISE_COEF - (1.0 - HP_NOISE_COEF) * HARM_HP_NOISE_COEF * cfg.speech_activity_q8 as f32 * (1.0 / 256.0);
    } else {
        let b = 1.3 / cfg.fs_khz as f32;
        lf_ma[0] = -1.0 + b;
        lf_ar[0] = 1.0 - b - b * strength * 0.6;
        for k in 1..cfg.nb_subfr {
            lf_ma[k] = lf_ma[0];
            lf_ar[k] = lf_ar[0];
        }
        tilt = -HP_NOISE_COEF;
    }

    // --- Harmonic shaping ---
    let harm_shape_gain = if cfg.signal_type == TYPE_VOICED {
        let mut g = HARMONIC_SHAPING;
        g += HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING * (1.0 - (1.0 - coding_quality) * input_quality);
        g * cfg.ltp_corr.max(0.0).sqrt()
    } else {
        0.0
    };

    // --- Smooth over subframes and convert to the NSQ's Q formats ---
    for k in 0..cfg.nb_subfr {
        state.harm_shape_gain_smth += SUBFR_SMTH_COEF * (harm_shape_gain - state.harm_shape_gain_smth);
        state.tilt_smth += SUBFR_SMTH_COEF * (tilt - state.tilt_smth);
        res.harm_shape_gain_q14[k] = float2int(state.harm_shape_gain_smth * 16384.0);
        res.tilt_q14[k] = float2int(state.tilt_smth * 16384.0);
        res.lf_shp_q14[k] = (float2int(lf_ar[k] * 16384.0) << 16) | (float2int(lf_ma[k] * 16384.0) & 0xffff);
        for j in 0..order {
            res.ar_q13[k * MAX_SHAPE_LPC_ORDER + j] = float2int(ar_f[k * MAX_SHAPE_LPC_ORDER + j] * 8192.0) as i16;
        }
    }

    res
}

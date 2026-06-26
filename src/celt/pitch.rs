//! Pitch analysis for the CELT pre-filter (RFC 6716 §5.3).
//!
//! The encoder estimates a pitch period and gain so the comb pre-filter can
//! attenuate the strong harmonic structure of voiced/tonal signals before
//! the MDCT, leaving a smaller residual to code. The chain is:
//!
//! 1. [`pitch_downsample`] - low-pass and decimate by two (a 4-tap LPC whitening filter plus a zero), folding both
//!    channels to mono.
//! 2. [`pitch_search`] - a two-stage decimated normalised cross-correlation giving a coarse integer lag.
//! 3. [`remove_doubling`] - refine the lag, rejecting octave errors, and return the pitch gain.
//!
//! Everything here is the float build; the values feed only encoder
//! *decisions*, so they need not be bit-exact with the fixed-point build.
#![allow(
    clippy::needless_range_loop,
    reason = "sample/lag indices mirror the reference loops"
)]

use alloc::vec;
use alloc::vec::Vec;

/// Longest comb-filter period (`COMBFILTER_MAXPERIOD`).
pub(crate) const COMBFILTER_MAXPERIOD: usize = 1024;
/// Shortest comb-filter period (`COMBFILTER_MINPERIOD`).
pub(crate) const COMBFILTER_MINPERIOD: usize = 15;

/// Inner product `Σ x[i]·y[i]` (`celt_inner_prod`), via the SIMD kernel.
fn inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
    crate::simd::dot(&x[..n], y)
}

/// Two inner products sharing the first operand (`dual_inner_prod`).
fn dual_inner_prod(x: &[f32], y01: &[f32], y02: &[f32], n: usize) -> (f32, f32) {
    crate::simd::dual_dot(&x[..n], y01, y02)
}

/// Cross-correlation `xcorr[i] = Σ_j x[j]·y[i+j]` (`celt_pitch_xcorr`), via the
/// batched 4-lag SIMD kernel.
fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    crate::simd::pitch_xcorr(x, y, &mut xcorr[..max_pitch], len);
}

/// Picks the two best lags by normalised correlation (`find_best_pitch`).
fn find_best_pitch(xcorr: &[f32], y: &[f32], len: usize, max_pitch: usize) -> [usize; 2] {
    let mut best_num = [-1.0f32; 2];
    let mut best_den = [0.0f32; 2];
    let mut best_pitch = [0usize, 1];
    let mut syy = 1.0f32;
    for j in 0..len {
        syy += y[j] * y[j];
    }
    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            // Scaled down to avoid over/underflow when squaring.
            let xcorr16 = xcorr[i] * 1e-12;
            let num = xcorr16 * xcorr16;
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        syy = syy.max(1.0);
    }
    best_pitch
}

/// In-place 5-tap FIR (`celt_fir5`), used by the downsampler's whitening.
///
/// `out[i] = x[i] + Σ_k num[k]·x[i-1-k]` (zero history before the start). The
/// filter is non-recursive, so with the input copied behind a 5-sample zero
/// history it is a forward 6-tap convolution the SIMD kernel handles. The
/// result feeds the pitch search (an encoder decision), so FMA rounding is fine.
fn celt_fir5(x: &mut [f32], num: &[f32; 5]) {
    let n = x.len();
    let mut inp = vec![0.0f32; n + 5];
    inp[5..].copy_from_slice(x);
    // Coefficients ordered for the forward convolution: oldest tap first, then
    // the identity term for x[i] itself.
    let c = [num[4], num[3], num[2], num[1], num[0], 1.0];
    crate::simd::fir6(x, &inp, c);
}

/// Autocorrelation `ac[k] = Σ_{i≥k} x[i]·x[i-k]` for `k = 0..=lag`
/// (`_celt_autocorr` with no window).
fn autocorr(x: &[f32], lag: usize) -> Vec<f32> {
    let n = x.len();
    let mut ac = vec![0.0f32; lag + 1];
    for (k, a) in ac.iter_mut().enumerate() {
        // ac[k] = Σ_j x[j]·x[j+k] over j in 0..n-k.
        *a = crate::simd::dot(&x[..n - k], &x[k..]);
    }
    ac
}

/// Levinson-Durbin LPC of order `p` (`_celt_lpc`, float build).
fn lpc(ac: &[f32], p: usize) -> Vec<f32> {
    let mut lpc = vec![0.0f32; p];
    let mut error = ac[0];
    if ac[0] > 1e-10 {
        for i in 0..p {
            let mut rr = 0.0f32;
            for j in 0..i {
                rr += lpc[j] * ac[i - j];
            }
            rr += ac[i + 1];
            let r = -rr / error;
            lpc[i] = r;
            for j in 0..(i + 1) >> 1 {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + r * tmp2;
                lpc[i - 1 - j] = tmp2 + r * tmp1;
            }
            error -= r * r * error;
            // Bail out once we hit ~30 dB of prediction gain.
            if error <= 0.001 * ac[0] {
                break;
            }
        }
    }
    lpc
}

/// Low-passes and decimates the (mono or stereo) signal by two, then applies
/// a 4th-order LPC whitening filter plus a zero (`pitch_downsample`).
/// Returns `x_lp` of length `len / 2`.
#[must_use]
pub(crate) fn pitch_downsample(pre: &[&[f32]], len: usize) -> Vec<f32> {
    const C1: f32 = 0.8;
    let half = len >> 1;
    let mut x_lp = vec![0.0f32; half];
    for i in 1..half {
        x_lp[i] = 0.25 * pre[0][2 * i - 1] + 0.25 * pre[0][2 * i + 1] + 0.5 * pre[0][2 * i];
    }
    x_lp[0] = 0.25 * pre[0][1] + 0.5 * pre[0][0];
    if pre.len() == 2 {
        for i in 1..half {
            x_lp[i] += 0.25 * pre[1][2 * i - 1] + 0.25 * pre[1][2 * i + 1] + 0.5 * pre[1][2 * i];
        }
        x_lp[0] += 0.25 * pre[1][1] + 0.5 * pre[1][0];
    }

    let mut ac = autocorr(&x_lp, 4);
    // Noise floor at -40 dB and a light lag window.
    ac[0] *= 1.0001;
    for i in 1..=4 {
        ac[i] -= ac[i] * (0.008 * i as f32) * (0.008 * i as f32);
    }
    let mut coef = lpc(&ac, 4);
    let mut tmp = 1.0f32;
    for c in &mut coef {
        tmp *= 0.9;
        *c *= tmp;
    }
    // Add a zero to flatten the response.
    let lpc2 = [
        coef[0] + 0.8,
        coef[1] + C1 * coef[0],
        coef[2] + C1 * coef[1],
        coef[3] + C1 * coef[2],
        C1 * coef[3],
    ];
    celt_fir5(&mut x_lp, &lpc2);
    x_lp
}

/// Pitch gain `xy / sqrt(1 + xx·yy)` (`compute_pitch_gain`, float build).
fn compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    xy / (1.0 + xx * yy).sqrt()
}

/// Two-stage decimated cross-correlation pitch search (`pitch_search`).
/// `x_lp` is the current frame's down-sampled signal, `y` the full
/// down-sampled history+frame; `len` is the *original* frame length and
/// `max_pitch` the search range. Returns the coarse integer lag.
#[must_use]
pub(crate) fn pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
    let lag = len + max_pitch;
    // Decimate by two again for the coarse search.
    let x_lp4: Vec<f32> = (0..len >> 2).map(|j| x_lp[2 * j]).collect();
    let y_lp4: Vec<f32> = (0..lag >> 2).map(|j| y[2 * j]).collect();

    let mut xcorr = vec![0.0f32; max_pitch >> 2];
    pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr, len >> 2, max_pitch >> 2);
    let best = find_best_pitch(&xcorr, &y_lp4, len >> 2, max_pitch >> 2);

    // Finer search at 2× decimation, only near the coarse candidates.
    let mut xcorr2 = vec![0.0f32; max_pitch >> 1];
    for i in 0..max_pitch >> 1 {
        if (i as i32 - 2 * best[0] as i32).abs() > 2 && (i as i32 - 2 * best[1] as i32).abs() > 2 {
            continue;
        }
        let sum = inner_prod(x_lp, &y[i..], len >> 1);
        xcorr2[i] = sum.max(-1.0);
    }
    let best2 = find_best_pitch(&xcorr2, y, len >> 1, max_pitch >> 1);

    // Pseudo-interpolation around the peak.
    let offset = if best2[0] > 0 && best2[0] < (max_pitch >> 1) - 1 {
        let a = xcorr2[best2[0] - 1];
        let b = xcorr2[best2[0]];
        let c = xcorr2[best2[0] + 1];
        if (c - a) > 0.7 * (b - a) {
            1
        } else if (a - c) > 0.7 * (b - c) {
            -1
        } else {
            0
        }
    } else {
        0
    };
    (2 * best2[0] as i32 - offset) as usize
}

/// `second_check` table for the octave-error search in [`remove_doubling`].
const SECOND_CHECK: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

/// Refines the pitch lag and rejects octave errors (`remove_doubling`).
/// `x` is the down-sampled history+frame (the search reads backwards from
/// `x[maxperiod..]`); `t0` is the coarse lag in. Returns `(pitch_gain,
/// refined_lag)` at the original sample rate.
#[must_use]
pub(crate) fn remove_doubling(
    x: &[f32],
    maxperiod: usize,
    minperiod: usize,
    n: usize,
    t0: usize,
    prev_period: usize,
    prev_gain: f32,
) -> (f32, usize) {
    let minperiod0 = minperiod;
    let maxperiod = maxperiod / 2;
    let minperiod = minperiod / 2;
    let mut t0 = t0 / 2;
    let prev_period = prev_period / 2;
    let n = n / 2;
    let xoff = maxperiod;
    if t0 >= maxperiod {
        t0 = maxperiod - 1;
    }
    let t0_init = t0;
    let mut t = t0;

    let (xx, xy0) = dual_inner_prod(&x[xoff..], &x[xoff..], &x[xoff - t0..], n);
    let mut yy_lookup = vec![0.0f32; maxperiod + 1];
    yy_lookup[0] = xx;
    let mut yy = xx;
    for i in 1..=maxperiod {
        yy = yy + x[xoff - i] * x[xoff - i] - x[xoff + n - i] * x[xoff + n - i];
        yy_lookup[i] = yy.max(0.0);
    }
    yy = yy_lookup[t0];
    let mut best_xy = xy0;
    let mut best_yy = yy;
    let g0 = compute_pitch_gain(xy0, xx, yy);
    let mut g = g0;

    // Look for a stronger correlation at T/k (an octave or fraction).
    for k in 2..=15usize {
        let t1 = (2 * t0_init + k) / (2 * k);
        if t1 < minperiod {
            break;
        }
        let t1b = if k == 2 {
            if t1 + t0_init > maxperiod {
                t0_init
            } else {
                t0_init + t1
            }
        } else {
            (2 * SECOND_CHECK[k] * t0_init + k) / (2 * k)
        };
        let (xy_a, xy_b) = dual_inner_prod(&x[xoff..], &x[xoff - t1..], &x[xoff - t1b..], n);
        let xy = 0.5 * (xy_a + xy_b);
        let yy_k = 0.5 * (yy_lookup[t1] + yy_lookup[t1b]);
        let g1 = compute_pitch_gain(xy, xx, yy_k);
        let dt = (t1 as i32 - prev_period as i32).abs();
        let cont = if dt <= 1 {
            prev_gain
        } else if dt <= 2 && 5 * k * k < t0_init {
            0.5 * prev_gain
        } else {
            0.0
        };
        let mut thresh = 0.3f32.max(0.7 * g0 - cont);
        // Bias against very short periods (false positives from short-term
        // correlation).
        if t1 < 3 * minperiod {
            thresh = 0.4f32.max(0.85 * g0 - cont);
        } else if t1 < 2 * minperiod {
            thresh = 0.5f32.max(0.9 * g0 - cont);
        }
        if g1 > thresh {
            best_xy = xy;
            best_yy = yy_k;
            t = t1;
            g = g1;
        }
    }
    best_xy = best_xy.max(0.0);
    let mut pg = if best_yy <= best_xy {
        1.0
    } else {
        best_xy / (best_yy + 1.0)
    };

    let mut xcorr = [0.0f32; 3];
    for (k, xc) in xcorr.iter_mut().enumerate() {
        *xc = inner_prod(&x[xoff..], &x[xoff - (t + k - 1)..], n);
    }
    let offset = if (xcorr[2] - xcorr[0]) > 0.7 * (xcorr[1] - xcorr[0]) {
        1
    } else if (xcorr[0] - xcorr[2]) > 0.7 * (xcorr[1] - xcorr[2]) {
        -1
    } else {
        0
    };
    if pg > g {
        pg = g;
    }
    let mut t0_out = (2 * t as i32 + offset).max(0) as usize;
    if t0_out < minperiod0 {
        t0_out = minperiod0;
    }
    (pg, t0_out)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn inner_prod_matches_manual() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        assert_eq!(inner_prod(&x, &y, 3), 4.0 + 10.0 + 18.0);
    }

    #[test]
    fn lpc_predicts_a_pure_tone() {
        // A strongly periodic signal yields a stable predictor (no NaNs,
        // bounded coefficients).
        let n = 256;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).sin()).collect();
        let ac = autocorr(&x, 4);
        let c = lpc(&ac, 4);
        assert!(c.iter().all(|v| v.is_finite() && v.abs() < 4.0), "{c:?}");
    }

    #[test]
    fn pitch_search_finds_a_known_period() {
        // Build a buffer periodic at 200 samples and confirm the detected
        // lag (after doubling refinement) is close to 200.
        let period = 200usize;
        let total = COMBFILTER_MAXPERIOD + 960;
        let sig: Vec<f32> = (0..total)
            .map(|i| {
                let p = (i % period) as f32 / period as f32;
                (2.0 * core::f32::consts::PI * p).sin() + 0.5 * (4.0 * core::f32::consts::PI * p).sin()
            })
            .collect();
        let pre: [&[f32]; 1] = [&sig];
        let x_lp = pitch_downsample(&pre, total);
        let max_pitch = COMBFILTER_MAXPERIOD - 3 * COMBFILTER_MINPERIOD;
        let mut pitch = pitch_search(&x_lp[COMBFILTER_MAXPERIOD >> 1..], &x_lp, 960, max_pitch);
        pitch = COMBFILTER_MAXPERIOD - pitch;
        let (gain, refined) = remove_doubling(&x_lp, COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD, 960, pitch, 0, 0.0);
        // The refined lag should be a multiple/divisor near the true period.
        let err = (refined as i32 - period as i32)
            .abs()
            .min((refined as i32 - 2 * period as i32).abs());
        assert!(err <= 4, "refined lag {refined} (raw {pitch}) for period {period}");
        assert!(gain > 0.5, "gain {gain} on a strongly periodic signal");
    }
}

//! Fixed-point LPC coefficient utilities.
//!
//! These keep the short-term predictor representable and stable:
//! bandwidth expansion shrinks coefficients geometrically, `lpc_fit`
//! squeezes 32-bit coefficients into Q12 `i16` without wrap-around, and the
//! inverse-prediction-gain check walks the reflection-coefficient
//! recursion, returning 0 for (nearly) unstable filters.

#![allow(dead_code, reason = "consumed incrementally as the SILK decoder stages land")]

use super::math::{clz32, inverse32_var_q, mul, rshift_round, rshift_round64, smmul, smull, smulww, sub_sat32};

/// Maximum SILK LPC order.
pub(crate) const SILK_MAX_ORDER_LPC: usize = 24;

/// Chirps the AR filter `ar` by `chirp_q16`.
pub(crate) fn bwexpander_32(ar: &mut [i32], chirp_q16: i32) {
    let d = ar.len();
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    let mut chirp_q16 = chirp_q16;
    for coef in ar.iter_mut().take(d - 1) {
        *coef = smulww(chirp_q16, *coef);
        chirp_q16 += rshift_round(mul(chirp_q16, chirp_minus_one_q16), 16);
    }
    ar[d - 1] = smulww(chirp_q16, ar[d - 1]);
}

/// Converts `a_qin` (Q`qin`) to i16 output in Q`qout`,
/// applying bandwidth expansion until the values fit (clipping as a last
/// resort).
pub(crate) fn lpc_fit(a_qout: &mut [i16], a_qin: &mut [i32], qout: i32, qin: i32) {
    let mut i = 0;
    while i < 10 {
        // Find the maximum absolute value and its index.
        let mut maxabs = 0i32;
        let mut idx = 0usize;
        for (k, &v) in a_qin.iter().enumerate() {
            let absval = v.abs();
            if absval > maxabs {
                maxabs = absval;
                idx = k;
            }
        }
        let maxabs = rshift_round(maxabs, qin - qout);

        if maxabs > i32::from(i16::MAX) {
            // Reduce the magnitude of the prediction coefficients.
            let maxabs = maxabs.min(163_838); // (int32_MAX >> 14) + int16_MAX
            let chirp_q16 = 65470 // SILK_FIX_CONST(0.999, 16)
                - ((maxabs - i32::from(i16::MAX)) << 14) / (mul(maxabs, idx as i32 + 1) >> 2);
            bwexpander_32(a_qin, chirp_q16);
        } else {
            break;
        }
        i += 1;
    }

    if i == 10 {
        // Last iteration: clip the coefficients.
        for (out, qin_v) in a_qout.iter_mut().zip(a_qin.iter_mut()) {
            let v = rshift_round(*qin_v, qin - qout).clamp(i32::from(i16::MIN), i32::from(i16::MAX));
            *out = v as i16;
            *qin_v = v << (qin - qout);
        }
    } else {
        for (out, &qin_v) in a_qout.iter_mut().zip(a_qin.iter()) {
            *out = rshift_round(qin_v, qin - qout) as i16;
        }
    }
}

/// Fixed-point precision of the inverse-prediction-gain recursion.
const QA: i32 = 24;
/// Coefficient stability limit, `SILK_FIX_CONST(0.99975, 24)`.
const A_LIMIT: i32 = 16_773_022;
/// `SILK_FIX_CONST(1.0 / MAX_PREDICTION_POWER_GAIN, 30)` with the maximum
/// power gain 1e4.
const INV_GAIN_MIN_Q30: i32 = 107_374;

/// `(a * b) >> Q` with rounding, computed in 64 bits.
#[inline]
const fn mul32_frac_q(a: i32, b: i32, q: i32) -> i32 {
    rshift_round64(smull(a, b), q) as i32
}

/// The reflection recursion over QA(24) coefficients; 0 means unstable.
fn lpc_inverse_pred_gain_qa(a_qa: &mut [i32]) -> i32 {
    let order = a_qa.len();
    let mut inv_gain_q30 = 1i32 << 30;
    for k in (1..order).rev() {
        // Stability check on the highest coefficient.
        if a_qa[k] > A_LIMIT || a_qa[k] < -A_LIMIT {
            return 0;
        }
        // Reflection coefficient = negated AR coefficient.
        let rc_q31 = -(a_qa[k] << (31 - QA));
        // rc_mult1 in [1, 2^30].
        let rc_mult1_q30 = (1i32 << 30) - smmul(rc_q31, rc_q31);
        inv_gain_q30 = smmul(inv_gain_q30, rc_mult1_q30) << 2;
        if inv_gain_q30 < INV_GAIN_MIN_Q30 {
            return 0;
        }
        // rc_mult2 in [2^30, int32_MAX].
        let mult2q = 32 - clz32(rc_mult1_q30.abs());
        let rc_mult2 = inverse32_var_q(rc_mult1_q30, mult2q + 30);

        // Update the AR coefficients.
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a_qa[n];
            let tmp2 = a_qa[k - n - 1];
            let t1 = rshift_round64(smull(sub_sat32(tmp1, mul32_frac_q(tmp2, rc_q31, 31)), rc_mult2), mult2q);
            if t1 > i64::from(i32::MAX) || t1 < i64::from(i32::MIN) {
                return 0;
            }
            a_qa[n] = t1 as i32;
            let t2 = rshift_round64(smull(sub_sat32(tmp2, mul32_frac_q(tmp1, rc_q31, 31)), rc_mult2), mult2q);
            if t2 > i64::from(i32::MAX) || t2 < i64::from(i32::MIN) {
                return 0;
            }
            a_qa[k - n - 1] = t2 as i32;
        }
    }

    // The final first-order stage.
    if a_qa[0] > A_LIMIT || a_qa[0] < -A_LIMIT {
        return 0;
    }
    let rc_q31 = -(a_qa[0] << (31 - QA));
    let rc_mult1_q30 = (1i32 << 30) - smmul(rc_q31, rc_q31);
    inv_gain_q30 = smmul(inv_gain_q30, rc_mult1_q30) << 2;
    if inv_gain_q30 < INV_GAIN_MIN_Q30 {
        return 0;
    }
    inv_gain_q30
}

/// Inverse prediction gain (Q30) of Q12 coefficients; 0 means the filter is
/// (too close to) unstable.
pub(crate) fn lpc_inverse_pred_gain(a_q12: &[i16]) -> i32 {
    let mut a_qa = [0i32; SILK_MAX_ORDER_LPC];
    let mut dc_resp = 0i32;
    for (qa, &q12) in a_qa.iter_mut().zip(a_q12.iter()) {
        dc_resp += i32::from(q12);
        *qa = i32::from(q12) << (QA - 12);
    }
    // An unstable DC response short-circuits the recursion.
    if dc_resp >= 4096 {
        return 0;
    }
    lpc_inverse_pred_gain_qa(&mut a_qa[..a_q12.len()])
}

/// MA whitening filter (Q12 taps); the first `order` outputs are zeroed.
/// Wrap-around in the accumulator is allowed -
/// only invalid streams can trigger it, and two wraps cancel.
pub(crate) fn lpc_analysis_filter(out: &mut [i16], input: &[i16], b: &[i16]) {
    use super::math::{rshift_round, smlabb};
    let d = b.len();
    debug_assert!(d >= 6 && d & 1 == 0 && d <= input.len() && d <= 16);
    debug_assert_eq!(out.len(), input.len());
    // Reverse the taps once so the per-sample prediction is a forward windowed
    // dot (`input[ix-d..ix]` zipped with `b_rev`) with no bounds checks. Integer
    // wrapping accumulation is order-independent, so this stays bit-identical.
    let mut b_rev = [0i16; 16];
    for (j, &tap) in b.iter().enumerate() {
        b_rev[d - 1 - j] = tap;
    }
    let b_rev = &b_rev[..d];
    for ix in d..input.len() {
        let win = &input[ix - d..ix];
        let mut out32_q12 = 0i32;
        for (&w, &tap) in win.iter().zip(b_rev.iter()) {
            out32_q12 = smlabb(out32_q12, i32::from(w), i32::from(tap));
        }
        // Subtract the prediction from the input (wrapping), scale to Q0.
        let out32_q12 = (i32::from(input[ix]) << 12).wrapping_sub(out32_q12);
        let out32 = rshift_round(out32_q12, 12);
        out[ix] = out32.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
    out[..d].fill(0);
}

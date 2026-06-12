//! NLSF dequantisation and conversion to LPC coefficients
//! (RFC 6716 §4.2.7.5; normative `NLSF_decode.c`, `NLSF_stabilize.c`,
//! `NLSF2A.c`).
//!
//! The decoded two-stage VQ indices become a Q15 normalised-LSF vector:
//! the stage-two residuals run through a backwards one-tap predictor and
//! inverse weighting, add onto the stage-one codebook vector, and the
//! result is *stabilised* (minimum distances enforced with minimal
//! movement). [`nlsf2a`] then maps the NLSFs through a piecewise-linear
//! cosine, builds the even/odd polynomials, and fits the result into Q12
//! `i16` coefficients, bandwidth-expanding until the filter passes the
//! inverse-prediction-gain stability check.

#![allow(dead_code, reason = "consumed incrementally as the SILK decoder stages land")]

use super::indices::{MAX_LPC_ORDER, NlsfCodebook, nlsf_unpack};
use super::lpc::{SILK_MAX_ORDER_LPC, bwexpander_32, lpc_fit, lpc_inverse_pred_gain};
use super::math::{add_sat32, mul, rshift_round, rshift_round64, smlawb, smulbb, smull};
use super::tables::LSF_COS_TAB_FIX_Q12;

/// `NLSF_QUANT_LEVEL_ADJ` in Q10 (`SILK_FIX_CONST(0.1, 10)`).
const NLSF_QUANT_LEVEL_ADJ_Q10: i32 = 102;
/// `MAX_LPC_STABILIZE_ITERATIONS`.
const MAX_LPC_STABILIZE_ITERATIONS: i32 = 16;
/// `MAX_LOOPS` of the stabiliser.
const MAX_LOOPS: usize = 20;

/// `silk_NLSF_residual_dequant`: backwards one-tap predictive dequantiser
/// of the stage-two residuals (Q10).
fn nlsf_residual_dequant(
    indices: &[i8],
    pred_coef_q8: &[u8],
    quant_step_size_q16: i32,
    order: usize,
) -> [i16; MAX_LPC_ORDER] {
    let mut x_q10 = [0i16; MAX_LPC_ORDER];
    let mut out_q10 = 0i32;
    for i in (0..order).rev() {
        let pred_q10 = smulbb(out_q10, i32::from(pred_coef_q8[i])) >> 8;
        out_q10 = i32::from(indices[i]) << 10;
        if out_q10 > 0 {
            out_q10 -= NLSF_QUANT_LEVEL_ADJ_Q10;
        } else if out_q10 < 0 {
            out_q10 += NLSF_QUANT_LEVEL_ADJ_Q10;
        }
        out_q10 = smlawb(pred_q10, out_q10, quant_step_size_q16);
        x_q10[i] = out_q10 as i16;
    }
    x_q10
}

/// `silk_NLSF_stabilize`: enforces the codebook's minimum distances with
/// minimal movement; falls back to a sort-and-clamp after 20 loops.
pub(crate) fn nlsf_stabilize(nlsf_q15: &mut [i16], delta_min_q15: &[i16]) {
    let l = nlsf_q15.len();
    debug_assert!(delta_min_q15[l] >= 1);

    for _ in 0..MAX_LOOPS {
        // Find the smallest slack against the minimum distances.
        let mut min_diff_q15 = i32::from(nlsf_q15[0]) - i32::from(delta_min_q15[0]);
        let mut idx = 0usize;
        for i in 1..l {
            let diff = i32::from(nlsf_q15[i]) - (i32::from(nlsf_q15[i - 1]) + i32::from(delta_min_q15[i]));
            if diff < min_diff_q15 {
                min_diff_q15 = diff;
                idx = i;
            }
        }
        let diff = (1i32 << 15) - (i32::from(nlsf_q15[l - 1]) + i32::from(delta_min_q15[l]));
        if diff < min_diff_q15 {
            min_diff_q15 = diff;
            idx = l;
        }

        if min_diff_q15 >= 0 {
            return;
        }

        if idx == 0 {
            // Move away from the lower limit.
            nlsf_q15[0] = delta_min_q15[0];
        } else if idx == l {
            // Move away from the upper limit.
            nlsf_q15[l - 1] = ((1i32 << 15) - i32::from(delta_min_q15[l])) as i16;
        } else {
            // Move the closest pair apart around their (bounded) centre.
            let mut min_center_q15 = 0i32;
            for &d in &delta_min_q15[..idx] {
                min_center_q15 += i32::from(d);
            }
            min_center_q15 += i32::from(delta_min_q15[idx]) >> 1;
            let mut max_center_q15 = 1i32 << 15;
            for &d in &delta_min_q15[idx + 1..=l] {
                max_center_q15 -= i32::from(d);
            }
            max_center_q15 -= i32::from(delta_min_q15[idx]) >> 1;

            let center_freq_q15 = rshift_round(i32::from(nlsf_q15[idx - 1]) + i32::from(nlsf_q15[idx]), 1)
                .clamp(min_center_q15, max_center_q15);
            nlsf_q15[idx - 1] = (center_freq_q15 - (i32::from(delta_min_q15[idx]) >> 1)) as i16;
            nlsf_q15[idx] = (i32::from(nlsf_q15[idx - 1]) + i32::from(delta_min_q15[idx])) as i16;
        }
    }

    // Fallback: sort, then clamp the distances from both ends.
    nlsf_q15.sort_unstable();
    nlsf_q15[0] = nlsf_q15[0].max(delta_min_q15[0]);
    for i in 1..l {
        nlsf_q15[i] = nlsf_q15[i].max(
            add_sat32(i32::from(nlsf_q15[i - 1]), i32::from(delta_min_q15[i]))
                .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
        );
    }
    nlsf_q15[l - 1] = nlsf_q15[l - 1].min(((1i32 << 15) - i32::from(delta_min_q15[l])) as i16);
    for i in (0..l - 1).rev() {
        nlsf_q15[i] = nlsf_q15[i].min(nlsf_q15[i + 1] - delta_min_q15[i + 1]);
    }
}

/// `silk_NLSF_decode`: NLSF vector (Q15) from the codebook path indices.
pub(crate) fn nlsf_decode(nlsf_indices: &[i8], cb: &NlsfCodebook) -> [i16; MAX_LPC_ORDER] {
    let (_, pred_q8) = nlsf_unpack(cb, nlsf_indices[0] as usize);
    let res_q10 = nlsf_residual_dequant(&nlsf_indices[1..], &pred_q8, cb.quant_step_size_q16, cb.order);

    let cb1 = &cb.cb1_nlsf_q8[nlsf_indices[0] as usize * cb.order..];
    let wght = &cb.cb1_wght_q9[nlsf_indices[0] as usize * cb.order..];
    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
    for i in 0..cb.order {
        // Inverse-weighted residual plus the stage-one vector.
        let tmp = (i32::from(res_q10[i]) << 14) / i32::from(wght[i]) + (i32::from(cb1[i]) << 7);
        nlsf_q15[i] = tmp.clamp(0, 32767) as i16;
    }
    nlsf_stabilize(&mut nlsf_q15[..cb.order], cb.delta_min_q15);
    nlsf_q15
}

/// `QA` of the polynomial construction in `NLSF2A.c`.
const QA16: i32 = 16;

/// `silk_NLSF2A_find_poly`.
fn nlsf2a_find_poly(out: &mut [i32], c_lsf: &[i32], dd: usize) {
    out[0] = 1 << QA16;
    out[1] = -c_lsf[0];
    for k in 1..dd {
        let ftmp = c_lsf[2 * k]; // QA
        out[k + 1] = (out[k - 1] << 1) - rshift_round64(smull(ftmp, out[k]), QA16) as i32;
        for n in (2..=k).rev() {
            out[n] += out[n - 2] - rshift_round64(smull(ftmp, out[n - 1]), QA16) as i32;
        }
        out[1] -= ftmp;
    }
}

/// `silk_NLSF2A`: monic whitening-filter coefficients (Q12) from NLSFs
/// (Q15); `d` is 10 or 16.
pub(crate) fn nlsf2a(a_q12: &mut [i16], nlsf: &[i16]) {
    // This ordering improves the numerical accuracy of the polynomial
    // construction.
    const ORDERING16: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
    const ORDERING10: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

    let d = nlsf.len();
    debug_assert!(d == 10 || d == 16);
    let ordering: &[usize] = if d == 16 { &ORDERING16 } else { &ORDERING10 };

    // LSF → 2*cos(LSF) via the piecewise-linear table.
    let mut cos_lsf_qa = [0i32; SILK_MAX_ORDER_LPC];
    for (k, &f) in nlsf.iter().enumerate() {
        let f_int = i32::from(f) >> (15 - 7);
        let f_frac = i32::from(f) - (f_int << (15 - 7));
        let cos_val = i32::from(LSF_COS_TAB_FIX_Q12[f_int as usize]);
        let delta = i32::from(LSF_COS_TAB_FIX_Q12[f_int as usize + 1]) - cos_val;
        cos_lsf_qa[ordering[k]] = rshift_round((cos_val << 8) + mul(delta, f_frac), 20 - QA16);
    }

    let dd = d >> 1;
    let mut p = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];
    let mut q = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];
    // Even and odd polynomials from the interleaved cosines.
    nlsf2a_find_poly(&mut p, &cos_lsf_qa, dd);
    nlsf2a_find_poly(&mut q, &cos_lsf_qa[1..], dd);

    let mut a32_qa1 = [0i32; SILK_MAX_ORDER_LPC];
    for k in 0..dd {
        let ptmp = p[k + 1] + p[k];
        let qtmp = q[k + 1] - q[k];
        a32_qa1[k] = -qtmp - ptmp; // QA+1
        a32_qa1[d - k - 1] = qtmp - ptmp; // QA+1
    }

    lpc_fit(a_q12, &mut a32_qa1[..d], 12, QA16 + 1);

    let mut i = 0;
    while lpc_inverse_pred_gain(&a_q12[..d]) == 0 && i < MAX_LPC_STABILIZE_ITERATIONS {
        // (Too close to) unstable: bandwidth-expand the unscaled
        // coefficients and try again.
        bwexpander_32(&mut a32_qa1[..d], 65536 - (2 << i));
        for (out, &v) in a_q12.iter_mut().zip(a32_qa1.iter()) {
            *out = rshift_round(v, QA16 + 1 - 12) as i16;
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::indices::{NLSF_CB_NB_MB, NLSF_CB_WB};
    use super::*;

    /// Pins generated by compiling the reference NLSF chain
    /// (`NLSF_decode.c` → `NLSF2A.c`, with stabilisation and the
    /// inverse-prediction-gain loop) and recording its outputs.
    #[test]
    fn decode_and_nlsf2a_match_reference_pins() {
        // NB/MB codebook, order 10.
        let ind10: [i8; 11] = [17, 3, -2, 0, 1, -4, 2, 0, -1, 5, 0];
        let nlsf = nlsf_decode(&ind10, &NLSF_CB_NB_MB);
        assert_eq!(
            &nlsf[..10],
            [3537, 3540, 6587, 10094, 10679, 19662, 20618, 24008, 30174, 30177]
        );
        let mut a = [0i16; 10];
        nlsf2a(&mut a, &nlsf[..10]);
        assert_eq!(a, [1927, 3805, -2430, -1018, -459, 2085, 1202, -4536, 221, 1400]);

        // WB codebook, order 16.
        let ind16: [i8; 17] = [8, -9, 4, -2, 0, 1, -1, 2, 0, 3, -3, 1, 0, -1, 2, 0, 1];
        let nlsf = nlsf_decode(&ind16, &NLSF_CB_WB);
        assert_eq!(
            &nlsf[..16],
            [
                100, 3611, 3651, 4748, 8266, 10666, 13540, 14769, 15971, 15981, 20674, 22109, 24216, 28088, 28637,
                31173
            ]
        );
        let mut a = [0i16; 16];
        nlsf2a(&mut a, &nlsf[..16]);
        assert_eq!(
            a,
            [
                4455, -556, -840, 3177, -1964, -1564, 585, -187, 1345, -1791, 714, 1720, -2721, 2154, 366, -803
            ]
        );
    }

    /// Stabiliser stress pin (forces both the minimal-movement loops and
    /// the sort fallback), recorded from the compiled reference.
    #[test]
    fn stabilize_matches_reference_pin() {
        let mut v: [i16; 10] = [30, 20, 10, 5, 5, 5, 32000, 32100, 100, 50];
        nlsf_stabilize(&mut v, NLSF_CB_NB_MB.delta_min_q15);
        assert_eq!(v, [250, 253, 259, 262, 265, 268, 15947, 15950, 16068, 16314]);
    }
}

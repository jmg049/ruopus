//! LPC → NLSF conversion for the SILK encoder (RFC 6716 §5.2.3).
//!
//! [`a2nlsf`] is the exact inverse of the decoder's [`super::super::nlsf::nlsf2a`]:
//! it finds the roots of the even/odd whitening-filter polynomials in the
//! cos domain (a piecewise-linear LSF↔cos map, so the NLSFs aren't exact but
//! the two maps invert each other), bandwidth-expanding the input until all
//! roots are found. Fixed-point throughout.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::super::indices::{MAX_LPC_ORDER, NlsfCodebook, nlsf_unpack};
use super::super::lpc::bwexpander_32;
use super::super::math::{div32_var_q, lin2log, rshift_round, smlabb, smlaww, smulbb};
use super::super::nlsf::{nlsf_decode, nlsf_stabilize};
use super::super::tables::LSF_COS_TAB_FIX_Q12;

/// Trellis (delayed-decision) survivor states (`NLSF_QUANT_DEL_DEC_STATES`).
const DEL_DEC_STATES: usize = 4;
const DEL_DEC_STATES_LOG2: i32 = 2;
/// Quantiser amplitude bounds (`NLSF_QUANT_MAX_AMPLITUDE[_EXT]`).
const MAX_AMP: i32 = 4;
const MAX_AMP_EXT: i32 = 10;
/// Level adjustment `NLSF_QUANT_LEVEL_ADJ` in Q10.
const LEVEL_ADJ_Q10: i32 = 102; // round(0.1 * 1024)

/// `a + b·c` (32-bit, wrapping).
#[inline]
const fn mla(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(b.wrapping_mul(c))
}

/// Binary-division refinement steps (`BIN_DIV_STEPS_A2NLSF_FIX`).
const BIN_DIV_STEPS: i32 = 3;
/// Maximum bandwidth-expansion retries (`MAX_ITERATIONS_A2NLSF_FIX`).
const MAX_ITERATIONS: i32 = 16;
/// Cos-table size (`LSF_COS_TAB_SZ_FIX`).
const LSF_COS_TAB_SZ: i32 = 128;

/// Transform a polynomial from the cos(n·f) basis to the cos(f)^n basis.
fn trans_poly(p: &mut [i32], dd: usize) {
    for k in 2..=dd {
        for n in (k + 1..=dd).rev() {
            p[n - 2] -= p[n];
        }
        p[k - 2] -= p[k] << 1;
    }
}

/// Evaluate `p` at Q12 point `x`, result Q16.
fn eval_poly(p: &[i32], x: i32, dd: usize) -> i32 {
    let x_q16 = x << 4;
    let mut y32 = p[dd];
    for n in (0..dd).rev() {
        y32 = smlaww(p[n], y32, x_q16);
    }
    y32
}

/// Build the even (`P`) and odd (`Q`) polynomials from
/// the Q16 whitening coefficients, dividing out the fixed z = ±1 roots and
/// transforming to the cos(f)^n basis.
fn init(a_q16: &[i32], p: &mut [i32], q: &mut [i32], dd: usize) {
    p[dd] = 1 << 16;
    q[dd] = 1 << 16;
    for k in 0..dd {
        p[k] = -a_q16[dd - k - 1] - a_q16[dd + k];
        q[k] = -a_q16[dd - k - 1] + a_q16[dd + k];
    }
    for k in (1..=dd).rev() {
        p[k - 1] -= p[k];
        q[k - 1] += q[k];
    }
    trans_poly(p, dd);
    trans_poly(q, dd);
}

/// NLSFs (Q15) from monic whitening-filter coefficients (Q16,
/// modified in place by bandwidth expansion on non-convergence). `d` even.
fn a2nlsf_fix(nlsf: &mut [i16], a_q16: &mut [i32], d: usize) {
    let dd = d >> 1;
    let mut p = vec![0i32; dd + 1];
    let mut q = vec![0i32; dd + 1];
    init(a_q16, &mut p, &mut q, dd);

    // `use_p` selects P (true) or Q (false) for the current root.
    let mut use_p = true;
    let mut xlo = i32::from(LSF_COS_TAB_FIX_Q12[0]);
    let mut ylo = eval_poly(&p, xlo, dd);
    let mut root_ix;
    if ylo < 0 {
        nlsf[0] = 0;
        use_p = false;
        ylo = eval_poly(&q, xlo, dd);
        root_ix = 1;
    } else {
        root_ix = 0;
    }
    let mut k = 1i32;
    let mut iters = 0i32;
    let mut thr = 0i32;

    loop {
        let poly: &[i32] = if use_p { &p } else { &q };
        let xhi = i32::from(LSF_COS_TAB_FIX_Q12[k as usize]);
        let mut yhi = eval_poly(poly, xhi, dd);

        if (ylo <= 0 && yhi >= thr) || (ylo >= 0 && yhi <= -thr) {
            thr = i32::from(yhi == 0);
            let (mut xlo_b, mut xhi_b, mut ylo_b) = (xlo, xhi, ylo);
            let mut ffrac = -256i32;
            for m in 0..BIN_DIV_STEPS {
                let xmid = rshift_round(xlo_b + xhi_b, 1);
                let ymid = eval_poly(poly, xmid, dd);
                if (ylo_b <= 0 && ymid >= 0) || (ylo_b >= 0 && ymid <= 0) {
                    xhi_b = xmid;
                    yhi = ymid;
                } else {
                    xlo_b = xmid;
                    ylo_b = ymid;
                    ffrac += 128 >> m;
                }
            }
            // Interpolate the crossing.
            if ylo_b.abs() < 65536 {
                let den = ylo_b - yhi;
                let nom = (ylo_b << (8 - BIN_DIV_STEPS)) + (den >> 1);
                if den != 0 {
                    ffrac += nom / den;
                }
            } else {
                ffrac += ylo_b / ((ylo_b - yhi) >> (8 - BIN_DIV_STEPS));
            }
            nlsf[root_ix] = i32::min((k << 8) + ffrac, i32::from(i16::MAX)) as i16;

            root_ix += 1;
            if root_ix >= d {
                break;
            }
            use_p = root_ix & 1 == 0;
            xlo = i32::from(LSF_COS_TAB_FIX_Q12[(k - 1) as usize]);
            ylo = (1 - (root_ix as i32 & 2)) << 12;
        } else {
            k += 1;
            xlo = xhi;
            ylo = yhi;
            thr = 0;
            if k > LSF_COS_TAB_SZ {
                iters += 1;
                if iters > MAX_ITERATIONS {
                    // Give up: a flat (white) spectrum.
                    nlsf[0] = ((1 << 15) / (d as i32 + 1)) as i16;
                    for kk in 1..d {
                        nlsf[kk] = nlsf[kk - 1] + nlsf[0];
                    }
                    return;
                }
                bwexpander_32(a_q16, 65536 - (1 << iters));
                init(a_q16, &mut p, &mut q, dd);
                use_p = true;
                xlo = i32::from(LSF_COS_TAB_FIX_Q12[0]);
                ylo = eval_poly(&p, xlo, dd);
                if ylo < 0 {
                    nlsf[0] = 0;
                    use_p = false;
                    ylo = eval_poly(&q, xlo, dd);
                    root_ix = 1;
                } else {
                    root_ix = 0;
                }
                k = 1;
            }
        }
    }
}

/// Converts float LPC coefficients to NLSFs in Q15 (`silk_A2NLSF_FLP`):
/// rounds the coefficients to Q16 and runs the fixed-point root finder.
#[must_use]
pub(crate) fn a2nlsf(lpc: &[f32]) -> Vec<i16> {
    let order = lpc.len();
    debug_assert!(order % 2 == 0 && order <= MAX_LPC_ORDER);
    let mut a_q16: Vec<i32> = lpc.iter().map(|&v| (f64::from(v) * 65536.0).round() as i32).collect();
    let mut nlsf = vec![0i16; order];
    a2nlsf_fix(&mut nlsf, &mut a_q16, order);
    nlsf
}

/// Perceptual NLSF weights (Q2) from the
/// inverse spacing between adjacent line spectral frequencies.
pub(crate) fn nlsf_vq_weights_laroia(w_q2: &mut [i16], nlsf_q15: &[i16], d: usize) {
    // 1 << (15 + NLSF_W_Q), NLSF_W_Q = 2.
    const NUM: i32 = 1 << 17;
    let mut tmp1 = NUM / i32::from(nlsf_q15[0]).max(1);
    let mut tmp2 = NUM / (i32::from(nlsf_q15[1]) - i32::from(nlsf_q15[0])).max(1);
    w_q2[0] = (tmp1 + tmp2).min(i32::from(i16::MAX)) as i16;
    let mut k = 1;
    while k < d - 1 {
        tmp1 = NUM / (i32::from(nlsf_q15[k + 1]) - i32::from(nlsf_q15[k])).max(1);
        w_q2[k] = (tmp1 + tmp2).min(i32::from(i16::MAX)) as i16;
        tmp2 = NUM / (i32::from(nlsf_q15[k + 2]) - i32::from(nlsf_q15[k + 1])).max(1);
        w_q2[k + 1] = (tmp1 + tmp2).min(i32::from(i16::MAX)) as i16;
        k += 2;
    }
    tmp1 = NUM / ((1 << 15) - i32::from(nlsf_q15[d - 1])).max(1);
    w_q2[d - 1] = (tmp1 + tmp2).min(i32::from(i16::MAX)) as i16;
}

/// Weighted predictive quantisation error of each first-
/// stage codebook vector against `in_q15`.
fn nlsf_vq(err_q24: &mut [i32], in_q15: &[i16], cb_q8: &[u8], wght_q9: &[i16], k: usize, order: usize) {
    for i in 0..k {
        let cb = &cb_q8[i * order..];
        let w = &wght_q9[i * order..];
        let mut sum_error_q24 = 0i32;
        let mut pred_q24 = 0i32;
        let mut m = order as isize - 2;
        while m >= 0 {
            let mu = m as usize;
            for j in [mu + 1, mu] {
                let diff_q15 = i32::from(in_q15[j]) - (i32::from(cb[j]) << 7);
                let diffw_q24 = smulbb(diff_q15, i32::from(w[j]));
                sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24 - (pred_q24 >> 1)).abs());
                pred_q24 = diffw_q24;
            }
            m -= 2;
        }
        err_q24[i] = sum_error_q24;
    }
}

/// Sort so the first `k` of `a` are the
/// `k` smallest in increasing order, tracking original indices in `idx`.
fn insertion_sort_increasing(a: &mut [i32], idx: &mut [usize], l: usize, k: usize) {
    for (i, ix) in idx.iter_mut().enumerate().take(k) {
        *ix = i;
    }
    for i in 1..k {
        let value = a[i];
        let mut j = i as isize - 1;
        while j >= 0 && value < a[j as usize] {
            a[j as usize + 1] = a[j as usize];
            idx[j as usize + 1] = idx[j as usize];
            j -= 1;
        }
        a[(j + 1) as usize] = value;
        idx[(j + 1) as usize] = i;
    }
    for i in k..l {
        let value = a[i];
        if value < a[k - 1] {
            let mut j = k as isize - 2;
            while j >= 0 && value < a[j as usize] {
                a[j as usize + 1] = a[j as usize];
                idx[j as usize + 1] = idx[j as usize];
                j -= 1;
            }
            a[(j + 1) as usize] = value;
            idx[(j + 1) as usize] = i;
        }
    }
}

/// Delayed-decision (trellis) quantiser of the
/// first-stage residual. Fills `indices` (length `order`) and returns the
/// rate-distortion value in Q25.
#[allow(clippy::too_many_arguments, reason = "mirrors the reference signature")]
fn nlsf_del_dec_quant(
    indices: &mut [i8],
    x_q10: &[i16],
    w_q5: &[i16],
    pred_coef_q8: &[u8],
    ec_ix: &[i16],
    ec_rates_q5: &[u8],
    quant_step_size_q16: i32,
    inv_quant_step_size_q6: i32,
    mu_q20: i32,
    order: usize,
) -> i32 {
    // Precompute the two candidate output levels per quantiser index.
    let span = 2 * MAX_AMP_EXT as usize;
    let mut out0_table = vec![0i32; span];
    let mut out1_table = vec![0i32; span];
    for i in -MAX_AMP_EXT..MAX_AMP_EXT {
        let mut out0 = i << 10;
        let mut out1 = out0 + 1024;
        if i > 0 {
            out0 -= LEVEL_ADJ_Q10;
            out1 -= LEVEL_ADJ_Q10;
        } else if i == 0 {
            out1 -= LEVEL_ADJ_Q10;
        } else if i == -1 {
            out0 += LEVEL_ADJ_Q10;
        } else {
            out0 += LEVEL_ADJ_Q10;
            out1 += LEVEL_ADJ_Q10;
        }
        let idx = (i + MAX_AMP_EXT) as usize;
        out0_table[idx] = smulbb(out0, quant_step_size_q16) >> 16;
        out1_table[idx] = smulbb(out1, quant_step_size_q16) >> 16;
    }

    let mut ind = [[0i8; MAX_LPC_ORDER]; DEL_DEC_STATES];
    let mut prev_out_q10 = [0i32; 2 * DEL_DEC_STATES];
    let mut rd_q25 = [0i32; 2 * DEL_DEC_STATES];
    let mut rd_min_q25 = [0i32; DEL_DEC_STATES];
    let mut rd_max_q25 = [0i32; DEL_DEC_STATES];
    let mut ind_sort = [0usize; DEL_DEC_STATES];
    let mut n_states = 1usize;

    for i in (0..order).rev() {
        let rates = &ec_rates_q5[ec_ix[i] as usize..];
        let in_q10 = i32::from(x_q10[i]);
        for j in 0..n_states {
            let pred_q10 = smulbb(i32::from(pred_coef_q8[i]), prev_out_q10[j]) >> 8;
            let res_q10 = in_q10 - pred_q10;
            let ind_tmp = (smulbb(inv_quant_step_size_q6, res_q10) >> 16).clamp(-MAX_AMP_EXT, MAX_AMP_EXT - 1);
            ind[j][i] = ind_tmp as i8;

            let ti = (ind_tmp + MAX_AMP_EXT) as usize;
            let out0_q10 = out0_table[ti] + pred_q10;
            let out1_q10 = out1_table[ti] + pred_q10;
            prev_out_q10[j] = out0_q10;
            prev_out_q10[j + n_states] = out1_q10;

            let (rate0_q5, rate1_q5) = if ind_tmp + 1 >= MAX_AMP {
                if ind_tmp + 1 == MAX_AMP {
                    (i32::from(rates[(ind_tmp + MAX_AMP) as usize]), 280)
                } else {
                    let r0 = smlabb(280 - 43 * MAX_AMP, 43, ind_tmp);
                    (r0, r0 + 43)
                }
            } else if ind_tmp <= -MAX_AMP {
                if ind_tmp == -MAX_AMP {
                    (280, i32::from(rates[(ind_tmp + 1 + MAX_AMP) as usize]))
                } else {
                    let r0 = smlabb(280 - 43 * MAX_AMP, -43, ind_tmp);
                    (r0, r0 - 43)
                }
            } else {
                (
                    i32::from(rates[(ind_tmp + MAX_AMP) as usize]),
                    i32::from(rates[(ind_tmp + 1 + MAX_AMP) as usize]),
                )
            };
            let rd_tmp = rd_q25[j];
            let diff0 = in_q10 - out0_q10;
            rd_q25[j] = smlabb(mla(rd_tmp, smulbb(diff0, diff0), i32::from(w_q5[i])), mu_q20, rate0_q5);
            let diff1 = in_q10 - out1_q10;
            rd_q25[j + n_states] = smlabb(mla(rd_tmp, smulbb(diff1, diff1), i32::from(w_q5[i])), mu_q20, rate1_q5);
        }

        if n_states <= DEL_DEC_STATES / 2 {
            for j in 0..n_states {
                ind[j + n_states][i] = ind[j][i] + 1;
            }
            n_states <<= 1;
            for j in n_states..DEL_DEC_STATES {
                ind[j][i] = ind[j - n_states][i];
            }
        } else {
            // Sort the lower/upper halves pairwise, then merge to keep the
            // DEL_DEC_STATES best survivors.
            for j in 0..DEL_DEC_STATES {
                if rd_q25[j] > rd_q25[j + DEL_DEC_STATES] {
                    rd_max_q25[j] = rd_q25[j];
                    rd_min_q25[j] = rd_q25[j + DEL_DEC_STATES];
                    rd_q25[j] = rd_min_q25[j];
                    rd_q25[j + DEL_DEC_STATES] = rd_max_q25[j];
                    prev_out_q10.swap(j, j + DEL_DEC_STATES);
                    ind_sort[j] = j + DEL_DEC_STATES;
                } else {
                    rd_min_q25[j] = rd_q25[j];
                    rd_max_q25[j] = rd_q25[j + DEL_DEC_STATES];
                    ind_sort[j] = j;
                }
            }
            loop {
                let mut min_max_q25 = i32::MAX;
                let mut max_min_q25 = 0i32;
                let mut ind_min_max = 0usize;
                let mut ind_max_min = 0usize;
                for j in 0..DEL_DEC_STATES {
                    if min_max_q25 > rd_max_q25[j] {
                        min_max_q25 = rd_max_q25[j];
                        ind_min_max = j;
                    }
                    if max_min_q25 < rd_min_q25[j] {
                        max_min_q25 = rd_min_q25[j];
                        ind_max_min = j;
                    }
                }
                if min_max_q25 >= max_min_q25 {
                    break;
                }
                ind_sort[ind_max_min] = ind_sort[ind_min_max] ^ DEL_DEC_STATES;
                rd_q25[ind_max_min] = rd_q25[ind_min_max + DEL_DEC_STATES];
                prev_out_q10[ind_max_min] = prev_out_q10[ind_min_max + DEL_DEC_STATES];
                rd_min_q25[ind_max_min] = 0;
                rd_max_q25[ind_min_max] = i32::MAX;
                let src = ind[ind_min_max];
                ind[ind_max_min] = src;
            }
            for j in 0..DEL_DEC_STATES {
                ind[j][i] += (ind_sort[j] >> DEL_DEC_STATES_LOG2) as i8;
            }
        }
    }

    // Pick the winning trellis state.
    let mut ind_tmp = 0usize;
    let mut min_q25 = i32::MAX;
    for (j, &rd) in rd_q25.iter().enumerate() {
        if min_q25 > rd {
            min_q25 = rd;
            ind_tmp = j;
        }
    }
    indices[..order].copy_from_slice(&ind[ind_tmp & (DEL_DEC_STATES - 1)][..order]);
    indices[0] += (ind_tmp >> DEL_DEC_STATES_LOG2) as i8;
    min_q25
}

/// Stabilise, run the first-stage VQ, refine the best
/// survivors with the trellis quantiser, pick the lowest rate-distortion
/// path, and decode it back into `nlsf_q15`. Returns the chosen path
/// `indices` (length `order + 1`) and the RD value (Q25).
#[allow(clippy::too_many_arguments, reason = "mirrors the reference signature")]
pub(crate) fn nlsf_encode(
    nlsf_q15: &mut [i16],
    cb: &NlsfCodebook,
    w_q2: &[i16],
    nlsf_mu_q20: i32,
    n_survivors: usize,
    signal_type: usize,
) -> (Vec<i8>, i32) {
    let order = cb.order;
    nlsf_stabilize(nlsf_q15, cb.delta_min_q15);

    let mut err_q24 = vec![0i32; cb.n_vectors];
    nlsf_vq(
        &mut err_q24,
        nlsf_q15,
        cb.cb1_nlsf_q8,
        cb.cb1_wght_q9,
        cb.n_vectors,
        order,
    );

    let mut temp_idx1 = vec![0usize; cb.n_vectors.max(n_survivors)];
    insertion_sort_increasing(&mut err_q24, &mut temp_idx1, cb.n_vectors, n_survivors);

    let mut rd = vec![0i32; n_survivors];
    let mut temp_idx2 = vec![0i8; n_survivors * MAX_LPC_ORDER];
    let icdf_off = (signal_type >> 1) * cb.n_vectors;

    for s in 0..n_survivors {
        let ind1 = temp_idx1[s];
        let cb_elem = &cb.cb1_nlsf_q8[ind1 * order..];
        let cb_wght = &cb.cb1_wght_q9[ind1 * order..];
        let mut res_q10 = [0i16; MAX_LPC_ORDER];
        let mut w_adj_q5 = [0i16; MAX_LPC_ORDER];
        for i in 0..order {
            let nlsf_tmp_q15 = i32::from(cb_elem[i]) << 7;
            let w_tmp_q9 = i32::from(cb_wght[i]);
            res_q10[i] = (smulbb(i32::from(nlsf_q15[i]) - nlsf_tmp_q15, w_tmp_q9) >> 14) as i16;
            w_adj_q5[i] = div32_var_q(i32::from(w_q2[i]), smulbb(w_tmp_q9, w_tmp_q9), 21) as i16;
        }

        let (ec_ix, pred_q8) = nlsf_unpack(cb, ind1);

        let path = &mut temp_idx2[s * MAX_LPC_ORDER..s * MAX_LPC_ORDER + order];
        let mut rd_s = nlsf_del_dec_quant(
            path,
            &res_q10[..order],
            &w_adj_q5[..order],
            &pred_q8[..order],
            &ec_ix[..order],
            cb.ec_rates_q5,
            cb.quant_step_size_q16,
            cb.inv_quant_step_size_q6,
            nlsf_mu_q20,
            order,
        );

        // Add the first-stage rate.
        let prob_q8 = if ind1 == 0 {
            256 - i32::from(cb.cb1_icdf[icdf_off])
        } else {
            i32::from(cb.cb1_icdf[icdf_off + ind1 - 1]) - i32::from(cb.cb1_icdf[icdf_off + ind1])
        };
        let bits_q7 = (8 << 7) - lin2log(prob_q8);
        rd_s = smlabb(rd_s, bits_q7, nlsf_mu_q20 >> 2);
        rd[s] = rd_s;
    }

    let mut best = [0usize; 1];
    insertion_sort_increasing(&mut rd, &mut best, n_survivors, 1);
    let best = best[0];

    let mut indices = vec![0i8; order + 1];
    indices[0] = temp_idx1[best] as i8;
    indices[1..=order].copy_from_slice(&temp_idx2[best * MAX_LPC_ORDER..best * MAX_LPC_ORDER + order]);

    let decoded = nlsf_decode(&indices, cb);
    nlsf_q15[..order].copy_from_slice(&decoded[..order]);
    (indices, rd[0])
}

#[cfg(test)]
mod tests {
    use super::super::super::nlsf::nlsf2a;
    use super::super::lpc::burg_modified;
    use super::*;
    extern crate alloc;
    use alloc::vec::Vec;

    /// `a2nlsf` and the decoder's `nlsf2a` are inverses: NLSFs are valid
    /// (monotonically increasing, in range) and the recovered LPC matches.
    #[test]
    fn a2nlsf_inverts_nlsf2a() {
        // A stable order-16 LPC from Burg on a synthetic signal.
        let n = 320usize;
        let order = 16usize;
        let mut seed = 777u32;
        let mut prev = [0.0f32; 4];
        let x: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let e = ((seed >> 16) as f32 / 32_768.0) - 1.0;
                let s = 1.1 * prev[0] - 0.5 * prev[1] + 0.2 * prev[2] + 0.1 * e;
                prev = [s, prev[0], prev[1], prev[2]];
                s
            })
            .collect();
        let mut lpc = [0.0f32; 16];
        burg_modified(&mut lpc, &x, 1.0 / 1e4, n, 1, order);

        let nlsf = a2nlsf(&lpc);
        // NLSFs strictly increasing within (0, 32768).
        for w in nlsf.windows(2) {
            assert!(w[0] < w[1], "non-monotonic NLSFs {nlsf:?}");
        }
        assert!(nlsf[0] > 0 && (nlsf[order - 1] as i32) < 32768);

        // Round trip back to LPC via the decoder and compare.
        let mut a_q12 = [0i16; 16];
        nlsf2a(&mut a_q12, &nlsf);
        for (k, &q12) in a_q12.iter().enumerate() {
            let back = f32::from(q12) / 4096.0;
            assert!((back - lpc[k]).abs() < 0.02, "coef {k}: {back} vs {}", lpc[k]);
        }
    }

    /// A pathological (white-ish) input must still yield ordered NLSFs.
    #[test]
    fn handles_flat_input() {
        let lpc = [0.0f32; 10];
        let nlsf = a2nlsf(&lpc);
        for w in nlsf.windows(2) {
            assert!(w[0] < w[1], "non-monotonic {nlsf:?}");
        }
    }

    /// The NLSF VQ quantiser produces valid indices, the encode/decode pair
    /// agree, and the requantised NLSFs track the input closely.
    #[test]
    fn nlsf_encode_quantises_and_tracks() {
        use super::super::super::indices::NLSF_CB_WB;

        // A voiced-like signal (a few harmonics) → well-spaced formant
        // NLSFs, as real speech produces.
        let n = 400usize;
        let order = 16usize;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.6 * (2.0 * core::f32::consts::PI * 200.0 * t).sin()
                    + 0.3 * (2.0 * core::f32::consts::PI * 800.0 * t).sin()
                    + 0.15 * (2.0 * core::f32::consts::PI * 2600.0 * t).sin()
            })
            .collect();
        let mut lpc = [0.0f32; 16];
        burg_modified(&mut lpc, &x, 1.0 / 1e4, n, 1, order);
        let input = a2nlsf(&lpc);

        // Laroia perceptual weights from the input NLSFs.
        let mut w_q2 = [0i16; MAX_LPC_ORDER];
        nlsf_vq_weights_laroia(&mut w_q2[..order], &input, order);
        let mut nlsf = input.clone();
        let (indices, _rd) = nlsf_encode(&mut nlsf, &NLSF_CB_WB, &w_q2[..order], 1 << 14, 4, 2);

        // Indices in range.
        assert!((indices[0] as usize) < NLSF_CB_WB.n_vectors);
        for &ix in &indices[1..] {
            assert!(
                (-MAX_AMP_EXT..=MAX_AMP_EXT).contains(&i32::from(ix)),
                "index {ix} out of range"
            );
        }

        // The requantised NLSFs equal the decode of the chosen path, are
        // ordered/in range, and track the input (coarse but bounded).
        assert_eq!(&nlsf[..order], &nlsf_decode(&indices, &NLSF_CB_WB)[..order]);
        for w in nlsf.windows(2) {
            assert!(w[0] < w[1], "non-monotonic requantised NLSFs {nlsf:?}");
        }
        let max_err = input
            .iter()
            .zip(&nlsf)
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
            .max()
            .unwrap();
        assert!(max_err < 2000, "max NLSF error {max_err} (Q15) too large");
    }
}

//! LPC → NLSF conversion for the SILK encoder (RFC 6716 §5.2.3; normative
//! `silk/A2NLSF.c` and `silk/float/wrappers_FLP.c`).
//!
//! [`a2nlsf`] is the exact inverse of the decoder's [`super::super::nlsf::nlsf2a`]:
//! it finds the roots of the even/odd whitening-filter polynomials in the
//! cos domain (a piecewise-linear LSF↔cos map, so the NLSFs aren't exact but
//! the two maps invert each other), bandwidth-expanding the input until all
//! roots are found. Fixed-point throughout, mirroring `silk_A2NLSF`.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::super::indices::MAX_LPC_ORDER;
use super::super::lpc::bwexpander_32;
use super::super::math::{rshift_round, smlaww};
use super::super::tables::LSF_COS_TAB_FIX_Q12;

/// Binary-division refinement steps (`BIN_DIV_STEPS_A2NLSF_FIX`).
const BIN_DIV_STEPS: i32 = 3;
/// Maximum bandwidth-expansion retries (`MAX_ITERATIONS_A2NLSF_FIX`).
const MAX_ITERATIONS: i32 = 16;
/// Cos-table size (`LSF_COS_TAB_SZ_FIX`).
const LSF_COS_TAB_SZ: i32 = 128;

/// `silk_A2NLSF_trans_poly`: cos(n·f) basis → cos(f)^n basis.
fn trans_poly(p: &mut [i32], dd: usize) {
    for k in 2..=dd {
        for n in (k + 1..=dd).rev() {
            p[n - 2] -= p[n];
        }
        p[k - 2] -= p[k] << 1;
    }
}

/// `silk_A2NLSF_eval_poly`: evaluate `p` at Q12 point `x`, result Q16.
fn eval_poly(p: &[i32], x: i32, dd: usize) -> i32 {
    let x_q16 = x << 4;
    let mut y32 = p[dd];
    for n in (0..dd).rev() {
        y32 = smlaww(p[n], y32, x_q16);
    }
    y32
}

/// `silk_A2NLSF_init`: build the even (`P`) and odd (`Q`) polynomials from
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

/// `silk_A2NLSF`: NLSFs (Q15) from monic whitening-filter coefficients (Q16,
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
}

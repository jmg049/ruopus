//! Long-term (pitch) prediction analysis and gain quantisation for the SILK
//! encoder (RFC 6716 §5.2).
//!
//! For a voiced subframe the encoder predicts the LPC residual from its own
//! past (one pitch period back) with a 5-tap FIR. [`find_ltp`] forms the
//! per-subframe correlation matrix `XX` and vector `xX` of the lagged
//! residual; [`quant_ltp_gains`] then searches the three LTP gain codebooks
//! with [`vq_wmat_ec`] (a weighted-matrix rate-distortion VQ) and emits the
//! quantised taps plus the periodicity/codebook indices the decoder reads
//! back. The chosen taps are exactly the decoder's `LTP_GAIN_VQ` entries, so
//! they round-trip bit-exactly.

extern crate alloc;

use super::super::indices::MAX_NB_SUBFR;
use super::super::math::{lin2log, log2lin, smlawb, smulbb};
use super::super::params::LTP_ORDER;
use super::super::tables::{
    LTP_GAIN_VQ_0, LTP_GAIN_VQ_0_GAIN, LTP_GAIN_VQ_1, LTP_GAIN_VQ_1_GAIN, LTP_GAIN_VQ_2, LTP_GAIN_VQ_2_GAIN,
};

/// `MAX_SUM_LOG_GAIN_DB`.
const MAX_SUM_LOG_GAIN_DB: f32 = 250.0;
/// `LTP_CORR_INV_MAX`.
const LTP_CORR_INV_MAX: f32 = 0.03;
const NB_LTP_CBKS: usize = 3;

/// Per-vector code length (Q5 bits) for each LTP gain codebook.
const LTP_GAIN_BITS_Q5_0: [u8; 8] = [15, 131, 138, 138, 155, 155, 173, 173];
const LTP_GAIN_BITS_Q5_1: [u8; 16] = [
    69, 93, 115, 118, 131, 138, 141, 138, 150, 150, 155, 150, 155, 160, 166, 160,
];
const LTP_GAIN_BITS_Q5_2: [u8; 32] = [
    131, 128, 134, 141, 141, 141, 145, 145, 145, 150, 155, 155, 155, 155, 160, 160, 160, 160, 166, 166, 173, 173, 182,
    192, 182, 192, 192, 192, 205, 192, 205, 224,
];

/// The codebook (Q7 taps), effective gains (Q7), and code lengths (Q5) for
/// one periodicity index.
fn codebook(per: usize) -> (&'static [[i8; LTP_ORDER]], &'static [u8], &'static [u8]) {
    match per {
        0 => (&LTP_GAIN_VQ_0, &LTP_GAIN_VQ_0_GAIN, &LTP_GAIN_BITS_Q5_0),
        1 => (&LTP_GAIN_VQ_1, &LTP_GAIN_VQ_1_GAIN, &LTP_GAIN_BITS_Q5_1),
        _ => (&LTP_GAIN_VQ_2, &LTP_GAIN_VQ_2_GAIN, &LTP_GAIN_BITS_Q5_2),
    }
}

/// Sum of squares in double precision.
fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
}

/// Inner product of the first `len` elements of `a` and `b`, in double precision.
fn inner_product(a: &[f32], b: &[f32], len: usize) -> f64 {
    (0..len).map(|i| f64::from(a[i]) * f64::from(b[i])).sum()
}

/// The symmetric `Order×Order` correlation matrix of
/// the columns of the lagged data `x` (`x` has `L + Order - 1` samples;
/// index 0 is the oldest). Written row-major into `xx`.
fn corr_matrix(x: &[f32], l: usize, order: usize, xx: &mut [f32]) {
    // ptr1 = &x[order-1]; energy of column 0.
    let p1 = order - 1;
    let mut e = energy(&x[p1..p1 + l]);
    xx[0] = e as f32;
    for j in 1..order {
        e += f64::from(x[p1 - j]) * f64::from(x[p1 - j]) - f64::from(x[p1 + l - j]) * f64::from(x[p1 + l - j]);
        xx[j * order + j] = e as f32;
    }
    // Off-diagonals. `p2` walks back one column per lag: order-2 down to 0.
    for lag in 1..order {
        let p2 = order - 1 - lag;
        let mut e = inner_product(&x[p1..], &x[p2..], l);
        xx[lag * order] = e as f32;
        xx[lag] = e as f32;
        for j in 1..order - lag {
            e += f64::from(x[p1 - j]) * f64::from(x[p2 - j]) - f64::from(x[p1 + l - j]) * f64::from(x[p2 + l - j]);
            xx[(lag + j) * order + j] = e as f32;
            xx[j * order + (lag + j)] = e as f32;
        }
    }
}

/// `X'·t` for the lagged data `x` (`Order` taps).
fn corr_vector(x: &[f32], t: &[f32], l: usize, order: usize, xt: &mut [f32]) {
    for (lag, out) in xt.iter_mut().enumerate().take(order) {
        // ptr1 starts at x[order-1] and walks back one column per lag.
        *out = inner_product(&x[order - 1 - lag..], t, l) as f32;
    }
}

/// Per-subframe weighted correlation matrix `XX`
/// (`nb_subfr * 25`) and vector `xX` (`nb_subfr * 5`) for the LPC residual
/// `r` and pitch `lag`s. `r` is indexed so each subframe `k` starts at
/// `r_base + k*subfr_length`; enough history precedes `r_base` to reach
/// `lag[k] + LTP_ORDER/2` samples back.
pub(crate) fn find_ltp(
    r: &[f32],
    r_base: usize,
    lag: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    xx: &mut [f32],
    x_x: &mut [f32],
) {
    for k in 0..nb_subfr {
        let r_ptr = r_base + k * subfr_length;
        let lag_ptr = r_ptr - (lag[k] as usize + LTP_ORDER / 2);
        let xx_k = &mut xx[k * LTP_ORDER * LTP_ORDER..(k + 1) * LTP_ORDER * LTP_ORDER];
        let xx_off = &mut x_x[k * LTP_ORDER..(k + 1) * LTP_ORDER];
        corr_matrix(&r[lag_ptr..], subfr_length, LTP_ORDER, xx_k);
        corr_vector(&r[lag_ptr..], &r[r_ptr..], subfr_length, LTP_ORDER, xx_off);
        let xx_energy = energy(&r[r_ptr..r_ptr + subfr_length + LTP_ORDER]) as f32;
        let temp = 1.0 / (xx_energy.max(LTP_CORR_INV_MAX * 0.5 * (xx_k[0] + xx_k[24]) + 1.0));
        for v in xx_k.iter_mut() {
            *v *= temp;
        }
        for v in xx_off.iter_mut() {
            *v *= temp;
        }
    }
}

/// Round to nearest integer.
fn float2int(x: f32) -> i32 {
    x.round() as i32
}

/// Search the codebook `cb_q7` (with effective gains
/// `cb_gain_q7` and code lengths `cl_q5`) for the vector minimising the
/// weighted quantisation error plus rate, subject to a maximum effective
/// gain. Returns `(index, res_nrg_q15, rate_dist_q8, gain_q7)`.
#[allow(clippy::too_many_arguments, reason = "mirrors the reference signature")]
fn vq_wmat_ec(
    xx_q17: &[i32],
    x_x_q17: &[i32],
    cb_q7: &[[i8; LTP_ORDER]],
    cb_gain_q7: &[u8],
    cl_q5: &[u8],
    subfr_len: i32,
    max_gain_q7: i32,
    l: usize,
) -> (i8, i32, i32, i32) {
    let neg_x_x_q24: [i32; LTP_ORDER] = core::array::from_fn(|i| -(x_x_q17[i] << 7));

    let mut rate_dist_q8 = i32::MAX;
    let mut res_nrg_q15 = i32::MAX;
    let mut ind = 0i8;
    let mut gain_q7_out = 0i32;

    for k in 0..l {
        let cb = &cb_q7[k];
        let row: [i32; LTP_ORDER] = core::array::from_fn(|i| i32::from(cb[i]));
        let gain_tmp_q7 = i32::from(cb_gain_q7[k]);
        let mut sum1_q15 = 32801; // SILK_FIX_CONST(1.001, 15)
        let penalty = (gain_tmp_q7 - max_gain_q7).max(0) << 11;

        // sum1 = 1.001 + c' (XX c - 2 xX), accumulated row by row (XX_Q17 is
        // the 5×5 matrix, row-major; the [.] indices match the reference).
        let mut sum2 = neg_x_x_q24[0]
            .wrapping_add(xx_q17[1].wrapping_mul(row[1]))
            .wrapping_add(xx_q17[2].wrapping_mul(row[2]))
            .wrapping_add(xx_q17[3].wrapping_mul(row[3]))
            .wrapping_add(xx_q17[4].wrapping_mul(row[4]));
        sum2 = (sum2 << 1).wrapping_add(xx_q17[0].wrapping_mul(row[0]));
        sum1_q15 = smlawb(sum1_q15, sum2, row[0]);

        let mut sum2 = neg_x_x_q24[1]
            .wrapping_add(xx_q17[7].wrapping_mul(row[2]))
            .wrapping_add(xx_q17[8].wrapping_mul(row[3]))
            .wrapping_add(xx_q17[9].wrapping_mul(row[4]));
        sum2 = (sum2 << 1).wrapping_add(xx_q17[6].wrapping_mul(row[1]));
        sum1_q15 = smlawb(sum1_q15, sum2, row[1]);

        let mut sum2 = neg_x_x_q24[2]
            .wrapping_add(xx_q17[13].wrapping_mul(row[3]))
            .wrapping_add(xx_q17[14].wrapping_mul(row[4]));
        sum2 = (sum2 << 1).wrapping_add(xx_q17[12].wrapping_mul(row[2]));
        sum1_q15 = smlawb(sum1_q15, sum2, row[2]);

        let mut sum2 = neg_x_x_q24[3].wrapping_add(xx_q17[19].wrapping_mul(row[4]));
        sum2 = (sum2 << 1).wrapping_add(xx_q17[18].wrapping_mul(row[3]));
        sum1_q15 = smlawb(sum1_q15, sum2, row[3]);

        let sum2 = (neg_x_x_q24[4] << 1).wrapping_add(xx_q17[24].wrapping_mul(row[4]));
        sum1_q15 = smlawb(sum1_q15, sum2, row[4]);

        if sum1_q15 >= 0 {
            let bits_res_q8 = smulbb(subfr_len, lin2log(sum1_q15 + penalty) - (15 << 7));
            let bits_tot_q8 = bits_res_q8.wrapping_add(i32::from(cl_q5[k]) << 2);
            if bits_tot_q8 <= rate_dist_q8 {
                rate_dist_q8 = bits_tot_q8;
                res_nrg_q15 = sum1_q15 + penalty;
                ind = k as i8;
                gain_q7_out = gain_tmp_q7;
            }
        }
    }

    (ind, res_nrg_q15, rate_dist_q8, gain_q7_out)
}

/// The outputs of [`quant_ltp_gains`].
pub(crate) struct LtpGains {
    /// Quantised LTP taps in Q14 (`nb_subfr * LTP_ORDER`).
    pub b_q14: [i16; MAX_NB_SUBFR * LTP_ORDER],
    /// Per-subframe codebook indices.
    pub cbk_index: [i8; MAX_NB_SUBFR],
    /// Periodicity (codebook) index 0-2.
    pub periodicity_index: i8,
    /// LTP prediction gain in dB.
    pub pred_gain_db: f32,
}

/// Convert the float
/// correlations to Q17, search the three periodicity codebooks, and pick the
/// one minimising total rate-distortion. `sum_log_gain_q7` is the cumulative
/// max-gain accumulator, updated in place.
pub(crate) fn quant_ltp_gains(
    xx: &[f32],
    x_x: &[f32],
    subfr_len: i32,
    nb_subfr: usize,
    sum_log_gain_q7: &mut i32,
) -> LtpGains {
    let len_xx = nb_subfr * LTP_ORDER * LTP_ORDER;
    let len_x = nb_subfr * LTP_ORDER;
    let xx_q17: alloc::vec::Vec<i32> = (0..len_xx).map(|i| float2int(xx[i] * 131072.0)).collect();
    let x_x_q17: alloc::vec::Vec<i32> = (0..len_x).map(|i| float2int(x_x[i] * 131072.0)).collect();

    const GAIN_SAFETY_Q7: i32 = 51; // SILK_FIX_CONST(0.4, 7)
    let max_sum_log_gain_q7 = (MAX_SUM_LOG_GAIN_DB / 6.0 * 128.0).round() as i32;

    let mut min_rate_dist_q8 = i32::MAX;
    let mut best_sum_log_gain_q7 = 0i32;
    let mut periodicity_index = 0i8;
    let mut cbk_index = [0i8; MAX_NB_SUBFR];
    let mut best_res_nrg_q15 = 0i32;

    for per in 0..NB_LTP_CBKS {
        let (cb, cb_gain, cl) = codebook(per);
        let mut res_nrg_q15 = 0i32;
        let mut rate_dist_q8 = 0i32;
        let mut sum_log_gain_tmp_q7 = *sum_log_gain_q7;
        let mut temp_idx = [0i8; MAX_NB_SUBFR];

        for j in 0..nb_subfr {
            let max_gain_q7 = log2lin((max_sum_log_gain_q7 - sum_log_gain_tmp_q7) + (7 << 7)) - GAIN_SAFETY_Q7;
            let (idx, res_subfr, rd_subfr, gain_q7) = vq_wmat_ec(
                &xx_q17[j * LTP_ORDER * LTP_ORDER..],
                &x_x_q17[j * LTP_ORDER..],
                cb,
                cb_gain,
                cl,
                subfr_len,
                max_gain_q7,
                cb.len(),
            );
            temp_idx[j] = idx;
            res_nrg_q15 = res_nrg_q15.saturating_add(res_subfr);
            rate_dist_q8 = rate_dist_q8.saturating_add(rd_subfr);
            sum_log_gain_tmp_q7 = (sum_log_gain_tmp_q7 + lin2log(GAIN_SAFETY_Q7 + gain_q7) - (7 << 7)).max(0);
        }

        if rate_dist_q8 <= min_rate_dist_q8 {
            min_rate_dist_q8 = rate_dist_q8;
            periodicity_index = per as i8;
            cbk_index[..nb_subfr].copy_from_slice(&temp_idx[..nb_subfr]);
            best_sum_log_gain_q7 = sum_log_gain_tmp_q7;
            best_res_nrg_q15 = res_nrg_q15;
        }
    }

    let (cb, _, _) = codebook(periodicity_index as usize);
    let mut b_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
    for j in 0..nb_subfr {
        for k in 0..LTP_ORDER {
            b_q14[j * LTP_ORDER + k] = (i32::from(cb[cbk_index[j] as usize][k]) << 7) as i16;
        }
    }

    best_res_nrg_q15 = if nb_subfr == 2 {
        best_res_nrg_q15 >> 1
    } else {
        best_res_nrg_q15 >> 2
    };
    *sum_log_gain_q7 = best_sum_log_gain_q7;
    let pred_gain_db_q7 = smulbb(-3, lin2log(best_res_nrg_q15) - (15 << 7));

    LtpGains {
        b_q14,
        cbk_index,
        periodicity_index,
        pred_gain_db: pred_gain_db_q7 as f32 / 128.0,
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    /// A strongly periodic residual yields a positive LTP prediction gain,
    /// and the quantised taps it selects are exactly the decoder's codebook
    /// entries for the chosen (periodicity, codebook) indices - i.e. they
    /// round-trip bit-exactly through `LTP_GAIN_VQ`.
    #[test]
    fn periodic_residual_quantises_and_round_trips() {
        let fs_khz = 16i32;
        let subfr = 5 * fs_khz as usize; // 80
        let nb_subfr = 4usize;
        let lag = 40i32;
        // History needed before the first subframe.
        let hist = lag as usize + LTP_ORDER / 2 + LTP_ORDER;
        let total = hist + nb_subfr * subfr + LTP_ORDER;

        // A periodic residual (period = lag) plus light noise.
        let mut seed = 0x1234_u32;
        let r: Vec<f32> = (0..total)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let noise = ((seed >> 20) as i32 - 2048) as f32 * 0.05;
                ((i as f32 * core::f32::consts::TAU / lag as f32).sin() * 100.0) + noise
            })
            .collect();

        let lags = [lag; MAX_NB_SUBFR];
        let mut xx = vec![0.0f32; nb_subfr * LTP_ORDER * LTP_ORDER];
        let mut x_x = vec![0.0f32; nb_subfr * LTP_ORDER];
        find_ltp(&r, hist, &lags, subfr, nb_subfr, &mut xx, &mut x_x);

        let mut sum_log_gain_q7 = 0i32;
        let g = quant_ltp_gains(&xx, &x_x, subfr as i32, nb_subfr, &mut sum_log_gain_q7);

        // Strong periodicity => positive prediction gain.
        assert!(g.pred_gain_db > 0.0, "pred gain {} should be positive", g.pred_gain_db);
        assert!(
            (0..3).contains(&g.periodicity_index),
            "per index {}",
            g.periodicity_index
        );

        // The quantised taps equal the decoder's codebook entries.
        let cb: &[[i8; LTP_ORDER]] = match g.periodicity_index {
            0 => &LTP_GAIN_VQ_0,
            1 => &LTP_GAIN_VQ_1,
            _ => &LTP_GAIN_VQ_2,
        };
        for j in 0..nb_subfr {
            for (k, &c) in cb[g.cbk_index[j] as usize].iter().enumerate() {
                let decoded = i32::from(c) << 7;
                assert_eq!(i32::from(g.b_q14[j * LTP_ORDER + k]), decoded, "tap {j},{k} mismatch");
            }
        }
    }

    /// `vq_wmat_ec` agrees with a brute-force evaluation of the same
    /// rate-distortion metric over the whole codebook (no max-gain penalty).
    #[test]
    fn vq_wmat_matches_brute_force() {
        // A simple, well-conditioned correlation: identity-ish matrix, modest
        // target correlation favouring tap 2.
        let mut xx_q17 = [0i32; LTP_ORDER * LTP_ORDER];
        for i in 0..LTP_ORDER {
            xx_q17[i * LTP_ORDER + i] = 131072; // 1.0 in Q17
        }
        let mut x_x_q17 = [0i32; LTP_ORDER];
        x_x_q17[2] = 131072 / 2;

        let (cb, cb_gain, cl) = codebook(1);
        let max_gain_q7 = 1 << 20; // effectively unbounded
        let (idx, _, rd, _) = vq_wmat_ec(&xx_q17, &x_x_q17, cb, cb_gain, cl, 80, max_gain_q7, cb.len());

        // Brute force: replicate the exact integer metric for every vector.
        let neg: [i32; LTP_ORDER] = core::array::from_fn(|i| -(x_x_q17[i] << 7));
        let mut best_rd = i32::MAX;
        let mut best = -1i32;
        for (k, row8) in cb.iter().enumerate() {
            let row: [i32; LTP_ORDER] = core::array::from_fn(|i| i32::from(row8[i]));
            let mut sum1 = 32801i32;
            for r in 0..LTP_ORDER {
                let mut sum2 = neg[r];
                for c in (r + 1)..LTP_ORDER {
                    sum2 = sum2.wrapping_add(xx_q17[r * LTP_ORDER + c].wrapping_mul(row[c]));
                }
                sum2 = (sum2 << 1).wrapping_add(xx_q17[r * LTP_ORDER + r].wrapping_mul(row[r]));
                sum1 = smlawb(sum1, sum2, row[r]);
            }
            if sum1 >= 0 {
                let bits_res = smulbb(80, lin2log(sum1) - (15 << 7));
                let bits_tot = bits_res.wrapping_add(i32::from(cl[k]) << 2);
                if bits_tot <= best_rd {
                    best_rd = bits_tot;
                    best = k as i32;
                }
            }
        }
        assert_eq!(i32::from(idx), best, "selected index disagrees with brute force");
        assert_eq!(rd, best_rd, "rate-distortion disagrees with brute force");
    }
}

//! The noise-shaping quantiser (RFC 6716 §5.2.3).
//!
//! NSQ turns the per-subframe input into the coded excitation `pulses`: for
//! each sample it forms the short-term (LPC) and long-term (LTP) prediction,
//! adds the noise-shaping feedback (an AR shaper, spectral tilt, low-
//! frequency shaper, and - for voiced frames - a harmonic FIR), then picks
//! between two adjacent quantisation levels by a rate-distortion measure.
//! The reconstructed signal `xq` it produces is exactly what the decoder
//! synthesises from the same pulses, gains and prediction coefficients -
//! which is the round-trip oracle used to test it.
//!
//! Fixed-point throughout.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::super::indices::{MAX_LPC_ORDER, MAX_NB_SUBFR, TYPE_VOICED};
use super::super::lpc::lpc_analysis_filter;
use super::super::math::{add_sat32, div32_var_q, inverse32_var_q, rshift_round, smlawb, smlawt, smulwb, smulww};
use super::super::tables::QUANTIZATION_OFFSETS_Q10;

/// Samples of LPC synthesis history kept (`NSQ_LPC_BUF_LENGTH`).
const NSQ_LPC_BUF_LENGTH: usize = MAX_LPC_ORDER;
/// `MAX_SHAPE_LPC_ORDER`, `LTP_ORDER`, `HARM_SHAPE_FIR_TAPS`.
const MAX_SHAPE_LPC_ORDER: usize = 24;
const LTP_ORDER: usize = 5;
const HARM_SHAPE_FIR_TAPS: usize = 3;
/// `QUANT_LEVEL_ADJUST_Q10`.
const QUANT_LEVEL_ADJUST_Q10: i32 = 80;
/// `MAX_FRAME_LENGTH` (5 ms × 4 subframes × 16 kHz).
const MAX_FRAME_LENGTH: usize = 320;
const RAND_MULTIPLIER: i32 = 196_314_165;
const RAND_INCREMENT: i32 = 907_633_515;

/// Saturate to i16.
#[inline]
fn sat16(a: i32) -> i16 {
    a.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// Linear-congruential pseudo-random step from `seed`.
#[inline]
fn rand(seed: i32) -> i32 {
    RAND_INCREMENT.wrapping_add(seed.wrapping_mul(RAND_MULTIPLIER))
}

/// Per-channel NSQ state (`silk_nsq_state`), carried across frames.
#[derive(Clone)]
pub(crate) struct NsqState {
    /// Quantised output buffer (`xq`).
    pub xq: Vec<i16>,
    /// Long-term shaping state (`sLTP_shp_Q14`).
    pub s_ltp_shp_q14: Vec<i32>,
    /// Short-term LPC synthesis state (`sLPC_Q14`).
    pub s_lpc_q14: [i32; MAX_LPC_ORDER + MAX_FRAME_LENGTH / MAX_NB_SUBFR],
    /// Noise-shaping AR state (`sAR2_Q14`).
    pub s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    pub s_lf_ar_shp_q14: i32,
    pub s_diff_shp_q14: i32,
    pub lag_prev: i32,
    pub s_ltp_buf_idx: usize,
    pub s_ltp_shp_buf_idx: usize,
    pub rand_seed: i32,
    pub prev_gain_q16: i32,
    pub rewhite_flag: bool,
    /// Per-call scratch reused across frames (and across the rate-control
    /// loop's re-quantisations) to avoid three heap allocations per `nsq`
    /// call; `nsq` re-initialises them on entry, so their carried contents
    /// are irrelevant.
    scratch_s_ltp_q15: Vec<i32>,
    scratch_s_ltp: Vec<i16>,
    scratch_x_sc_q10: Vec<i32>,
}

impl NsqState {
    /// A reset NSQ state for the given LTP memory and frame length.
    #[must_use]
    pub(crate) fn new() -> Self {
        NsqState {
            xq: vec![0; 2 * MAX_FRAME_LENGTH],
            s_ltp_shp_q14: vec![0; 2 * MAX_FRAME_LENGTH],
            s_lpc_q14: [0; MAX_LPC_ORDER + MAX_FRAME_LENGTH / MAX_NB_SUBFR],
            s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
            s_lf_ar_shp_q14: 0,
            s_diff_shp_q14: 0,
            lag_prev: 0,
            s_ltp_buf_idx: 0,
            s_ltp_shp_buf_idx: 0,
            rand_seed: 0,
            prev_gain_q16: 65536,
            rewhite_flag: false,
            scratch_s_ltp_q15: Vec::new(),
            scratch_s_ltp: Vec::new(),
            scratch_x_sc_q10: Vec::new(),
        }
    }
}

/// Scalar configuration the NSQ needs from the encoder state.
#[derive(Clone, Copy)]
pub(crate) struct NsqConfig {
    pub frame_length: usize,
    pub subfr_length: usize,
    pub nb_subfr: usize,
    pub ltp_mem_length: usize,
    pub predict_lpc_order: usize,
    pub shaping_lpc_order: usize,
}

/// The LPC prediction (Q10).
///
/// `coef_rev` is the prediction coefficients reversed (so the descending
/// history access `buf[base-j]·coef[j]` becomes a forward windowed dot),
/// reversed once per subframe by the caller. This is a short (≤16-tap) dot
/// called once per sample, where the scalar SMLAWB chain beats a SIMD kernel
/// (the call + horizontal-fold overhead does not amortise over so few taps).
#[inline]
fn short_prediction(buf: &[i32], base: usize, coef_rev: &[i16]) -> i32 {
    let order = coef_rev.len();
    let w = &buf[base + 1 - order..=base];
    // `w` and `coef_rev` are both exactly `order` long; zipping drops the
    // per-tap bounds checks in this once-per-sample dot.
    let mut out = (order >> 1) as i32;
    for (&wj, &cj) in w.iter().zip(coef_rev.iter()) {
        out = smlawb(out, wj, i32::from(cj));
    }
    out
}

/// AR shaping with the shift-register
/// state `data1` (`sAR2_Q14`), driven by the new sample `data0`. Returns Q12.
///
/// The right-shift of
/// the AR state (prepend `data0`, drop the last tap) is fused with the dot in a
/// single in-place pass using two rolling temporaries - no scratch array and no
/// `copy_from_slice`. The accumulation order (`coef[0]·data0`, then
/// `coef[k]·old_state[k-1]` ascending) is identical to the materialise-then-dot
/// form, so it stays bit-exact. `order` is always even for SILK shaping.
fn noise_shape_feedback_loop(data0: i32, data1: &mut [i32], coef: &[i16], order: usize) -> i32 {
    debug_assert!(order >= 2 && order & 1 == 0 && order <= data1.len() && order <= coef.len());
    let mut tmp2 = data0;
    let mut tmp1 = data1[0];
    data1[0] = tmp2;
    let mut out = (order >> 1) as i32;
    out = smlawb(out, tmp2, i32::from(coef[0]));
    let mut j = 2;
    while j < order {
        tmp2 = data1[j - 1];
        data1[j - 1] = tmp1;
        out = smlawb(out, tmp1, i32::from(coef[j - 1]));
        tmp1 = data1[j];
        data1[j] = tmp2;
        out = smlawb(out, tmp2, i32::from(coef[j]));
        j += 2;
    }
    data1[order - 1] = tmp1;
    out = smlawb(out, tmp1, i32::from(coef[order - 1]));
    out << 1
}

/// Scale this subframe's input by 1/gain into
/// `x_sc_q10`, rescale the (re-whitened) LTP state, and adjust all states
/// for a gain change since the previous subframe.
#[allow(clippy::too_many_arguments, reason = "mirrors the reference signature")]
#[allow(clippy::needless_range_loop, reason = "computed index ranges mirror the reference")]
fn nsq_scale_states(
    nsq: &mut NsqState,
    cfg: &NsqConfig,
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: usize,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
) {
    let lag = pitch_l[subfr];
    let mut inv_gain_q31 = inverse32_var_q(gains_q16[subfr].max(1), 47);
    let inv_gain_q26 = rshift_round(inv_gain_q31, 5);
    for i in 0..cfg.subfr_length {
        x_sc_q10[i] = smulww(i32::from(x16[i]), inv_gain_q26);
    }

    if nsq.rewhite_flag {
        if subfr == 0 {
            inv_gain_q31 = smulwb(inv_gain_q31, ltp_scale_q14) << 2;
        }
        for i in nsq.s_ltp_buf_idx - lag as usize - LTP_ORDER / 2..nsq.s_ltp_buf_idx {
            s_ltp_q15[i] = smulwb(inv_gain_q31, i32::from(s_ltp[i]));
        }
    }

    if gains_q16[subfr] != nsq.prev_gain_q16 {
        let gain_adj_q16 = div32_var_q(nsq.prev_gain_q16, gains_q16[subfr], 16);
        for i in nsq.s_ltp_shp_buf_idx - cfg.ltp_mem_length..nsq.s_ltp_shp_buf_idx {
            nsq.s_ltp_shp_q14[i] = smulww(gain_adj_q16, nsq.s_ltp_shp_q14[i]);
        }
        if signal_type == TYPE_VOICED && !nsq.rewhite_flag {
            for i in nsq.s_ltp_buf_idx - lag as usize - LTP_ORDER / 2..nsq.s_ltp_buf_idx {
                s_ltp_q15[i] = smulww(gain_adj_q16, s_ltp_q15[i]);
            }
        }
        nsq.s_lf_ar_shp_q14 = smulww(gain_adj_q16, nsq.s_lf_ar_shp_q14);
        nsq.s_diff_shp_q14 = smulww(gain_adj_q16, nsq.s_diff_shp_q14);
        for i in 0..NSQ_LPC_BUF_LENGTH {
            nsq.s_lpc_q14[i] = smulww(gain_adj_q16, nsq.s_lpc_q14[i]);
        }
        for i in 0..MAX_SHAPE_LPC_ORDER {
            nsq.s_ar2_q14[i] = smulww(gain_adj_q16, nsq.s_ar2_q14[i]);
        }
        nsq.prev_gain_q16 = gains_q16[subfr];
    }
}

/// Rate-distortion candidate table, indexed by `q1_Q0 + 32` (`q1_Q0` ∈
/// [-32, 31]). Each row is
/// `[q1_Q10, q2_Q10, 2·(q1-q2), (rd1-rd2)+(q1²-q2²)]`, so the per-sample choice
/// collapses to one lookup, one multiply and one compare. The decision
/// `r·row[2] - row[3] < 0` is algebraically identical to the branchy
/// `rd2 < rd1` (with `rd = level·λ + (r-level)²`), so the pulses are bit-exact.
fn build_rd_table(offset_q10: i32, lambda_q10: i32) -> [[i32; 4]; 64] {
    let mut table = [[0i32; 4]; 64];
    for (idx, row) in table.iter_mut().enumerate() {
        let k = idx as i32 - 32;
        let (q1, q2, rd1, rd2) = if k > 0 {
            let q1 = offset_q10 + (k << 10) - QUANT_LEVEL_ADJUST_Q10;
            let q2 = q1 + 1024;
            (q1, q2, q1 * lambda_q10, q2 * lambda_q10)
        } else if k == 0 {
            let q1 = offset_q10;
            let q2 = q1 + (1024 - QUANT_LEVEL_ADJUST_Q10);
            (q1, q2, q1 * lambda_q10, q2 * lambda_q10)
        } else if k == -1 {
            let q2 = offset_q10;
            let q1 = q2 - (1024 - QUANT_LEVEL_ADJUST_Q10);
            (q1, q2, -q1 * lambda_q10, q2 * lambda_q10)
        } else {
            let q1 = offset_q10 + (k << 10) + QUANT_LEVEL_ADJUST_Q10;
            let q2 = q1 + 1024;
            (q1, q2, -q1 * lambda_q10, -q2 * lambda_q10)
        };
        row[0] = q1;
        row[1] = q2;
        row[2] = 2 * (q1 - q2);
        row[3] = (rd1 - rd2) + (q1 * q1 - q2 * q2);
    }
    table
}

/// The per-sample RD quantiser for one
/// subframe. `pxq_base` indexes `nsq.xq`; `pulses`/`xq_out` receive this
/// subframe's results. `rd_table` is the precomputed RD candidate table
/// ([`build_rd_table`]).
#[allow(clippy::too_many_arguments, reason = "mirrors the reference signature")]
fn noise_shape_quantizer(
    nsq: &mut NsqState,
    signal_type: i32,
    x_sc_q10: &[i32],
    pulses: &mut [i8],
    pxq_base: usize,
    s_ltp_q15: &mut [i32],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    harm_shape_fir_packed_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    rd_table: &[[i32; 4]; 64],
    length: usize,
    shaping_lpc_order: usize,
    predict_lpc_order: usize,
) {
    let mut shp_lag_ptr = nsq.s_ltp_shp_buf_idx - lag as usize + HARM_SHAPE_FIR_TAPS / 2;
    let mut pred_lag_ptr = nsq.s_ltp_buf_idx - lag as usize + LTP_ORDER / 2;
    let gain_q10 = gain_q16 >> 6;
    // `psLPC` index into s_lpc_q14 (starts at NSQ_LPC_BUF_LENGTH - 1).
    let mut p_lpc = NSQ_LPC_BUF_LENGTH - 1;

    // The prediction coefficients are constant over the subframe; reverse them
    // once so the per-sample prediction is a forward windowed dot.
    let mut a_rev = [0i16; MAX_LPC_ORDER];
    for j in 0..predict_lpc_order {
        a_rev[j] = a_q12[predict_lpc_order - 1 - j];
    }
    let a_rev = &a_rev[..predict_lpc_order];

    for i in 0..length {
        nsq.rand_seed = rand(nsq.rand_seed);

        let lpc_pred_q10 = short_prediction(&nsq.s_lpc_q14, p_lpc, a_rev);

        let ltp_pred_q13 = if signal_type == TYPE_VOICED {
            let mut p = 2i32;
            p = smlawb(p, s_ltp_q15[pred_lag_ptr], i32::from(b_q14[0]));
            p = smlawb(p, s_ltp_q15[pred_lag_ptr - 1], i32::from(b_q14[1]));
            p = smlawb(p, s_ltp_q15[pred_lag_ptr - 2], i32::from(b_q14[2]));
            p = smlawb(p, s_ltp_q15[pred_lag_ptr - 3], i32::from(b_q14[3]));
            p = smlawb(p, s_ltp_q15[pred_lag_ptr - 4], i32::from(b_q14[4]));
            pred_lag_ptr += 1;
            p
        } else {
            0
        };

        // Noise-shape feedback.
        let mut n_ar_q12 =
            noise_shape_feedback_loop(nsq.s_diff_shp_q14, &mut nsq.s_ar2_q14, ar_shp_q13, shaping_lpc_order);
        n_ar_q12 = smlawb(n_ar_q12, nsq.s_lf_ar_shp_q14, tilt_q14);
        let mut n_lf_q12 = smulwb(nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx - 1], lf_shp_q14);
        n_lf_q12 = smlawt(n_lf_q12, nsq.s_lf_ar_shp_q14, lf_shp_q14);

        let mut tmp1 = (lpc_pred_q10 << 2).wrapping_sub(n_ar_q12);
        tmp1 = tmp1.wrapping_sub(n_lf_q12);
        if lag > 0 {
            let mut n_ltp_q13 = smulwb(
                add_sat32(nsq.s_ltp_shp_q14[shp_lag_ptr], nsq.s_ltp_shp_q14[shp_lag_ptr - 2]),
                harm_shape_fir_packed_q14,
            );
            n_ltp_q13 = smlawt(n_ltp_q13, nsq.s_ltp_shp_q14[shp_lag_ptr - 1], harm_shape_fir_packed_q14);
            n_ltp_q13 <<= 1;
            shp_lag_ptr += 1;
            let tmp2 = ltp_pred_q13.wrapping_sub(n_ltp_q13);
            tmp1 = tmp2.wrapping_add(tmp1 << 1);
            tmp1 = rshift_round(tmp1, 3);
        } else {
            tmp1 = rshift_round(tmp1, 2);
        }

        let mut r_q10 = x_sc_q10[i].wrapping_sub(tmp1);
        if nsq.rand_seed < 0 {
            r_q10 = -r_q10;
        }
        r_q10 = r_q10.clamp(-(31 << 10), 30 << 10);

        // Two quantisation candidates and their rate-distortion, via the
        // precomputed table: compute `q1_Q0`, then one lookup + one multiply +
        // one compare pick the level (bit-exact with the branchy `rd2 < rd1`).
        let q1_q10_tmp = r_q10 - offset_q10;
        let mut q1_q0 = q1_q10_tmp >> 10;
        if lambda_q10 > 2048 {
            let rdo_offset = lambda_q10 / 2 - 512;
            if q1_q10_tmp > rdo_offset {
                q1_q0 = (q1_q10_tmp - rdo_offset) >> 10;
            } else if q1_q10_tmp < -rdo_offset {
                q1_q0 = (q1_q10_tmp + rdo_offset) >> 10;
            } else if q1_q10_tmp < 0 {
                q1_q0 = -1;
            } else {
                q1_q0 = 0;
            }
        }
        debug_assert!((-32..=31).contains(&q1_q0));
        let row = &rd_table[(q1_q0 + 32) as usize];
        let q1_q10 = if r_q10 * row[2] - row[3] < 0 { row[1] } else { row[0] };

        pulses[i] = rshift_round(q1_q10, 10) as i8;

        let mut exc_q14 = q1_q10 << 4;
        if nsq.rand_seed < 0 {
            exc_q14 = -exc_q14;
        }
        let lpc_exc_q14 = exc_q14.wrapping_add(ltp_pred_q13 << 1);
        let xq_q14 = lpc_exc_q14.wrapping_add(lpc_pred_q10 << 4);

        nsq.xq[pxq_base + i] = sat16(rshift_round(smulww(xq_q14, gain_q10), 8));

        // State updates.
        p_lpc += 1;
        nsq.s_lpc_q14[p_lpc] = xq_q14;
        nsq.s_diff_shp_q14 = xq_q14.wrapping_sub(x_sc_q10[i] << 4);
        let s_lf = nsq.s_diff_shp_q14.wrapping_sub(n_ar_q12 << 2);
        nsq.s_lf_ar_shp_q14 = s_lf;
        nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx] = s_lf.wrapping_sub(n_lf_q12 << 2);
        s_ltp_q15[nsq.s_ltp_buf_idx] = lpc_exc_q14 << 1;
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;
        nsq.rand_seed = nsq.rand_seed.wrapping_add(i32::from(pulses[i]));
    }

    // Shift the LPC synthesis history down for the next subframe.
    nsq.s_lpc_q14.copy_within(length..length + NSQ_LPC_BUF_LENGTH, 0);
}

/// Quantise one frame's input `x16` into `pulses`, writing the
/// chosen LTP-scaling index. `signal_type`, `quant_offset_type`,
/// `nlsf_interp_coef_q2` and `seed` come from the frame's side info.
#[allow(clippy::too_many_arguments, reason = "mirrors the reference signature")]
pub(crate) fn nsq(
    nsq: &mut NsqState,
    cfg: &NsqConfig,
    signal_type: i32,
    quant_offset_type: i32,
    nlsf_interp_coef_q2: i32,
    seed: i32,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltp_coef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    nsq.rand_seed = seed;
    let mut lag = nsq.lag_prev;
    let offset_q10 = i32::from(QUANTIZATION_OFFSETS_Q10[(signal_type >> 1) as usize][quant_offset_type as usize]);
    let lsf_interp_flag = i32::from(nlsf_interp_coef_q2 != 4);
    // `offset_q10` and `lambda_q10` are constant across the frame, so build the
    // per-sample RD candidate table once here (not per subframe).
    let rd_table = build_rd_table(offset_q10, lambda_q10);

    // Reuse the scratch buffers (taken out so the sub-functions can borrow
    // `&mut nsq`); clear+resize zero-fills to match the reference's fresh
    // stack arrays. Restored before returning.
    let buf_len = cfg.ltp_mem_length + cfg.frame_length;
    let mut s_ltp_q15 = core::mem::take(&mut nsq.scratch_s_ltp_q15);
    let mut s_ltp = core::mem::take(&mut nsq.scratch_s_ltp);
    let mut x_sc_q10 = core::mem::take(&mut nsq.scratch_x_sc_q10);
    s_ltp_q15.clear();
    s_ltp_q15.resize(buf_len, 0);
    s_ltp.clear();
    s_ltp.resize(buf_len, 0);
    x_sc_q10.clear();
    x_sc_q10.resize(cfg.subfr_length, 0);

    nsq.s_ltp_shp_buf_idx = cfg.ltp_mem_length;
    nsq.s_ltp_buf_idx = cfg.ltp_mem_length;
    let mut pxq_base = cfg.ltp_mem_length;
    let mut x_off = 0usize;

    for k in 0..cfg.nb_subfr {
        let a_q12 = &pred_coef_q12[(((k >> 1) | (1 - lsf_interp_flag as usize)) * MAX_LPC_ORDER)..];
        let b_q14 = &ltp_coef_q14[k * LTP_ORDER..];
        let ar_shp = &ar_q13[k * MAX_SHAPE_LPC_ORDER..];
        let harm_packed = (harm_shape_gain_q14[k] >> 2) | ((harm_shape_gain_q14[k] >> 1) << 16);

        nsq.rewhite_flag = false;
        if signal_type == TYPE_VOICED {
            lag = pitch_l[k];
            if k & (3 - (lsf_interp_flag << 1) as usize) == 0 {
                let start_idx = cfg.ltp_mem_length - lag as usize - cfg.predict_lpc_order - LTP_ORDER / 2;
                let xq_off = start_idx + k * cfg.subfr_length;
                let len = cfg.ltp_mem_length - start_idx;
                // `s_ltp` (output) and `nsq.xq` (input) are distinct buffers, so the
                // filter reads and writes without aliasing - no input copy needed.
                lpc_analysis_filter(
                    &mut s_ltp[start_idx..start_idx + len],
                    &nsq.xq[xq_off..xq_off + len],
                    &a_q12[..cfg.predict_lpc_order],
                );
                nsq.rewhite_flag = true;
                nsq.s_ltp_buf_idx = cfg.ltp_mem_length;
            }
        }

        nsq_scale_states(
            nsq,
            cfg,
            &x16[x_off..],
            &mut x_sc_q10,
            &s_ltp,
            &mut s_ltp_q15,
            k,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            signal_type,
        );

        noise_shape_quantizer(
            nsq,
            signal_type,
            &x_sc_q10,
            &mut pulses[k * cfg.subfr_length..],
            pxq_base,
            &mut s_ltp_q15,
            &a_q12[..cfg.predict_lpc_order],
            &b_q14[..LTP_ORDER],
            &ar_shp[..cfg.shaping_lpc_order],
            lag,
            harm_packed,
            tilt_q14[k],
            lf_shp_q14[k],
            gains_q16[k],
            lambda_q10,
            offset_q10,
            &rd_table,
            cfg.subfr_length,
            cfg.shaping_lpc_order,
            cfg.predict_lpc_order,
        );

        x_off += cfg.subfr_length;
        pxq_base += cfg.subfr_length;
    }

    nsq.lag_prev = pitch_l[cfg.nb_subfr - 1];
    // Carry the LTP memory's worth of history into the next frame.
    nsq.xq
        .copy_within(cfg.frame_length..cfg.frame_length + cfg.ltp_mem_length, 0);
    nsq.s_ltp_shp_q14
        .copy_within(cfg.frame_length..cfg.frame_length + cfg.ltp_mem_length, 0);

    // Return the scratch allocations for the next call to reuse.
    nsq.scratch_s_ltp_q15 = s_ltp_q15;
    nsq.scratch_s_ltp = s_ltp;
    nsq.scratch_x_sc_q10 = x_sc_q10;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// NSQ's reconstructed `xq` must equal an independent synthesis from the
    /// pulses it emitted (the decoder's formula): excitation (dithered) +
    /// LPC prediction, scaled by the gain. Unvoiced, single subframe, so no
    /// LTP - isolating the short-term predictor and the dither/seed chain.
    #[test]
    fn unvoiced_xq_matches_resynthesis_from_pulses() {
        let order = 10usize;
        let subfr = 80usize;
        let cfg = NsqConfig {
            frame_length: subfr,
            subfr_length: subfr,
            nb_subfr: 1,
            ltp_mem_length: 256,
            predict_lpc_order: order,
            shaping_lpc_order: 16,
        };
        // A gentle stable LPC (Q12) in both interpolation sets.
        let mut a_q12 = [0i16; 2 * MAX_LPC_ORDER];
        for half in 0..2 {
            a_q12[half * MAX_LPC_ORDER] = 4096; // 1.0
            a_q12[half * MAX_LPC_ORDER + 1] = -1500;
            a_q12[half * MAX_LPC_ORDER + 2] = 600;
        }
        let ar_q13 = [0i16; MAX_SHAPE_LPC_ORDER];
        let ltp = [0i16; LTP_ORDER * MAX_NB_SUBFR];
        let gains = [1 << 18; MAX_NB_SUBFR];
        let pitch = [0i32; MAX_NB_SUBFR];
        let x16: Vec<i16> = (0..subfr).map(|i| ((i as f32 * 0.4).sin() * 8000.0) as i16).collect();
        let mut pulses = vec![0i8; subfr];

        let mut st = NsqState::new();
        let gain_q10 = gains[0] >> 6;
        nsq(
            &mut st,
            &cfg,
            1,
            0,
            4,
            12345,
            &x16,
            &mut pulses,
            &a_q12,
            &ltp,
            &ar_q13,
            &[0; MAX_NB_SUBFR],
            &[0; MAX_NB_SUBFR],
            &[0; MAX_NB_SUBFR],
            &gains,
            &pitch,
            0,
            0,
        );

        // Independent resynthesis using the decoder's own excitation
        // reconstruction (offset + level-adjust + dither) followed by the
        // LPC synthesis - exactly what `silk_decode_core` does for an
        // unvoiced subframe.
        let offset_q10 = i32::from(QUANTIZATION_OFFSETS_Q10[0][0]); // unvoiced, offset type 0
        let mut s_lpc = [0i32; MAX_LPC_ORDER];
        let mut seed = 12345i32;
        let mut xq_ref = vec![0i16; subfr];
        for i in 0..subfr {
            seed = rand(seed);
            let mut exc = i32::from(pulses[i]) << 14;
            if exc > 0 {
                exc -= QUANT_LEVEL_ADJUST_Q10 << 4;
            } else if exc < 0 {
                exc += QUANT_LEVEL_ADJUST_Q10 << 4;
            }
            exc += offset_q10 << 4;
            if seed < 0 {
                exc = -exc;
            }
            let mut pred = (order >> 1) as i32;
            for j in 0..order {
                pred = smlawb(pred, s_lpc[MAX_LPC_ORDER - 1 - j], i32::from(a_q12[j]));
            }
            let xq_q14 = exc.wrapping_add(pred << 4);
            xq_ref[i] = sat16(rshift_round(smulww(xq_q14, gain_q10), 8));
            s_lpc.copy_within(1..MAX_LPC_ORDER, 0);
            s_lpc[MAX_LPC_ORDER - 1] = xq_q14;
            seed = seed.wrapping_add(i32::from(pulses[i]));
        }

        let xq_nsq = &st.xq[cfg.ltp_mem_length..cfg.ltp_mem_length + subfr];
        assert_eq!(xq_nsq, &xq_ref[..], "NSQ xq disagrees with resynthesis from pulses");
        // Sanity: it actually produced some excitation.
        assert!(pulses.iter().any(|&p| p != 0), "all-zero pulses");
    }
}

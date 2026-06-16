//! Stereo prediction quantisation for the SILK encoder (RFC 6716 §5.2;
//! normative `silk/stereo_quant_pred.c`, `silk/stereo_encode_pred.c`).
//!
//! The mid/side stereo predictor weights are vector-quantised on a
//! sub-stepped grid of the shared `STEREO_PRED_QUANT_Q13` levels and coded
//! as a joint index plus two per-predictor refinements. [`stereo_quant_pred`]
//! is the exact inverse of the decoder's `stereo_decode_pred`, so the
//! quantised weights round-trip; [`stereo_encode_pred`] writes the indices
//! the decoder reads.

extern crate alloc;
use alloc::vec;

use crate::range::RangeEncoder;

use super::super::math::{div32_var_q, rshift_round, smlabb, smlawb, smulbb, smulwb, sqrt_approx};
use super::super::plc::sum_sqr_shift;
use super::super::tables::{
    STEREO_ONLY_CODE_MID_ICDF, STEREO_PRED_JOINT_ICDF, STEREO_PRED_QUANT_Q13, UNIFORM3_ICDF, UNIFORM5_ICDF,
};

const STEREO_QUANT_TAB_SIZE: usize = 16;
const STEREO_QUANT_SUB_STEPS: i32 = 5;
/// `SILK_FIX_CONST(0.5 / STEREO_QUANT_SUB_STEPS, 16)`.
const HALF_SUB_STEP_Q16: i32 = 6554;
/// `STEREO_INTERP_LEN_MS` (predictor cross-fade length, even).
const STEREO_INTERP_LEN_MS: usize = 8;
/// `SILK_FIX_CONST(STEREO_RATIO_SMOOTH_COEF, 16)` = round(0.01 * 65536).
const RATIO_SMOOTH_COEF_Q16: i32 = 655;
const LA_SHAPE_MS: i32 = 5;

/// `silk_ADD_LSHIFT32(a, b, s)` / `silk_SUB_LSHIFT32` / `silk_ADD_RSHIFT32`.
#[inline]
const fn add_lshift32(a: i32, b: i32, s: u32) -> i32 {
    a.wrapping_add(b << s)
}
#[inline]
const fn sub_lshift32(a: i32, b: i32, s: u32) -> i32 {
    a.wrapping_sub(b << s)
}

/// `silk_SAT16`.
#[inline]
fn sat16(a: i32) -> i16 {
    a.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// `silk_inner_prod_aligned_scale`: `Σ (x[i]·y[i]) >> scale`.
fn inner_prod_aligned_scale(x: &[i16], y: &[i16], scale: u32, len: usize) -> i32 {
    let mut sum = 0i32;
    for i in 0..len {
        sum = sum.wrapping_add(smulbb(i32::from(x[i]), i32::from(y[i])) >> scale);
    }
    sum
}

/// Cross-frame stereo encoder state (`stereo_enc_state`).
#[derive(Clone, Default)]
pub(crate) struct StereoEncState {
    pred_prev_q13: [i16; 2],
    s_mid: [i16; 2],
    s_side: [i16; 2],
    mid_side_amp_q0: [i32; 4],
    smth_width_q14: i16,
    width_prev_q14: i16,
    silent_side_len: i16,
}

/// `silk_stereo_find_predictor`: the Q13 prediction weight of `y` from `x`
/// and the residual/mid energy ratio (Q14), updating the smoothed mid/
/// residual norms in `mid_res_amp_q0` (2 elements).
fn find_predictor(x: &[i16], y: &[i16], mid_res_amp_q0: &mut [i32], length: usize, smooth_coef_q16: i32) -> (i32, i32) {
    let (mut nrgx, scale1) = sum_sqr_shift(&x[..length]);
    let (mut nrgy, scale2) = sum_sqr_shift(&y[..length]);
    let mut scale = scale1.max(scale2);
    scale += scale & 1; // make even
    nrgy >>= (scale - scale2) as u32;
    nrgx >>= (scale - scale1) as u32;
    nrgx = nrgx.max(1);
    let corr = inner_prod_aligned_scale(x, y, scale as u32, length);
    let mut pred_q13 = div32_var_q(corr, nrgx, 13);
    pred_q13 = pred_q13.clamp(-(1 << 14), 1 << 14);
    let pred2_q10 = smulwb(pred_q13, pred_q13);

    let smooth_coef_q16 = smooth_coef_q16.max(pred2_q10.abs());
    let scale_h = (scale >> 1) as u32;
    mid_res_amp_q0[0] = smlawb(
        mid_res_amp_q0[0],
        (sqrt_approx(nrgx) << scale_h) - mid_res_amp_q0[0],
        smooth_coef_q16,
    );
    nrgy = sub_lshift32(nrgy, smulwb(corr, pred_q13), 3 + 1);
    nrgy = add_lshift32(nrgy, smulwb(nrgx, pred2_q10), 6);
    mid_res_amp_q0[1] = smlawb(
        mid_res_amp_q0[1],
        (sqrt_approx(nrgy) << scale_h) - mid_res_amp_q0[1],
        smooth_coef_q16,
    );
    let ratio_q14 = div32_var_q(mid_res_amp_q0[1], mid_res_amp_q0[0].max(1), 14).clamp(0, 32767);
    (pred_q13, ratio_q14)
}

/// `silk_stereo_quant_pred`: quantise the two predictor weights (Q13) in
/// place, returning the codebook indices `ix[2][3]`. On return `pred_q13[0]`
/// holds the first weight minus the second (the form the NSQ/decoder use).
pub(crate) fn stereo_quant_pred(pred_q13: &mut [i32; 2]) -> [[i8; 3]; 2] {
    let mut ix = [[0i8; 3]; 2];
    for n in 0..2 {
        let mut err_min_q13 = i32::MAX;
        let mut quant_pred_q13 = 0i32;
        'outer: for i in 0..STEREO_QUANT_TAB_SIZE - 1 {
            let low_q13 = i32::from(STEREO_PRED_QUANT_Q13[i]);
            let step_q13 = smulwb(i32::from(STEREO_PRED_QUANT_Q13[i + 1]) - low_q13, HALF_SUB_STEP_Q16);
            for j in 0..STEREO_QUANT_SUB_STEPS {
                let lvl_q13 = smlabb(low_q13, step_q13, 2 * j + 1);
                let err_q13 = (pred_q13[n] - lvl_q13).abs();
                if err_q13 < err_min_q13 {
                    err_min_q13 = err_q13;
                    quant_pred_q13 = lvl_q13;
                    ix[n][0] = i as i8;
                    ix[n][1] = j as i8;
                } else {
                    // The error is monotone away from the best level.
                    break 'outer;
                }
            }
        }
        ix[n][2] = ix[n][0] / 3;
        ix[n][0] -= ix[n][2] * 3;
        pred_q13[n] = quant_pred_q13;
    }
    pred_q13[0] -= pred_q13[1];
    ix
}

/// `silk_stereo_encode_pred`: code the predictor indices (joint index then
/// two uniform refinements per predictor).
pub(crate) fn stereo_encode_pred(enc: &mut RangeEncoder, ix: &[[i8; 3]; 2]) {
    let n = 5 * ix[0][2] + ix[1][2];
    debug_assert!(n < 25);
    enc.encode_icdf(n as usize, &STEREO_PRED_JOINT_ICDF, 8);
    for row in ix {
        enc.encode_icdf(row[0] as usize, &UNIFORM3_ICDF, 8);
        enc.encode_icdf(row[1] as usize, &UNIFORM5_ICDF, 8);
    }
}

/// `silk_stereo_encode_mid_only`: code the mid-only flag.
pub(crate) fn stereo_encode_mid_only(enc: &mut RangeEncoder, mid_only_flag: i8) {
    enc.encode_icdf(mid_only_flag as usize, &STEREO_ONLY_CODE_MID_ICDF, 8);
}

/// `silk_stereo_LR_to_MS`: convert L/R to mid/side, choosing the predictor
/// weights, the mid/side rate split and the mid-only flag. `x1` and `x2` each
/// hold 2 history samples then the `frame_length` frame (length
/// `frame_length + 2`). On return `x1` holds the mid signal (frame at `[2..]`,
/// `[0..2]` the next-frame history) and `x2[1..=frame_length]` the side
/// residual. Returns the predictor indices, the mid-only flag, and the
/// mid/side bitrates.
#[allow(clippy::too_many_arguments, reason = "mirrors the reference signature")]
#[allow(clippy::needless_range_loop, reason = "computed index ranges mirror the reference")]
pub(crate) fn lr_to_ms(
    state: &mut StereoEncState,
    x1: &mut [i16],
    x2: &mut [i16],
    total_rate_bps: i32,
    prev_speech_act_q8: i32,
    to_mono: bool,
    fs_khz: i32,
    frame_length: usize,
) -> ([[i8; 3]; 2], i8, [i32; 2]) {
    let fs = fs_khz as usize;
    let mut total_rate_bps = total_rate_bps;

    // Mid/side from L/R; mid overwrites x1 in place, side into a local buffer.
    let mut side = vec![0i16; frame_length + 2];
    for n in 0..frame_length + 2 {
        let l = i32::from(x1[n]);
        let r = i32::from(x2[n]);
        x1[n] = rshift_round(l + r, 1) as i16;
        side[n] = sat16(rshift_round(l - r, 1));
    }
    // Prepend the saved history, then save this frame's tail for next time.
    x1[0] = state.s_mid[0];
    x1[1] = state.s_mid[1];
    side[0] = state.s_side[0];
    side[1] = state.s_side[1];
    state.s_mid[0] = x1[frame_length];
    state.s_mid[1] = x1[frame_length + 1];
    state.s_side[0] = side[frame_length];
    state.s_side[1] = side[frame_length + 1];

    // Low-pass / high-pass split of mid and side.
    let (mut lp_mid, mut hp_mid) = (vec![0i16; frame_length], vec![0i16; frame_length]);
    let (mut lp_side, mut hp_side) = (vec![0i16; frame_length], vec![0i16; frame_length]);
    for n in 0..frame_length {
        let s = rshift_round(
            add_lshift32(i32::from(x1[n]) + i32::from(x1[n + 2]), i32::from(x1[n + 1]), 1),
            2,
        );
        lp_mid[n] = s as i16;
        hp_mid[n] = (i32::from(x1[n + 1]) - s) as i16;
    }
    for n in 0..frame_length {
        let s = rshift_round(
            add_lshift32(i32::from(side[n]) + i32::from(side[n + 2]), i32::from(side[n + 1]), 1),
            2,
        );
        lp_side[n] = s as i16;
        hp_side[n] = (i32::from(side[n + 1]) - s) as i16;
    }

    let is_10ms = frame_length == 10 * fs;
    let base = if is_10ms {
        RATIO_SMOOTH_COEF_Q16 / 2
    } else {
        RATIO_SMOOTH_COEF_Q16
    };
    let smooth_coef_q16 = smulwb(smulbb(prev_speech_act_q8, prev_speech_act_q8), base);

    let mut pred_q13 = [0i32; 2];
    let (p0, lp_ratio_q14) = find_predictor(
        &lp_mid,
        &lp_side,
        &mut state.mid_side_amp_q0[0..2],
        frame_length,
        smooth_coef_q16,
    );
    let (p1, hp_ratio_q14) = find_predictor(
        &hp_mid,
        &hp_side,
        &mut state.mid_side_amp_q0[2..4],
        frame_length,
        smooth_coef_q16,
    );
    pred_q13[0] = p0;
    pred_q13[1] = p1;
    let frac_q16 = smlabb(hp_ratio_q14, lp_ratio_q14, 3).min(1 << 16);

    total_rate_bps -= if is_10ms { 1200 } else { 600 };
    total_rate_bps = total_rate_bps.max(1);
    let min_mid_rate_bps = smlabb(2000, fs_khz, 600);
    let frac_3_q16 = 3 * frac_q16;
    let mut rates = [0i32; 2];
    rates[0] = div32_var_q(total_rate_bps, (13 << 16) + frac_3_q16, 16 + 3);
    let mut width_q14;
    if rates[0] < min_mid_rate_bps {
        rates[0] = min_mid_rate_bps;
        rates[1] = total_rate_bps - rates[0];
        width_q14 = div32_var_q(
            (rates[1] << 1) - min_mid_rate_bps,
            smulwb((1 << 16) + frac_3_q16, min_mid_rate_bps),
            14 + 2,
        )
        .clamp(0, 1 << 14);
    } else {
        rates[1] = total_rate_bps - rates[0];
        width_q14 = 1 << 14;
    }

    state.smth_width_q14 = smlawb(
        i32::from(state.smth_width_q14),
        width_q14 - i32::from(state.smth_width_q14),
        smooth_coef_q16,
    ) as i16;
    let smth_width = i32::from(state.smth_width_q14);

    let mut mid_only_flag = 0i8;
    let ix;
    if to_mono {
        width_q14 = 0;
        pred_q13 = [0, 0];
        ix = stereo_quant_pred(&mut pred_q13);
    } else if state.width_prev_q14 == 0
        && (8 * total_rate_bps < 13 * min_mid_rate_bps || smulwb(frac_q16, smth_width) < 819)
    {
        pred_q13[0] = smulbb(smth_width, pred_q13[0]) >> 14;
        pred_q13[1] = smulbb(smth_width, pred_q13[1]) >> 14;
        ix = stereo_quant_pred(&mut pred_q13);
        width_q14 = 0;
        pred_q13 = [0, 0];
        rates[0] = total_rate_bps;
        rates[1] = 0;
        mid_only_flag = 1;
    } else if state.width_prev_q14 != 0
        && (8 * total_rate_bps < 11 * min_mid_rate_bps || smulwb(frac_q16, smth_width) < 328)
    {
        pred_q13[0] = smulbb(smth_width, pred_q13[0]) >> 14;
        pred_q13[1] = smulbb(smth_width, pred_q13[1]) >> 14;
        ix = stereo_quant_pred(&mut pred_q13);
        width_q14 = 0;
        pred_q13 = [0, 0];
    } else if state.smth_width_q14 > 15565 {
        ix = stereo_quant_pred(&mut pred_q13);
        width_q14 = 1 << 14;
    } else {
        pred_q13[0] = smulbb(smth_width, pred_q13[0]) >> 14;
        pred_q13[1] = smulbb(smth_width, pred_q13[1]) >> 14;
        ix = stereo_quant_pred(&mut pred_q13);
        width_q14 = smth_width;
    }

    // Keep encoding the side until the tapered output has flushed.
    if mid_only_flag == 1 {
        state.silent_side_len += (frame_length - STEREO_INTERP_LEN_MS * fs) as i16;
        if i32::from(state.silent_side_len) < LA_SHAPE_MS * fs_khz {
            mid_only_flag = 0;
        } else {
            state.silent_side_len = 10000;
        }
    } else {
        state.silent_side_len = 0;
    }
    if mid_only_flag == 0 && rates[1] < 1 {
        rates[1] = 1;
        rates[0] = (total_rate_bps - rates[1]).max(1);
    }

    // Apply the (interpolated) predictor to produce the side residual.
    let interp = STEREO_INTERP_LEN_MS * fs;
    let mut pred0 = -i32::from(state.pred_prev_q13[0]);
    let mut pred1 = -i32::from(state.pred_prev_q13[1]);
    let mut w_q24 = i32::from(state.width_prev_q14) << 10;
    let denom_q16 = (1 << 16) / (STEREO_INTERP_LEN_MS as i32 * fs_khz);
    let delta0 = -rshift_round(smulbb(pred_q13[0] - i32::from(state.pred_prev_q13[0]), denom_q16), 16);
    let delta1 = -rshift_round(smulbb(pred_q13[1] - i32::from(state.pred_prev_q13[1]), denom_q16), 16);
    let deltaw = smulwb(width_q14 - i32::from(state.width_prev_q14), denom_q16) << 10;
    for n in 0..interp {
        pred0 += delta0;
        pred1 += delta1;
        w_q24 += deltaw;
        let mut sum = add_lshift32(i32::from(x1[n]) + i32::from(x1[n + 2]), i32::from(x1[n + 1]), 1) << 9;
        sum = smlawb(smulwb(w_q24, i32::from(side[n + 1])), sum, pred0);
        sum = smlawb(sum, i32::from(x1[n + 1]) << 11, pred1);
        x2[n + 1] = sat16(rshift_round(sum, 8));
    }
    pred0 = -pred_q13[0];
    pred1 = -pred_q13[1];
    w_q24 = width_q14 << 10;
    for n in interp..frame_length {
        let mut sum = add_lshift32(i32::from(x1[n]) + i32::from(x1[n + 2]), i32::from(x1[n + 1]), 1) << 9;
        sum = smlawb(smulwb(w_q24, i32::from(side[n + 1])), sum, pred0);
        sum = smlawb(sum, i32::from(x1[n + 1]) << 11, pred1);
        x2[n + 1] = sat16(rshift_round(sum, 8));
    }
    state.pred_prev_q13[0] = pred_q13[0] as i16;
    state.pred_prev_q13[1] = pred_q13[1] as i16;
    state.width_prev_q14 = width_q14 as i16;

    (ix, mid_only_flag, rates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::{RangeDecoder, RangeEncoder};
    use crate::silk::stereo::{stereo_decode_mid_only, stereo_decode_pred};

    /// Quantised predictor weights round-trip exactly through the decoder's
    /// `stereo_decode_pred`, and the mid-only flag round-trips.
    #[test]
    fn stereo_pred_round_trips_through_the_decoder() {
        for &(p0, p1, mid_only) in &[
            (0i32, 0i32, 0i8),
            (4096, -2048, 1),
            (8000, 3000, 0),
            (-7000, 6000, 1),
            (1234, -5678, 0),
        ] {
            let mut pred = [p0, p1];
            let ix = stereo_quant_pred(&mut pred);
            // `pred` is now the quantised (difference, second) pair.

            let mut enc = RangeEncoder::new(16);
            stereo_encode_pred(&mut enc, &ix);
            stereo_encode_mid_only(&mut enc, mid_only);
            let bytes = enc.finalize().expect("fits");

            let mut dec = RangeDecoder::new(&bytes);
            let dec_pred = stereo_decode_pred(&mut dec);
            let dec_mid_only = stereo_decode_mid_only(&mut dec);

            assert_eq!(dec_pred, pred, "predictor weights disagree for ({p0},{p1})");
            assert_eq!(dec_mid_only, mid_only == 1, "mid-only flag disagrees");
        }
    }

    /// Bit-exact pin of `silk_stereo_LR_to_MS` against the compiled reference
    /// over 50 frames (state evolves; mid-only flips to 0 around frame 49).
    #[test]
    fn lr_to_ms_matches_reference_pins() {
        let (fs_khz, frame) = (16i32, 320usize);
        let sample = |n: i32| -> (i16, i16) {
            if n < 0 {
                return (0, 0);
            }
            let t = core::f64::consts::TAU * f64::from(n);
            let l = 6000.0 * (t / 90.0).sin();
            let r = 3000.0 * (t / 90.0 + 0.3).sin() + 5000.0 * (t / 53.0).sin();
            (l as i16, r as i16)
        };

        let mut st = StereoEncState::default();
        for f in 0..50i32 {
            let mut x1 = vec![0i16; frame + 2];
            let mut x2 = vec![0i16; frame + 2];
            for k in 0..frame + 2 {
                let (l, r) = sample(f * frame as i32 + k as i32 - 2);
                x1[k] = l;
                x2[k] = r;
            }
            let (ix, mid_only, rates) = lr_to_ms(&mut st, &mut x1, &mut x2, 30000, 128, false, fs_khz, frame);

            if f == 0 {
                assert_eq!(ix, [[1, 2, 2], [1, 2, 2]]);
                assert_eq!((mid_only, rates), (1, [29400, 0]));
                assert_eq!(&x1[..6], &[0, 0, 443, 1047, 1643, 2226]);
                assert_eq!(&x2[1..6], &[0, 0, 0, 0, 0]);
                assert_eq!(st.smth_width_q14, 40);
                assert_eq!(st.mid_side_amp_q0, [160, 92, 1, 0]);
            } else if f == 1 {
                assert_eq!((mid_only, rates), (1, [29400, 0]));
                assert_eq!(&x1[..6], &[-1355, -1351, -1345, -1337, -1331, -1328]);
                assert_eq!(st.smth_width_q14, 80);
                assert_eq!(st.mid_side_amp_q0, [313, 183, 1, 0]);
            } else if f == 49 {
                assert_eq!(ix, [[1, 2, 2], [1, 0, 2]]);
                assert_eq!((mid_only, rates), (0, [16450, 12950]));
                assert_eq!(&x1[..6], &[2037, 2219, 2411, 2611, 2814, 3018]);
                assert_eq!(&x2[1..6], &[3, 6, 9, 11, 13]);
                assert_eq!(st.pred_prev_q13, [656, -656]);
                assert_eq!((st.smth_width_q14, st.width_prev_q14), (1895, 1895));
                assert_eq!(st.mid_side_amp_q0, [7465, 4304, 1, 0]);
            }
        }
    }
}

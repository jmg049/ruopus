//! Stereo prediction decoding and mid/side unmixing (RFC 6716 §4.2.7.1-
//! 4.2.7.2; normative `stereo_decode_pred.c`, `stereo_MS_to_LR.c`).
//!
//! Stereo SILK codes a mid channel and (optionally) a side channel plus
//! two predictor weights; the weights interpolate over the first 8 ms of
//! each frame, the predicted side adds onto the decoded side, and mid/side
//! convert to left/right with one sample of look-ahead (hence the
//! two-sample state buffers).

#![allow(dead_code, reason = "consumed incrementally as the SILK decoder stages land")]

use crate::range::RangeDecoder;

use super::math::{rshift_round, smlabb, smlawb, smulbb, smulwb};
use super::tables::{
    STEREO_ONLY_CODE_MID_ICDF, STEREO_PRED_JOINT_ICDF, STEREO_PRED_QUANT_Q13, UNIFORM3_ICDF, UNIFORM5_ICDF,
};

/// `STEREO_INTERP_LEN_MS`.
const STEREO_INTERP_LEN_MS: i32 = 8;
/// `SILK_FIX_CONST(0.5 / STEREO_QUANT_SUB_STEPS, 16)`.
const HALF_SUB_STEP_Q16: i32 = 6554;

/// Cross-frame stereo state (`stereo_dec_state`).
#[derive(Debug, Clone, Default)]
pub(crate) struct StereoDecState {
    /// `pred_prev_Q13`.
    pub pred_prev_q13: [i16; 2],
    /// `sMid` / `sSide`: two-sample history of each channel.
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
}

/// `silk_stereo_decode_pred`: the two stereo predictor weights (Q13).
pub(crate) fn stereo_decode_pred(dec: &mut RangeDecoder) -> [i32; 2] {
    // Entropy decoding: joint index then per-predictor refinements.
    let n = dec.decode_icdf(&STEREO_PRED_JOINT_ICDF, 8) as i32;
    let mut ix = [[0i32; 3]; 2];
    ix[0][2] = n / 5;
    ix[1][2] = n - 5 * ix[0][2];
    for row in &mut ix {
        row[0] = dec.decode_icdf(&UNIFORM3_ICDF, 8) as i32;
        row[1] = dec.decode_icdf(&UNIFORM5_ICDF, 8) as i32;
    }

    // Dequantise.
    let mut pred_q13 = [0i32; 2];
    for (pred, row) in pred_q13.iter_mut().zip(&mut ix) {
        row[0] += 3 * row[2];
        let low_q13 = i32::from(STEREO_PRED_QUANT_Q13[row[0] as usize]);
        let step_q13 = smulwb(
            i32::from(STEREO_PRED_QUANT_Q13[row[0] as usize + 1]) - low_q13,
            HALF_SUB_STEP_Q16,
        );
        *pred = smlabb(low_q13, step_q13, 2 * row[1] + 1);
    }

    // Subtract the second from the first (simplifies application).
    pred_q13[0] -= pred_q13[1];
    pred_q13
}

/// `silk_stereo_decode_mid_only`: whether only the mid channel is coded.
pub(crate) fn stereo_decode_mid_only(dec: &mut RangeDecoder) -> bool {
    dec.decode_icdf(&STEREO_ONLY_CODE_MID_ICDF, 8) == 1
}

/// `silk_stereo_MS_to_LR`: mid/side → left/right in place.
///
/// `x1`/`x2` hold two history samples followed by `frame_length` new
/// samples (and `x1` one further look-ahead slot); on return
/// `x1[1..=frame_length]` is left and `x2[1..=frame_length]` is right.
pub(crate) fn stereo_ms_to_lr(
    state: &mut StereoDecState,
    x1: &mut [i16],
    x2: &mut [i16],
    pred_q13: &[i32; 2],
    fs_khz: i32,
    frame_length: usize,
) {
    // Buffering: restore the two-sample history, save the new tail.
    x1[..2].copy_from_slice(&state.s_mid);
    x2[..2].copy_from_slice(&state.s_side);
    state.s_mid.copy_from_slice(&x1[frame_length..frame_length + 2]);
    state.s_side.copy_from_slice(&x2[frame_length..frame_length + 2]);

    // Interpolate the predictors over the first 8 ms while adding the
    // prediction onto the side channel.
    let mut pred0_q13 = i32::from(state.pred_prev_q13[0]);
    let mut pred1_q13 = i32::from(state.pred_prev_q13[1]);
    let interp_len = (STEREO_INTERP_LEN_MS * fs_khz) as usize;
    let denom_q16 = (1i32 << 16) / (STEREO_INTERP_LEN_MS * fs_khz);
    let delta0_q13 = rshift_round(smulbb(pred_q13[0] - i32::from(state.pred_prev_q13[0]), denom_q16), 16);
    let delta1_q13 = rshift_round(smulbb(pred_q13[1] - i32::from(state.pred_prev_q13[1]), denom_q16), 16);
    let side_predict = |x1: &[i16], x2: &mut [i16], n: usize, p0: i32, p1: i32| {
        // Q11 three-tap smoothing of mid, then both predictor taps in Q8.
        let sum = (i32::from(x1[n]) + i32::from(x1[n + 2]) + (i32::from(x1[n + 1]) << 1)) << 9;
        let sum = smlawb(i32::from(x2[n + 1]) << 8, sum, p0);
        let sum = smlawb(sum, i32::from(x1[n + 1]) << 11, p1);
        x2[n + 1] = rshift_round(sum, 8).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    };
    for n in 0..interp_len {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        side_predict(x1, x2, n, pred0_q13, pred1_q13);
    }
    for n in interp_len..frame_length {
        side_predict(x1, x2, n, pred_q13[0], pred_q13[1]);
    }
    state.pred_prev_q13[0] = pred_q13[0] as i16;
    state.pred_prev_q13[1] = pred_q13[1] as i16;

    // Mid/side → left/right.
    for n in 0..frame_length {
        let sum = i32::from(x1[n + 1]) + i32::from(x2[n + 1]);
        let diff = i32::from(x1[n + 1]) - i32::from(x2[n + 1]);
        x1[n + 1] = sum.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        x2[n + 1] = diff.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins generated by compiling the reference `stereo_MS_to_LR.c` with
    /// this exact state and input.
    #[test]
    fn ms_to_lr_matches_reference_pins() {
        let mut state = StereoDecState {
            pred_prev_q13: [1000, -500],
            s_mid: [11, -22],
            s_side: [33, -44],
        };
        let mut x1 = [0i16; 163];
        let mut x2 = [0i16; 163];
        for i in 0..163 {
            x1[i] = ((i as i32 * 73) % 501 - 250) as i16;
            x2[i] = ((i as i32 * 37) % 301 - 150) as i16;
        }
        stereo_ms_to_lr(&mut state, &mut x1, &mut x2, &[4000, -2500], 8, 160);

        assert_eq!(
            &x1[1..25],
            [
                -69, -182, -72, 43, 158, 254, -129, -34, -219, -104, 12, 129, 220, -158, -67, 49, -135, -18, 99, 187,
                -187, -100, 17, 135
            ]
        );
        assert_eq!(
            &x2[1..25],
            [
                25, -26, 10, 41, 72, 122, -351, -300, 31, 62, 92, 121, 176, -302, -247, -217, 113, 142, 171, 229, -253,
                -194, -165, -137
            ]
        );
        assert_eq!([x1[157], x1[160], x2[157], x2[160]], [102, -59, 276, -127]);
        assert_eq!(state.pred_prev_q13, [4000, -2500]);
        assert_eq!(state.s_mid, [-93, -20]);
        assert_eq!(state.s_side, [51, 88]);
    }
}

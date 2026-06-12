//! Per-frame parameter assembly (RFC 6716 §4.2.7.4-4.2.7.6; normative
//! `decode_parameters.c`, `bwexpander.c`).
//!
//! Turns the decoded side information into the synthesis controls: linear
//! subframe gains, the two half-frame LPC coefficient sets (with NLSF
//! interpolation), per-subframe pitch lags, the LTP filter taps (Q14) from
//! the periodicity codebook, and the LTP scaling factor.

#![allow(dead_code, reason = "consumed incrementally as the SILK decoder stages land")]

use super::gains::gains_dequant;
use super::indices::{CondCoding, MAX_LPC_ORDER, MAX_NB_SUBFR, SideInfoIndices, TYPE_VOICED, nlsf_codebook};
use super::math::{mul, rshift_round};
use super::nlsf::{nlsf_decode, nlsf2a};
use super::pitch::decode_pitch;
use super::tables::{LTP_GAIN_VQ_0, LTP_GAIN_VQ_1, LTP_GAIN_VQ_2, LTPSCALES_TABLE_Q14};

/// `LTP_ORDER`: taps per LTP filter.
pub(crate) const LTP_ORDER: usize = 5;
/// `BWE_AFTER_LOSS_Q16`.
const BWE_AFTER_LOSS_Q16: i32 = 63570;

/// `silk_bwexpander`: chirps an i16 AR filter. The reference deliberately
/// avoids `silk_SMULWB` here - its bias can destabilise filters.
pub(crate) fn bwexpander(ar: &mut [i16], chirp_q16: i32) {
    let d = ar.len();
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    let mut chirp_q16 = chirp_q16;
    for coef in ar.iter_mut().take(d - 1) {
        *coef = rshift_round(mul(chirp_q16, i32::from(*coef)), 16) as i16;
        chirp_q16 += rshift_round(mul(chirp_q16, chirp_minus_one_q16), 16);
    }
    ar[d - 1] = rshift_round(mul(chirp_q16, i32::from(ar[d - 1])), 16) as i16;
}

/// Synthesis controls for one frame (`silk_decoder_control`).
#[derive(Debug, Clone, Default)]
pub(crate) struct DecoderControl {
    /// Per-subframe pitch lags in samples (`pitchL`).
    pub pitch_l: [i32; MAX_NB_SUBFR],
    /// Linear gains, Q16 (`Gains_Q16`).
    pub gains_q16: [i32; MAX_NB_SUBFR],
    /// LPC coefficients (Q12) for each half frame (`PredCoef_Q12`).
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    /// LTP taps, Q14, `nb_subfr * LTP_ORDER` (`LTPCoef_Q14`).
    pub ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    /// LTP scale, Q14 (`LTP_scale_Q14`).
    pub ltp_scale_q14: i32,
}

/// The cross-frame state read and updated by [`decode_parameters`].
#[derive(Debug, Clone)]
pub(crate) struct ParamState {
    /// `LastGainIndex`.
    pub last_gain_index: i8,
    /// `prevNLSF_Q15`.
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    /// `first_frame_after_reset`.
    pub first_frame_after_reset: bool,
    /// `lossCnt` (0 outside packet-loss concealment).
    pub loss_cnt: i32,
}

impl Default for ParamState {
    fn default() -> Self {
        ParamState {
            last_gain_index: 10,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            first_frame_after_reset: true,
            loss_cnt: 0,
        }
    }
}

/// `silk_decode_parameters`: side information → synthesis controls.
///
/// `indices.per_index` is reset to 0 for unvoiced frames exactly as the
/// reference mutates its copy.
pub(crate) fn decode_parameters(
    indices: &mut SideInfoIndices,
    state: &mut ParamState,
    fs_khz: i32,
    nb_subfr: usize,
    cond_coding: CondCoding,
) -> DecoderControl {
    let cb = nlsf_codebook(fs_khz);
    let order = cb.order;
    let mut ctrl = DecoderControl {
        gains_q16: gains_dequant(
            &indices.gains_indices,
            &mut state.last_gain_index,
            cond_coding == CondCoding::Conditionally,
            nb_subfr,
        ),
        ..DecoderControl::default()
    };

    // NLSFs for the second half frame, then their LPC coefficients.
    let nlsf_q15 = nlsf_decode(&indices.nlsf_indices, cb);
    nlsf2a(&mut ctrl.pred_coef_q12[1][..order], &nlsf_q15[..order]);

    // A reset forbids interpolation (helps loss right after a switch).
    if state.first_frame_after_reset {
        indices.nlsf_interp_coef_q2 = 4;
    }

    if indices.nlsf_interp_coef_q2 < 4 {
        // Interpolate the first-half NLSFs between previous and current.
        let mut nlsf0_q15 = [0i16; MAX_LPC_ORDER];
        for i in 0..order {
            nlsf0_q15[i] = (i32::from(state.prev_nlsf_q15[i])
                + ((i32::from(indices.nlsf_interp_coef_q2)
                    * (i32::from(nlsf_q15[i]) - i32::from(state.prev_nlsf_q15[i])))
                    >> 2)) as i16;
        }
        nlsf2a(&mut ctrl.pred_coef_q12[0][..order], &nlsf0_q15[..order]);
    } else {
        ctrl.pred_coef_q12[0] = ctrl.pred_coef_q12[1];
    }
    state.prev_nlsf_q15 = nlsf_q15;

    // Bandwidth expansion after packet loss.
    if state.loss_cnt != 0 {
        bwexpander(&mut ctrl.pred_coef_q12[0][..order], BWE_AFTER_LOSS_Q16);
        bwexpander(&mut ctrl.pred_coef_q12[1][..order], BWE_AFTER_LOSS_Q16);
    }

    if i32::from(indices.signal_type) == TYPE_VOICED {
        ctrl.pitch_l = decode_pitch(indices.lag_index, indices.contour_index, fs_khz, nb_subfr);

        // LTP taps from the periodicity codebook (Q7 → Q14).
        for k in 0..nb_subfr {
            let ix = indices.ltp_index[k] as usize;
            let row: &[i8; LTP_ORDER] = match indices.per_index {
                0 => &LTP_GAIN_VQ_0[ix],
                1 => &LTP_GAIN_VQ_1[ix],
                _ => &LTP_GAIN_VQ_2[ix],
            };
            for (i, &c) in row.iter().enumerate() {
                ctrl.ltp_coef_q14[k * LTP_ORDER + i] = i16::from(c) << 7;
            }
        }

        ctrl.ltp_scale_q14 = i32::from(LTPSCALES_TABLE_Q14[indices.ltp_scale_index as usize]);
    } else {
        indices.per_index = 0;
        ctrl.ltp_scale_q14 = 0;
    }
    ctrl
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Glue-level smoke test: the components are pinned individually
    /// (gains, NLSF chain, pitch); this checks the assembly paths - copy
    /// vs interpolation of the first-half LPC set, voiced vs unvoiced LTP.
    #[test]
    fn assembles_voiced_and_unvoiced_frames() {
        let mut state = ParamState::default();

        let mut ind = SideInfoIndices {
            signal_type: 2,
            gains_indices: [32, 20, 20, 25],
            nlsf_indices: [17, 3, -2, 0, 1, -4, 2, 0, -1, 5, 0, 0, 0, 0, 0, 0, 0],
            nlsf_interp_coef_q2: 2,
            lag_index: 100,
            contour_index: 5,
            per_index: 1,
            ltp_index: [3, 1, 0, 2],
            ltp_scale_index: 1,
            ..Default::default()
        };
        let ctrl = decode_parameters(&mut ind, &mut state, 8, 4, CondCoding::Independently);
        // First frame after reset: interpolation suppressed → both halves
        // identical.
        assert_eq!(ind.nlsf_interp_coef_q2, 4);
        assert_eq!(ctrl.pred_coef_q12[0], ctrl.pred_coef_q12[1]);
        assert_eq!(ctrl.pitch_l, [116, 116, 116, 117]);
        assert_eq!(ctrl.ltp_scale_q14, i32::from(LTPSCALES_TABLE_Q14[1]));
        // Q7 → Q14 of LTP_GAIN_VQ_1 row 3.
        assert_eq!(
            &ctrl.ltp_coef_q14[..LTP_ORDER],
            LTP_GAIN_VQ_1[3].map(|c| i16::from(c) << 7)
        );
        assert!(ctrl.gains_q16.iter().all(|&g| g > 0));

        // Second frame: interpolation now allowed and must differ from the
        // copy path for a different NLSF vector.
        state.first_frame_after_reset = false;
        let mut ind2 = SideInfoIndices {
            signal_type: 1,
            gains_indices: [4, 0, 5, 8],
            nlsf_indices: [3, -1, 2, 0, -3, 1, 0, 2, -1, 0, 1, 0, 0, 0, 0, 0, 0],
            nlsf_interp_coef_q2: 1,
            ..Default::default()
        };
        let ctrl2 = decode_parameters(&mut ind2, &mut state, 8, 4, CondCoding::Conditionally);
        assert_eq!(ind2.nlsf_interp_coef_q2, 1);
        assert_ne!(ctrl2.pred_coef_q12[0], ctrl2.pred_coef_q12[1]);
        // Unvoiced: no pitch, no LTP.
        assert_eq!(ctrl2.pitch_l, [0; 4]);
        assert_eq!(ctrl2.ltp_coef_q14, [0i16; 20]);
        assert_eq!(ctrl2.ltp_scale_q14, 0);
        assert_eq!(ind2.per_index, 0);
    }
}

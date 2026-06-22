//! Gain processing for the SILK encoder (RFC 6716 §5.2).
//!
//! [`process_gains`] turns the raw per-subframe gains (and residual
//! energies) into the quantised gains the noise-shaping quantiser uses: it
//! reduces the gain when the LTP coding gain is high, soft-limits the
//! residual-to-gain ratio, quantises via the shared [`gains_quant`], and
//! computes the rate-distortion `lambda` and (for voiced frames) the
//! quantiser offset type.

extern crate alloc;

use super::super::gains::gains_quant;
use super::super::indices::{CondCoding, MAX_NB_SUBFR, TYPE_VOICED};
use super::super::tables::QUANTIZATION_OFFSETS_Q10;

// Rate-distortion `lambda` tuning constants.
const LAMBDA_OFFSET: f32 = 1.2;
const LAMBDA_SPEECH_ACT: f32 = -0.2;
const LAMBDA_DELAYED_DECISIONS: f32 = -0.05;
const LAMBDA_INPUT_QUALITY: f32 = -0.1;
const LAMBDA_CODING_QUALITY: f32 = -0.2;
const LAMBDA_QUANT_OFFSET: f32 = 0.8;

/// Logistic sigmoid `1 / (1 + e^-x)`.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The outputs of [`process_gains`].
pub(crate) struct GainsResult {
    /// Quantised per-subframe gains in Q16 (input to NSQ).
    pub gains_q16: [i32; MAX_NB_SUBFR],
    /// Unquantised (limited) per-subframe gains in Q16 (`GainsUnq_Q16`), the
    /// base the `max_bits` rate-control loop rescales by a gain multiplier.
    pub gains_unq_q16: [i32; MAX_NB_SUBFR],
    /// The coded gain indices.
    pub gains_indices: [i8; MAX_NB_SUBFR],
    /// Rate-distortion trade-off (NSQ's `Lambda`, float).
    pub lambda: f32,
    /// Quantiser offset type, possibly updated for voiced frames.
    pub quant_offset_type: i32,
}

/// Limit and quantise the gains and derive the RD
/// lambda. `gains` are the raw per-subframe gains (linear); `res_nrg` the
/// per-subframe residual energies. `last_gain_index` is the cross-frame
/// accumulator (`sShape.LastGainIndex`), updated in place.
#[allow(clippy::too_many_arguments, reason = "mirrors the reference encoder state inputs")]
pub(crate) fn process_gains(
    gains: &mut [f32; MAX_NB_SUBFR],
    res_nrg: &[f32; MAX_NB_SUBFR],
    signal_type: i32,
    quant_offset_type: i32,
    nb_subfr: usize,
    subfr_length: usize,
    snr_db_q7: i32,
    ltp_pred_cod_gain: f32,
    input_tilt_q15: i32,
    n_states_delayed_decision: i32,
    speech_activity_q8: i32,
    input_quality: f32,
    coding_quality: f32,
    last_gain_index: &mut i8,
    cond_coding: CondCoding,
) -> GainsResult {
    let mut quant_offset_type = quant_offset_type;

    // Reduce the gain when the LTP coding gain is high (voiced).
    if signal_type == TYPE_VOICED {
        let s = 1.0 - 0.5 * sigmoid(0.25 * (ltp_pred_cod_gain - 12.0));
        for g in gains.iter_mut().take(nb_subfr) {
            *g *= s;
        }
    }

    // Soft limit on the residual-energy/squared-gain ratio.
    let inv_max_sqr_val = 2.0f32.powf(0.33 * (21.0 - snr_db_q7 as f32 / 128.0)) / subfr_length as f32;
    for k in 0..nb_subfr {
        let g = gains[k];
        gains[k] = (g * g + res_nrg[k] * inv_max_sqr_val).sqrt().min(32767.0);
    }

    // Quantise the gains. Keep the pre-quantisation values (`GainsUnq_Q16`)
    // for the `max_bits` rate-control loop.
    let mut gains_q16 = [0i32; MAX_NB_SUBFR];
    for k in 0..nb_subfr {
        gains_q16[k] = (gains[k] * 65536.0) as i32;
    }
    let gains_unq_q16 = gains_q16;
    let mut gains_indices = [0i8; MAX_NB_SUBFR];
    gains_quant(
        &mut gains_indices,
        &mut gains_q16,
        last_gain_index,
        cond_coding == CondCoding::Conditionally,
        nb_subfr,
    );
    for k in 0..nb_subfr {
        gains[k] = gains_q16[k] as f32 / 65536.0;
    }

    // Quantiser offset for voiced signals.
    if signal_type == TYPE_VOICED {
        quant_offset_type = i32::from(ltp_pred_cod_gain + input_tilt_q15 as f32 / 32768.0 <= 1.0);
    }

    let quant_offset =
        f32::from(QUANTIZATION_OFFSETS_Q10[(signal_type >> 1) as usize][quant_offset_type as usize]) / 1024.0;
    let lambda = LAMBDA_OFFSET
        + LAMBDA_DELAYED_DECISIONS * n_states_delayed_decision as f32
        + LAMBDA_SPEECH_ACT * speech_activity_q8 as f32 / 256.0
        + LAMBDA_INPUT_QUALITY * input_quality
        + LAMBDA_CODING_QUALITY * coding_quality
        + LAMBDA_QUANT_OFFSET * quant_offset;

    GainsResult {
        gains_q16,
        gains_unq_q16,
        gains_indices,
        lambda,
        quant_offset_type,
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::gains::gains_dequant;
    use super::*;

    /// The processed gains are bounded, the quantised gains round-trip
    /// through the decoder's dequantiser, and lambda is sensible.
    #[test]
    fn process_gains_limits_and_quantises() {
        let mut gains = [2000.0f32, 8000.0, 500.0, 30000.0];
        let res_nrg = [1.0e7f32, 5.0e6, 2.0e7, 1.0e6];
        let mut last_idx = 20i8;
        let out = process_gains(
            &mut gains,
            &res_nrg,
            1, // unvoiced
            0,
            4,
            80,
            18 << 7, // SNR ~18 dB in Q7
            0.0,
            0,
            0,
            128,
            0.5,
            0.5,
            &mut last_idx,
            CondCoding::Independently,
        );

        // Gains are bounded and the Q16 form matches the float form.
        for (k, &g) in gains.iter().enumerate() {
            assert!(g > 0.0 && g <= 32767.0, "gain {k} = {g}");
            assert_eq!(out.gains_q16[k], (g * 65536.0) as i32);
        }
        // The chosen indices decode (via the decoder) to the same gains.
        let mut dec_idx = 20i8;
        let dec = gains_dequant(&out.gains_indices, &mut dec_idx, false, 4);
        assert_eq!(dec, out.gains_q16, "gains must round-trip through gains_dequant");
        // Lambda is in a sane positive range.
        assert!(out.lambda > 0.0 && out.lambda < 4.0, "lambda {}", out.lambda);
    }
}

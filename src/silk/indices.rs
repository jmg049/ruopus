//! Side-information decoding (RFC 6716 §4.2.7.3-4.2.7.6).
//!
//! Per SILK frame this decodes, in bitstream order: signal type and
//! quantisation offset type, subframe gain indices (independent MSB+LSB or
//! delta-coded), the two-stage NLSF VQ indices with extension escapes, the
//! NLSF interpolation factor (20 ms frames), and - for voiced frames -
//! pitch lag (absolute or delta), pitch contour, the LTP filter codebook
//! indices, and the LTP scaling index; finally the LCG seed.

#![allow(dead_code, reason = "consumed incrementally as the SILK decoder stages land")]

use super::tables::{
    DELTA_GAIN_ICDF, GAIN_ICDF, LTP_GAIN_ICDF_0, LTP_GAIN_ICDF_1, LTP_GAIN_ICDF_2, LTP_PER_INDEX_ICDF, LTPSCALE_ICDF,
    NLSF_CB1_ICDF_NB_MB, NLSF_CB1_ICDF_WB, NLSF_CB1_NB_MB_Q8, NLSF_CB1_WB_Q8, NLSF_CB1_WB_WGHT_Q9, NLSF_CB1_WGHT_Q9,
    NLSF_CB2_BITS_NB_MB_Q5, NLSF_CB2_BITS_WB_Q5, NLSF_CB2_ICDF_NB_MB, NLSF_CB2_ICDF_WB, NLSF_CB2_SELECT_NB_MB,
    NLSF_CB2_SELECT_WB, NLSF_DELTA_MIN_NB_MB_Q15, NLSF_DELTA_MIN_WB_Q15, NLSF_EXT_ICDF, NLSF_INTERPOLATION_FACTOR_ICDF,
    NLSF_PRED_NB_MB_Q8, NLSF_PRED_WB_Q8, PITCH_CONTOUR_10_MS_ICDF, PITCH_CONTOUR_10_MS_NB_ICDF, PITCH_CONTOUR_ICDF,
    PITCH_CONTOUR_NB_ICDF, PITCH_DELTA_ICDF, PITCH_LAG_ICDF, TYPE_OFFSET_NO_VAD_ICDF, TYPE_OFFSET_VAD_ICDF,
    UNIFORM4_ICDF, UNIFORM6_ICDF, UNIFORM8_ICDF,
};
use crate::range::RangeDecoder;

/// `MAX_NB_SUBFR`: subframes per 20 ms frame.
pub(crate) const MAX_NB_SUBFR: usize = 4;
/// `MAX_LPC_ORDER` (wideband; narrowband/mediumband use 10).
pub(crate) const MAX_LPC_ORDER: usize = 16;
/// `NLSF_QUANT_MAX_AMPLITUDE`: stage-two residuals beyond ±4 escape to the
/// extension iCDF.
pub(crate) const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;

/// Signal types (`TYPE_NO_VOICE_ACTIVITY`/`TYPE_UNVOICED`/`TYPE_VOICED`).
pub(crate) const TYPE_NO_VOICE_ACTIVITY: i32 = 0;
pub(crate) const TYPE_UNVOICED: i32 = 1;
pub(crate) const TYPE_VOICED: i32 = 2;

/// Conditional-coding mode for a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CondCoding {
    /// `CODE_INDEPENDENTLY`.
    Independently,
    /// `CODE_INDEPENDENTLY_NO_LTP_SCALING`.
    IndependentlyNoLtpScaling,
    /// `CODE_CONDITIONALLY`.
    Conditionally,
}

/// One NLSF codebook: NB/MB (order 10) or WB (order 16).
pub(crate) struct NlsfCodebook {
    pub n_vectors: usize,
    pub order: usize,
    /// `SILK_FIX_CONST(0.18 | 0.15, 16)`.
    pub quant_step_size_q16: i32,
    /// `SILK_FIX_CONST(1/0.18 | 1/0.15, 6)`.
    pub inv_quant_step_size_q6: i32,
    pub cb1_nlsf_q8: &'static [u8],
    pub cb1_wght_q9: &'static [i16],
    pub cb1_icdf: &'static [u8],
    pub pred_q8: &'static [u8],
    pub ec_sel: &'static [u8],
    pub ec_icdf: &'static [u8],
    pub ec_rates_q5: &'static [u8],
    pub delta_min_q15: &'static [i16],
}

/// The NB/MB NLSF codebook (order 10).
pub(crate) const NLSF_CB_NB_MB: NlsfCodebook = NlsfCodebook {
    n_vectors: 32,
    order: 10,
    quant_step_size_q16: 11796,  // SILK_FIX_CONST(0.18, 16)
    inv_quant_step_size_q6: 355, // SILK_FIX_CONST(1.0/0.18, 6)
    cb1_nlsf_q8: &NLSF_CB1_NB_MB_Q8,
    cb1_wght_q9: &NLSF_CB1_WGHT_Q9,
    cb1_icdf: &NLSF_CB1_ICDF_NB_MB,
    pred_q8: &NLSF_PRED_NB_MB_Q8,
    ec_sel: &NLSF_CB2_SELECT_NB_MB,
    ec_icdf: &NLSF_CB2_ICDF_NB_MB,
    ec_rates_q5: &NLSF_CB2_BITS_NB_MB_Q5,
    delta_min_q15: &NLSF_DELTA_MIN_NB_MB_Q15,
};

/// The WB NLSF codebook (order 16).
pub(crate) const NLSF_CB_WB: NlsfCodebook = NlsfCodebook {
    n_vectors: 32,
    order: 16,
    quant_step_size_q16: 9830,   // SILK_FIX_CONST(0.15, 16)
    inv_quant_step_size_q6: 426, // SILK_FIX_CONST(1.0/0.15, 6)
    cb1_nlsf_q8: &NLSF_CB1_WB_Q8,
    cb1_wght_q9: &NLSF_CB1_WB_WGHT_Q9,
    cb1_icdf: &NLSF_CB1_ICDF_WB,
    pred_q8: &NLSF_PRED_WB_Q8,
    ec_sel: &NLSF_CB2_SELECT_WB,
    ec_icdf: &NLSF_CB2_ICDF_WB,
    ec_rates_q5: &NLSF_CB2_BITS_WB_Q5,
    delta_min_q15: &NLSF_DELTA_MIN_WB_Q15,
};

/// The codebook for an internal rate.
pub(crate) const fn nlsf_codebook(fs_khz: i32) -> &'static NlsfCodebook {
    if fs_khz == 16 { &NLSF_CB_WB } else { &NLSF_CB_NB_MB }
}

/// The pitch-lag low-bits iCDF for an internal rate.
pub(crate) const fn pitch_lag_low_bits_icdf(fs_khz: i32) -> &'static [u8] {
    match fs_khz {
        16 => &UNIFORM8_ICDF,
        12 => &UNIFORM6_ICDF,
        _ => &UNIFORM4_ICDF,
    }
}

/// The pitch-contour iCDF for an internal rate and subframe count.
pub(crate) const fn pitch_contour_icdf(fs_khz: i32, nb_subfr: usize) -> &'static [u8] {
    match (fs_khz == 8, nb_subfr == MAX_NB_SUBFR) {
        (true, true) => &PITCH_CONTOUR_NB_ICDF,
        (true, false) => &PITCH_CONTOUR_10_MS_NB_ICDF,
        (false, true) => &PITCH_CONTOUR_ICDF,
        (false, false) => &PITCH_CONTOUR_10_MS_ICDF,
    }
}

/// The LTP gain iCDF for a periodicity index.
pub(crate) const fn ltp_gain_icdf(per_index: usize) -> &'static [u8] {
    match per_index {
        0 => &LTP_GAIN_ICDF_0,
        1 => &LTP_GAIN_ICDF_1,
        _ => &LTP_GAIN_ICDF_2,
    }
}

/// Decoded side information for one frame.
#[derive(Debug, Clone, Default)]
pub(crate) struct SideInfoIndices {
    pub gains_indices: [i8; MAX_NB_SUBFR],
    pub ltp_index: [i8; MAX_NB_SUBFR],
    pub nlsf_indices: [i8; MAX_LPC_ORDER + 1],
    pub lag_index: i16,
    pub contour_index: i8,
    pub signal_type: i8,
    pub quant_offset_type: i8,
    pub nlsf_interp_coef_q2: i8,
    pub per_index: i8,
    pub ltp_scale_index: i8,
    pub seed: i8,
}

/// Cross-frame entropy-coding state used by [`decode_indices`]: the previous
/// frame's signal type and lag index.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct EcPrevState {
    pub signal_type: i32,
    pub lag_index: i16,
}

/// Per-coefficient entropy-table offsets and predictor values selected by
/// the stage-one index.
pub(crate) fn nlsf_unpack(cb: &NlsfCodebook, cb1_index: usize) -> ([i16; MAX_LPC_ORDER], [u8; MAX_LPC_ORDER]) {
    let mut ec_ix = [0i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    let sel = &cb.ec_sel[cb1_index * cb.order / 2..];
    for i in (0..cb.order).step_by(2) {
        let entry = i32::from(sel[i / 2]);
        ec_ix[i] = (((entry >> 1) & 7) * (2 * NLSF_QUANT_MAX_AMPLITUDE + 1)) as i16;
        pred_q8[i] = cb.pred_q8[i + (entry & 1) as usize * (cb.order - 1)];
        ec_ix[i + 1] = (((entry >> 5) & 7) * (2 * NLSF_QUANT_MAX_AMPLITUDE + 1)) as i16;
        pred_q8[i + 1] = cb.pred_q8[i + ((entry >> 4) & 1) as usize * (cb.order - 1) + 1];
    }
    (ec_ix, pred_q8)
}

/// Decodes all side information for one frame.
///
/// `vad_flag` is this frame's VAD bit (or ignored when `decode_lbrr`);
/// `prev` carries the previous frame's signal type and lag index for
/// conditional coding and is updated in place.
pub(crate) fn decode_indices(
    dec: &mut RangeDecoder,
    fs_khz: i32,
    nb_subfr: usize,
    vad_flag: bool,
    decode_lbrr: bool,
    cond_coding: CondCoding,
    prev: &mut EcPrevState,
) -> SideInfoIndices {
    let cb = nlsf_codebook(fs_khz);
    let mut ind = SideInfoIndices::default();

    // Signal type and quantiser offset type.
    let ix = if decode_lbrr || vad_flag {
        dec.decode_icdf(&TYPE_OFFSET_VAD_ICDF, 8) as i32 + 2
    } else {
        dec.decode_icdf(&TYPE_OFFSET_NO_VAD_ICDF, 8) as i32
    };
    ind.signal_type = (ix >> 1) as i8;
    ind.quant_offset_type = (ix & 1) as i8;

    // Gains: first subframe conditional or independent (MSBs then 3 LSBs).
    if cond_coding == CondCoding::Conditionally {
        ind.gains_indices[0] = dec.decode_icdf(&DELTA_GAIN_ICDF, 8) as i8;
    } else {
        ind.gains_indices[0] = (dec.decode_icdf(&GAIN_ICDF[ind.signal_type as usize], 8) << 3) as i8;
        ind.gains_indices[0] += dec.decode_icdf(&UNIFORM8_ICDF, 8) as i8;
    }
    for i in 1..nb_subfr {
        ind.gains_indices[i] = dec.decode_icdf(&DELTA_GAIN_ICDF, 8) as i8;
    }

    // NLSF indices: stage one, then per-coefficient stage-two residuals
    // with extension escapes.
    ind.nlsf_indices[0] = dec.decode_icdf(&cb.cb1_icdf[(ind.signal_type as usize >> 1) * cb.n_vectors..], 8) as i8;
    let (ec_ix, _pred) = nlsf_unpack(cb, ind.nlsf_indices[0] as usize);
    for (i, &ec_off) in ec_ix.iter().enumerate().take(cb.order) {
        let mut ix = dec.decode_icdf(&cb.ec_icdf[ec_off as usize..], 8) as i32;
        if ix == 0 {
            ix -= dec.decode_icdf(&NLSF_EXT_ICDF, 8) as i32;
        } else if ix == 2 * NLSF_QUANT_MAX_AMPLITUDE {
            ix += dec.decode_icdf(&NLSF_EXT_ICDF, 8) as i32;
        }
        ind.nlsf_indices[i + 1] = (ix - NLSF_QUANT_MAX_AMPLITUDE) as i8;
    }

    // NLSF interpolation factor (20 ms frames only).
    ind.nlsf_interp_coef_q2 = if nb_subfr == MAX_NB_SUBFR {
        dec.decode_icdf(&NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i8
    } else {
        4
    };

    if i32::from(ind.signal_type) == TYPE_VOICED {
        // Pitch lag: delta from the previous frame when possible,
        // otherwise absolute (high then low bits).
        let mut decode_absolute = true;
        if cond_coding == CondCoding::Conditionally && prev.signal_type == TYPE_VOICED {
            let delta = dec.decode_icdf(&PITCH_DELTA_ICDF, 8) as i32;
            if delta > 0 {
                ind.lag_index = prev.lag_index + (delta - 9) as i16;
                decode_absolute = false;
            }
        }
        if decode_absolute {
            ind.lag_index = (dec.decode_icdf(&PITCH_LAG_ICDF, 8) as i32 * (fs_khz >> 1)) as i16;
            ind.lag_index += dec.decode_icdf(pitch_lag_low_bits_icdf(fs_khz), 8) as i16;
        }
        prev.lag_index = ind.lag_index;

        ind.contour_index = dec.decode_icdf(pitch_contour_icdf(fs_khz, nb_subfr), 8) as i8;

        // LTP filter: periodicity index, then per-subframe codebook
        // indices, then the scaling index for independent coding.
        ind.per_index = dec.decode_icdf(&LTP_PER_INDEX_ICDF, 8) as i8;
        for k in 0..nb_subfr {
            ind.ltp_index[k] = dec.decode_icdf(ltp_gain_icdf(ind.per_index as usize), 8) as i8;
        }
        ind.ltp_scale_index = if cond_coding == CondCoding::Independently {
            dec.decode_icdf(&LTPSCALE_ICDF, 8) as i8
        } else {
            0
        };
    }
    prev.signal_type = i32::from(ind.signal_type);

    ind.seed = dec.decode_icdf(&UNIFORM4_ICDF, 8) as i8;
    ind
}

/// Writes one frame's side information - type/offset,
/// gains (delta or absolute), the NLSF VQ path, NLSF interpolation, and (for
/// voiced frames) the pitch lag/contour, LTP gains/scaling - plus the LCG
/// seed. The exact inverse of [`decode_indices`].
#[allow(clippy::too_many_lines, reason = "follows the bitstream sequence")]
#[allow(clippy::too_many_arguments, reason = "decodes many independent fields")]
pub(crate) fn encode_indices(
    enc: &mut crate::range::RangeEncoder,
    ind: &SideInfoIndices,
    fs_khz: i32,
    nb_subfr: usize,
    encode_lbrr: bool,
    vad_flag: bool,
    cond_coding: CondCoding,
    prev: &mut EcPrevState,
) {
    let cb = nlsf_codebook(fs_khz);
    let typ = i32::from(ind.signal_type) * 2 + i32::from(ind.quant_offset_type);
    if encode_lbrr || vad_flag {
        enc.encode_icdf((typ - 2) as usize, &TYPE_OFFSET_VAD_ICDF, 8);
    } else {
        enc.encode_icdf(typ as usize, &TYPE_OFFSET_NO_VAD_ICDF, 8);
    }

    if cond_coding == CondCoding::Conditionally {
        enc.encode_icdf(ind.gains_indices[0] as usize, &DELTA_GAIN_ICDF, 8);
    } else {
        enc.encode_icdf(
            (ind.gains_indices[0] >> 3) as usize,
            &GAIN_ICDF[ind.signal_type as usize],
            8,
        );
        enc.encode_icdf((ind.gains_indices[0] & 7) as usize, &UNIFORM8_ICDF, 8);
    }
    for i in 1..nb_subfr {
        enc.encode_icdf(ind.gains_indices[i] as usize, &DELTA_GAIN_ICDF, 8);
    }

    enc.encode_icdf(
        ind.nlsf_indices[0] as usize,
        &cb.cb1_icdf[(ind.signal_type as usize >> 1) * cb.n_vectors..],
        8,
    );
    let (ec_ix, _) = nlsf_unpack(cb, ind.nlsf_indices[0] as usize);
    for (i, &ec_off) in ec_ix.iter().enumerate().take(cb.order) {
        let v = i32::from(ind.nlsf_indices[i + 1]);
        let table = &cb.ec_icdf[ec_off as usize..];
        if v >= NLSF_QUANT_MAX_AMPLITUDE {
            enc.encode_icdf(2 * NLSF_QUANT_MAX_AMPLITUDE as usize, table, 8);
            enc.encode_icdf((v - NLSF_QUANT_MAX_AMPLITUDE) as usize, &NLSF_EXT_ICDF, 8);
        } else if v <= -NLSF_QUANT_MAX_AMPLITUDE {
            enc.encode_icdf(0, table, 8);
            enc.encode_icdf((-v - NLSF_QUANT_MAX_AMPLITUDE) as usize, &NLSF_EXT_ICDF, 8);
        } else {
            enc.encode_icdf((v + NLSF_QUANT_MAX_AMPLITUDE) as usize, table, 8);
        }
    }

    if nb_subfr == MAX_NB_SUBFR {
        enc.encode_icdf(ind.nlsf_interp_coef_q2 as usize, &NLSF_INTERPOLATION_FACTOR_ICDF, 8);
    }

    if i32::from(ind.signal_type) == TYPE_VOICED {
        let mut encode_absolute = true;
        if cond_coding == CondCoding::Conditionally && prev.signal_type == TYPE_VOICED {
            let delta = i32::from(ind.lag_index) - i32::from(prev.lag_index);
            let symbol = if (-8..=11).contains(&delta) {
                encode_absolute = false;
                delta + 9
            } else {
                0
            };
            enc.encode_icdf(symbol as usize, &PITCH_DELTA_ICDF, 8);
        }
        if encode_absolute {
            let half = fs_khz >> 1;
            enc.encode_icdf((i32::from(ind.lag_index) / half) as usize, &PITCH_LAG_ICDF, 8);
            enc.encode_icdf(
                (i32::from(ind.lag_index) % half) as usize,
                pitch_lag_low_bits_icdf(fs_khz),
                8,
            );
        }
        prev.lag_index = ind.lag_index;

        enc.encode_icdf(ind.contour_index as usize, pitch_contour_icdf(fs_khz, nb_subfr), 8);

        enc.encode_icdf(ind.per_index as usize, &LTP_PER_INDEX_ICDF, 8);
        for k in 0..nb_subfr {
            enc.encode_icdf(ind.ltp_index[k] as usize, ltp_gain_icdf(ind.per_index as usize), 8);
        }
        if cond_coding == CondCoding::Independently {
            enc.encode_icdf(ind.ltp_scale_index as usize, &LTPSCALE_ICDF, 8);
        }
    }
    prev.signal_type = i32::from(ind.signal_type);

    enc.encode_icdf(ind.seed as usize, &UNIFORM4_ICDF, 8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::{RangeDecoder, RangeEncoder};

    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *seed >> 8
    }

    /// Random valid side info for the given configuration.
    fn random_indices(seed: &mut u32, fs_khz: i32, nb_subfr: usize, vad: bool, cond: CondCoding) -> SideInfoIndices {
        let cb = nlsf_codebook(fs_khz);
        let mut ind = SideInfoIndices::default();
        if vad {
            ind.signal_type = (1 + lcg(seed) % 2) as i8; // unvoiced or voiced
        } else {
            ind.signal_type = 0;
        }
        ind.quant_offset_type = (lcg(seed) % 2) as i8;
        if cond == CondCoding::Conditionally {
            ind.gains_indices[0] = (lcg(seed) % 41) as i8;
        } else {
            ind.gains_indices[0] = (lcg(seed) % 64) as i8;
        }
        for i in 1..nb_subfr {
            ind.gains_indices[i] = (lcg(seed) % 41) as i8;
        }
        ind.nlsf_indices[0] = (lcg(seed) % cb.n_vectors as u32) as i8;
        for i in 0..cb.order {
            // Stay within the extension range the tables can express.
            ind.nlsf_indices[i + 1] = ((lcg(seed) % 19) as i32 - 9) as i8;
        }
        ind.nlsf_interp_coef_q2 = if nb_subfr == MAX_NB_SUBFR {
            (lcg(seed) % 5) as i8
        } else {
            4
        };
        if i32::from(ind.signal_type) == TYPE_VOICED {
            let max_lag = 32 * (fs_khz >> 1) - 1;
            ind.lag_index = (lcg(seed) % (max_lag as u32 + 1)) as i16;
            let contour_len = pitch_contour_icdf(fs_khz, nb_subfr).len();
            ind.contour_index = (lcg(seed) % contour_len as u32) as i8;
            ind.per_index = (lcg(seed) % 3) as i8;
            for k in 0..nb_subfr {
                let n = ltp_gain_icdf(ind.per_index as usize).len();
                ind.ltp_index[k] = (lcg(seed) % n as u32) as i8;
            }
            if cond == CondCoding::Independently {
                ind.ltp_scale_index = (lcg(seed) % 3) as i8;
            }
        }
        ind.seed = (lcg(seed) % 4) as i8;
        ind
    }

    #[test]
    fn indices_round_trip_across_configurations() {
        let mut seed = 0xfeed_u32;
        for fs_khz in [8i32, 12, 16] {
            for nb_subfr in [2usize, 4] {
                for cond in [
                    CondCoding::Independently,
                    CondCoding::IndependentlyNoLtpScaling,
                    CondCoding::Conditionally,
                ] {
                    for vad in [false, true] {
                        for _ in 0..25 {
                            let ind = random_indices(&mut seed, fs_khz, nb_subfr, vad, cond);
                            let mut enc_prev = EcPrevState {
                                signal_type: TYPE_VOICED,
                                lag_index: 50,
                            };
                            let mut dec_prev = enc_prev;

                            let mut enc = RangeEncoder::new(256);
                            encode_indices(&mut enc, &ind, fs_khz, nb_subfr, false, vad, cond, &mut enc_prev);
                            let bytes = enc.finalize().expect("fits");

                            let mut dec = RangeDecoder::new(&bytes);
                            let got = decode_indices(&mut dec, fs_khz, nb_subfr, vad, false, cond, &mut dec_prev);

                            assert_eq!(got.signal_type, ind.signal_type);
                            assert_eq!(got.quant_offset_type, ind.quant_offset_type);
                            assert_eq!(got.gains_indices[..nb_subfr], ind.gains_indices[..nb_subfr]);
                            assert_eq!(got.nlsf_indices, ind.nlsf_indices, "fs={fs_khz} cond={cond:?}");
                            assert_eq!(got.nlsf_interp_coef_q2, ind.nlsf_interp_coef_q2);
                            if i32::from(ind.signal_type) == TYPE_VOICED {
                                assert_eq!(got.lag_index, ind.lag_index, "fs={fs_khz} cond={cond:?}");
                                assert_eq!(got.contour_index, ind.contour_index);
                                assert_eq!(got.per_index, ind.per_index);
                                assert_eq!(got.ltp_index[..nb_subfr], ind.ltp_index[..nb_subfr]);
                                assert_eq!(got.ltp_scale_index, ind.ltp_scale_index);
                            }
                            assert_eq!(got.seed, ind.seed);
                            assert_eq!(dec_prev.signal_type, enc_prev.signal_type);
                            assert_eq!(dec_prev.lag_index, enc_prev.lag_index);
                        }
                    }
                }
            }
        }
    }
}

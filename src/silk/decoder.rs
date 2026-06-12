//! The SILK channel decoder state and synthesis core (RFC 6716 §4.2.7.8-
//! 4.2.7.9; normative `decode_core.c`, state from `structs.h`).
//!
//! [`SilkChannelDecoder::decode_core`] is the inverse NSQ operation: the
//! pulse excitation is offset/sign-randomised into Q14, voiced subframes
//! add a 5-tap long-term prediction over a rewhitened history, and the
//! short-term LPC filter plus gain scaling produce the 16-bit output at
//! the internal rate. All arithmetic is bit-exact fixed point.

#![allow(dead_code, reason = "consumed incrementally as the SILK decoder stages land")]

use alloc::vec;

use crate::range::RangeDecoder;

use super::indices::{CondCoding, EcPrevState, MAX_LPC_ORDER, SideInfoIndices, TYPE_VOICED, decode_indices};
use super::lpc::lpc_analysis_filter;
use super::math::{add_sat32, div32_var_q, inverse32_var_q, lshift_sat32, rshift_round, smlawb, smulwb, smulww};
use super::params::{DecoderControl, LTP_ORDER, ParamState, decode_parameters};
use super::pulses::decode_pulses;
use super::tables::QUANTIZATION_OFFSETS_Q10;

/// `MAX_FRAME_LENGTH`: 20 ms at 16 kHz.
pub(crate) const MAX_FRAME_LENGTH: usize = 320;
/// `MAX_SUB_FRAME_LENGTH`: 5 ms at 16 kHz.
pub(crate) const MAX_SUB_FRAME_LENGTH: usize = 80;
/// `LTP_MEM_LENGTH_MS`.
pub(crate) const LTP_MEM_LENGTH_MS: usize = 20;
/// `QUANT_LEVEL_ADJUST_Q10`.
const QUANT_LEVEL_ADJUST_Q10: i32 = 80;
/// `silk_RAND` constants (`RAND_MULTIPLIER`, `RAND_INCREMENT`).
const RAND_MULTIPLIER: i32 = 196_314_165;
const RAND_INCREMENT: i32 = 907_633_515;

/// `silk_RAND`.
#[inline]
const fn silk_rand(seed: i32) -> i32 {
    RAND_INCREMENT.wrapping_add(seed.wrapping_mul(RAND_MULTIPLIER))
}

/// Per-channel decoder state (`silk_decoder_state`, decode-relevant
/// fields).
pub(crate) struct SilkChannelDecoder {
    /// Internal rate in kHz (8, 12 or 16).
    pub fs_khz: i32,
    /// Subframes per frame (2 for 10 ms, 4 for 20 ms).
    pub nb_subfr: usize,
    /// Samples per frame at the internal rate.
    pub frame_length: usize,
    /// Samples per 5 ms subframe.
    pub subfr_length: usize,
    /// LTP history length (20 ms).
    pub ltp_mem_length: usize,
    /// LPC order (10 for NB/MB, 16 for WB).
    pub lpc_order: usize,
    /// Synthesis history (`outBuf`).
    pub out_buf: [i16; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
    /// LPC filter state (`sLPC_Q14_buf`).
    pub slpc_q14_buf: [i32; MAX_LPC_ORDER],
    /// Excitation (`exc_Q14`).
    pub exc_q14: [i32; MAX_FRAME_LENGTH],
    /// `prev_gain_Q16`.
    pub prev_gain_q16: i32,
    /// `lagPrev`.
    pub lag_prev: i32,
    /// `lossCnt`.
    pub loss_cnt: i32,
    /// `prevSignalType`.
    pub prev_signal_type: i32,
    /// Decoded side information of the current frame.
    pub indices: SideInfoIndices,
    /// Entropy-coding history for `decode_indices`.
    pub ec_prev: EcPrevState,
    /// Parameter-stage history (gain index, previous NLSFs, reset flag).
    pub params: ParamState,
}

impl SilkChannelDecoder {
    /// Creates a decoder for the given internal rate and frame duration
    /// (`silk_decoder_set_fs` reset semantics).
    pub fn new(fs_khz: i32, nb_subfr: usize) -> Self {
        debug_assert!(fs_khz == 8 || fs_khz == 12 || fs_khz == 16);
        debug_assert!(nb_subfr == 2 || nb_subfr == 4);
        let subfr_length = 5 * fs_khz as usize;
        SilkChannelDecoder {
            fs_khz,
            nb_subfr,
            frame_length: nb_subfr * subfr_length,
            subfr_length,
            ltp_mem_length: LTP_MEM_LENGTH_MS * fs_khz as usize,
            lpc_order: if fs_khz == 16 { 16 } else { 10 },
            out_buf: [0; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
            slpc_q14_buf: [0; MAX_LPC_ORDER],
            exc_q14: [0; MAX_FRAME_LENGTH],
            prev_gain_q16: 65536,
            lag_prev: 100,
            loss_cnt: 0,
            prev_signal_type: 0,
            indices: SideInfoIndices::default(),
            ec_prev: EcPrevState::default(),
            params: ParamState::default(),
        }
    }

    /// `silk_decode_frame` (normal decode path): side information,
    /// excitation, parameters, synthesis, and history update for one frame.
    ///
    /// Packet-loss concealment and comfort-noise state updates are not yet
    /// ported; they only affect output after lost packets, never a
    /// loss-free decode.
    pub fn decode_frame(
        &mut self,
        dec: &mut RangeDecoder,
        xq: &mut [i16],
        vad_flag: bool,
        decode_lbrr: bool,
        cond_coding: CondCoding,
    ) {
        debug_assert!(xq.len() >= self.frame_length);

        self.indices = decode_indices(
            dec,
            self.fs_khz,
            self.nb_subfr,
            vad_flag,
            decode_lbrr,
            cond_coding,
            &mut self.ec_prev,
        );
        let pulses = decode_pulses(
            dec,
            i32::from(self.indices.signal_type),
            i32::from(self.indices.quant_offset_type),
            self.frame_length,
        );

        self.params.loss_cnt = self.loss_cnt;
        let mut indices = self.indices.clone();
        let mut ctrl = decode_parameters(&mut indices, &mut self.params, self.fs_khz, self.nb_subfr, cond_coding);
        self.indices = indices;

        self.decode_core(&mut ctrl, xq, &pulses);

        // Shift the synthesis history and append this frame.
        let mv_len = self.ltp_mem_length - self.frame_length;
        self.out_buf.copy_within(self.frame_length..self.ltp_mem_length, 0);
        self.out_buf[mv_len..self.ltp_mem_length].copy_from_slice(&xq[..self.frame_length]);

        self.loss_cnt = 0;
        self.prev_signal_type = i32::from(self.indices.signal_type);
        self.params.first_frame_after_reset = false;
        self.lag_prev = ctrl.pitch_l[self.nb_subfr - 1];
    }

    /// `silk_decode_core`: excitation → LTP → LPC synthesis for one frame;
    /// writes `frame_length` samples into `xq`.
    #[allow(
        clippy::needless_range_loop,
        clippy::explicit_counter_loop,
        reason = "the loops mirror the reference decode_core sequence index-for-index"
    )]
    pub fn decode_core(&mut self, ctrl: &mut DecoderControl, xq: &mut [i16], pulses: &[i16]) {
        debug_assert!(self.prev_gain_q16 != 0);
        let frame_length = self.frame_length;
        let subfr_length = self.subfr_length;
        let ltp_mem_length = self.ltp_mem_length;
        let lpc_order = self.lpc_order;

        let mut s_ltp = vec![0i16; ltp_mem_length];
        let mut s_ltp_q15 = vec![0i32; ltp_mem_length + frame_length];
        let mut res_q14 = vec![0i32; subfr_length];
        let mut s_lpc_q14 = vec![0i32; subfr_length + MAX_LPC_ORDER];

        let offset_q10 = i32::from(
            QUANTIZATION_OFFSETS_Q10[(self.indices.signal_type >> 1) as usize][self.indices.quant_offset_type as usize],
        );

        let nlsf_interpolation_flag = self.indices.nlsf_interp_coef_q2 < 4;

        // Decode the excitation: offsets, then pseudorandom sign flips.
        let mut rand_seed = i32::from(self.indices.seed);
        for i in 0..frame_length {
            rand_seed = silk_rand(rand_seed);
            let mut exc = i32::from(pulses[i]) << 14;
            if exc > 0 {
                exc -= QUANT_LEVEL_ADJUST_Q10 << 4;
            } else if exc < 0 {
                exc += QUANT_LEVEL_ADJUST_Q10 << 4;
            }
            exc += offset_q10 << 4;
            if rand_seed < 0 {
                exc = -exc;
            }
            self.exc_q14[i] = exc;
            rand_seed = rand_seed.wrapping_add(i32::from(pulses[i]));
        }

        s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&self.slpc_q14_buf);

        let mut s_ltp_buf_idx = ltp_mem_length;
        let mut lag = 0i32;
        for k in 0..self.nb_subfr {
            let xq_off = k * subfr_length;
            let exc_off = k * subfr_length;
            let a_q12: [i16; MAX_LPC_ORDER] = ctrl.pred_coef_q12[k >> 1];
            let b_off = k * LTP_ORDER;
            let mut signal_type = i32::from(self.indices.signal_type);

            let gain_q10 = ctrl.gains_q16[k] >> 6;
            let mut inv_gain_q31 = inverse32_var_q(ctrl.gains_q16[k], 47);

            // Gain adjustment when the gain changes between subframes.
            let gain_adj_q16 = if ctrl.gains_q16[k] != self.prev_gain_q16 {
                let adj = div32_var_q(self.prev_gain_q16, ctrl.gains_q16[k], 16);
                for v in s_lpc_q14.iter_mut().take(MAX_LPC_ORDER) {
                    *v = smulww(adj, *v);
                }
                adj
            } else {
                1 << 16
            };
            debug_assert!(inv_gain_q31 != 0);
            self.prev_gain_q16 = ctrl.gains_q16[k];

            // Avoid an abrupt transition from voiced PLC to unvoiced
            // normal decoding.
            if self.loss_cnt != 0
                && self.prev_signal_type == TYPE_VOICED
                && i32::from(self.indices.signal_type) != TYPE_VOICED
                && k < 2
            {
                ctrl.ltp_coef_q14[b_off..b_off + LTP_ORDER].fill(0);
                ctrl.ltp_coef_q14[b_off + LTP_ORDER / 2] = 4096; // 0.25 in Q14
                signal_type = TYPE_VOICED;
                ctrl.pitch_l[k] = self.lag_prev;
            }

            if signal_type == TYPE_VOICED {
                lag = ctrl.pitch_l[k];

                // Rewhitening of the synthesis history.
                if k == 0 || (k == 2 && nlsf_interpolation_flag) {
                    let start_idx = ltp_mem_length as i32 - lag - lpc_order as i32 - (LTP_ORDER / 2) as i32;
                    debug_assert!(start_idx > 0);
                    let start_idx = start_idx as usize;

                    if k == 2 {
                        self.out_buf[ltp_mem_length..ltp_mem_length + 2 * subfr_length]
                            .copy_from_slice(&xq[..2 * subfr_length]);
                    }

                    lpc_analysis_filter(
                        &mut s_ltp[start_idx..ltp_mem_length],
                        &self.out_buf
                            [start_idx + k * subfr_length..start_idx + k * subfr_length + (ltp_mem_length - start_idx)],
                        &a_q12[..lpc_order],
                    );

                    // LTP downscaling on the first subframe reduces
                    // inter-packet dependency.
                    if k == 0 {
                        inv_gain_q31 = smulwb(inv_gain_q31, ctrl.ltp_scale_q14) << 2;
                    }
                    for i in 0..(lag as usize + LTP_ORDER / 2) {
                        s_ltp_q15[s_ltp_buf_idx - i - 1] =
                            smulwb(inv_gain_q31, i32::from(s_ltp[ltp_mem_length - i - 1]));
                    }
                } else if gain_adj_q16 != 1 << 16 {
                    // Rescale the LTP state when the gain changes.
                    for i in 0..(lag as usize + LTP_ORDER / 2) {
                        s_ltp_q15[s_ltp_buf_idx - i - 1] = smulww(gain_adj_q16, s_ltp_q15[s_ltp_buf_idx - i - 1]);
                    }
                }
            }

            // Long-term prediction.
            if signal_type == TYPE_VOICED {
                let mut pred_lag_idx = s_ltp_buf_idx - lag as usize + LTP_ORDER / 2;
                for i in 0..subfr_length {
                    // The +2 bias compensates SMLAWB's round-to--inf.
                    let mut ltp_pred_q13 = 2i32;
                    for t in 0..LTP_ORDER {
                        ltp_pred_q13 = smlawb(
                            ltp_pred_q13,
                            s_ltp_q15[pred_lag_idx - t],
                            i32::from(ctrl.ltp_coef_q14[b_off + t]),
                        );
                    }
                    pred_lag_idx += 1;

                    // LPC excitation.
                    res_q14[i] = self.exc_q14[exc_off + i].wrapping_add(ltp_pred_q13 << 1);

                    s_ltp_q15[s_ltp_buf_idx] = res_q14[i] << 1;
                    s_ltp_buf_idx += 1;
                }
            } else {
                res_q14[..subfr_length].copy_from_slice(&self.exc_q14[exc_off..exc_off + subfr_length]);
            }

            // Short-term prediction and gain scaling.
            for i in 0..subfr_length {
                // The order/2 bias compensates SMLAWB's round-to--inf.
                let mut lpc_pred_q10 = (lpc_order as i32) >> 1;
                for (t, &coef) in a_q12.iter().enumerate().take(lpc_order) {
                    lpc_pred_q10 = smlawb(lpc_pred_q10, s_lpc_q14[MAX_LPC_ORDER + i - 1 - t], i32::from(coef));
                }

                s_lpc_q14[MAX_LPC_ORDER + i] = add_sat32(res_q14[i], lshift_sat32(lpc_pred_q10, 4));
                xq[xq_off + i] = rshift_round(smulww(s_lpc_q14[MAX_LPC_ORDER + i], gain_q10), 8)
                    .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
            }

            // Carry the LPC state into the next subframe.
            s_lpc_q14.copy_within(subfr_length..subfr_length + MAX_LPC_ORDER, 0);
        }

        self.slpc_q14_buf.copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::super::params::DecoderControl;
    use super::*;

    /// Pins generated by compiling the reference decode_core.c and driving
    /// it with this exact hand-built state: a voiced frame (rewhitening,
    /// LTP, per-subframe gain changes) followed by an unvoiced frame
    /// (state carry, gain adjustment).
    #[test]
    fn decode_core_matches_reference_pins() {
        let mut dec = SilkChannelDecoder::new(8, 4);
        dec.indices.signal_type = 2;
        dec.indices.quant_offset_type = 0;
        dec.indices.seed = 3;
        dec.indices.nlsf_interp_coef_q2 = 4;
        for i in 0..dec.ltp_mem_length {
            dec.out_buf[i] = ((i as i32 * 37) % 1001 - 500) as i16;
        }
        for i in 0..MAX_LPC_ORDER {
            dec.slpc_q14_buf[i] = (i as i32 + 1) * 100_000;
        }

        let a10: [i16; 10] = [1927, 3805, -2430, -1018, -459, 2085, 1202, -4536, 221, 1400];
        let mut ctrl = DecoderControl::default();
        for k in 0..4 {
            ctrl.gains_q16[k] = 100_000 + 30_000 * k as i32;
            ctrl.pitch_l[k] = 60 + 3 * k as i32;
        }
        ctrl.pred_coef_q12[0][..10].copy_from_slice(&a10);
        ctrl.pred_coef_q12[1][..10].copy_from_slice(&a10);
        for k in 0..4 * LTP_ORDER {
            ctrl.ltp_coef_q14[k] = ((((k as i32 * 7) % 32) - 16) << 7) as i16;
        }
        ctrl.ltp_scale_q14 = 15565;

        let pulses: Vec<i16> = (0..160).map(|i| ((i * 13) % 11 - 5) as i16).collect();
        let mut xq = [0i16; 160];

        dec.decode_core(&mut ctrl, &mut xq, &pulses);
        let want_voiced: [i16; 160] = [
            40, 24, -47, -66, -122, -103, -149, -28, 29, 58, 51, 6, 55, 56, 97, 73, 93, 66, 84, 26, 5, -41, -74, -101,
            -111, -109, -135, -108, -119, -73, -56, 2, -1, 24, 146, 207, 229, 167, 77, 1, -102, -126, -135, -122, -249,
            -188, -191, -19, 81, 155, 117, 78, 96, 83, 171, 119, 68, -94, -133, -196, -155, -104, -89, -81, -83, 0, 43,
            172, 176, 183, 100, 110, 72, 87, 49, -22, -106, -186, -184, -206, -122, -161, -70, -106, 31, 40, 173, 165,
            191, 162, 118, 124, 36, 70, -77, -51, -194, -131, -200, -109, -121, -44, -15, 3, 88, 81, 184, 100, 186, 70,
            92, -13, 1, -73, -86, -116, -183, -129, -164, -38, -81, 86, 15, 140, 95, 172, 116, 143, 92, 15, 1, -84,
            -52, -156, -68, -180, -73, -110, 10, 5, 89, 117, 116, 185, 130, 187, 67, 101, -72, -26, -153, -114, -180,
            -163, -175, -153, -57, -45, 88, 51,
        ];
        assert_eq!(xq, want_voiced);

        dec.indices.signal_type = 1;
        dec.indices.seed = 1;
        for k in 0..4 {
            ctrl.gains_q16[k] = 80_000 + 10_000 * k as i32;
        }
        dec.decode_core(&mut ctrl, &mut xq, &pulses);
        let want_unvoiced: [i16; 160] = [
            163, 88, 194, 118, 160, 70, 27, -41, -111, -98, -173, -101, -192, -85, -145, 3, -17, 105, 96, 138, 143,
            127, 163, 70, 109, -47, -8, -159, -81, -187, -111, -157, -102, -79, -39, 62, 49, 166, 87, 194, 75, 165, 42,
            67, -46, -74, -120, -168, -112, -173, -58, -130, 20, -38, 114, 85, 175, 148, 144, 132, 61, 79, -42, -6,
            -155, -90, -197, -104, -149, -66, -62, -21, 45, 53, 156, 107, 190, 72, 130, 8, 43, -47, -61, -120, -158,
            -134, -172, -69, -101, 15, -22, 101, 66, 151, 129, 136, 109, 52, 56, -42, -2, -119, -71, -177, -91, -139,
            -47, -40, 8, 52, 49, 140, 83, 180, 71, 126, -10, 33, -71, -51, -107, -116, -124, -141, -66, -94, 38, -1,
            120, 60, 165, 108, 145, 100, 59, 26, -54, -27, -143, -73, -180, -94, -153, -31, -43, 41, 70, 90, 129, 112,
            173, 81, 128, -8, 21, -95, -55, -132, -120, -142,
        ];
        assert_eq!(xq, want_unvoiced);
    }
}

#[cfg(test)]
mod frame_tests {
    use alloc::vec;

    use super::*;

    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *seed
    }

    /// The frame driver must decode arbitrary byte streams without
    /// panicking for every rate/duration/coding configuration, advancing
    /// state across frames (robustness; the conformance vectors are the
    /// correctness oracle once the packet layer lands).
    #[test]
    fn decode_frame_handles_arbitrary_streams() {
        let mut seed = 0x0531_u32;
        for fs_khz in [8i32, 12, 16] {
            for nb_subfr in [2usize, 4] {
                let mut decoder = SilkChannelDecoder::new(fs_khz, nb_subfr);
                let mut xq = [0i16; MAX_FRAME_LENGTH];
                for frame in 0..8 {
                    let data: vec::Vec<u8> = (0..200).map(|_| (lcg(&mut seed) >> 13) as u8).collect();
                    let mut dec = RangeDecoder::new(&data);
                    let cond = if frame == 0 {
                        CondCoding::Independently
                    } else {
                        CondCoding::Conditionally
                    };
                    decoder.decode_frame(&mut dec, &mut xq, frame % 2 == 0, false, cond);
                }
            }
        }
    }
}

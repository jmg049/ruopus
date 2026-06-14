//! Per-frame SILK encode assembly (RFC 6716 §5.2; normative
//! `silk/float/encode_frame_FLP.c`).
//!
//! [`SilkChannelEncoder::encode_frame`] is the end-to-end SILK encode path:
//! it ties the analysis and quantisation kernels into a single coded frame
//! and is validated by round-tripping through the bit-exact decoder. Both
//! the unvoiced path (short-term prediction only) and the voiced path
//! (adding long-term/pitch prediction) are assembled here; the higher-level
//! mode/stereo glue and the pitch *search* (which would choose the lag
//! indices the voiced path currently takes as input) build on top.
//!
//! The chain is: Burg LPC → NLSF VQ → `nlsf2a` requantise → (voiced) LTP
//! correlation + gain VQ → noise-shaping analysis → gain quantisation →
//! NSQ → index/pulse bitstream. The decoder rebuilds the prediction
//! coefficients and gains from the coded indices, so feeding NSQ those same
//! quantised values makes its reconstruction equal the decoder's output.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::range::RangeEncoder;

use super::super::indices::{
    CondCoding, EcPrevState, MAX_LPC_ORDER, MAX_NB_SUBFR, SideInfoIndices, TYPE_UNVOICED, TYPE_VOICED, encode_indices,
    nlsf_codebook,
};
use super::super::lpc::lpc_analysis_filter;
use super::super::nlsf::nlsf2a;
use super::super::params::LTP_ORDER;
use super::super::pitch::decode_pitch;
use super::super::pulses::encode_pulses;
use super::super::tables::LTPSCALES_TABLE_Q14;
use super::gains::process_gains;
use super::lpc::burg_modified;
use super::ltp::{find_ltp, quant_ltp_gains};
use super::nlsf::{a2nlsf, nlsf_encode, nlsf_vq_weights_laroia};
use super::noise_shape::{NoiseShapeConfig, ShapeState, noise_shape_analysis};
use super::nsq::{NsqConfig, NsqState, nsq};

/// Voiced-frame parameters: the pitch lag/contour indices (which a later
/// pitch *search* will choose; supplied here) and the normalised long-term
/// correlation that tunes the harmonic noise shaper.
#[derive(Clone, Copy)]
pub(crate) struct VoicedParams {
    pub lag_index: i16,
    pub contour_index: i8,
    pub ltp_corr: f32,
}

/// One channel's SILK encoder state.
pub(crate) struct SilkChannelEncoder {
    pub nsq: NsqState,
    /// Cross-frame noise-shaping smoothing state (`sShape`).
    pub shape: ShapeState,
    /// Gain-quantiser accumulator (`sShape.LastGainIndex`).
    pub last_gain_index: i8,
    /// Cumulative LTP max-gain accumulator (`sum_log_gain_Q7`).
    pub sum_log_gain_q7: i32,
    /// Entropy-coding history for [`encode_indices`].
    pub ec_prev: EcPrevState,
    pub fs_khz: i32,
    pub nb_subfr: usize,
}

impl SilkChannelEncoder {
    /// A reset encoder for the given internal rate and subframe count.
    #[must_use]
    pub(crate) fn new(fs_khz: i32, nb_subfr: usize) -> Self {
        SilkChannelEncoder {
            nsq: NsqState::new(),
            shape: ShapeState::default(),
            last_gain_index: 10,
            sum_log_gain_q7: 0,
            ec_prev: EcPrevState::default(),
            fs_khz,
            nb_subfr,
        }
    }

    /// Encodes one unvoiced frame (short-term prediction only).
    pub(crate) fn encode_frame_unvoiced(
        &mut self,
        enc: &mut RangeEncoder,
        input: &[i16],
        cond_coding: CondCoding,
    ) -> SideInfoIndices {
        self.encode_frame(enc, input, cond_coding, None)
    }

    /// Encodes one voiced frame, taking the pitch lag/contour from `voiced`.
    pub(crate) fn encode_frame_voiced(
        &mut self,
        enc: &mut RangeEncoder,
        input: &[i16],
        cond_coding: CondCoding,
        voiced: VoicedParams,
    ) -> SideInfoIndices {
        self.encode_frame(enc, input, cond_coding, Some(voiced))
    }

    /// The shared encode pipeline. `voiced` selects the long-term-prediction
    /// path and supplies its pitch indices.
    fn encode_frame(
        &mut self,
        enc: &mut RangeEncoder,
        input: &[i16],
        cond_coding: CondCoding,
        voiced: Option<VoicedParams>,
    ) -> SideInfoIndices {
        let order = if self.fs_khz == 16 { 16 } else { 10 };
        let subfr_length = 5 * self.fs_khz as usize;
        let frame_length = self.nb_subfr * subfr_length;
        let ltp_mem_length = 20 * self.fs_khz as usize;
        debug_assert_eq!(input.len(), frame_length);
        let signal_type = if voiced.is_some() { TYPE_VOICED } else { TYPE_UNVOICED };

        // Short-term analysis: Burg LPC over the frame → NLSF → VQ-quantised
        // indices (with the requantised NLSF written back), then the Q12 LPC
        // the decoder rebuilds.
        let x_f: Vec<f32> = input.iter().map(|&v| f32::from(v)).collect();
        let mut lpc = [0.0f32; MAX_LPC_ORDER];
        burg_modified(&mut lpc[..order], &x_f, 1.0 / 1e4, frame_length, 1, order);

        let cb = nlsf_codebook(self.fs_khz);
        let mut nlsf_q15: Vec<i16> = a2nlsf(&lpc[..order]);
        let mut w_q2 = [0i16; MAX_LPC_ORDER];
        nlsf_vq_weights_laroia(&mut w_q2[..order], &nlsf_q15, order);
        let (nlsf_indices, _) = nlsf_encode(&mut nlsf_q15, cb, &w_q2[..order], 1 << 14, 4, signal_type as usize);

        let mut pred_coef = [0i16; 2 * MAX_LPC_ORDER];
        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        nlsf2a(&mut a_q12[..order], &nlsf_q15[..order]);
        pred_coef[..order].copy_from_slice(&a_q12[..order]);
        pred_coef[MAX_LPC_ORDER..MAX_LPC_ORDER + order].copy_from_slice(&a_q12[..order]);

        // LPC residual over the frame (stands in for the pitch-analysis
        // residual `res_pitch`, used by the sparseness measure and LTP).
        let mut residual = vec![0i16; frame_length];
        lpc_analysis_filter(&mut residual, input, &a_q12[..order]);
        let res_f: Vec<f32> = residual.iter().map(|&r| f32::from(r)).collect();

        // Pitch lags from the supplied indices (round-tripped via the
        // decoder's own `decode_pitch`, so encoder and decoder agree).
        let pitch_l = match voiced {
            Some(v) => decode_pitch(v.lag_index, v.contour_index, self.fs_khz, self.nb_subfr),
            None => [0; MAX_NB_SUBFR],
        };

        // Long-term prediction: correlation + gain VQ (voiced only).
        let mut ltp_coef = [0i16; LTP_ORDER * MAX_NB_SUBFR];
        let mut ltp_index = [0i8; MAX_NB_SUBFR];
        let mut per_index = 0i8;
        let mut pred_gain_db = 0.0f32;
        if voiced.is_some() {
            // Residual buffer with `ltp_mem_length` of (zero) history so the
            // analysis can reach a full pitch period back, plus `LTP_ORDER`
            // trailing samples for the last subframe's energy window.
            let mut r = vec![0.0f32; ltp_mem_length + frame_length + LTP_ORDER];
            r[ltp_mem_length..ltp_mem_length + frame_length].copy_from_slice(&res_f);
            let mut xx = vec![0.0f32; self.nb_subfr * LTP_ORDER * LTP_ORDER];
            let mut x_x = vec![0.0f32; self.nb_subfr * LTP_ORDER];
            find_ltp(
                &r,
                ltp_mem_length,
                &pitch_l,
                subfr_length,
                self.nb_subfr,
                &mut xx,
                &mut x_x,
            );
            let g = quant_ltp_gains(&xx, &x_x, subfr_length as i32, self.nb_subfr, &mut self.sum_log_gain_q7);
            ltp_coef = g.b_q14;
            ltp_index[..self.nb_subfr].copy_from_slice(&g.cbk_index[..self.nb_subfr]);
            per_index = g.periodicity_index;
            pred_gain_db = g.pred_gain_db;
        }

        // Noise-shaping analysis (complexity-0 configuration: no warping, so
        // the plain NSQ sees ordinary shaping coefficients).
        let la_shape = 3 * self.fs_khz as usize;
        let shaping_lpc_order = 12.min(order);
        let snr_db_q7 = 18 << 7;
        let mut x_buf = vec![0.0f32; frame_length + 2 * la_shape];
        for (i, &v) in input.iter().enumerate() {
            x_buf[la_shape + i] = f32::from(v);
        }
        let shape_cfg = NoiseShapeConfig {
            fs_khz: self.fs_khz,
            nb_subfr: self.nb_subfr,
            subfr_length,
            la_shape,
            shape_win_length: subfr_length + 2 * la_shape,
            shaping_lpc_order,
            warping_q16: 0,
            signal_type,
            snr_db_q7,
            speech_activity_q8: 256,
            input_quality_bands_q15: [32768, 32768],
            use_cbr: true,
            ltp_corr: voiced.map_or(0.0, |v| v.ltp_corr),
            pred_gain: 0.0,
            input_tilt_q15: 0,
            pitch_l,
        };
        let shp = noise_shape_analysis(&mut self.shape, &shape_cfg, &res_f, &x_buf);

        // Residual energy on the gain-normalised signal (`silk_residual_
        // energy_FLP`): ResNrg[k] = Gains[k]^2 * energy(LPC residual of x/Gain).
        let a_f: Vec<f32> = a_q12[..order].iter().map(|&c| f32::from(c) / 4096.0).collect();
        let mut x_hist = vec![0.0f32; order + frame_length];
        for (i, &v) in input.iter().enumerate() {
            x_hist[order + i] = f32::from(v);
        }
        let mut gains = shp.gains;
        let mut res_nrg = [0.0f32; MAX_NB_SUBFR];
        for k in 0..self.nb_subfr {
            let inv_gain = 1.0 / gains[k];
            let base = k * subfr_length;
            let mut nrg = 0.0f64;
            for n in 0..subfr_length {
                let p = base + order + n;
                let mut acc = x_hist[p] * inv_gain;
                for (j, &aj) in a_f.iter().enumerate() {
                    acc -= aj * x_hist[p - 1 - j] * inv_gain;
                }
                nrg += f64::from(acc) * f64::from(acc);
            }
            res_nrg[k] = (f64::from(gains[k]) * f64::from(gains[k]) * nrg) as f32;
        }

        let gres = process_gains(
            &mut gains,
            &res_nrg,
            signal_type,
            shp.quant_offset_type,
            self.nb_subfr,
            subfr_length,
            snr_db_q7,
            pred_gain_db,
            0,
            1,
            256,
            shp.input_quality,
            shp.coding_quality,
            &mut self.last_gain_index,
            cond_coding,
        );

        // LTP scaling: independent coding with no packet loss selects index 0.
        let ltp_scale_index = 0i8;
        let ltp_scale_q14 = if voiced.is_some() {
            i32::from(LTPSCALES_TABLE_Q14[ltp_scale_index as usize])
        } else {
            0
        };
        let seed = 0i32;

        let cfg = NsqConfig {
            frame_length,
            subfr_length,
            nb_subfr: self.nb_subfr,
            ltp_mem_length,
            predict_lpc_order: order,
            shaping_lpc_order,
        };
        let mut pulses = vec![0i8; frame_length];
        let lambda_q10 = (gres.lambda * 1024.0) as i32;
        nsq(
            &mut self.nsq,
            &cfg,
            signal_type,
            gres.quant_offset_type,
            4,
            seed,
            input,
            &mut pulses,
            &pred_coef,
            &ltp_coef,
            &shp.ar_q13,
            &shp.harm_shape_gain_q14,
            &shp.tilt_q14,
            &shp.lf_shp_q14,
            &gres.gains_q16,
            &pitch_l,
            lambda_q10,
            ltp_scale_q14,
        );

        // Assemble the side info and write the frame.
        let mut indices = SideInfoIndices {
            signal_type: signal_type as i8,
            quant_offset_type: gres.quant_offset_type as i8,
            nlsf_interp_coef_q2: 4,
            seed: seed as i8,
            ..SideInfoIndices::default()
        };
        indices.gains_indices[..self.nb_subfr].copy_from_slice(&gres.gains_indices[..self.nb_subfr]);
        indices.nlsf_indices[..=order].copy_from_slice(&nlsf_indices[..=order]);
        if let Some(v) = voiced {
            indices.lag_index = v.lag_index;
            indices.contour_index = v.contour_index;
            indices.per_index = per_index;
            indices.ltp_index = ltp_index;
            indices.ltp_scale_index = ltp_scale_index;
        }

        encode_indices(
            enc,
            &indices,
            self.fs_khz,
            self.nb_subfr,
            false,
            true,
            cond_coding,
            &mut self.ec_prev,
        );
        encode_pulses(enc, signal_type, gres.quant_offset_type, &pulses, frame_length);
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::decoder::SilkChannelDecoder;
    use super::*;
    use crate::range::RangeDecoder;

    /// Encodes `input`, captures the encoder's NSQ reconstruction, decodes the
    /// coded frame, and returns (decoder xq, encoder xq, input-correlation).
    fn round_trip(
        e: &mut SilkChannelEncoder,
        input: &[i16],
        voiced: Option<VoicedParams>,
    ) -> (Vec<i16>, Vec<i16>, f64) {
        let fs_khz = e.fs_khz;
        let nb_subfr = e.nb_subfr;
        let frame_length = nb_subfr * 5 * fs_khz as usize;
        let ltp_mem = 20 * fs_khz as usize;

        let mut enc = RangeEncoder::new(512);
        match voiced {
            Some(v) => {
                e.encode_frame_voiced(&mut enc, input, CondCoding::Independently, v);
            },
            None => {
                e.encode_frame_unvoiced(&mut enc, input, CondCoding::Independently);
            },
        }
        let xq_enc: Vec<i16> = e.nsq.xq[ltp_mem..ltp_mem + frame_length].to_vec();
        let bytes = enc.finalize().expect("frame fits");
        assert!(!bytes.is_empty());

        let mut d = SilkChannelDecoder::new(fs_khz, nb_subfr);
        let mut dec = RangeDecoder::new(&bytes);
        let mut xq = vec![0i16; frame_length];
        d.decode_frame(&mut dec, &mut xq, true, false, CondCoding::Independently);

        let (mut sig, mut dot, mut e_out) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..frame_length {
            let a = f64::from(input[i]);
            let b = f64::from(xq[i]);
            sig += a * a;
            dot += a * b;
            e_out += b * b;
        }
        let corr = dot / (sig.sqrt() * e_out.sqrt()).max(1.0);
        (xq, xq_enc, corr)
    }

    /// An unvoiced (noise+tone) frame decodes to the encoder's own NSQ
    /// reconstruction and tracks the input.
    #[test]
    fn unvoiced_frame_round_trips_through_the_decoder() {
        let (fs_khz, nb_subfr) = (16i32, 4usize);
        let frame_length = nb_subfr * 5 * fs_khz as usize;
        let mut seed = 0x9e37_u32;
        let input: Vec<i16> = (0..frame_length)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let n = ((seed >> 16) as i32 - 32768) / 12;
                let tone = ((i as f32 * 0.11).sin() * 2500.0) as i32;
                (n + tone).clamp(-30000, 30000) as i16
            })
            .collect();

        let mut e = SilkChannelEncoder::new(fs_khz, nb_subfr);
        let (xq, xq_enc, corr) = round_trip(&mut e, &input, None);
        assert_eq!(xq, xq_enc, "decoder disagrees with the encoder's NSQ reconstruction");
        assert!(corr > 0.7, "reconstruction correlation {corr:.3} too low");
    }

    /// A strongly periodic (voiced) frame round-trips through the decoder with
    /// the long-term predictor engaged.
    #[test]
    fn voiced_frame_round_trips_through_the_decoder() {
        let (fs_khz, nb_subfr) = (16i32, 4usize);
        let frame_length = nb_subfr * 5 * fs_khz as usize;
        // Pitch period ≈ 80 samples (200 Hz at 16 kHz): lag_index = 80 - min_lag.
        let min_lag = 2 * fs_khz;
        let period = 80i32;
        let voiced = VoicedParams {
            lag_index: (period - min_lag) as i16,
            contour_index: 0,
            ltp_corr: 0.8,
        };

        // A periodic glottal-like pulse train shaped by a soft formant.
        let mut seed = 0x51ed_u32;
        let input: Vec<i16> = (0..frame_length)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let noise = ((seed >> 18) as i32 - 8192) as f32 * 0.02;
                let phase = (i as i32 % period) as f32 / period as f32;
                let pulse = (-((phase - 0.0).powi(2)) * 30.0).exp() * 9000.0;
                let formant = (i as f32 * 0.30).sin() * 1500.0;
                (pulse + formant + noise * 100.0).clamp(-30000.0, 30000.0) as i16
            })
            .collect();

        let mut e = SilkChannelEncoder::new(fs_khz, nb_subfr);
        let (xq, xq_enc, corr) = round_trip(&mut e, &input, Some(voiced));
        assert_eq!(
            xq, xq_enc,
            "decoder disagrees with the encoder's NSQ reconstruction (voiced)"
        );
        assert!(corr > 0.5, "voiced reconstruction correlation {corr:.3} too low");
    }
}

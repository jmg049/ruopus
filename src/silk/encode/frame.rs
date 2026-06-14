//! Per-frame SILK encode assembly (RFC 6716 §5.2; normative
//! `silk/float/encode_frame_FLP.c`).
//!
//! [`encode_frame_unvoiced`] is the first end-to-end SILK encode path: it
//! ties together the analysis and quantisation kernels into a single coded
//! frame and is validated by round-tripping through the bit-exact decoder.
//! It targets the simplest legal configuration - unvoiced, no NLSF
//! interpolation, flat noise shaping - which exercises the whole chain
//! (LPC → NLSF VQ → gains → NSQ → index/pulse bitstream) and produces a
//! frame the decoder reconstructs sample-for-sample. Voiced coding, real
//! noise shaping, and the higher-level mode/stereo glue build on top.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::range::RangeEncoder;

use super::super::indices::{
    CondCoding, EcPrevState, MAX_LPC_ORDER, MAX_NB_SUBFR, SideInfoIndices, TYPE_UNVOICED, encode_indices, nlsf_codebook,
};
use super::super::lpc::lpc_analysis_filter;
use super::super::nlsf::nlsf2a;
use super::super::pulses::encode_pulses;
use super::gains::process_gains;
use super::lpc::burg_modified;
use super::nlsf::{a2nlsf, nlsf_encode, nlsf_vq_weights_laroia};
use super::nsq::{NsqConfig, NsqState, nsq};

/// One channel's SILK encoder state for the (unvoiced) frame path.
pub(crate) struct SilkChannelEncoder {
    pub nsq: NsqState,
    /// Gain-quantiser accumulator (`sShape.LastGainIndex`).
    pub last_gain_index: i8,
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
            last_gain_index: 10,
            ec_prev: EcPrevState::default(),
            fs_khz,
            nb_subfr,
        }
    }

    /// Encodes one unvoiced frame of `input` (i16 PCM at the internal rate,
    /// `frame_length` samples) into `enc`. Returns the coded `SideInfoIndices`
    /// (for inspection/round-trip checks).
    pub(crate) fn encode_frame_unvoiced(
        &mut self,
        enc: &mut RangeEncoder,
        input: &[i16],
        cond_coding: CondCoding,
    ) -> SideInfoIndices {
        let order = if self.fs_khz == 16 { 16 } else { 10 };
        let subfr_length = 5 * self.fs_khz as usize;
        let frame_length = self.nb_subfr * subfr_length;
        debug_assert_eq!(input.len(), frame_length);

        // Short-term analysis: Burg LPC over the frame.
        let x_f: Vec<f32> = input.iter().map(|&v| f32::from(v)).collect();
        let mut lpc = [0.0f32; MAX_LPC_ORDER];
        burg_modified(&mut lpc[..order], &x_f, 1.0 / 1e4, frame_length, 1, order);

        // LPC → NLSF → VQ-quantised indices (req. NLSF written back).
        let cb = nlsf_codebook(self.fs_khz);
        let mut nlsf_q15: Vec<i16> = a2nlsf(&lpc[..order]);
        let mut w_q2 = [0i16; MAX_LPC_ORDER];
        nlsf_vq_weights_laroia(&mut w_q2[..order], &nlsf_q15, order);
        let (nlsf_indices, _) = nlsf_encode(&mut nlsf_q15, cb, &w_q2[..order], 1 << 14, 4, TYPE_UNVOICED as usize);

        // Quantised LPC (Q12) for NSQ - exactly what the decoder rebuilds.
        let mut pred_coef = [0i16; 2 * MAX_LPC_ORDER];
        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        nlsf2a(&mut a_q12[..order], &nlsf_q15[..order]);
        pred_coef[..order].copy_from_slice(&a_q12[..order]);
        pred_coef[MAX_LPC_ORDER..MAX_LPC_ORDER + order].copy_from_slice(&a_q12[..order]);

        // Per-subframe gains from the LPC residual energy.
        let mut residual = vec![0i16; frame_length];
        lpc_analysis_filter(&mut residual, input, &a_q12[..order]);
        let mut gains = [0.0f32; MAX_NB_SUBFR];
        let mut res_nrg = [0.0f32; MAX_NB_SUBFR];
        for k in 0..self.nb_subfr {
            let nrg: f64 = residual[k * subfr_length..(k + 1) * subfr_length]
                .iter()
                .map(|&r| f64::from(r) * f64::from(r))
                .sum();
            res_nrg[k] = nrg as f32;
            gains[k] = ((nrg / subfr_length as f64).sqrt() as f32).max(4.0);
        }
        let gres = process_gains(
            &mut gains,
            &res_nrg,
            TYPE_UNVOICED,
            0,
            self.nb_subfr,
            subfr_length,
            18 << 7,
            0.0,
            0,
            0,
            0,
            0.5,
            0.5,
            &mut self.last_gain_index,
            cond_coding,
        );

        // Flat noise shaping (the decoder ignores it; NSQ degenerates to a
        // scalar quantiser with LPC prediction).
        let ar_q13 = [0i16; MAX_NB_SUBFR * 24];
        let ltp_coef = [0i16; 5 * MAX_NB_SUBFR];
        let zeros = [0i32; MAX_NB_SUBFR];
        let pitch_l = [0i32; MAX_NB_SUBFR];
        let seed = 0i32;

        let cfg = NsqConfig {
            frame_length,
            subfr_length,
            nb_subfr: self.nb_subfr,
            ltp_mem_length: 20 * self.fs_khz as usize,
            predict_lpc_order: order,
            shaping_lpc_order: 16,
        };
        let mut pulses = vec![0i8; frame_length];
        let lambda_q10 = (gres.lambda * 1024.0) as i32;
        nsq(
            &mut self.nsq,
            &cfg,
            TYPE_UNVOICED,
            gres.quant_offset_type,
            4,
            seed,
            input,
            &mut pulses,
            &pred_coef,
            &ltp_coef,
            &ar_q13,
            &zeros,
            &zeros,
            &zeros,
            &gres.gains_q16,
            &pitch_l,
            lambda_q10,
            0,
        );

        // Assemble the side info and write the frame.
        let mut indices = SideInfoIndices {
            signal_type: TYPE_UNVOICED as i8,
            quant_offset_type: gres.quant_offset_type as i8,
            nlsf_interp_coef_q2: 4,
            seed: seed as i8,
            ..SideInfoIndices::default()
        };
        indices.gains_indices[..self.nb_subfr].copy_from_slice(&gres.gains_indices[..self.nb_subfr]);
        indices.nlsf_indices[..=order].copy_from_slice(&nlsf_indices[..=order]);

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
        encode_pulses(enc, TYPE_UNVOICED, gres.quant_offset_type, &pulses, frame_length);
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::decoder::SilkChannelDecoder;
    use super::*;
    use crate::range::RangeDecoder;

    /// End-to-end: an unvoiced SILK frame encoded by our encoder decodes
    /// through the bit-exact decoder, reconstructing the encoder's own
    /// quantised signal and tracking the input.
    #[test]
    fn unvoiced_frame_round_trips_through_the_decoder() {
        let fs_khz = 16i32;
        let nb_subfr = 4usize;
        let subfr = 5 * fs_khz as usize;
        let frame_length = nb_subfr * subfr;

        // A noise-like (unvoiced) input.
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
        let mut enc = RangeEncoder::new(512);
        let _ind = e.encode_frame_unvoiced(&mut enc, &input, CondCoding::Independently);
        // The encoder's own reconstruction of this frame.
        let xq_enc: Vec<i16> = e.nsq.xq[20 * fs_khz as usize..20 * fs_khz as usize + frame_length].to_vec();
        let bytes = enc.finalize().expect("frame fits");
        assert!(!bytes.is_empty());

        // Decode it back.
        let mut d = SilkChannelDecoder::new(fs_khz, nb_subfr);
        let mut dec = RangeDecoder::new(&bytes);
        let mut xq = vec![0i16; frame_length];
        d.decode_frame(&mut dec, &mut xq, true, false, CondCoding::Independently);

        // The decoder reproduces the encoder's quantised signal.
        assert_eq!(
            xq, xq_enc,
            "decoder output disagrees with the encoder's NSQ reconstruction"
        );
        // And it tracks the input (lossy, but correlated and bounded).
        let (mut sig, mut dot, mut e_out) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..frame_length {
            let a = f64::from(input[i]);
            let b = f64::from(xq[i]);
            sig += a * a;
            dot += a * b;
            e_out += b * b;
        }
        let corr = dot / (sig.sqrt() * e_out.sqrt()).max(1.0);
        assert!(corr > 0.7, "reconstruction correlation {corr:.3} too low");
    }
}

//! Per-frame SILK encode assembly (RFC 6716 §5.2).
//!
//! [`SilkChannelEncoder::encode_frame`] is the end-to-end SILK encode path:
//! it ties the analysis and quantisation kernels into a single coded frame
//! and is validated by round-tripping through the bit-exact decoder. It
//! decides voiced/unvoiced itself (via the pitch analysis) and handles both
//! paths; the remaining work is the higher-level rate control, the resampler
//! front-end, stereo, and the public Opus mode glue.
//!
//! The chain is: pitch analysis (whitening + lag search, deciding voicing) →
//! Burg LPC → NLSF VQ → `nlsf2a` requantise → (voiced) LTP correlation +
//! gain VQ → noise-shaping analysis → gain quantisation → NSQ → index/pulse
//! bitstream. The decoder rebuilds the prediction coefficients and gains
//! from the coded indices, so feeding NSQ those same quantised values makes
//! its reconstruction equal the decoder's output. Cross-frame state (NSQ
//! history, pitch lag, input history, shaping smoothers) is carried in the
//! encoder so consecutive frames stay in sync with the decoder.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::range::RangeEncoder;

use super::super::gains::{gains_dequant, gains_quant};
use super::super::indices::{
    CondCoding, EcPrevState, MAX_LPC_ORDER, MAX_NB_SUBFR, SideInfoIndices, TYPE_UNVOICED, TYPE_VOICED, encode_indices,
    nlsf_codebook,
};
use super::super::math::smulwb;
use super::super::nlsf::nlsf2a;
use super::super::params::LTP_ORDER;
use super::super::pulses::encode_pulses;
use super::super::tables::LTPSCALES_TABLE_Q14;
use super::control::control_snr;
use super::dsp::{energy, lpc_analysis_filter_flp};
use super::gains::process_gains;
use super::lpc::burg_modified;
use super::ltp::{find_ltp, quant_ltp_gains};
use super::nlsf::{a2nlsf, nlsf_encode, nlsf_vq_weights_laroia};
use super::noise_shape::{NoiseShapeConfig, ShapeState, noise_shape_analysis};
use super::nsq::{NsqConfig, NsqState, nsq};
use super::pitch_analysis::find_pitch_lags;
use super::vad::VadState;

/// One channel's SILK encoder state.
#[derive(Clone)]
pub(crate) struct SilkChannelEncoder {
    pub nsq: NsqState,
    /// Cross-frame noise-shaping smoothing state (`sShape`).
    pub shape: ShapeState,
    /// Gain-quantiser accumulator (`sShape.LastGainIndex`).
    pub last_gain_index: i8,
    /// Cumulative LTP max-gain accumulator (`sum_log_gain_Q7`).
    pub sum_log_gain_q7: i32,
    /// Previous frame's final pitch lag (`prevLag`, 0 if unvoiced).
    pub prev_lag: i32,
    /// Previous frame's signal type (`prevSignalType`).
    pub prev_signal_type: i32,
    /// Normalised long-term correlation carried across frames (`LTPCorr`).
    pub ltp_corr: f32,
    /// The previous `ltp_mem_length` input samples, used as pitch-analysis
    /// history for the next frame.
    pub prev_input: Vec<i16>,
    /// Voice-activity detector (noise-floor estimation) state.
    pub vad: VadState,
    /// Target bitrate (bps), mapped to the coding SNR per frame.
    pub target_rate_bps: i32,
    /// Expected packet-loss percentage 0-100 (`PacketLoss_perc`). When > 0, an
    /// independently-coded voiced frame raises its LTP scaling index to reduce
    /// inter-frame prediction dependency (loss-robust coding).
    pub packet_loss_perc: i32,
    /// Number of SILK frames per Opus packet (`nFramesPerPacket`: 1 for
    /// 10/20 ms, 2 for 40 ms, 3 for 60 ms), used by the LTP scaling selection.
    pub n_frames_per_packet: i32,
    /// Gain increase applied to the LBRR copy's first-subframe gain index
    /// (`LBRR_GainIncreases`), so the redundant FEC copy costs fewer bits. 0
    /// disables the reduced-rate LBRR pass (the copy then reuses the full-rate
    /// pulses).
    pub lbrr_gain_increases: i32,
    /// Cross-frame LBRR gain-index accumulator (`LBRRprevLastGainIndex`), reset
    /// at the first LBRR frame of a packet and carried across conditionally
    /// coded LBRR frames.
    pub lbrr_prev_last_gain_index: i8,
    /// True until the first frame after a reset has been coded; relaxes the
    /// maximum-prediction-gain cap (`first_frame_after_reset`).
    pub first_frame_after_reset: bool,
    /// Entropy-coding history for [`encode_indices`].
    pub ec_prev: EcPrevState,
    pub fs_khz: i32,
    pub nb_subfr: usize,
    /// Encode complexity 0-10; selects the pitch-search depth/threshold/order
    /// (the rest of the analysis is already its low-complexity configuration:
    /// plain NSQ, no warping).
    pub complexity: u8,
    /// Per-frame work buffers, reused across frames to avoid reallocating (and
    /// re-zeroing) them every call: pitch-analysis input/residual, the aligned
    /// residual, the noise-shape input, the LPC-analysis input, and the pulse
    /// output. Fully overwritten each frame.
    scratch_pitch: Vec<f32>,
    scratch_res: Vec<f32>,
    scratch_res_f: Vec<f32>,
    scratch_x_buf: Vec<f32>,
    scratch_lpc_in: Vec<f32>,
    scratch_pulses: Vec<i8>,
    /// Reused rate-control snapshot (NSQ state + entropy history), so the
    /// gain-bisection fallback does not clone the NSQ `Vec`s every frame.
    snap_nsq: NsqState,
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
            prev_lag: 0,
            prev_signal_type: TYPE_UNVOICED,
            ltp_corr: 0.0,
            prev_input: vec![0; 20 * fs_khz as usize],
            vad: VadState::new(),
            target_rate_bps: 30_000,
            packet_loss_perc: 0,
            n_frames_per_packet: 1,
            lbrr_gain_increases: 0,
            lbrr_prev_last_gain_index: 0,
            first_frame_after_reset: true,
            ec_prev: EcPrevState::default(),
            fs_khz,
            nb_subfr,
            complexity: 10,
            scratch_pitch: Vec::new(),
            scratch_res: Vec::new(),
            scratch_res_f: Vec::new(),
            scratch_x_buf: Vec::new(),
            scratch_lpc_in: Vec::new(),
            scratch_pulses: Vec::new(),
            snap_nsq: NsqState::new(),
        }
    }

    /// Sets the target bitrate (bps), which maps to the coding SNR per frame.
    pub(crate) fn set_bitrate(&mut self, bps: i32) {
        self.target_rate_bps = bps;
    }

    /// Sets the expected packet-loss percentage 0-100 (`PacketLoss_perc`),
    /// clamped; > 0 enables loss-robust LTP scaling on independently coded
    /// voiced frames.
    pub(crate) fn set_packet_loss_perc(&mut self, perc: i32) {
        self.packet_loss_perc = perc.clamp(0, 100);
    }

    /// Sets the number of SILK frames per packet (`nFramesPerPacket`), used by
    /// the LTP scaling selection.
    pub(crate) fn set_n_frames_per_packet(&mut self, n: i32) {
        self.n_frames_per_packet = n.max(1);
    }

    /// Sets the LBRR first-subframe gain increase (`LBRR_GainIncreases`); 0
    /// disables the reduced-rate LBRR pass (the copy reuses the full-rate
    /// pulses).
    pub(crate) fn set_lbrr_gain_increases(&mut self, g: i32) {
        self.lbrr_gain_increases = g.max(0);
    }

    /// Sets the encode complexity 0-10 (clamped), selecting the pitch search.
    pub(crate) fn set_complexity(&mut self, complexity: u8) {
        self.complexity = complexity.min(10);
    }

    /// Resets the prediction memory for the first coded side frame after a
    /// mid-only stretch, mirroring the decoder's `reset_side_prediction`.
    pub(crate) fn reset_side_prediction(&mut self) {
        self.nsq = NsqState::new();
        self.nsq.lag_prev = 100;
        self.shape = ShapeState::default();
        self.last_gain_index = 10;
        self.sum_log_gain_q7 = 0;
        self.prev_lag = 0;
        self.prev_signal_type = TYPE_UNVOICED;
        self.ltp_corr = 0.0;
        self.first_frame_after_reset = true;
        for v in &mut self.prev_input {
            *v = 0;
        }
    }

    /// Encodes one frame of `input` (i16 PCM at the internal rate,
    /// `frame_length` samples), deciding voiced/unvoiced and (when voiced)
    /// the pitch lags itself via the pitch analysis. Returns the coded
    /// `SideInfoIndices`.
    pub(crate) fn encode_frame(
        &mut self,
        enc: &mut RangeEncoder,
        input: &[i16],
        cond_coding: CondCoding,
        max_bits: Option<i32>,
    ) -> SideInfoIndices {
        self.encode_frame_inner(enc, input, cond_coding, max_bits, false).0
    }

    /// Codes the regular frame into `enc` honouring the cumulative `max_bits`
    /// cap (so hybrid packets reserve the CELT high band its room) and captures
    /// the chosen `indices`/`pulses`, advancing the cross-frame analysis state
    /// (NSQ history, pitch/voicing, input history, gain accumulator, entropy
    /// history) and `ec_prev` exactly as a regular [`encode_frame`] would. This
    /// is the in-band FEC capture path: the captured frame is stored to re-emit
    /// as the *next* packet's LBRR copy via [`emit_frame`](Self::emit_frame).
    ///
    /// Returns `(indices, pulses)` for the regular frame and, when
    /// `lbrr_gain_increases > 0`, a reduced-rate LBRR copy `(indices, pulses)`
    /// produced by a second NSQ pass with a raised first-subframe gain index
    /// (`silk_LBRR_encode`). When the gain increase is 0 the LBRR copy is
    /// `None` and the caller reuses the regular frame as the LBRR copy.
    #[allow(clippy::type_complexity)]
    pub(crate) fn encode_frame_capture(
        &mut self,
        enc: &mut RangeEncoder,
        input: &[i16],
        cond_coding: CondCoding,
        max_bits: Option<i32>,
    ) -> ((SideInfoIndices, Vec<i8>), Option<(SideInfoIndices, Vec<i8>)>) {
        let want_lbrr = self.lbrr_gain_increases > 0;
        let (indices, pulses, lbrr) = self.encode_frame_inner(enc, input, cond_coding, max_bits, want_lbrr);
        ((indices, pulses), lbrr)
    }

    /// Emits a previously [`encode_frame_capture`](Self::encode_frame_capture)d
    /// frame into `enc` with the given conditional-coding mode, as the LBRR copy
    /// when
    /// `lbrr` is true (which forces the VAD-present type coding). Advances the
    /// entropy history (`ec_prev`) only - no analysis state.
    pub(crate) fn emit_frame(
        &mut self,
        enc: &mut RangeEncoder,
        indices: &SideInfoIndices,
        pulses: &[i8],
        cond_coding: CondCoding,
        lbrr: bool,
    ) {
        let frame_length = self.nb_subfr * 5 * self.fs_khz as usize;
        encode_indices(
            enc,
            indices,
            self.fs_khz,
            self.nb_subfr,
            lbrr,
            true,
            cond_coding,
            &mut self.ec_prev,
        );
        encode_pulses(
            enc,
            i32::from(indices.signal_type),
            i32::from(indices.quant_offset_type),
            pulses,
            frame_length,
        );
    }

    #[allow(clippy::type_complexity)]
    fn encode_frame_inner(
        &mut self,
        enc: &mut RangeEncoder,
        input: &[i16],
        cond_coding: CondCoding,
        max_bits: Option<i32>,
        want_lbrr: bool,
    ) -> (SideInfoIndices, Vec<i8>, Option<(SideInfoIndices, Vec<i8>)>) {
        let order = if self.fs_khz == 16 { 16 } else { 10 };
        let subfr_length = 5 * self.fs_khz as usize;
        let frame_length = self.nb_subfr * subfr_length;
        let ltp_mem_length = 20 * self.fs_khz as usize;
        let la_pitch = 2 * self.fs_khz as usize;
        debug_assert_eq!(input.len(), frame_length);

        // Voice-activity analysis: speech-activity, spectral tilt and per-band
        // input quality, which tune the pitch threshold, noise shaping and gains.
        let vad = {
            self.vad.get_sa_q8(input, frame_length, self.fs_khz)
        };

        // Pitch analysis: whiten and search for the lag. `pitch_x_buf` holds
        // `ltp_mem_length` of history, the frame, then `la_pitch` lookahead;
        // an isolated frame zero-pads the history and lookahead.
        // Pitch-search settings from the complexity: the
        // estimation complexity (0/1/2), the voicing threshold, and the
        // whitening LPC order (capped at the prediction order).
        let (pe_complexity, search_thres, pe_lpc_order): (usize, f32, usize) = match self.complexity {
            0 => (0, 0.8, 6),
            1 => (1, 0.76, 8),
            2 => (0, 0.8, 6),
            3 => (1, 0.76, 8),
            4 | 5 => (1, 0.74, 10),
            6 | 7 => (1, 0.72, 12),
            _ => (2, 0.7, 16),
        };
        let pe_order = pe_lpc_order.min(order);
        let buf_len = la_pitch + frame_length + ltp_mem_length;
        // Reuse the work buffers (no per-frame allocation). `pitch_x_buf` is
        // history + frame, with the trailing `la_pitch` lookahead zeroed; the
        // zipped writes carry no bounds checks.
        let mut pitch_x_buf = core::mem::take(&mut self.scratch_pitch);
        pitch_x_buf.resize(buf_len, 0.0);
        for (dst, &v) in pitch_x_buf[..ltp_mem_length].iter_mut().zip(self.prev_input.iter()) {
            *dst = f32::from(v);
        }
        for (dst, &v) in pitch_x_buf[ltp_mem_length..ltp_mem_length + frame_length]
            .iter_mut()
            .zip(input.iter())
        {
            *dst = f32::from(v);
        }
        pitch_x_buf[ltp_mem_length + frame_length..].fill(0.0);
        let mut res = core::mem::take(&mut self.scratch_res);
        res.clear();
        res.resize(buf_len, 0.0);
        let pl = find_pitch_lags(
            &pitch_x_buf,
            &mut res,
            self.fs_khz,
            self.nb_subfr,
            pe_order,
            pe_complexity,
            search_thres,
            self.prev_lag,
            self.prev_signal_type,
            vad.speech_activity_q8,
            vad.input_tilt_q15,
            &mut self.ltp_corr,
        );
        let is_voiced = pl.voicing == 0;
        let signal_type = if is_voiced { TYPE_VOICED } else { TYPE_UNVOICED };
        let pitch_l = pl.pitch_l;
        // The whitened residual aligned to the frame (`res_pitch_frame`).
        let mut res_f = core::mem::take(&mut self.scratch_res_f);
        res_f.clear();
        res_f.extend_from_slice(&res[ltp_mem_length..ltp_mem_length + frame_length]);

        // Noise-shaping analysis (complexity-0 configuration: no warping, so
        // the plain NSQ sees ordinary shaping coefficients). The reference runs
        // this before `find_pred_coefs` because it produces the per-subframe
        // gains the prediction analysis normalises by.
        let la_shape = 3 * self.fs_khz as usize;
        let shaping_lpc_order = 12.min(order);
        let snr_db_q7 = control_snr(self.fs_khz, self.nb_subfr, self.target_rate_bps);
        let mut x_buf = core::mem::take(&mut self.scratch_x_buf);
        x_buf.clear();
        x_buf.resize(frame_length + 2 * la_shape, 0.0);
        for (dst, &v) in x_buf[la_shape..la_shape + frame_length].iter_mut().zip(input.iter()) {
            *dst = f32::from(v);
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
            speech_activity_q8: vad.speech_activity_q8,
            input_quality_bands_q15: [vad.input_quality_bands_q15[0], vad.input_quality_bands_q15[1]],
            use_cbr: true,
            ltp_corr: self.ltp_corr,
            pred_gain: pl.pred_gain,
            pitch_l,
        };
        let shp = {
            noise_shape_analysis(&mut self.shape, &shape_cfg, &res_f, &x_buf)
        };

        // `silk_find_pred_coefs_FLP`: short- and long-term prediction analysis
        // on the gain-normalised input. `LPC_in_pre` holds, per subframe,
        // `order` history samples plus the subframe scaled by the inverse gain
        // (and, when voiced, whitened by the LTP first); the short-term LPC is
        // estimated from that signal rather than the raw frame.
        let mut inv_gains = [0.0f32; MAX_NB_SUBFR];
        for (k, ig) in inv_gains.iter_mut().enumerate().take(self.nb_subfr) {
            *ig = 1.0 / shp.gains[k];
        }
        let pre = order; // preceding samples per subframe (predictLPCOrder)
        let shift = subfr_length + pre;
        let mut lpc_in_pre = core::mem::take(&mut self.scratch_lpc_in);
        lpc_in_pre.clear();
        lpc_in_pre.resize(self.nb_subfr * shift, 0.0);

        let mut ltp_coef = [0i16; LTP_ORDER * MAX_NB_SUBFR];
        let mut ltp_index = [0i8; MAX_NB_SUBFR];
        let mut per_index = 0i8;
        let mut pred_gain_db = 0.0f32;
        if is_voiced {
            // LTP correlation + gain VQ on the whitened residual.
            let mut xx = vec![0.0f32; self.nb_subfr * LTP_ORDER * LTP_ORDER];
            let mut x_x = vec![0.0f32; self.nb_subfr * LTP_ORDER];
            find_ltp(
                &res,
                ltp_mem_length,
                &pitch_l,
                subfr_length,
                self.nb_subfr,
                &mut xx,
                &mut x_x,
            );
            let g = {
                quant_ltp_gains(&xx, &x_x, subfr_length as i32, self.nb_subfr, &mut self.sum_log_gain_q7)
            };
            ltp_coef = g.b_q14;
            ltp_index[..self.nb_subfr].copy_from_slice(&g.cbk_index[..self.nb_subfr]);
            per_index = g.periodicity_index;
            pred_gain_db = g.pred_gain_db;

            // LTP analysis filter: subtract the long-term prediction (lagged by
            // `pitch_l[k]`, taps centred at `LTP_ORDER/2`) and scale by the
            // inverse gain. `pitch_x_buf` holds the raw input with history.
            for k in 0..self.nb_subfr {
                let x_ptr = ltp_mem_length + k * subfr_length - pre;
                let inv = inv_gains[k];
                let btmp = &ltp_coef[k * LTP_ORDER..k * LTP_ORDER + LTP_ORDER];
                for i in 0..shift {
                    let xi = x_ptr + i;
                    let lag = xi - pitch_l[k] as usize;
                    let mut v = pitch_x_buf[xi];
                    for (j, &b) in btmp.iter().enumerate() {
                        v -= (f32::from(b) / 16384.0) * pitch_x_buf[lag + LTP_ORDER / 2 - j];
                    }
                    lpc_in_pre[k * shift + i] = v * inv;
                }
            }
        } else {
            // Unvoiced: gain-normalised input with `pre` history samples.
            for k in 0..self.nb_subfr {
                let x_ptr = ltp_mem_length + k * subfr_length - pre;
                let inv = inv_gains[k];
                for i in 0..shift {
                    lpc_in_pre[k * shift + i] = pitch_x_buf[x_ptr + i] * inv;
                }
            }
        }

        // Maximum prediction-gain cap (`minInvGain`): looser right after a
        // reset, otherwise scaled by the LTP coding gain and the coding quality
        // so a strong long-term predictor permits a sharper LPC.
        let min_inv_gain = if self.first_frame_after_reset {
            1.0 / 100.0 // 1 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET
        } else {
            let g = 2.0f32.powf(pred_gain_db / 3.0) / 1e4; // / MAX_PREDICTION_POWER_GAIN
            g / (0.25 + 0.75 * shp.coding_quality)
        };

        // Short-term analysis: Burg LPC over the per-subframe `LPC_in_pre`
        // blocks → NLSF → VQ-quantised indices (requantised NLSF written back),
        // then the Q12 LPC the decoder rebuilds.
        let mut lpc = [0.0f32; MAX_LPC_ORDER];
        {
            burg_modified(
                &mut lpc[..order],
                &lpc_in_pre,
                min_inv_gain,
                shift,
                self.nb_subfr,
                order,
            );
        }

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

        // Residual energy on `LPC_in_pre` with the quantised LPC
        // (`silk_residual_energy_FLP`): ResNrg[k] = Gains[k]^2 · energy(residual).
        let a_f: Vec<f32> = a_q12[..order].iter().map(|&c| f32::from(c) / 4096.0).collect();
        let mut gains = shp.gains;
        let mut res_nrg = [0.0f32; MAX_NB_SUBFR];
        let mut lpc_res = vec![0.0f32; shift];
        for k in 0..self.nb_subfr {
            let sub = &lpc_in_pre[k * shift..k * shift + shift];
            lpc_analysis_filter_flp(&mut lpc_res, &a_f, sub, shift, order);
            let nrg = energy(&lpc_res[order..shift]);
            res_nrg[k] = (f64::from(gains[k]) * f64::from(gains[k]) * nrg) as f32;
        }

        // `lastGainIndexPrev`: the gain-quantiser accumulator before this
        // frame, restored each iteration of the `max_bits` rate-control loop.
        let last_gain_prev = self.last_gain_index;
        let gres = process_gains(
            &mut gains,
            &res_nrg,
            signal_type,
            shp.quant_offset_type,
            self.nb_subfr,
            subfr_length,
            snr_db_q7,
            pred_gain_db,
            vad.input_tilt_q15,
            1,
            vad.speech_activity_q8,
            shp.input_quality,
            shp.coding_quality,
            &mut self.last_gain_index,
            cond_coding,
        );

        // LTP scaling (`silk_LTP_scale_ctrl`): the first (independently coded)
        // frame of a packet raises the LTP scaling index when packet loss is
        // expected, reducing inter-frame prediction dependency so a lost frame
        // damages fewer following frames. Conditionally coded frames always use
        // index 0 (minimum scaling). With `packet_loss_perc == 0` (the default)
        // we keep index 0 to stay byte-identical to the prior behaviour; the
        // reference's `round_loss = perc + nFramesPerPacket` term would raise it
        // even at zero loss, but that divergence is gated off here.
        let ltp_scale_index = if self.packet_loss_perc > 0
            && is_voiced
            && cond_coding == CondCoding::Independently
        {
            // round_loss = PacketLoss_perc + nFramesPerPacket;
            // LTP_scaleIndex = LIMIT(round_loss * LTPredCodGain * 0.1, 0, 2)
            // (truncated toward zero on assignment to int8). `pred_gain_db`
            // is the LTP prediction coding gain (`LTPredCodGain`).
            let round_loss = self.packet_loss_perc + self.n_frames_per_packet;
            let v = round_loss as f32 * pred_gain_db * 0.1;
            v.clamp(0.0, 2.0) as i8
        } else {
            0i8
        };
        let ltp_scale_q14 = if is_voiced {
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
        // Lambda (RD trade-off) and the quantiser offset can be adjusted by the
        // rate-control loop when a frame is stuck above the cap.
        let mut lambda_q10 = (gres.lambda * 1024.0) as i32;
        let mut quant_offset_type = gres.quant_offset_type;

        // Assemble the side info; `gains_indices` and `quant_offset_type` are
        // refreshed per rate-control iteration below.
        let mut indices = SideInfoIndices {
            signal_type: signal_type as i8,
            quant_offset_type: quant_offset_type as i8,
            nlsf_interp_coef_q2: 4,
            seed: seed as i8,
            ..SideInfoIndices::default()
        };
        indices.nlsf_indices[..=order].copy_from_slice(&nlsf_indices[..=order]);
        if is_voiced {
            indices.lag_index = pl.lag_index;
            indices.contour_index = pl.contour_index;
            indices.per_index = per_index;
            indices.ltp_index = ltp_index;
            indices.ltp_scale_index = ltp_scale_index;
        }
        indices.gains_indices[..self.nb_subfr].copy_from_slice(&gres.gains_indices[..self.nb_subfr]);

        // LBRR (FEC redundant copy), `silk_LBRR_encode`: encode the excitation
        // at a lower bitrate by raising the first-subframe gain index by
        // `lbrr_gain_increases` (first LBRR frame of the packet only), then
        // re-running NSQ on the *pre-frame* NSQ state with the re-dequantised,
        // coarser gains. This runs before the regular NSQ loop so it sees the
        // same pre-frame NSQ history the reference does. The LBRR copy is only
        // decoded on packet loss (`decode_fec`); the regular frame below is
        // unaffected.
        let lbrr_out = if want_lbrr {
            let mut lbrr_indices = indices.clone();
            // First LBRR frame in the packet (independent coding) resets the
            // accumulator and applies the gain bump; conditionally coded LBRR
            // frames carry the accumulator and skip the bump.
            if cond_coding == CondCoding::Independently {
                self.lbrr_prev_last_gain_index = self.last_gain_index;
                let bumped = i32::from(lbrr_indices.gains_indices[0]) + self.lbrr_gain_increases;
                lbrr_indices.gains_indices[0] = bumped.min(63) as i8; // N_LEVELS_QGAIN - 1
            }
            // Re-dequantise the (bumped) gain indices to keep the encoder's
            // gains in sync with what `decode_fec` will reconstruct.
            let lbrr_gains_q16 = gains_dequant(
                &lbrr_indices.gains_indices,
                &mut self.lbrr_prev_last_gain_index,
                cond_coding == CondCoding::Conditionally,
                self.nb_subfr,
            );
            // Second NSQ pass on a clone of the pre-frame NSQ state, so the
            // regular NSQ loop below still starts from the untouched state.
            let mut lbrr_nsq = self.nsq.clone();
            let mut lbrr_pulses = vec![0i8; frame_length];
            nsq(
                &mut lbrr_nsq,
                &cfg,
                signal_type,
                quant_offset_type,
                4,
                seed,
                input,
                &mut lbrr_pulses,
                &pred_coef,
                &ltp_coef,
                &shp.ar_q13,
                &shp.harm_shape_gain_q14,
                &shp.tilt_q14,
                &shp.lf_shp_q14,
                &lbrr_gains_q16,
                &pitch_l,
                lambda_q10,
                ltp_scale_q14,
            );
            Some((lbrr_indices, lbrr_pulses))
        } else {
            None
        };

        // Carry forward the pitch/voicing state and input history.
        self.prev_lag = if is_voiced { pitch_l[self.nb_subfr - 1] } else { 0 };
        self.prev_signal_type = signal_type;
        self.first_frame_after_reset = false;
        if frame_length >= ltp_mem_length {
            self.prev_input.copy_from_slice(&input[frame_length - ltp_mem_length..]);
        } else {
            self.prev_input.copy_within(frame_length.., 0);
            self.prev_input[ltp_mem_length - frame_length..].copy_from_slice(input);
        }

        // NSQ + entropy coding. With `max_bits` set (hybrid), this is the
        // per-frame rate-control loop: the first
        // attempt uses the gains from `process_gains`; if it busts the cap, the
        // unquantised gains are scaled coarser by a multiplier - geometrically
        // until the over/under budget is bracketed, then bisection-interpolated
        // toward the cap - re-running NSQ and re-coding from a snapshot each
        // time. The best fitting attempt is restored at the end.
        //
        // When a frame stays stuck above the cap the loop also raises the RD
        // lambda + drops the dither and locks each subframe to its sparsest
        // multiplier, both from the reference. Capped at the reference's 1024
        // (4×) ceiling over 6 iterations; a rare loud transient that still busts
        // at 4× is left over budget and the caller (encode_auto) falls back to
        // CELT, which codes it far better than extreme-gain SILK. The reference's
        // zero-pulse damage control (which would desync our decoder without a
        // synthesis resync) is the one remaining refinement.
        let mut pulses = core::mem::take(&mut self.scratch_pulses);
        pulses.clear();
        pulses.resize(frame_length, 0);
        let mut gains_q16 = gres.gains_q16;
        let mut gains_indices = gres.gains_indices;
        // Snapshot for the gain-bisection fallback. The bulky NSQ state is
        // copied into the reused `snap_nsq` (no per-frame allocation); only the
        // small range-coder state is cloned.
        let snap_enc = max_bits.map(|_| {
            self.snap_nsq.clone_from(&self.nsq);
            (enc.clone(), self.ec_prev)
        });
        let bits_margin = max_bits.map_or(0, |m| m / 4); // VBR: within 25% is close enough
        let mut best_fit: Option<(RangeEncoder, NsqState, EcPrevState, i8, [i8; MAX_NB_SUBFR])> = None;
        let mut gain_mult_q8 = 256i32;
        let (mut found_lower, mut found_upper) = (false, false);
        let (mut gm_lower, mut gm_upper) = (0i32, 0i32);
        let (mut nb_lower, mut nb_upper) = (0i32, 0i32);
        let mut gain_lock = [false; MAX_NB_SUBFR];
        let mut best_gain_mult = [256i32; MAX_NB_SUBFR];
        let mut best_sum = [0i32; MAX_NB_SUBFR];
        let mut iter = 0;
        loop {
            if iter > 0 {
                // Restore and re-quantise the gains scaled by the multiplier.
                let (enc0, ec0) = snap_enc.as_ref().expect("snapshot present when capping");
                enc.clone_from(enc0);
                self.nsq.clone_from(&self.snap_nsq);
                self.ec_prev = *ec0;
                self.last_gain_index = last_gain_prev;
                let mut pg = [0i32; MAX_NB_SUBFR];
                for (k, pg_k) in pg.iter_mut().enumerate().take(self.nb_subfr) {
                    // pGains_Q16 = LSHIFT_SAT32(SMULWB(GainsUnq_Q16, gainMult_Q8), 8),
                    // using each subframe's locked multiplier when it has one.
                    let gm = if gain_lock[k] { best_gain_mult[k] } else { gain_mult_q8 };
                    let v = i64::from(smulwb(gres.gains_unq_q16[k], gm)) << 8;
                    *pg_k = v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                }
                let mut gi = [0i8; MAX_NB_SUBFR];
                gains_quant(
                    &mut gi,
                    &mut pg,
                    &mut self.last_gain_index,
                    cond_coding == CondCoding::Conditionally,
                    self.nb_subfr,
                );
                gains_q16 = pg;
                gains_indices = gi;
            }

            nsq(
                &mut self.nsq,
                &cfg,
                signal_type,
                quant_offset_type,
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
                &gains_q16,
                &pitch_l,
                lambda_q10,
                ltp_scale_q14,
            );
            indices.gains_indices[..self.nb_subfr].copy_from_slice(&gains_indices[..self.nb_subfr]);
            indices.quant_offset_type = quant_offset_type as i8;
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
            encode_pulses(enc, signal_type, quant_offset_type, &pulses, frame_length);

            let Some(max_bits) = max_bits else { break };
            let n_bits = enc.tell() as i32;
            // VBR: the first attempt at the target gains is accepted if it fits.
            if iter == 0 && n_bits <= max_bits {
                break;
            }
            if n_bits > max_bits {
                if !found_lower && iter >= 2 {
                    // Stuck above the cap: trade more distortion for rate (raise
                    // lambda) and drop the dither (quantOffsetType = 0), then
                    // discard the stale upper bracket.
                    lambda_q10 = (lambda_q10 * 3 / 2).max(1536);
                    quant_offset_type = 0;
                    found_upper = false;
                } else {
                    found_upper = true;
                    nb_upper = n_bits;
                    gm_upper = gain_mult_q8;
                }
            } else if n_bits < max_bits - bits_margin {
                best_fit = Some((
                    enc.clone(),
                    self.nsq.clone(),
                    self.ec_prev,
                    self.last_gain_index,
                    indices.gains_indices,
                ));
                found_lower = true;
                nb_lower = n_bits;
                gm_lower = gain_mult_q8;
            } else {
                // Close enough to the cap - accept this attempt.
                break;
            }

            // Per-subframe gain lock: while still over budget without a lower
            // bracket, remember the multiplier that gave each subframe its
            // sparsest pulses, and lock subframes that stop improving so the
            // global multiplier does not over-coarsen them.
            if !found_lower && n_bits > max_bits {
                for k in 0..self.nb_subfr {
                    let sum: i32 = pulses[k * subfr_length..(k + 1) * subfr_length]
                        .iter()
                        .map(|&p| i32::from(p).abs())
                        .sum();
                    if iter == 0 || (sum < best_sum[k] && !gain_lock[k]) {
                        best_sum[k] = sum;
                        best_gain_mult[k] = gain_mult_q8;
                    } else {
                        gain_lock[k] = true;
                    }
                }
            }

            if iter >= 6 {
                break;
            }
            iter += 1;
            gain_mult_q8 = if found_lower && found_upper {
                // Interpolate to the cap, then clamp to the middle half of the
                // bracket (gm_upper < gm_lower since more gain → fewer bits).
                let interp = gm_lower + (gm_upper - gm_lower) * (max_bits - nb_lower) / (nb_upper - nb_lower);
                let span = gm_upper - gm_lower;
                interp.clamp(gm_upper - span / 4, gm_lower + span / 4)
            } else if n_bits > max_bits {
                (gain_mult_q8 * 3 / 2).min(1024)
            } else {
                (gain_mult_q8 * 4 / 5).max(64)
            };
        }

        // If the final attempt still busts but an earlier one fit, restore it.
        if let (Some(max_bits), Some(bf)) = (max_bits, &best_fit) {
            if enc.tell() as i32 > max_bits {
                let (enc0, nsq0, ec0, lg0, gi0) = bf;
                enc.clone_from(enc0);
                self.nsq.clone_from(nsq0);
                self.ec_prev = *ec0;
                self.last_gain_index = *lg0;
                indices.gains_indices = *gi0;
            }
        }
        // `indices`/`pulses`/`quant_offset_type` now hold the chosen attempt and
        // `self.ec_prev` reflects exactly one regular emission. The caller may
        // re-emit these captured indices/pulses as an LBRR copy (FEC).

        // Capture the chosen pulses for the caller (the analysis/emission split
        // and the LBRR copy both need them after the work buffer is returned).
        let pulses_out = pulses.clone();

        // Return the work buffers for the next frame to reuse.
        self.scratch_pitch = pitch_x_buf;
        self.scratch_res = res;
        self.scratch_res_f = res_f;
        self.scratch_x_buf = x_buf;
        self.scratch_lpc_in = lpc_in_pre;
        self.scratch_pulses = pulses;
        (indices, pulses_out, lbrr_out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::decoder::SilkChannelDecoder;
    use super::*;
    use crate::range::RangeDecoder;

    /// Encodes `input` on `e`, captures the encoder's NSQ reconstruction, and
    /// decodes the coded frame on the persistent decoder `d` (so cross-frame
    /// state stays in sync). Returns (decoder xq, encoder xq, coded signal
    /// type, input-correlation).
    fn round_trip(
        e: &mut SilkChannelEncoder,
        d: &mut SilkChannelDecoder,
        input: &[i16],
    ) -> (Vec<i16>, Vec<i16>, i32, f64) {
        let frame_length = e.nb_subfr * 5 * e.fs_khz as usize;
        let ltp_mem = 20 * e.fs_khz as usize;

        let mut enc = RangeEncoder::new(512);
        let ind = e.encode_frame(&mut enc, input, CondCoding::Independently, None);
        let signal_type = i32::from(ind.signal_type);
        let xq_enc: Vec<i16> = e.nsq.xq[ltp_mem..ltp_mem + frame_length].to_vec();
        let bytes = enc.finalize().expect("frame fits");
        assert!(!bytes.is_empty());

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
        (xq, xq_enc, signal_type, corr)
    }

    /// A noise frame is detected as unvoiced and decodes to the encoder's own
    /// NSQ reconstruction.
    #[test]
    fn unvoiced_frame_round_trips_through_the_decoder() {
        let (fs_khz, nb_subfr) = (16i32, 4usize);
        let frame_length = nb_subfr * 5 * fs_khz as usize;
        let mut seed = 0x9e37_u32;
        let input: Vec<i16> = (0..frame_length)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((seed >> 16) as i32 - 32768) as i16 / 12
            })
            .collect();

        let mut e = SilkChannelEncoder::new(fs_khz, nb_subfr);
        let mut d = SilkChannelDecoder::new(fs_khz, nb_subfr);
        let (xq, xq_enc, signal_type, _corr) = round_trip(&mut e, &mut d, &input);
        assert_eq!(signal_type, TYPE_UNVOICED, "noise should be detected unvoiced");
        assert_eq!(xq, xq_enc, "decoder disagrees with the encoder's NSQ reconstruction");
    }

    /// A strongly periodic frame is detected as voiced and round-trips through
    /// the decoder with the long-term predictor engaged.
    #[test]
    fn voiced_frame_round_trips_through_the_decoder() {
        let (fs_khz, nb_subfr) = (16i32, 4usize);
        let frame_length = nb_subfr * 5 * fs_khz as usize;

        // A continuous, strongly periodic tone (period 100 samples) spanning
        // two frames, so the second frame's pitch history is phase-continuous.
        let full: Vec<i16> = (0..2 * frame_length)
            .map(|i| {
                let mut s = 2500.0 * (core::f32::consts::TAU * i as f32 / 100.0).sin();
                s += 900.0 * (core::f32::consts::TAU * i as f32 / 50.0).sin();
                s += ((i as i32 * 1733 + 3) % 173 - 86) as f32 * 1.2;
                s.clamp(-30000.0, 30000.0) as i16
            })
            .collect();

        let mut e = SilkChannelEncoder::new(fs_khz, nb_subfr);
        let mut d = SilkChannelDecoder::new(fs_khz, nb_subfr);
        // First frame primes the pitch-analysis history; the second's history
        // is then the (phase-continuous) first frame. The decoder is shared so
        // its synthesis history matches the encoder's going into frame two.
        round_trip(&mut e, &mut d, &full[..frame_length]);
        let (xq, xq_enc, signal_type, corr) = round_trip(&mut e, &mut d, &full[frame_length..]);
        assert_eq!(signal_type, TYPE_VOICED, "periodic signal should be detected voiced");
        assert_eq!(
            xq, xq_enc,
            "decoder disagrees with the encoder's NSQ reconstruction (voiced)"
        );
        assert!(corr > 0.5, "voiced reconstruction correlation {corr:.3} too low");
    }
}

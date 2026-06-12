//! The CELT frame decoder (RFC 6716 §4.3; normative `celt_decoder.c`, float
//! build) - the driver that sequences every stage into PCM.
//!
//! Per frame, in bitstream order: silence flag → post-filter parameters →
//! transient flag → intra flag → coarse energy → time/frequency resolution →
//! spreading → dynamic allocation boosts → allocation trim → the implicit
//! bit allocation → fine energy → PVQ band shapes → anti-collapse →
//! final energy bits; then synthesis: denormalisation → inverse MDCT →
//! comb post-filter → de-emphasis.
//!
//! The decoder carries cross-frame state: the energy predictor
//! ([`EnergyState`]), two frames of energy history for anti-collapse, the
//! MDCT overlap/post-filter history ring, the de-emphasis filter memory, and
//! the noise seed. Mono streams decoded by a stereo decoder (and vice versa)
//! follow the reference's up/down-mix paths.

use alloc::vec;
use alloc::vec::Vec;

use crate::range::RangeDecoder;

use super::bands::{anti_collapse, quant_all_bands};
use super::energy::{EnergyState, decode_coarse_energy, decode_energy_finalise, decode_fine_energy};
use super::mdct::MdctLookup;
use super::modes::{E_MEANS, EBANDS, NB_EBANDS};
use super::plc::{
    CELT_LPC_ORDER, MAX_PERIOD, PLC_PITCH_LAG_MAX, PLC_PITCH_LAG_MIN, celt_autocorr, celt_fir, celt_iir, celt_lpc,
    pitch_downsample, pitch_search,
};
use super::rate::{BITRES, compute_allocation, init_caps};
use super::tables::WINDOW120;
use super::vq::Spread;

/// Synthesis history length (`DECODE_BUFFER_SIZE`).
const DECODE_BUFFER_SIZE: usize = 2048;

/// MDCT overlap of the standard mode.
const OVERLAP: usize = 120;

/// Samples per shortest MDCT block (`shortMdctSize`).
const SHORT_MDCT_SIZE: usize = 120;

/// Comb-filter pitch period bounds (`COMBFILTER_*`).
const COMBFILTER_MINPERIOD: usize = 15;

/// First-order pre-emphasis coefficient of the standard mode (`preemph`).
const PREEMPH_COEF: f32 = 0.850_006_1;

/// Spreading decision ICDF (`spread_icdf`, celt.h).
const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];

/// Allocation trim ICDF (`trim_icdf`, celt.h).
const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];

/// Post-filter tapset ICDF (`tapset_icdf`, celt.h).
const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// Time/frequency resolution change table (`tf_select_table`, celt.c),
/// indexed `[LM][4*isTransient + 2*tf_select + per_band_flag]`.
pub(super) const TF_SELECT_TABLE: [[i32; 8]; 4] = [
    [0, -1, 0, -1, 0, -1, 0, -1],
    [0, -1, 0, -2, 1, 0, 1, -1],
    [0, -2, 0, -3, 2, 0, 1, -1],
    [0, -2, 0, -3, 3, 0, 1, -1],
];

/// Comb-filter tap gains per tapset (`gains`, celt.c).
const COMB_GAINS: [[f32; 3]; 3] = [
    [0.306_640_62, 0.217_041_02, 0.129_638_67],
    [0.463_867_2, 0.268_066_4, 0.0],
    [0.799_804_7, 0.100_097_656, 0.0],
];

/// The CELT decoder state for one stream.
#[derive(Debug, Clone)]
pub struct CeltDecoder {
    /// Output channel count `CC` (1 or 2); the per-frame stream channel
    /// count `C` may differ and is converted per the reference.
    channels: usize,
    /// Synthesis + overlap history per channel (`decode_mem`).
    decode_mem: Vec<Vec<f32>>,
    /// Cross-frame energy predictor state (`oldBandE`).
    energy: EnergyState,
    /// Energy of the previous frame (`oldLogE`), per channel.
    old_log_e: [[f32; NB_EBANDS]; 2],
    /// Energy of the frame before that (`oldLogE2`).
    old_log_e2: [[f32; NB_EBANDS]; 2],
    /// Long-term background energy (`backgroundLogE`).
    background_log_e: [[f32; NB_EBANDS]; 2],
    /// De-emphasis filter memory per channel (`preemph_memD`).
    preemph_mem: [f32; 2],
    /// Post-filter state.
    postfilter_period: usize,
    postfilter_period_old: usize,
    postfilter_gain: f32,
    postfilter_gain_old: f32,
    postfilter_tapset: usize,
    postfilter_tapset_old: usize,
    /// Noise seed (`rng`), carried across frames.
    rng: u32,
    /// Samples concealed since the last good frame (`loss_duration`).
    loss_duration: usize,
    /// Suppress pitch-based PLC until two consecutive good frames
    /// (`skip_plc`).
    skip_plc: bool,
    /// Pitch period of the last concealment (`last_pitch_index`).
    last_pitch_index: usize,
    /// Concealment LPC per channel (`lpc`), kept across consecutive losses.
    plc_lpc: [[f32; CELT_LPC_ORDER]; 2],
    /// The next frame must pre-filter and TDAC-fold the concealed overlap
    /// (`prefilter_and_fold`).
    prefilter_and_fold: bool,
    /// Output decimation factor (`downsample`): 48000 / output rate.
    downsample: usize,
    /// The MDCT family of the standard mode.
    mdct: MdctLookup,
}

impl CeltDecoder {
    /// Creates a decoder producing `channels` output channels (1 or 2).
    ///
    /// # Panics
    ///
    /// Panics if `channels` is not 1 or 2.
    #[must_use]
    pub fn new(channels: usize) -> Self {
        Self::with_rate(channels, 48_000)
    }

    /// Creates a decoder producing output at `fs_hz`
    /// (48/24/16/12/8 kHz; `celt_decoder_init` + `resampling_factor`).
    ///
    /// # Panics
    ///
    /// Panics on unsupported rates or channel counts.
    #[must_use]
    pub fn with_rate(channels: usize, fs_hz: u32) -> Self {
        assert!(
            matches!(fs_hz, 48_000 | 24_000 | 16_000 | 12_000 | 8_000),
            "unsupported CELT output rate"
        );
        assert!(channels == 1 || channels == 2, "CELT supports 1 or 2 channels");
        CeltDecoder {
            channels,
            decode_mem: vec![vec![0.0; DECODE_BUFFER_SIZE + OVERLAP]; channels],
            energy: EnergyState::default(),
            old_log_e: [[-28.0; NB_EBANDS]; 2],
            old_log_e2: [[-28.0; NB_EBANDS]; 2],
            background_log_e: [[0.0; NB_EBANDS]; 2],
            preemph_mem: [0.0; 2],
            postfilter_period: 0,
            postfilter_period_old: 0,
            postfilter_gain: 0.0,
            postfilter_gain_old: 0.0,
            postfilter_tapset: 0,
            postfilter_tapset_old: 0,
            rng: 0,
            loss_duration: 0,
            skip_plc: false,
            last_pitch_index: 0,
            plc_lpc: [[0.0; CELT_LPC_ORDER]; 2],
            prefilter_and_fold: false,
            downsample: (48_000 / fs_hz) as usize,
            mdct: MdctLookup::new(1920),
        }
    }

    /// The range coder's `rng` after the most recent frame - the
    /// bit-exactness oracle (`OPUS_GET_FINAL_RANGE`); the noise seed is
    /// reseeded from this same value per the reference.
    #[must_use]
    pub const fn final_range(&self) -> u32 {
        self.rng
    }

    /// Decodes one CELT frame of `frame_size` samples per channel (at 48 kHz)
    /// from `dec` over `frame_bytes` of input, with the band range
    /// `start..end` set by the Opus layer (0/17 and 13..=21 per bandwidth).
    ///
    /// Returns interleaved f32 PCM in `[-1, 1]`, `frame_size * channels`
    /// samples.
    ///
    /// # Panics
    ///
    /// Panics if `frame_size` is not 120, 240, 480 or 960 - the Opus layer
    /// guarantees this from the TOC.
    #[allow(clippy::too_many_arguments, reason = "mirrors celt_decode_with_ec")]
    #[must_use]
    pub fn decode_frame(
        &mut self,
        dec: &mut RangeDecoder,
        frame_bytes: usize,
        frame_size: usize,
        stream_channels: usize,
        start: usize,
        end: usize,
    ) -> Vec<f32> {
        let cc = self.channels;
        let c = stream_channels;
        let lm = (0..=3)
            .find(|&lm| SHORT_MDCT_SIZE << lm == frame_size)
            .expect("frame size must be 120/240/480/960");
        let m = 1usize << lm;
        let n = frame_size;
        let eff_end = end.min(NB_EBANDS);
        let len = frame_bytes;

        // A stereo decoder fed a mono stream merges its energy history.
        if c == 1 {
            for i in 0..NB_EBANDS {
                self.energy.old_ebands[0][i] = self.energy.old_ebands[0][i].max(self.energy.old_ebands[1][i]);
            }
        }

        if self.loss_duration == 0 {
            self.skip_plc = false;
        }

        let total_bits = (len * 8) as u32;
        let mut tell = dec.tell();

        // Silence flag.
        let silence = if tell >= total_bits {
            true
        } else if tell == 1 {
            dec.decode_bit_logp(15)
        } else {
            false
        };
        if silence {
            // Pretend we've read all the remaining bits.
            dec.force_tell(total_bits);
            tell = total_bits;
        }

        // Post-filter parameters.
        let mut postfilter_pitch = 0usize;
        let mut postfilter_gain = 0.0f32;
        let mut postfilter_tapset = 0usize;
        if start == 0 && tell + 16 <= total_bits {
            if dec.decode_bit_logp(1) {
                let octave = dec.decode_uint(6).unwrap_or(0);
                postfilter_pitch = ((16u32 << octave) + dec.decode_raw_bits(4 + octave) - 1) as usize;
                let qg = dec.decode_raw_bits(3);
                if dec.tell() + 2 <= total_bits {
                    postfilter_tapset = dec.decode_icdf(&TAPSET_ICDF, 2);
                }
                postfilter_gain = 0.09375 * (qg + 1) as f32;
            }
            tell = dec.tell();
        }

        // Transient flag.
        let is_transient = if lm > 0 && tell + 3 <= total_bits {
            let t = dec.decode_bit_logp(3);
            tell = dec.tell();
            t
        } else {
            false
        };
        let short_blocks = is_transient;

        // Intra flag and coarse energy.
        let intra = tell + 3 <= total_bits && dec.decode_bit_logp(3);
        // Recovering from loss without intra energy: make the prediction
        // safe to avoid loud artifacts (opus_decoder.c loss-recovery clamp).
        if !intra && self.loss_duration != 0 {
            let safety = match lm {
                0 => 1.5f32,
                1 => 0.5,
                _ => 0.0,
            };
            let missing = 10.min(self.loss_duration >> lm) as f32;
            for ch in 0..2 {
                for i in start..end {
                    let e0 = self.energy.old_ebands[ch][i];
                    let e1 = self.old_log_e[ch][i];
                    let e2 = self.old_log_e2[ch][i];
                    if e0 < e1.max(e2) {
                        // Energy is going down already: continue the trend.
                        let slope = (e1 - e0).max(0.5 * (e2 - e0));
                        self.energy.old_ebands[ch][i] = (e0 - (1.0 + missing) * slope.max(0.0)).max(-20.0);
                    } else {
                        // Otherwise take the min of the last frames.
                        self.energy.old_ebands[ch][i] = e0.min(e1).min(e2);
                    }
                    self.energy.old_ebands[ch][i] -= safety;
                }
            }
        }
        decode_coarse_energy(dec, &mut self.energy, start, end, intra, c, lm, total_bits);

        // Time/frequency resolution.
        let tf_res = tf_decode(dec, start, end, is_transient, lm, total_bits);

        // Spreading decision.
        let spread = if dec.tell() + 4 <= total_bits {
            Spread::from_raw(dec.decode_icdf(&SPREAD_ICDF, 5) as u32)
        } else {
            Spread::Normal
        };

        let caps = init_caps(lm, c);

        // Dynamic allocation boosts.
        let mut offsets = [0i32; NB_EBANDS];
        let mut dynalloc_logp = 6u32;
        let mut total_bits_frac = (total_bits as i64) << BITRES;
        let mut tell_frac = i64::from(dec.tell_frac());
        for i in start..end {
            let width = (c as i32 * i32::from(EBANDS[i + 1] - EBANDS[i])) << lm;
            // 6 bits, but no more than 1 bit/sample and at least 1/8.
            let quanta = (width << BITRES).min((6 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            while tell_frac + ((i64::from(dynalloc_loop_logp)) << BITRES) < total_bits_frac && boost < caps[i] {
                let flag = dec.decode_bit_logp(dynalloc_loop_logp);
                tell_frac = i64::from(dec.tell_frac());
                if !flag {
                    break;
                }
                boost += quanta;
                total_bits_frac -= i64::from(quanta);
                dynalloc_loop_logp = 1;
            }
            offsets[i] = boost;
            if boost > 0 {
                dynalloc_logp = 2.max(dynalloc_logp - 1);
            }
        }

        // Allocation trim.
        let alloc_trim = if tell_frac + (6 << BITRES) <= total_bits_frac {
            dec.decode_icdf(&TRIM_ICDF, 7) as i32
        } else {
            5
        };

        // The implicit allocation.
        let mut bits = ((len as i32 * 8) << BITRES) - dec.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm as i32 + 2) << BITRES) {
            1 << BITRES
        } else {
            0
        };
        bits -= anti_collapse_rsv;

        let alloc = compute_allocation(
            &mut super::rate::AllocEc::Dec(dec),
            start,
            end,
            &offsets,
            &caps,
            alloc_trim,
            bits,
            c,
            lm,
        );

        decode_fine_energy(dec, &mut self.energy, start, end, &alloc.fine_quant, c);

        // PVQ shapes.
        let mut x = vec![0.0f32; c * n];
        let mut collapse_masks = vec![0u8; c * NB_EBANDS];
        {
            let (x0, x1) = x.split_at_mut(n);
            quant_all_bands(
                dec,
                start,
                end,
                x0,
                (c == 2).then_some(x1),
                &mut collapse_masks,
                &alloc.shape_bits,
                short_blocks,
                spread,
                alloc.dual_stereo,
                alloc.intensity,
                &tf_res,
                (len as i32) * (8 << BITRES) - anti_collapse_rsv,
                alloc.balance,
                lm,
                alloc.coded_bands,
                &mut self.rng,
            );
        }

        let anti_collapse_on = anti_collapse_rsv > 0 && dec.decode_raw_bits(1) == 1;

        decode_energy_finalise(
            dec,
            &mut self.energy,
            start,
            end,
            &alloc.fine_quant,
            &alloc.fine_priority,
            len as i32 * 8 - dec.tell() as i32,
            c,
        );

        if anti_collapse_on {
            anti_collapse(
                &mut x,
                &collapse_masks,
                lm,
                c,
                n,
                start,
                end,
                &self.energy.old_ebands,
                &self.old_log_e,
                &self.old_log_e2,
                &alloc.shape_bits,
                self.rng,
            );
        }

        if silence {
            for ch in &mut self.energy.old_ebands {
                ch.fill(-28.0);
            }
        }

        // Shift the synthesis history.
        for mem in &mut self.decode_mem {
            mem.copy_within(n..n + (DECODE_BUFFER_SIZE - n + OVERLAP / 2), 0);
        }

        // Blend in concealed audio left by a preceding loss.
        if self.prefilter_and_fold {
            self.run_prefilter_and_fold(n);
        }

        self.synthesis(&x, c, start, eff_end, lm, short_blocks, silence);
        let out_base = DECODE_BUFFER_SIZE - n;

        // Comb post-filter over the new samples.
        self.postfilter_period = self.postfilter_period.max(COMBFILTER_MINPERIOD);
        self.postfilter_period_old = self.postfilter_period_old.max(COMBFILTER_MINPERIOD);
        let pf_pitch = postfilter_pitch.max(COMBFILTER_MINPERIOD);
        for ch in 0..cc {
            comb_filter(
                &mut self.decode_mem[ch],
                out_base,
                self.postfilter_period_old,
                self.postfilter_period,
                SHORT_MDCT_SIZE,
                self.postfilter_gain_old,
                self.postfilter_gain,
                self.postfilter_tapset_old,
                self.postfilter_tapset,
            );
            if lm != 0 {
                comb_filter(
                    &mut self.decode_mem[ch],
                    out_base + SHORT_MDCT_SIZE,
                    self.postfilter_period,
                    pf_pitch,
                    n - SHORT_MDCT_SIZE,
                    self.postfilter_gain,
                    postfilter_gain,
                    self.postfilter_tapset,
                    postfilter_tapset,
                );
            }
        }
        self.postfilter_period_old = self.postfilter_period;
        self.postfilter_gain_old = self.postfilter_gain;
        self.postfilter_tapset_old = self.postfilter_tapset;
        self.postfilter_period = pf_pitch;
        self.postfilter_gain = postfilter_gain;
        self.postfilter_tapset = postfilter_tapset;
        if lm != 0 {
            self.postfilter_period_old = self.postfilter_period;
            self.postfilter_gain_old = self.postfilter_gain;
            self.postfilter_tapset_old = self.postfilter_tapset;
        }

        // Energy history bookkeeping.
        if c == 1 {
            self.energy.old_ebands[1] = self.energy.old_ebands[0];
        }
        if is_transient {
            for ch in 0..2 {
                for i in 0..NB_EBANDS {
                    self.old_log_e[ch][i] = self.old_log_e[ch][i].min(self.energy.old_ebands[ch][i]);
                }
            }
        } else {
            self.old_log_e2 = self.old_log_e;
            self.old_log_e = self.energy.old_ebands;
            for ch in 0..2 {
                for i in 0..NB_EBANDS {
                    self.background_log_e[ch][i] = (self.background_log_e[ch][i]
                        + 160.min(self.loss_duration + m) as f32 * 0.001)
                        .min(self.energy.old_ebands[ch][i]);
                }
            }
        }
        for ch in 0..2 {
            for i in 0..start {
                self.energy.old_ebands[ch][i] = 0.0;
                self.old_log_e[ch][i] = -28.0;
                self.old_log_e2[ch][i] = -28.0;
            }
            for i in end..NB_EBANDS {
                self.energy.old_ebands[ch][i] = 0.0;
                self.old_log_e[ch][i] = -28.0;
                self.old_log_e2[ch][i] = -28.0;
            }
        }

        // Reseed the noise generator from the range coder (the reference's
        // `st->rng = dec->rng`); this is also the frame's final range value.
        self.rng = dec.range_size();

        // De-emphasis into interleaved PCM (float API scale: 1/32768).
        let pcm = self.deemphasis(n);

        if c == 1 {
            self.energy.old_ebands[1] = self.energy.old_ebands[0];
        }

        self.loss_duration = 0;
        self.prefilter_and_fold = false;
        pcm
    }

    /// `celt_decode_lost`: conceals one lost frame of `frame_size` samples
    /// per channel, returning interleaved PCM like
    /// [`decode_frame`](Self::decode_frame).
    ///
    /// Short losses extrapolate the last pitch period in the excitation
    /// domain; long losses (or losses before two good frames) fade to
    /// comfort noise shaped by the long-term band energies. PLC output is
    /// not normative - this mirrors the reference float build.
    ///
    /// # Panics
    ///
    /// Panics if `frame_size` is not 120, 240, 480 or 960.
    #[must_use]
    pub fn decode_lost(&mut self, frame_size: usize, start: usize, end: usize) -> Vec<f32> {
        let cc = self.channels;
        let lm = (0..=3)
            .find(|&lm| SHORT_MDCT_SIZE << lm == frame_size)
            .expect("frame size must be 120/240/480/960");
        let n = frame_size;
        let eff_end = start.max(end.min(NB_EBANDS));
        let loss_duration = self.loss_duration;

        let noise_based = loss_duration >= 40 || start != 0 || self.skip_plc;
        if noise_based {
            // Noise-based PLC/CNG.
            for mem in &mut self.decode_mem {
                mem.copy_within(n..n + (DECODE_BUFFER_SIZE - n + OVERLAP), 0);
            }
            if self.prefilter_and_fold {
                self.run_prefilter_and_fold(n);
            }

            // Energy decay towards the background noise floor.
            let decay = if loss_duration == 0 { 1.5 } else { 0.5 };
            for ch in 0..cc {
                for i in start..end {
                    self.energy.old_ebands[ch][i] =
                        self.background_log_e[ch][i].max(self.energy.old_ebands[ch][i] - decay);
                }
            }

            // Fill the coded bands with normalised noise.
            let m = 1usize << lm;
            let mut seed = self.rng;
            let mut x = vec![0.0f32; cc * n];
            for ch in 0..cc {
                for i in start..eff_end {
                    let boffs = n * ch + ((EBANDS[i] as usize) << lm);
                    let blen = ((EBANDS[i + 1] - EBANDS[i]) as usize) << lm;
                    for v in &mut x[boffs..boffs + blen] {
                        seed = super::bands::celt_lcg_rand(seed);
                        *v = (seed as i32 >> 20) as f32;
                    }
                    super::vq::renormalise_vector(&mut x[boffs..boffs + blen], 1.0);
                }
            }
            self.rng = seed;
            let _ = m;

            self.synthesis(&x, cc, start, eff_end, lm, false, false);
            self.prefilter_and_fold = false;
            // Skip pitch-based PLC until two consecutive good frames.
            self.skip_plc = true;
        } else {
            // Pitch-based PLC.
            let fade = if loss_duration == 0 {
                let mut lp = vec![0.0f32; DECODE_BUFFER_SIZE >> 1];
                {
                    let refs: alloc::vec::Vec<&[f32]> =
                        self.decode_mem.iter().map(|mem| &mem[..DECODE_BUFFER_SIZE]).collect();
                    pitch_downsample(&refs, &mut lp, DECODE_BUFFER_SIZE);
                }
                let pitch = pitch_search(
                    &lp[PLC_PITCH_LAG_MAX >> 1..],
                    &lp,
                    DECODE_BUFFER_SIZE - PLC_PITCH_LAG_MAX,
                    PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN,
                );
                self.last_pitch_index = PLC_PITCH_LAG_MAX - pitch;
                1.0f32
            } else {
                0.8f32
            };
            let pitch_index = self.last_pitch_index;

            // Excitation for up to two pitch periods, to gauge decay.
            let exc_length = (2 * pitch_index).min(MAX_PERIOD);

            for ch in 0..cc {
                // The excitation with LPC_ORDER samples of history.
                let mut exc = [0.0f32; MAX_PERIOD + CELT_LPC_ORDER];
                {
                    let buf = &self.decode_mem[ch];
                    exc.copy_from_slice(&buf[DECODE_BUFFER_SIZE - MAX_PERIOD - CELT_LPC_ORDER..DECODE_BUFFER_SIZE]);
                }

                if loss_duration == 0 {
                    // LPC over the last MAX_PERIOD samples before the loss.
                    let mut ac = [0.0f32; CELT_LPC_ORDER + 1];
                    celt_autocorr(&exc[CELT_LPC_ORDER..], &mut ac, &WINDOW120, OVERLAP, CELT_LPC_ORDER);
                    // Noise floor -40 dB, then lag windowing.
                    ac[0] *= 1.0001;
                    for (i, a) in ac.iter_mut().enumerate().skip(1) {
                        *a -= *a * (0.008 * i as f32) * (0.008 * i as f32);
                    }
                    celt_lpc(&mut self.plc_lpc[ch], &ac);
                }

                // Whiten the last exc_length samples into the excitation
                // domain.
                {
                    let mut fir_tmp = vec![0.0f32; exc_length];
                    let base = MAX_PERIOD - exc_length;
                    celt_fir(
                        &exc[base..base + CELT_LPC_ORDER + exc_length],
                        &self.plc_lpc[ch],
                        &mut fir_tmp,
                    );
                    exc[CELT_LPC_ORDER + base..CELT_LPC_ORDER + base + exc_length].copy_from_slice(&fir_tmp);
                }

                // Energy decay of the excitation across the two periods.
                let decay = {
                    let decay_length = exc_length >> 1;
                    let mut e1 = 1.0f32;
                    let mut e2 = 1.0f32;
                    for i in 0..decay_length {
                        let e = exc[CELT_LPC_ORDER + MAX_PERIOD - decay_length + i];
                        e1 += e * e;
                        let e = exc[CELT_LPC_ORDER + MAX_PERIOD - 2 * decay_length + i];
                        e2 += e * e;
                    }
                    (e1.min(e2) / e2).sqrt()
                };

                let buf = &mut self.decode_mem[ch];
                buf.copy_within(n..n + (DECODE_BUFFER_SIZE - n), 0);

                // Extrapolate one pitch period at a time, decaying.
                let extrapolation_offset = MAX_PERIOD - pitch_index;
                let extrapolation_len = n + OVERLAP;
                let mut attenuation = fade * decay;
                let mut s1 = 0.0f32;
                let mut j = 0usize;
                for i in 0..extrapolation_len {
                    if j >= pitch_index {
                        j -= pitch_index;
                        attenuation *= decay;
                    }
                    buf[DECODE_BUFFER_SIZE - n + i] = attenuation * exc[CELT_LPC_ORDER + extrapolation_offset + j];
                    // Energy of the previously decoded signal whose
                    // excitation is being copied.
                    let tmp = buf[DECODE_BUFFER_SIZE - MAX_PERIOD - n + extrapolation_offset + j];
                    s1 += tmp * tmp;
                    j += 1;
                }

                // Back to the signal domain through the synthesis filter.
                {
                    let mut lpc_mem = [0.0f32; CELT_LPC_ORDER];
                    for (i, v) in lpc_mem.iter_mut().enumerate() {
                        *v = buf[DECODE_BUFFER_SIZE - n - 1 - i];
                    }
                    let region = &mut buf[DECODE_BUFFER_SIZE - n..DECODE_BUFFER_SIZE - n + extrapolation_len];
                    celt_iir(region, &self.plc_lpc[ch], &mut lpc_mem);
                }

                // Attenuate if the synthesis energy exceeds expectation.
                let mut s2 = 0.0f32;
                for i in 0..extrapolation_len {
                    let tmp = buf[DECODE_BUFFER_SIZE - n + i];
                    s2 += tmp * tmp;
                }
                // Written to also catch NaN from the IIR filter.
                #[allow(
                    clippy::neg_cmp_op_on_partial_ord,
                    reason = "the reference writes it this way to also catch NaN from the IIR"
                )]
                if !(s1 > 0.2 * s2) {
                    for v in &mut buf[DECODE_BUFFER_SIZE - n..DECODE_BUFFER_SIZE - n + extrapolation_len] {
                        *v = 0.0;
                    }
                } else if s1 < s2 {
                    let ratio = ((s1 + 1.0) / (s2 + 1.0)).sqrt();
                    for i in 0..OVERLAP {
                        let tmp_g = 1.0 - WINDOW120[i] * (1.0 - ratio);
                        buf[DECODE_BUFFER_SIZE - n + i] *= tmp_g;
                    }
                    for i in OVERLAP..extrapolation_len {
                        buf[DECODE_BUFFER_SIZE - n + i] *= ratio;
                    }
                }
            }
            self.prefilter_and_fold = true;
        }

        self.loss_duration = 10_000.min(loss_duration + (1usize << lm));
        self.deemphasis(n)
    }

    /// De-emphasis of the newest `n` history samples into interleaved PCM,
    /// decimating by `downsample` (`deemphasis` in celt_decoder.c).
    fn deemphasis(&mut self, n: usize) -> Vec<f32> {
        let cc = self.channels;
        let out_base = DECODE_BUFFER_SIZE - n;
        let nd = n / self.downsample;
        let mut pcm = vec![0.0f32; nd * cc];
        let mut scratch = vec![0.0f32; n];
        for ch in 0..cc {
            let mut mem = self.preemph_mem[ch];
            let x = &self.decode_mem[ch][out_base..out_base + n];
            if self.downsample > 1 {
                for (j, &v) in x.iter().enumerate() {
                    let tmp = v + mem + 1e-30;
                    mem = PREEMPH_COEF * tmp;
                    scratch[j] = tmp;
                }
                for (j, p) in pcm.iter_mut().skip(ch).step_by(cc).enumerate() {
                    *p = scratch[j * self.downsample] * (1.0 / 32768.0);
                }
            } else {
                for (j, &v) in x.iter().enumerate() {
                    let tmp = v + mem + 1e-30;
                    mem = PREEMPH_COEF * tmp;
                    pcm[j * cc + ch] = tmp * (1.0 / 32768.0);
                }
            }
            self.preemph_mem[ch] = mem;
        }
        pcm
    }

    /// `celt_synthesis`: denormalises the band shapes against the energy
    /// state, converts stream to decoder channels, and runs the inverse
    /// MDCTs into the (already shifted) history ring.
    #[allow(clippy::too_many_arguments, reason = "mirrors celt_synthesis")]
    fn synthesis(
        &mut self,
        x: &[f32],
        c: usize,
        start: usize,
        eff_end: usize,
        lm: usize,
        short_blocks: bool,
        silence: bool,
    ) {
        let cc = self.channels;
        let m = 1usize << lm;
        let n = SHORT_MDCT_SIZE << lm;

        let mut freq = vec![0.0f32; cc.max(c) * n];
        if !silence {
            for ch in 0..c {
                denormalise_band_energies(
                    &x[ch * n..(ch + 1) * n],
                    &mut freq[ch * n..(ch + 1) * n],
                    &self.energy.old_ebands[ch],
                    start,
                    eff_end,
                    m,
                );
            }
            // Zero the uncoded spectrum top (bounded by the output rate).
            let mut bound = m * EBANDS[eff_end] as usize;
            if self.downsample != 1 {
                bound = bound.min(n / self.downsample);
            }
            for ch in 0..c {
                for f in &mut freq[ch * n + bound..(ch + 1) * n] {
                    *f = 0.0;
                }
            }
        }

        // Stream/decoder channel-count conversion.
        if cc == 2 && c == 1 {
            let (f0, f1) = freq.split_at_mut(n);
            f1.copy_from_slice(f0);
        }
        if cc == 1 && c == 2 {
            let (f0, f1) = freq.split_at_mut(n);
            for (a, &b) in f0.iter_mut().zip(f1.iter()) {
                *a = 0.5 * (*a + b);
            }
        }

        // Inverse MDCTs into the history ring.
        let (b_blocks, nb, shift) = if short_blocks {
            (m, SHORT_MDCT_SIZE, 3usize)
        } else {
            (1, n, 3 - lm)
        };
        let out_base = DECODE_BUFFER_SIZE - n;
        for ch in 0..cc {
            for b in 0..b_blocks {
                self.mdct.backward(
                    &freq[ch * n + b..],
                    &mut self.decode_mem[ch][out_base + nb * b..],
                    &WINDOW120,
                    OVERLAP,
                    shift,
                    b_blocks,
                );
            }
        }
    }

    /// `prefilter_and_fold`: pre-filters the concealed tail (undoing the
    /// post-filter the next frame will re-apply) and simulates TDAC so the
    /// concealment blends with the next frame's MDCT.
    fn run_prefilter_and_fold(&mut self, n: usize) {
        let cc = self.channels;
        let t1 = self.postfilter_period;
        let g1 = -self.postfilter_gain;
        let taps = COMB_GAINS[self.postfilter_tapset];
        for ch in 0..cc {
            let buf = &mut self.decode_mem[ch];
            let base = DECODE_BUFFER_SIZE - n;
            let mut etmp = [0.0f32; OVERLAP];
            for (i, e) in etmp.iter_mut().enumerate() {
                let idx = base + i;
                *e = buf[idx]
                    + g1 * (taps[0] * buf[idx - t1]
                        + taps[1] * (buf[idx - t1 + 1] + buf[idx - t1 - 1])
                        + taps[2] * (buf[idx - t1 + 2] + buf[idx - t1 - 2]));
            }
            for i in 0..OVERLAP / 2 {
                buf[base + i] = WINDOW120[i] * etmp[OVERLAP - 1 - i] + WINDOW120[OVERLAP - 1 - i] * etmp[i];
            }
        }
    }
}

/// Per-band gain application (`denormalise_bands`, float build): the shape
/// is scaled by `2^(bandLogE + eMeans)` per band; spectrum outside
/// `start..end` is zeroed.
fn denormalise_band_energies(
    x: &[f32],
    freq: &mut [f32],
    band_log_e: &[f32; NB_EBANDS],
    start: usize,
    end: usize,
    m: usize,
) {
    for f in freq[..m * EBANDS[start] as usize].iter_mut() {
        *f = 0.0;
    }
    for i in start..end {
        let band_start = m * EBANDS[i] as usize;
        let band_end = m * EBANDS[i + 1] as usize;
        let lg = band_log_e[i] + E_MEANS[i];
        let g = exp2f(lg);
        for j in band_start..band_end {
            freq[j] = x[j] * g;
        }
    }
    for f in freq[m * EBANDS[end] as usize..].iter_mut() {
        *f = 0.0;
    }
}

/// `2^x` matching the reference float build's `celt_exp2`
/// (`exp(0.6931471805599453094 * x)`).
fn exp2f(x: f32) -> f32 {
    (core::f64::consts::LN_2 * f64::from(x)).exp() as f32
}

/// Time/frequency resolution decoding (`tf_decode`).
fn tf_decode(
    dec: &mut RangeDecoder,
    start: usize,
    end: usize,
    is_transient: bool,
    lm: usize,
    total_bits: u32,
) -> [i32; NB_EBANDS] {
    let mut budget = total_bits;
    let mut tell = dec.tell();
    let mut logp = if is_transient { 2 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    budget -= u32::from(tf_select_rsv);
    let mut tf_changed = false;
    let mut curr = false;

    let mut tf_res = [0i32; NB_EBANDS];
    for r in tf_res.iter_mut().take(end).skip(start) {
        if tell + logp <= budget {
            curr ^= dec.decode_bit_logp(logp);
            tell = dec.tell();
            tf_changed |= curr;
        }
        *r = i32::from(curr);
        logp = if is_transient { 4 } else { 5 };
    }

    let base = 4 * usize::from(is_transient);
    let mut tf_select = 0usize;
    if tf_select_rsv
        && TF_SELECT_TABLE[lm][base + usize::from(tf_changed)]
            != TF_SELECT_TABLE[lm][base + 2 + usize::from(tf_changed)]
    {
        tf_select = usize::from(dec.decode_bit_logp(1));
    }
    for r in tf_res.iter_mut().take(end).skip(start) {
        *r = TF_SELECT_TABLE[lm][base + 2 * tf_select + (*r as usize)];
    }
    tf_res
}

/// The pitch post-filter (`comb_filter`): a 5-tap comb at the old period
/// cross-faded (via the squared MDCT window) into one at the new period over
/// the first `overlap` samples, then constant.
///
/// Operates in place on `mem[base..base+n]`, reading up to
/// `max(t0, t1) + 2` samples of history before `base`.
#[allow(clippy::too_many_arguments, reason = "mirrors the reference comb_filter signature")]
fn comb_filter(
    mem: &mut [f32],
    base: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: usize,
    tapset1: usize,
) {
    if g0 == 0.0 && g1 == 0.0 {
        return;
    }
    let g00 = g0 * COMB_GAINS[tapset0][0];
    let g01 = g0 * COMB_GAINS[tapset0][1];
    let g02 = g0 * COMB_GAINS[tapset0][2];
    let g10 = g1 * COMB_GAINS[tapset1][0];
    let g11 = g1 * COMB_GAINS[tapset1][1];
    let g12 = g1 * COMB_GAINS[tapset1][2];

    let overlap = OVERLAP.min(n);
    let mut x1 = mem[base + 1 - t1];
    let mut x2 = mem[base - t1];
    let mut x3 = mem[base - t1 - 1];
    let mut x4 = mem[base - t1 - 2];

    let mut i = 0usize;
    while i < overlap {
        let x0 = mem[base + i + 2 - t1];
        let f = WINDOW120[i] * WINDOW120[i];
        mem[base + i] += (1.0 - f) * g00 * mem[base + i - t0]
            + (1.0 - f) * g01 * (mem[base + i + 1 - t0] + mem[base + i - 1 - t0])
            + (1.0 - f) * g02 * (mem[base + i + 2 - t0] + mem[base + i - 2 - t0])
            + f * g10 * x2
            + f * g11 * (x1 + x3)
            + f * g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
        i += 1;
    }
    if g1 == 0.0 {
        return;
    }
    // The constant-filter tail.
    while i < n {
        mem[base + i] += g10 * mem[base + i - t1]
            + g11 * (mem[base + i + 1 - t1] + mem[base + i - 1 - t1])
            + g12 * (mem[base + i + 2 - t1] + mem[base + i - 2 - t1]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;

    use super::*;
    use crate::range::RangeDecoder;

    /// The frame driver must terminate and produce finite PCM for arbitrary
    /// input bytes across all frame sizes and channel configurations.
    #[test]
    fn decodes_arbitrary_bytes_to_finite_pcm() {
        for lm in 0..4usize {
            for (cc, c) in [(1usize, 1usize), (2, 2), (2, 1)] {
                for fill in [0x00u8, 0xA5, 0xFF] {
                    let mut decoder = CeltDecoder::new(cc);
                    let frame_size = 120 << lm;
                    let data = vec![fill; 60];
                    for _ in 0..3 {
                        let mut dec = RangeDecoder::new(&data);
                        let pcm = decoder.decode_frame(&mut dec, data.len(), frame_size, c, 0, 21);
                        assert_eq!(pcm.len(), frame_size * cc);
                        for (i, v) in pcm.iter().enumerate() {
                            assert!(v.is_finite(), "lm={lm} cc={cc} c={c} fill={fill:#x} pcm[{i}]");
                        }
                    }
                }
            }
        }
    }

    /// An all-zero range stream decodes the silence path without panicking.
    #[test]
    fn silence_frame_is_quiet() {
        let mut decoder = CeltDecoder::new(1);
        // A 2-byte frame: tell quickly exceeds the budget, forcing silence
        // and fallback paths everywhere.
        let data = [0x00u8, 0x00];
        let mut dec = RangeDecoder::new(&data);
        let pcm = decoder.decode_frame(&mut dec, data.len(), 960, 1, 0, 21);
        let peak = pcm.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(peak < 1.0, "silence-ish output, got peak {peak}");
    }
}

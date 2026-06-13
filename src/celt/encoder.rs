//! A CELT encoder (RFC 6716 §5.3; normative `celt_encoder.c`,
//! `quant_bands.c` encoder paths): mono or stereo, with transient
//! detection and short blocks.
//!
//! The encoder now runs the full analysis chain - transient detection and
//! short blocks, the pitch pre-filter (comb whitening), per-band time/
//! frequency resolution, dynamic-allocation boosts, spreading and
//! allocation trim - leaving full stereo (no intensity collapse, no dual
//! stereo) and VBR as the remaining conservative choices. The bit-exact
//! machinery (energy quantisation, allocation, theta splits, PVQ) mirrors
//! the decoder's exactly, and every decision is a legal one, so the
//! bitstream is fully conformant.

use alloc::vec;
use alloc::vec::Vec;

use crate::range::RangeEncoder;

use super::bands::encode::quant_all_bands_enc;
use super::bands::haar1;
use super::decoder::{COMB_GAINS, TF_SELECT_TABLE};
use super::energy::EnergyState;
use super::laplace::ec_laplace_encode;
use super::mdct::MdctLookup;
use super::modes::{BETA_COEF, BETA_INTRA, E_MEANS, E_PROB_MODEL, EBANDS, LOG_N, MAX_FINE_BITS, NB_EBANDS, PRED_COEF};
use super::pitch::{COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD, pitch_downsample, pitch_search, remove_doubling};
use super::rate::{AllocEc, BITRES, compute_allocation, init_caps};
use super::tables::WINDOW120;
use super::vq::Spread;

/// Samples per shortest MDCT block.
const SHORT_MDCT_SIZE: usize = 120;
/// MDCT overlap.
const OVERLAP: usize = 120;
/// Pre-emphasis coefficient of the standard mode.
const PREEMPH_COEF: f32 = 0.850_006_1;
/// Spreading decision ICDF (`spread_icdf`).
const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];
/// Allocation trim ICDF (`trim_icdf`).
const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];
/// Post-filter tapset ICDF (`tapset_icdf`).
const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// A CELT encoder at 48 kHz (mono or stereo).
pub struct CeltEncoder {
    /// Channel count (1 or 2).
    channels: usize,
    /// Pre-emphasis filter memory, per channel.
    preemph_mem: [f32; 2],
    /// The previous frame's tail (`in_mem`), `OVERLAP` samples per channel.
    in_mem: [[f32; OVERLAP]; 2],
    /// Energy predictor state (`oldBandE`), shared semantics with the
    /// decoder.
    energy: EnergyState,
    /// Frames encoded (the first is coded intra).
    frames: u64,
    /// Consecutive transient frames (`consec_transient`), steering the
    /// anti-collapse decision.
    consec_transient: u32,
    /// Whether the last frame was coded with short blocks (diagnostic).
    last_transient: bool,
    /// Recursively averaged tonality metric (`tonal_average`) for the
    /// spreading decision.
    tonal_average: i32,
    /// The previous frame's spreading decision (`spread_decision`).
    spread_decision: i32,
    /// Pre-filter history (`prefilter_mem`): the last `COMBFILTER_MAXPERIOD`
    /// pre-emphasised samples per channel.
    prefilter_mem: [Vec<f32>; 2],
    /// The previous frame's pre-filter period, gain and tapset (continuity
    /// for the comb cross-fade).
    prefilter_period: usize,
    prefilter_gain: f32,
    prefilter_tapset: usize,
    /// The most recent pre-filter decision (diagnostic, see
    /// [`CeltEncoder::last_pitch`]).
    last_pitch: usize,
    last_pitch_gain: f32,
    /// Range state of the last encoded frame (the bit-exactness oracle).
    final_range: u32,
    mdct: MdctLookup,
}

impl Default for CeltEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CeltEncoder {
    /// Creates a mono 48 kHz encoder.
    #[must_use]
    pub fn new() -> Self {
        Self::with_channels(1)
    }

    /// Creates a 48 kHz encoder with 1 or 2 channels.
    ///
    /// # Panics
    ///
    /// Panics unless `channels` is 1 or 2.
    #[must_use]
    pub fn with_channels(channels: usize) -> Self {
        assert!(channels == 1 || channels == 2, "channels must be 1 or 2");
        CeltEncoder {
            channels,
            preemph_mem: [0.0; 2],
            in_mem: [[0.0; OVERLAP]; 2],
            energy: EnergyState::default(),
            frames: 0,
            consec_transient: 0,
            last_transient: false,
            tonal_average: 256,
            spread_decision: Spread::Normal as i32,
            prefilter_mem: [vec![0.0; COMBFILTER_MAXPERIOD], vec![0.0; COMBFILTER_MAXPERIOD]],
            prefilter_period: COMBFILTER_MINPERIOD,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            last_pitch: COMBFILTER_MINPERIOD,
            last_pitch_gain: 0.0,
            final_range: 0,
            mdct: MdctLookup::new(1920),
        }
    }

    /// Encodes one frame of `pcm` (interleaved f32 in `[-1, 1]`; 120, 240,
    /// 480 or 960 samples per channel at 48 kHz) into `nb_bytes` of output.
    ///
    /// # Panics
    ///
    /// Panics on invalid frame sizes or byte budgets outside 2..=1275.
    pub fn encode_frame(&mut self, pcm: &[f32], nb_bytes: usize) -> Vec<u8> {
        let channels = self.channels;
        assert!(pcm.len() % channels == 0, "interleaved frame length");
        let n = pcm.len() / channels;
        let lm = (0..=3)
            .find(|&lm| SHORT_MDCT_SIZE << lm == n)
            .expect("frame size must be 120/240/480/960 per channel");
        assert!((2..=1275).contains(&nb_bytes));
        let start = 0usize;
        let end = NB_EBANDS;
        let m = 1usize << lm;

        let mut enc = RangeEncoder::new(nb_bytes);
        let total_bits = (nb_bytes * 8) as u32;

        // Per channel: pre-emphasis into the signal domain (scale 32768),
        // including the previous frame's overlap. `inputs` is planar.
        let in_len = OVERLAP + n;
        let mut inputs = vec![0.0f32; in_len * channels];
        for c in 0..channels {
            let input = &mut inputs[c * in_len..(c + 1) * in_len];
            input[..OVERLAP].copy_from_slice(&self.in_mem[c]);
            let mut mem = self.preemph_mem[c];
            for (dst, &s) in input[OVERLAP..].iter_mut().zip(pcm.iter().skip(c).step_by(channels)) {
                let s = s * 32768.0;
                *dst = s - mem;
                mem = PREEMPH_COEF * s;
            }
            self.preemph_mem[c] = mem;
        }

        // Pitch pre-filter: estimate the pitch, comb-filter `inputs` in place
        // to whiten the harmonic structure, and return the decision to code.
        // Runs before the transient analysis and MDCT, which both see the
        // filtered signal (matching the reference order).
        let (pf_on, pitch_index, qg) = self.prefilter_analysis(&mut inputs, in_len, n, channels, nb_bytes);

        // The MDCT overlap for the next frame is the filtered tail.
        for c in 0..channels {
            let input = &inputs[c * in_len..(c + 1) * in_len];
            self.in_mem[c].copy_from_slice(&input[n..n + OVERLAP]);
        }

        // Transient decision (`transient_analysis`); the flag needs 3 bits.
        let (is_transient, tf_estimate, tf_chan) = if lm > 0 && enc.tell() + 3 <= total_bits {
            transient_analysis(&inputs, in_len, channels)
        } else {
            (false, 0.0, 0)
        };

        // Forward MDCT(s) per channel, then band energies (log domain
        // relative to eMeans) and unit-norm band shapes. `x` is planar.
        // `band_log_e2` is the long-block variant for dynalloc; it equals
        // `band_log_e` for non-transient frames, otherwise comes from a
        // second long-block MDCT.
        let mut x = vec![0.0f32; n * channels];
        let mut band_e = [[0.0f32; NB_EBANDS]; 2];
        let mut band_log_e = [[0.0f32; NB_EBANDS]; 2];
        let mut band_log_e2 = [[0.0f32; NB_EBANDS]; 2];
        for c in 0..channels {
            let input = &inputs[c * in_len..(c + 1) * in_len];
            let mut freq = vec![0.0f32; n];
            if is_transient {
                // `m` short MDCTs, interleaved into `freq`.
                let sub = n / m;
                for b in 0..m {
                    self.mdct.forward(
                        &input[b * sub..b * sub + sub + OVERLAP],
                        &mut freq[b..],
                        &WINDOW120,
                        OVERLAP,
                        3,
                        m,
                    );
                }
                // Second long-block MDCT for `band_log_e2`.
                let mut freq2 = vec![0.0f32; n];
                self.mdct.forward(input, &mut freq2, &WINDOW120, OVERLAP, 3 - lm, 1);
                for i in 0..end {
                    let lo = m * EBANDS[i] as usize;
                    let hi = m * EBANDS[i + 1] as usize;
                    let mut sum = 1e-27f32;
                    for &v in &freq2[lo..hi] {
                        sum += v * v;
                    }
                    band_log_e2[c][i] = sum.sqrt().log2() - E_MEANS[i] + 0.5 * lm as f32;
                }
            } else {
                self.mdct.forward(input, &mut freq, &WINDOW120, OVERLAP, 3 - lm, 1);
            }

            let xc = &mut x[c * n..(c + 1) * n];
            for i in 0..end {
                let lo = m * EBANDS[i] as usize;
                let hi = m * EBANDS[i + 1] as usize;
                let mut sum = 1e-27f32;
                for &v in &freq[lo..hi] {
                    sum += v * v;
                }
                band_e[c][i] = sum.sqrt();
                band_log_e[c][i] = band_e[c][i].log2() - E_MEANS[i];
                if !is_transient {
                    band_log_e2[c][i] = band_log_e[c][i];
                }
                let g = 1.0 / (1e-27 + band_e[c][i]);
                for (xv, &f) in xc[lo..hi].iter_mut().zip(freq[lo..hi].iter()) {
                    *xv = f * g;
                }
            }
        }

        // Dynalloc analysis (uses the previous frame's energies, so it must
        // run before coarse-energy quantisation overwrites them); yields the
        // boost targets plus the importance/spread weights for tf and spread.
        let dyn_an = dynalloc_analysis(
            &band_log_e,
            &band_log_e2,
            &self.energy.old_ebands,
            start,
            end,
            channels,
            lm,
            nb_bytes,
            is_transient,
        );
        let boost_targets = dyn_an.offsets;

        // Per-band time/frequency resolution (`tf_analysis`), enabled above a
        // low byte threshold. The result is the raw 0/1 flags plus tf_select.
        let n0 = n;
        let (mut tf_res, tf_select) = if nb_bytes >= 15 * channels && lm > 0 {
            let lambda = 80.max(20480 / nb_bytes as i32 + 2);
            tf_analysis(
                end,
                is_transient,
                lambda,
                &x,
                n0,
                lm,
                tf_estimate,
                tf_chan,
                &dyn_an.importance,
            )
        } else {
            ([i32::from(is_transient); NB_EBANDS], usize::from(is_transient))
        };

        // The CBR-equivalent rate in bits/s, for trim and dynalloc tuning.
        let equiv_rate =
            (nb_bytes as i32) * 8 * 50 * (1 << (3 - lm)) - (40 * channels as i32 + 20) * ((400 >> lm) - 50);

        // --- Bitstream, in the decoder's exact order. ---
        // Silence flag.
        if enc.tell() + 15 <= total_bits {
            enc.encode_bit_logp(false, 15);
        }
        // Post-filter parameters. When on, the byte budget that enabled the
        // pre-filter guarantees the 16-bit gate holds, so they are coded
        // unconditionally (mirroring the reference); when off, the flag is
        // gated like the decoder's read.
        if start == 0 {
            if pf_on {
                enc.encode_bit_logp(true, 1);
                let p = (pitch_index + 1) as u32;
                // EC_ILOG(p) - 5, where EC_ILOG(p) = 32 - leading_zeros.
                let octave = (32 - p.leading_zeros()) as i32 - 5;
                enc.encode_uint(octave as u32, 6);
                enc.encode_raw_bits(p - (16 << octave), (4 + octave) as u32);
                enc.encode_raw_bits(qg as u32, 3);
                if enc.tell() + 2 <= total_bits {
                    enc.encode_icdf(self.prefilter_tapset, &TAPSET_ICDF, 2);
                }
            } else if enc.tell() + 16 <= total_bits {
                enc.encode_bit_logp(false, 1);
            }
        }
        // Transient flag.
        if lm > 0 && enc.tell() + 3 <= total_bits {
            enc.encode_bit_logp(is_transient, 3);
        }
        // Intra only on the first frame.
        let intra = self.frames == 0;
        if enc.tell() + 3 <= total_bits {
            enc.encode_bit_logp(intra, 3);
        }

        // Coarse energy.
        let mut error = [[0.0f32; NB_EBANDS]; 2];
        self.quant_coarse_energy(&mut enc, start, end, &band_log_e, &mut error, intra, lm, total_bits);

        // Time-frequency coding (`tf_encode`): code each band's flag as the
        // change from the previous band, code tf_select when it matters, and
        // map the flags through the table to per-band tf_change values.
        {
            let mut budget = total_bits;
            let mut tell = enc.tell();
            let mut logp: u32 = if is_transient { 2 } else { 4 };
            let tf_select_rsv = lm > 0 && tell + logp < budget;
            budget -= u32::from(tf_select_rsv);
            let mut tf_changed = false;
            let mut curr = 0i32;
            for r in tf_res.iter_mut().take(end).skip(start) {
                if tell + logp <= budget {
                    enc.encode_bit_logp(*r != curr, logp);
                    tell = enc.tell();
                    curr = *r;
                    tf_changed |= curr != 0;
                } else {
                    *r = curr;
                }
                logp = if is_transient { 4 } else { 5 };
            }
            let base = 4 * usize::from(is_transient);
            // Only code tf_select if it would make a difference.
            let tf_select = if tf_select_rsv
                && TF_SELECT_TABLE[lm][base + usize::from(tf_changed)]
                    != TF_SELECT_TABLE[lm][base + 2 + usize::from(tf_changed)]
            {
                enc.encode_bit_logp(tf_select != 0, 1);
                tf_select
            } else {
                0
            };
            for r in tf_res.iter_mut().take(end).skip(start) {
                *r = TF_SELECT_TABLE[lm][base + 2 * tf_select + *r as usize];
            }
        }

        // Spreading decision.
        let mut spread = Spread::Normal;
        if enc.tell() + 4 <= total_bits {
            let s = spreading_decision(
                &x,
                n0,
                &mut self.tonal_average,
                self.spread_decision,
                end,
                channels,
                m,
                &dyn_an.spread_weight,
            );
            self.spread_decision = s;
            spread = Spread::from_raw(s as u32);
            enc.encode_icdf(s as usize, &SPREAD_ICDF, 5);
        }

        // Dynamic allocation: code each band's boost as a run of `1` flags
        // (one per increment in `boost_targets`) terminated by a `0`,
        // mirroring the decoder. `offsets[i]` becomes the boost in 8th bits.
        let caps = init_caps(lm, channels);
        let mut offsets = [0i32; NB_EBANDS];
        let total_bits_frac = (total_bits << 3) as i64;
        let mut total_boost = 0i64;
        {
            let mut dynalloc_logp = 6u32;
            let mut tell_frac = i64::from(enc.tell_frac());
            for i in start..end {
                let width = (channels as i32 * i32::from(EBANDS[i + 1] - EBANDS[i])) << lm;
                // 6 bits, but no more than 1 bit/sample and at least 1/8.
                let quanta = (width << BITRES).min((6 << BITRES).max(width));
                let mut dynalloc_loop_logp = dynalloc_logp;
                let mut boost = 0i32;
                let mut j = 0i32;
                while tell_frac + (i64::from(dynalloc_loop_logp) << BITRES) < total_bits_frac - total_boost
                    && boost < caps[i]
                {
                    let flag = j < boost_targets[i];
                    enc.encode_bit_logp(flag, dynalloc_loop_logp);
                    tell_frac = i64::from(enc.tell_frac());
                    if !flag {
                        break;
                    }
                    boost += quanta;
                    total_boost += i64::from(quanta);
                    dynalloc_loop_logp = 1;
                    j += 1;
                }
                if j > 0 {
                    dynalloc_logp = 2.max(dynalloc_logp - 1);
                }
                offsets[i] = boost;
            }
        }

        // Allocation trim from the spectral tilt and transient estimate.
        // The budget gate must discount the dynalloc boost, exactly as the
        // decoder does (its `total_bits_frac` is decremented per boost).
        let trim = if i64::from(enc.tell_frac()) + (6 << 3) <= total_bits_frac - total_boost {
            let trim = alloc_trim_analysis(&band_log_e, end, channels, tf_estimate, equiv_rate);
            enc.encode_icdf(trim as usize, &TRIM_ICDF, 7);
            trim
        } else {
            5
        };

        // The implicit allocation (shared with the decoder). Stereo
        // decisions are conservative: intensity at `end` (full stereo in
        // every band; the allocator clamps it to the coded bands) and no
        // dual stereo.
        let mut bits = (((nb_bytes * 8) << 3) as i32) - enc.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm as i32 + 2) << 3) {
            1 << 3
        } else {
            0
        };
        bits -= anti_collapse_rsv;
        let alloc = compute_allocation(
            &mut AllocEc::Enc {
                enc: &mut enc,
                signal_bandwidth: end - 1,
                intensity: end,
                dual_stereo: false,
            },
            start,
            end,
            &offsets,
            &caps,
            trim,
            bits,
            channels,
            lm,
        );

        // Fine energy.
        self.quant_fine_energy(&mut enc, start, end, &mut error, &alloc.fine_quant);

        // Band shapes.
        let total = ((nb_bytes * 8) << 3) as i32 - anti_collapse_rsv;
        let (xs, ys) = x.split_at_mut(n);
        quant_all_bands_enc(
            &mut enc,
            start,
            end,
            xs,
            if channels == 2 { Some(ys) } else { None },
            &alloc.shape_bits,
            is_transient,
            spread,
            alloc.dual_stereo,
            alloc.intensity,
            &tf_res,
            total,
            alloc.balance,
            lm,
            alloc.coded_bands,
            &band_e,
        );

        // Anti-collapse: on unless this is a long transient run.
        if anti_collapse_rsv > 0 {
            enc.encode_raw_bits(u32::from(self.consec_transient < 2), 1);
        }

        // Finalise the leftover bits into extra fine energy.
        let bits_left = nb_bytes as i32 * 8 - enc.tell() as i32;
        self.quant_energy_finalise(
            &mut enc,
            start,
            end,
            &mut error,
            &alloc.fine_quant,
            &alloc.fine_priority,
            bits_left,
        );

        if is_transient {
            self.consec_transient += 1;
        } else {
            self.consec_transient = 0;
        }
        self.last_transient = is_transient;
        self.frames += 1;
        self.final_range = enc.range_size();
        enc.finalize().expect("budget enforced by construction")
    }

    /// The range state after the last encoded frame; a conformant decoder
    /// finishes the frame with this exact value.
    #[must_use]
    pub const fn final_range(&self) -> u32 {
        self.final_range
    }

    /// Whether the last frame was detected as a transient and coded with
    /// short blocks.
    #[must_use]
    pub const fn last_transient(&self) -> bool {
        self.last_transient
    }

    /// The pitch period (samples) and gain the pre-filter analysis chose for
    /// the last frame. Currently informational - the comb filter and its
    /// post-filter bitstream coding are not yet applied.
    #[must_use]
    pub const fn last_pitch(&self) -> (usize, f32) {
        (self.last_pitch, self.last_pitch_gain)
    }

    /// Pitch pre-filter (`run_prefilter`): estimates the pitch period/gain,
    /// applies the rate/continuity gain threshold and the 8-level gain
    /// quantisation, comb-filters `inputs` in place (the FIR whitening, with
    /// a windowed cross-fade from the previous frame's parameters), advances
    /// the pre-filter history, and returns `(pf_on, pitch_index, qg)` for the
    /// bitstream.
    #[allow(
        clippy::needless_range_loop,
        reason = "channel index also addresses prefilter_mem/inputs"
    )]
    fn prefilter_analysis(
        &mut self,
        inputs: &mut [f32],
        in_len: usize,
        n: usize,
        channels: usize,
        nb_bytes: usize,
    ) -> (bool, usize, i32) {
        // pre[c] = [prefilter history | this frame's new samples].
        let pre_len = COMBFILTER_MAXPERIOD + n;
        let mut pre = [vec![0.0f32; pre_len], vec![0.0f32; pre_len]];
        for c in 0..channels {
            pre[c][..COMBFILTER_MAXPERIOD].copy_from_slice(&self.prefilter_mem[c]);
            pre[c][COMBFILTER_MAXPERIOD..].copy_from_slice(&inputs[c * in_len + OVERLAP..c * in_len + OVERLAP + n]);
        }

        let enabled = nb_bytes > 12 * channels;
        let (pitch_index, mut gain1) = if enabled {
            let refs: Vec<&[f32]> = pre[..channels].iter().map(Vec::as_slice).collect();
            let x_lp = pitch_downsample(&refs, pre_len);
            let max_pitch = COMBFILTER_MAXPERIOD - 3 * COMBFILTER_MINPERIOD;
            let coarse = pitch_search(&x_lp[COMBFILTER_MAXPERIOD >> 1..], &x_lp, n, max_pitch);
            let mut pitch_index = COMBFILTER_MAXPERIOD - coarse;
            let (gain, refined) = remove_doubling(
                &x_lp,
                COMBFILTER_MAXPERIOD,
                COMBFILTER_MINPERIOD,
                n,
                pitch_index,
                self.prefilter_period,
                self.prefilter_gain,
            );
            pitch_index = refined.min(COMBFILTER_MAXPERIOD - 2);
            (pitch_index, 0.7 * gain)
        } else {
            (COMBFILTER_MINPERIOD, 0.0)
        };

        // Gain threshold for enabling the (post-)filter, adjusted by rate and
        // continuity.
        let mut pf_threshold = 0.2f32;
        if (pitch_index as i32 - self.prefilter_period as i32).abs() * 10 > pitch_index as i32 {
            pf_threshold += 0.2;
        }
        if nb_bytes < 25 {
            pf_threshold += 0.1;
        }
        if nb_bytes < 35 {
            pf_threshold += 0.1;
        }
        if self.prefilter_gain > 0.4 {
            pf_threshold -= 0.1;
        }
        if self.prefilter_gain > 0.55 {
            pf_threshold -= 0.1;
        }
        pf_threshold = pf_threshold.max(0.2);
        let mut pf_on = false;
        let mut qg = 0i32;
        if gain1 < pf_threshold {
            gain1 = 0.0;
        } else {
            // Snap to the previous gain to avoid needless changes, then
            // quantise to one of eight levels.
            if (gain1 - self.prefilter_gain).abs() < 0.1 {
                gain1 = self.prefilter_gain;
            }
            qg = ((0.5 + gain1 * 32.0 / 3.0).floor() as i32 - 1).clamp(0, 7);
            gain1 = 0.093_75 * (qg + 1) as f32;
            pf_on = true;
        }

        // Comb-filter the new samples in place: a FIR whitening reading the
        // unfiltered `pre` history, cross-fading from the previous frame's
        // (period, -gain) to this frame's over the overlap. With the standard
        // mode's `offset == 0` this is a single call over all `n` samples.
        let old_period = self.prefilter_period.max(COMBFILTER_MINPERIOD);
        let old_tapset = self.prefilter_tapset;
        let new_tapset = self.prefilter_tapset;
        for c in 0..channels {
            let dst = &mut inputs[c * in_len + OVERLAP..c * in_len + OVERLAP + n];
            comb_filter_prefilter(
                dst,
                &pre[c],
                COMBFILTER_MAXPERIOD,
                old_period,
                pitch_index.max(COMBFILTER_MINPERIOD),
                n,
                -self.prefilter_gain,
                -gain1,
                old_tapset,
                new_tapset,
            );
        }

        // Advance the pre-filter history (new = last MAXPERIOD of pre[c]).
        for c in 0..channels {
            self.prefilter_mem[c].copy_from_slice(&pre[c][n..n + COMBFILTER_MAXPERIOD]);
        }
        self.prefilter_period = pitch_index.max(COMBFILTER_MINPERIOD);
        self.prefilter_gain = gain1;
        self.last_pitch = pitch_index;
        self.last_pitch_gain = gain1;
        (pf_on, pitch_index, qg)
    }

    /// `quant_coarse_energy` (float build): time/frequency-predicted,
    /// Laplace-coded 6 dB energy quantisation, channel-interleaved per band.
    #[allow(clippy::too_many_arguments, reason = "mirrors quant_coarse_energy_impl")]
    fn quant_coarse_energy(
        &mut self,
        enc: &mut RangeEncoder,
        start: usize,
        end: usize,
        band_log_e: &[[f32; NB_EBANDS]; 2],
        error: &mut [[f32; NB_EBANDS]; 2],
        intra: bool,
        lm: usize,
        budget: u32,
    ) {
        let channels = self.channels;
        let prob = &E_PROB_MODEL[lm][usize::from(intra)];
        let (coef, beta) = if intra {
            (0.0, BETA_INTRA)
        } else {
            (PRED_COEF[lm], BETA_COEF[lm])
        };
        let max_decay = 16.0f32.min((budget as f32 / 3.0).max(0.0));

        let mut prev = [0.0f32; 2];
        for i in start..end {
            for c in 0..channels {
                let x = band_log_e[c][i];
                let old_e = self.energy.old_ebands[c][i].max(-9.0);
                let f = x - coef * old_e - prev[c];
                let mut qi = (0.5 + f).floor() as i32;
                let decay_bound = self.energy.old_ebands[c][i].max(-28.0) - max_decay;
                // Prevent energy from dropping too fast.
                if qi < 0 && x < decay_bound {
                    qi += (decay_bound - x) as i32;
                    if qi > 0 {
                        qi = 0;
                    }
                }
                let tell = enc.tell();
                let bits_left = budget as i32 - tell as i32 - 3 * channels as i32 * (end - i) as i32;
                if i != start && bits_left < 30 {
                    if bits_left < 24 {
                        qi = qi.min(1);
                    }
                    if bits_left < 16 {
                        qi = qi.max(-1);
                    }
                }
                let qi = if budget - tell >= 15 {
                    let pi = 2 * i.min(20);
                    ec_laplace_encode(enc, qi, u32::from(prob[pi]) << 7, u32::from(prob[pi + 1]) << 6)
                } else if budget - tell >= 2 {
                    let qi = qi.clamp(-1, 1);
                    const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];
                    enc.encode_icdf(((2 * qi) ^ -i32::from(qi < 0)) as usize, &SMALL_ENERGY_ICDF, 2);
                    qi
                } else if budget - tell >= 1 {
                    let qi = qi.min(0);
                    enc.encode_bit_logp(qi != 0, 1);
                    qi
                } else {
                    -1
                };
                error[c][i] = f - qi as f32;
                let q = qi as f32;
                self.energy.old_ebands[c][i] = coef * old_e + prev[c] + q;
                prev[c] = prev[c] + q - beta * q;
            }
        }
        if channels == 1 {
            self.energy.old_ebands[1] = self.energy.old_ebands[0];
        }
    }

    #[allow(
        clippy::needless_range_loop,
        reason = "channel indices mirror the reference's c loop"
    )]
    fn quant_fine_energy(
        &mut self,
        enc: &mut RangeEncoder,
        start: usize,
        end: usize,
        error: &mut [[f32; NB_EBANDS]; 2],
        fine_quant: &[i32; NB_EBANDS],
    ) {
        let channels = self.channels;
        for i in start..end {
            if fine_quant[i] <= 0 {
                continue;
            }
            let frac = 1 << fine_quant[i];
            for c in 0..channels {
                let q2 = (((error[c][i] + 0.5) * frac as f32).floor() as i32).clamp(0, frac - 1);
                enc.encode_raw_bits(q2 as u32, fine_quant[i] as u32);
                let offset = (q2 as f32 + 0.5) * (1 << (14 - fine_quant[i])) as f32 / 16384.0 - 0.5;
                self.energy.old_ebands[c][i] += offset;
                error[c][i] -= offset;
            }
        }
        if channels == 1 {
            self.energy.old_ebands[1] = self.energy.old_ebands[0];
        }
    }

    #[allow(clippy::too_many_arguments, reason = "mirrors quant_energy_finalise")]
    #[allow(
        clippy::needless_range_loop,
        reason = "channel indices mirror the reference's c loop"
    )]
    fn quant_energy_finalise(
        &mut self,
        enc: &mut RangeEncoder,
        start: usize,
        end: usize,
        error: &mut [[f32; NB_EBANDS]; 2],
        fine_quant: &[i32; NB_EBANDS],
        fine_priority: &[bool; NB_EBANDS],
        mut bits_left: i32,
    ) {
        let channels = self.channels;
        for prio in [false, true] {
            let mut i = start;
            while i < end && bits_left >= channels as i32 {
                if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                    i += 1;
                    continue;
                }
                for c in 0..channels {
                    let q2 = i32::from(error[c][i] >= 0.0);
                    enc.encode_raw_bits(q2 as u32, 1);
                    let offset = (q2 as f32 - 0.5) * (1 << (14 - fine_quant[i] - 1)) as f32 / 16384.0;
                    self.energy.old_ebands[c][i] += offset;
                    error[c][i] -= offset;
                    bits_left -= 1;
                }
                i += 1;
            }
        }
        if channels == 1 {
            self.energy.old_ebands[1] = self.energy.old_ebands[0];
        }
    }
}

/// `transient_analysis` (float build): high-pass the input, apply forward
/// (6.7 dB/ms) and backward (13.9 dB/ms) masking decays, and compare the
/// frame energy against the harmonic mean of the masked energy - a
/// bitrate-normalised temporal noise-to-mask ratio. `inputs` is the planar
/// per-channel pre-emphasised signal including the overlap (`len` samples
/// per channel). Returns `(is_transient, tf_estimate, tf_chan)`: the
/// transient flag, an arbitrary VBR/trim metric, and the channel with the
/// strongest transient (the one `tf_analysis` examines).
fn transient_analysis(inputs: &[f32], len: usize, channels: usize) -> (bool, f32, usize) {
    /// `inv_table`: 6*64/x, trained on real data to minimise average error.
    const INV_TABLE: [u8; 128] = [
        255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18, 17, 16, 16, 15, 15,
        14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
    ];
    const EPSILON: f32 = 1e-15;
    // Forward masking: 6.7 dB/ms.
    const FORWARD_DECAY: f32 = 0.0625;

    let len2 = len / 2;
    let mut mask_metric = 0i32;
    let mut tf_chan = 0usize;
    let mut tmp = vec![0.0f32; len];
    for c in 0..channels {
        let input = &inputs[c * len..(c + 1) * len];

        // High-pass filter: (1 - 2z^-1 + z^-2) / (1 - z^-1 + 0.5 z^-2).
        let (mut mem0, mut mem1) = (0.0f32, 0.0f32);
        for (t, &x) in tmp.iter_mut().zip(input.iter()) {
            let y = mem0 + x;
            let mem00 = mem0;
            mem0 = mem0 - x + 0.5 * mem1;
            mem1 = x - mem00;
            *t = y;
        }
        // The first few samples are bad: the memory is not propagated.
        tmp[..12].fill(0.0);

        // Forward pass for the post-echo threshold, grouping by two.
        let mut mean = 0.0f32;
        let mut mem = 0.0f32;
        for i in 0..len2 {
            let x2 = tmp[2 * i] * tmp[2 * i] + tmp[2 * i + 1] * tmp[2 * i + 1];
            mean += x2;
            mem = x2 + (1.0 - FORWARD_DECAY) * mem;
            tmp[i] = FORWARD_DECAY * mem;
        }

        // Backward pass for the pre-echo threshold.
        let mut mem = 0.0f32;
        let mut max_e = 0.0f32;
        for i in (0..len2).rev() {
            mem = tmp[i] + 0.875 * mem;
            tmp[i] = 0.125 * mem;
            max_e = max_e.max(0.125 * mem);
        }

        // Frame energy: the geometric mean of the energy and half the max.
        let mean = (mean * max_e * 0.5 * len2 as f32).sqrt();
        let norm = len2 as f32 / (EPSILON + mean);
        // Harmonic mean over 1/4 of the samples, away from the boundaries.
        let mut unmask = 0i32;
        let mut i = 12;
        while i + 5 < len2 {
            let id = (64.0 * norm * (tmp[i] + EPSILON)).floor().clamp(0.0, 127.0) as usize;
            unmask += i32::from(INV_TABLE[id]);
            i += 4;
        }
        // Normalise for the 1/4 sampling and the factor 6 in the table.
        let unmask = 64 * unmask * 4 / (6 * (len2 as i32 - 17));
        if unmask > mask_metric {
            tf_chan = c;
            mask_metric = unmask;
        }
    }
    let is_transient = mask_metric > 200;
    // Arbitrary metric for VBR boost and trim (float build).
    let tf_max = 0.0f32.max((27.0 * mask_metric as f32).sqrt() - 42.0);
    let tf_estimate = 0.0f32.max(0.0069 * 163.0f32.min(tf_max) - 0.139).sqrt();
    (is_transient, tf_estimate, tf_chan)
}

/// The pre-filter comb (`comb_filter`, encoder direction): a 3-tap FIR
/// whitening `dst[i] = src[base+i] + g·(taps of src around base+i-T)`,
/// cross-fading from `(t0, g0, tapset0)` to `(t1, g1, tapset1)` over the
/// MDCT overlap (windowed), then constant. `src` is the *unfiltered* history
/// plus the new samples, so this is non-recursive (unlike the decoder's
/// in-place post-filter). Gains are passed negated by the caller.
#[allow(clippy::too_many_arguments, reason = "mirrors the reference comb_filter signature")]
fn comb_filter_prefilter(
    dst: &mut [f32],
    src: &[f32],
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
        dst[..n].copy_from_slice(&src[base..base + n]);
        return;
    }
    let t0 = t0.max(COMBFILTER_MINPERIOD);
    let t1 = t1.max(COMBFILTER_MINPERIOD);
    let g00 = g0 * COMB_GAINS[tapset0][0];
    let g01 = g0 * COMB_GAINS[tapset0][1];
    let g02 = g0 * COMB_GAINS[tapset0][2];
    let g10 = g1 * COMB_GAINS[tapset1][0];
    let g11 = g1 * COMB_GAINS[tapset1][1];
    let g12 = g1 * COMB_GAINS[tapset1][2];

    // No cross-fade needed when nothing changed.
    let overlap = if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        0
    } else {
        OVERLAP.min(n)
    };
    let mut x1 = src[base + 1 - t1];
    let mut x2 = src[base - t1];
    let mut x3 = src[base - t1 - 1];
    let mut x4 = src[base - t1 - 2];

    let mut i = 0usize;
    while i < overlap {
        let x0 = src[base + i + 2 - t1];
        let f = WINDOW120[i] * WINDOW120[i];
        dst[i] = src[base + i]
            + (1.0 - f) * g00 * src[base + i - t0]
            + (1.0 - f) * g01 * (src[base + i + 1 - t0] + src[base + i - 1 - t0])
            + (1.0 - f) * g02 * (src[base + i + 2 - t0] + src[base + i - 2 - t0])
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
        dst[overlap..n].copy_from_slice(&src[base + overlap..base + n]);
        return;
    }
    while i < n {
        dst[i] = src[base + i]
            + g10 * src[base + i - t1]
            + g11 * (src[base + i + 1 - t1] + src[base + i - 1 - t1])
            + g12 * (src[base + i + 2 - t1] + src[base + i - 2 - t1]);
        i += 1;
    }
}

/// `dynalloc_analysis` (float build, non-LFE, no surround, no analysis
/// module): computes per-band boost *targets* (the count of dynalloc
/// increments to code) from a follower of the band energies. `band_log_e`
/// is the current frame's per-channel log energy, `band_log_e2` the
/// long-block variant (equal to `band_log_e` for non-transient frames),
/// and `old_ebands` the previous frame's energies. Boosts only kick in
/// once the budget is large enough.
/// The per-band analysis outputs shared with the allocator and the
/// spreading/tf decisions.
struct Dynalloc {
    /// Per-band boost *targets* (count of dynalloc increments to code).
    offsets: [i32; NB_EBANDS],
    /// Per-band perceptual importance (`importance`), for `tf_analysis`.
    importance: [i32; NB_EBANDS],
    /// Per-band spreading weight (`spread_weight`), for `spreading_decision`.
    spread_weight: [i32; NB_EBANDS],
}

#[allow(clippy::too_many_arguments, reason = "mirrors the reference signature")]
#[allow(clippy::needless_range_loop, reason = "band indices mirror the reference loops")]
fn dynalloc_analysis(
    band_log_e: &[[f32; NB_EBANDS]; 2],
    band_log_e2: &[[f32; NB_EBANDS]; 2],
    old_ebands: &[[f32; NB_EBANDS]; 2],
    start: usize,
    end: usize,
    channels: usize,
    lm: usize,
    effective_bytes: usize,
    is_transient: bool,
) -> Dynalloc {
    /// `lsb_depth` for float input.
    const LSB_DEPTH: f32 = 24.0;
    let mut offsets = [0i32; NB_EBANDS];
    let mut importance = [13i32; NB_EBANDS];
    let mut spread_weight = [32i32; NB_EBANDS];

    // Noise floor: eMeans, depth, band width and the pre-emphasis tilt.
    let mut noise_floor = [0.0f32; NB_EBANDS];
    for (i, nf) in noise_floor.iter_mut().enumerate().take(end) {
        *nf = 0.0625 * f32::from(LOG_N[i]) + 0.5 + (9.0 - LSB_DEPTH) - E_MEANS[i] + 0.0062 * ((i + 5) * (i + 5)) as f32;
    }

    // The depth of the loudest band above the noise floor.
    let mut max_depth = -31.9f32;
    for c in 0..channels {
        for i in 0..end {
            max_depth = max_depth.max(band_log_e[c][i] - noise_floor[i]);
        }
    }

    // A simple masking model giving each band a spreading weight, so the
    // spreading decision ignores fully masked bands.
    {
        let mut mask = [0.0f32; NB_EBANDS];
        let mut sig = [0.0f32; NB_EBANDS];
        for i in 0..end {
            mask[i] = band_log_e[0][i] - noise_floor[i];
            if channels == 2 {
                mask[i] = mask[i].max(band_log_e[1][i] - noise_floor[i]);
            }
            sig[i] = mask[i];
        }
        for i in 1..end {
            mask[i] = mask[i].max(mask[i - 1] - 2.0);
        }
        for i in (0..end - 1).rev() {
            mask[i] = mask[i].max(mask[i + 1] - 3.0);
        }
        for i in 0..end {
            // SMR is never more than 72 dB below the peak nor below the floor.
            let smr = sig[i] - 0.0f32.max(max_depth - 12.0).max(mask[i]);
            let shift = (-(0.5 + smr).floor() as i32).clamp(0, 5);
            spread_weight[i] = 32 >> shift;
        }
    }

    // The gate: enable at ~24 kb/s for 20 ms, ~96 kb/s for 2.5 ms.
    if effective_bytes < 30 + 5 * lm {
        return Dynalloc {
            offsets,
            importance,
            spread_weight,
        };
    }

    let mut follower = [[0.0f32; NB_EBANDS]; 2];
    for c in 0..channels {
        let mut e3 = [0.0f32; NB_EBANDS];
        e3[..end].copy_from_slice(&band_log_e2[c][..end]);
        if lm == 0 {
            // 2.5 ms: the first 8 bands have one bin (unreliable); take the
            // max with the previous energy so 2 bins contribute.
            for i in 0..end.min(8) {
                e3[i] = band_log_e2[c][i].max(old_ebands[c][i]);
            }
        }
        let f = &mut follower[c];
        f[0] = e3[0];
        let mut last = 0usize;
        for i in 1..end {
            // The last band at least 0.5 dB above the previous is the last
            // we consider (avoids problems on band-limited signals).
            if e3[i] > e3[i - 1] + 0.5 {
                last = i;
            }
            f[i] = (f[i - 1] + 1.5).min(e3[i]);
        }
        for i in (0..last).rev() {
            f[i] = f[i].min((f[i + 1] + 2.0).min(e3[i]));
        }
        // A median filter avoids triggering dynalloc unnecessarily.
        const OFFSET: f32 = 1.0;
        for i in 2..end.saturating_sub(2) {
            f[i] = f[i].max(median_of_5(&e3[i - 2..i + 3]) - OFFSET);
        }
        let tmp = median_of_3(&e3[0..3]) - OFFSET;
        f[0] = f[0].max(tmp);
        f[1] = f[1].max(tmp);
        let tmp = median_of_3(&e3[end - 3..end]) - OFFSET;
        f[end - 2] = f[end - 2].max(tmp);
        f[end - 1] = f[end - 1].max(tmp);
        for i in 0..end {
            f[i] = f[i].max(noise_floor[i]);
        }
    }

    if channels == 2 {
        for i in start..end {
            // Consider 24 dB of cross-talk between channels.
            let (l, r) = (follower[0][i], follower[1][i]);
            follower[1][i] = r.max(l - 4.0);
            follower[0][i] = l.max(r - 4.0);
            follower[0][i] =
                0.5 * (0.0f32.max(band_log_e[0][i] - follower[0][i]) + 0.0f32.max(band_log_e[1][i] - follower[1][i]));
        }
    } else {
        for i in start..end {
            follower[0][i] = 0.0f32.max(band_log_e[0][i] - follower[0][i]);
        }
    }

    // Perceptual importance (before the dynalloc-specific scaling below).
    for i in start..end {
        importance[i] = (0.5 + 13.0 * celt_exp2(follower[0][i].min(4.0))).floor() as i32;
    }

    // For non-transient CBR frames, halve the dynalloc contribution.
    if !is_transient {
        for i in start..end {
            follower[0][i] *= 0.5;
        }
    }
    for i in start..end {
        if i < 8 {
            follower[0][i] *= 2.0;
        }
        if i >= 12 {
            follower[0][i] *= 0.5;
        }
    }

    let mut tot_boost = 0i32;
    let caps = init_caps(lm, channels);
    for i in start..end {
        follower[0][i] = follower[0][i].min(4.0);
        let width = (channels as i32 * i32::from(EBANDS[i + 1] - EBANDS[i])) << lm;
        let (boost, boost_bits) = if width < 6 {
            let boost = follower[0][i] as i32;
            (boost, (boost * width) << BITRES)
        } else if width > 48 {
            let boost = (follower[0][i] * 8.0) as i32;
            (boost, ((boost * width) << BITRES) / 8)
        } else {
            let boost = (follower[0][i] * width as f32 / 6.0) as i32;
            (boost, (boost * 6) << BITRES)
        };
        // For CBR, limit dynalloc to 2/3 of the bits.
        if (tot_boost + boost_bits) >> BITRES >> 3 > 2 * effective_bytes as i32 / 3 {
            let cap = (2 * effective_bytes as i32 / 3) << BITRES << 3;
            offsets[i] = cap - tot_boost;
            break;
        }
        offsets[i] = boost.min(caps[i]);
        tot_boost += boost_bits;
    }
    Dynalloc {
        offsets,
        importance,
        spread_weight,
    }
}

/// `celt_exp2` (float build): the reference's polynomial 2^x approximation,
/// reproduced so `importance` matches the reference bit-for-bit-ish.
#[allow(clippy::excessive_precision, reason = "verbatim reference polynomial constants")]
fn celt_exp2(x: f32) -> f32 {
    let integer = x.floor();
    if integer < -50.0 {
        return 0.0;
    }
    let frac = x - integer;
    let res = 0.999_925_22 + frac * (0.695_833_54 + frac * (0.226_067_16 + 0.078_024_52 * frac));
    // Scale by 2^integer via the IEEE-754 exponent field.
    let bits = (res.to_bits() as i32 + ((integer as i32) << 23)) & 0x7fff_ffff;
    f32::from_bits(bits as u32)
}

/// `median_of_5` (encoder helper).
fn median_of_5(x: &[f32]) -> f32 {
    let (t0, t1) = if x[0] > x[1] { (x[1], x[0]) } else { (x[0], x[1]) };
    let (t3, t4) = if x[3] > x[4] { (x[4], x[3]) } else { (x[3], x[4]) };
    let (_t0, t1, t3, t4) = if t0 > t3 { (t3, t4, t0, t1) } else { (t0, t1, t3, t4) };
    let t2 = x[2];
    if t2 > t1 {
        if t1 < t3 { t2.min(t3) } else { t4.min(t1) }
    } else if t2 < t3 {
        t1.min(t3)
    } else {
        t2.min(t4)
    }
}

/// `median_of_3` (encoder helper).
fn median_of_3(x: &[f32]) -> f32 {
    let (t0, t1) = if x[0] > x[1] { (x[1], x[0]) } else { (x[0], x[1]) };
    let t2 = x[2];
    if t1 < t2 {
        t1
    } else if t0 < t2 {
        t2
    } else {
        t0
    }
}

/// `alloc_trim_analysis` (float build, mono, no analysis module): tilts the
/// allocation by spectral slope, transient estimate and bitrate. Returns
/// the trim index 0..=10.
#[allow(clippy::needless_range_loop, reason = "band indices mirror the reference loop")]
fn alloc_trim_analysis(
    band_log_e: &[[f32; NB_EBANDS]; 2],
    end: usize,
    channels: usize,
    tf_estimate: f32,
    equiv_rate: i32,
) -> i32 {
    // At low bitrate, a lower trim helps.
    let mut trim: f32 = if equiv_rate < 64_000 {
        4.0
    } else if equiv_rate < 80_000 {
        4.0 + (1.0 / 16.0) * ((equiv_rate - 64_000) >> 10) as f32
    } else {
        5.0
    };

    // Spectral tilt across the bands.
    let mut diff = 0.0f32;
    for c in 0..channels {
        for i in 0..end - 1 {
            diff += band_log_e[c][i] * (2 + 2 * i as i32 - end as i32) as f32;
        }
    }
    diff /= (channels * (end - 1)) as f32;
    trim -= (-2.0f32).max(2.0f32.min((diff + 1.0) / 6.0));
    trim -= 2.0 * tf_estimate;

    (trim + 0.5).floor().clamp(0.0, 10.0) as i32
}

/// `spreading_decision` (float build, no analysis module, `update_hf`
/// disabled since there is no post-filter): chooses the PVQ spreading from a
/// rough CDF of the normalised band shapes, weighted by `spread_weight`, and
/// recursively averaged in `tonal_average` with hysteresis from
/// `last_decision`. `x` is the planar normalised spectrum.
#[allow(clippy::too_many_arguments, reason = "mirrors the reference signature")]
fn spreading_decision(
    x: &[f32],
    n0: usize,
    tonal_average: &mut i32,
    last_decision: i32,
    end: usize,
    channels: usize,
    m: usize,
    spread_weight: &[i32; NB_EBANDS],
) -> i32 {
    if m * (EBANDS[end] - EBANDS[end - 1]) as usize <= 8 {
        return Spread::None as i32;
    }
    let mut sum = 0i32;
    let mut nb_bands = 0i32;
    for c in 0..channels {
        for i in 0..end {
            let lo = c * n0 + m * EBANDS[i] as usize;
            let n = m * (EBANDS[i + 1] - EBANDS[i]) as usize;
            if n <= 8 {
                continue;
            }
            let band = &x[lo..lo + n];
            // Rough CDF of |x[j]| via three energy thresholds.
            let mut tcount = [0i32; 3];
            for &v in band {
                let x2n = v * v * n as f32;
                tcount[0] += i32::from(x2n < 0.25);
                tcount[1] += i32::from(x2n < 0.0625);
                tcount[2] += i32::from(x2n < 0.015_625);
            }
            let nn = n as i32;
            let tmp = i32::from(2 * tcount[2] >= nn) + i32::from(2 * tcount[1] >= nn) + i32::from(2 * tcount[0] >= nn);
            sum += tmp * spread_weight[i];
            nb_bands += spread_weight[i];
        }
    }
    let nb_bands = nb_bands.max(1);
    let mut sum = (sum << 8) / nb_bands;
    // Recursive averaging, then hysteresis around the previous decision.
    sum = (sum + *tonal_average) >> 1;
    *tonal_average = sum;
    sum = (3 * sum + (((3 - last_decision) << 7) + 64) + 2) >> 2;
    if sum < 80 {
        Spread::Aggressive as i32
    } else if sum < 256 {
        Spread::Normal as i32
    } else if sum < 384 {
        Spread::Light as i32
    } else {
        Spread::None as i32
    }
}

/// L1 norm of a band, biased toward good frequency resolution (`l1_metric`).
fn l1_metric(tmp: &[f32], lm: i32, bias: f32) -> f32 {
    let l1: f32 = tmp.iter().map(|v| v.abs()).sum();
    l1 + l1 * (lm as f32 * bias)
}

/// `tf_analysis` (float build): per band, find the time/frequency split that
/// minimises the L1 metric (`metric`), then a Viterbi pass weighted by
/// `importance` and the switching cost `lambda` chooses the per-band tf
/// resolution flags and `tf_select`. Returns `(tf_res, tf_select)` with the
/// raw 0/1 flags (the caller maps them through `TF_SELECT_TABLE`).
#[allow(clippy::too_many_arguments, reason = "mirrors the reference signature")]
#[allow(clippy::needless_range_loop, reason = "band indices mirror the reference loops")]
fn tf_analysis(
    end: usize,
    is_transient: bool,
    lambda: i32,
    x: &[f32],
    n0: usize,
    lm: usize,
    tf_estimate: f32,
    tf_chan: usize,
    importance: &[i32; NB_EBANDS],
) -> ([i32; NB_EBANDS], usize) {
    let bias = 0.04 * (-0.25f32).max(0.5 - tf_estimate);
    let mut metric = [0i32; NB_EBANDS];
    let mut path0 = [0i32; NB_EBANDS];
    let mut path1 = [0i32; NB_EBANDS];

    for i in 0..end {
        let n = (EBANDS[i + 1] - EBANDS[i]) as usize * (1 << lm);
        let narrow = (EBANDS[i + 1] - EBANDS[i]) == 1;
        let lo = tf_chan * n0 + (EBANDS[i] as usize) * (1 << lm);
        let mut tmp = x[lo..lo + n].to_vec();
        let mut best_l1 = l1_metric(&tmp, if is_transient { lm as i32 } else { 0 }, bias);
        let mut best_level = 0i32;
        // The -1 (recombine) case for transients.
        if is_transient && !narrow {
            let mut tmp1 = tmp.clone();
            haar1(&mut tmp1, n >> lm, 1 << lm);
            let l1 = l1_metric(&tmp1, lm as i32 + 1, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = -1;
            }
        }
        let levels = lm + usize::from(!(is_transient || narrow));
        for k in 0..levels {
            let b = if is_transient {
                lm as i32 - k as i32 - 1
            } else {
                k as i32 + 1
            };
            haar1(&mut tmp, n >> k, 1 << k);
            let l1 = l1_metric(&tmp, b, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = k as i32 + 1;
            }
        }
        // Q1 metric so a narrow band can sit at the -0.5 mid-point.
        metric[i] = if is_transient { 2 * best_level } else { -2 * best_level };
        if narrow && (metric[i] == 0 || metric[i] == -2 * lm as i32) {
            metric[i] -= 1;
        }
    }

    let tf_tab = &TF_SELECT_TABLE[lm];
    let base = 4 * usize::from(is_transient);
    let cost = |sel: usize, flag: usize, i: usize| -> i32 {
        importance[i] * (metric[i] - 2 * tf_tab[base + 2 * sel + flag]).abs()
    };

    // Pick tf_select by comparing the two candidate tables' total cost.
    let mut selcost = [0i32; 2];
    for sel in 0..2 {
        let mut cost0 = cost(sel, 0, 0);
        let mut cost1 = cost(sel, 1, 0) + if is_transient { 0 } else { lambda };
        for i in 1..end {
            let curr0 = cost0.min(cost1 + lambda);
            let curr1 = (cost0 + lambda).min(cost1);
            cost0 = curr0 + cost(sel, 0, i);
            cost1 = curr1 + cost(sel, 1, i);
        }
        selcost[sel] = cost0.min(cost1);
    }
    // Only allow tf_select=1 for transients (the reference's conservatism).
    let tf_select = usize::from(selcost[1] < selcost[0] && is_transient);

    // Viterbi forward pass recording the back-pointers.
    let mut cost0 = cost(tf_select, 0, 0);
    let mut cost1 = cost(tf_select, 1, 0) + if is_transient { 0 } else { lambda };
    for i in 1..end {
        let (curr0, p0) = if cost0 < cost1 + lambda {
            (cost0, 0)
        } else {
            (cost1 + lambda, 1)
        };
        let (curr1, p1) = if cost0 + lambda < cost1 {
            (cost0 + lambda, 0)
        } else {
            (cost1, 1)
        };
        path0[i] = p0;
        path1[i] = p1;
        cost0 = curr0 + cost(tf_select, 0, i);
        cost1 = curr1 + cost(tf_select, 1, i);
    }
    let mut tf_res = [0i32; NB_EBANDS];
    tf_res[end - 1] = i32::from(cost0 >= cost1);
    for i in (0..end - 1).rev() {
        tf_res[i] = if tf_res[i + 1] == 1 { path1[i + 1] } else { path0[i + 1] };
    }
    (tf_res, tf_select)
}

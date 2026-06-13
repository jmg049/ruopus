//! A CELT encoder (RFC 6716 §5.3; normative `celt_encoder.c`,
//! `quant_bands.c` encoder paths): mono or stereo, with transient
//! detection and short blocks.
//!
//! Encoder *decisions* are deliberately conservative - transient detection
//! but per-band tf flags all zero, no post-filter, no dynamic allocation
//! boosts, default trim, normal spreading, full stereo (no intensity
//! collapse, no dual stereo) - every one of which is a legal choice, so
//! the bitstream is fully conformant; quality-improving analysis lands
//! incrementally. The bit-exact machinery (energy quantisation,
//! allocation, theta splits, PVQ) mirrors the decoder's exactly.

use alloc::vec;
use alloc::vec::Vec;

use crate::range::RangeEncoder;

use super::bands::encode::quant_all_bands_enc;
use super::decoder::TF_SELECT_TABLE;
use super::energy::EnergyState;
use super::laplace::ec_laplace_encode;
use super::mdct::MdctLookup;
use super::modes::{BETA_COEF, BETA_INTRA, E_MEANS, E_PROB_MODEL, EBANDS, LOG_N, MAX_FINE_BITS, NB_EBANDS, PRED_COEF};
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
            self.in_mem[c].copy_from_slice(&input[n..n + OVERLAP]);
        }

        // Transient decision (`transient_analysis`); the flag needs 3 bits.
        let (is_transient, tf_estimate) = if lm > 0 && enc.tell() + 3 <= total_bits {
            transient_analysis(&inputs, in_len, channels)
        } else {
            (false, 0.0)
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

        // Dynalloc boost targets (uses the previous frame's energies, so
        // it must run before coarse-energy quantisation overwrites them).
        let boost_targets = dynalloc_analysis(
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

        // The CBR-equivalent rate in bits/s, for trim and dynalloc tuning.
        let equiv_rate =
            (nb_bytes as i32) * 8 * 50 * (1 << (3 - lm)) - (40 * channels as i32 + 20) * ((400 >> lm) - 50);

        // --- Bitstream, in the decoder's exact order. ---
        // Silence flag.
        if enc.tell() + 15 <= total_bits {
            enc.encode_bit_logp(false, 15);
        }
        // Post-filter off.
        if start == 0 && enc.tell() + 16 <= total_bits {
            enc.encode_bit_logp(false, 1);
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

        // Time-frequency: no per-band changes (`tf_encode` with all-zero
        // flags and tf_select 0); the effective per-band tf_change still
        // comes from the table - 3 for a default transient frame.
        let tf_res = {
            let mut budget = total_bits;
            let mut tell = enc.tell();
            let mut logp: u32 = if is_transient { 2 } else { 4 };
            let tf_select_rsv = lm > 0 && tell + logp < budget;
            budget -= u32::from(tf_select_rsv);
            for _ in start..end {
                if tell + logp <= budget {
                    enc.encode_bit_logp(false, logp);
                    tell = enc.tell();
                }
                logp = if is_transient { 4 } else { 5 };
            }
            // tf_select is only coded when the two candidate tables differ
            // for the coded flags (tf_changed == 0 here).
            let base = 4 * usize::from(is_transient);
            if tf_select_rsv && TF_SELECT_TABLE[lm][base] != TF_SELECT_TABLE[lm][base + 2] {
                enc.encode_bit_logp(false, 1);
            }
            let mut tf_res = [0i32; NB_EBANDS];
            for r in tf_res.iter_mut().take(end).skip(start) {
                *r = TF_SELECT_TABLE[lm][base];
            }
            tf_res
        };

        // Spreading: normal.
        if enc.tell() + 4 <= total_bits {
            enc.encode_icdf(Spread::Normal as usize, &SPREAD_ICDF, 5);
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
            Spread::Normal,
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
/// per channel). Returns `(is_transient, tf_estimate)`, the latter an
/// arbitrary VBR/trim metric derived from the mask metric.
fn transient_analysis(inputs: &[f32], len: usize, channels: usize) -> (bool, f32) {
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
        mask_metric = mask_metric.max(unmask);
    }
    let is_transient = mask_metric > 200;
    // Arbitrary metric for VBR boost and trim (float build).
    let tf_max = 0.0f32.max((27.0 * mask_metric as f32).sqrt() - 42.0);
    let tf_estimate = 0.0f32.max(0.0069 * 163.0f32.min(tf_max) - 0.139).sqrt();
    (is_transient, tf_estimate)
}

/// `dynalloc_analysis` (float build, non-LFE, no surround, no analysis
/// module): computes per-band boost *targets* (the count of dynalloc
/// increments to code) from a follower of the band energies. `band_log_e`
/// is the current frame's per-channel log energy, `band_log_e2` the
/// long-block variant (equal to `band_log_e` for non-transient frames),
/// and `old_ebands` the previous frame's energies. Boosts only kick in
/// once the budget is large enough.
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
) -> [i32; NB_EBANDS] {
    /// `lsb_depth` for float input.
    const LSB_DEPTH: f32 = 24.0;
    let mut offsets = [0i32; NB_EBANDS];

    // Noise floor: eMeans, depth, band width and the pre-emphasis tilt.
    let mut noise_floor = [0.0f32; NB_EBANDS];
    for (i, nf) in noise_floor.iter_mut().enumerate().take(end) {
        *nf = 0.0625 * f32::from(LOG_N[i]) + 0.5 + (9.0 - LSB_DEPTH) - E_MEANS[i] + 0.0062 * ((i + 5) * (i + 5)) as f32;
    }

    // The gate: enable at ~24 kb/s for 20 ms, ~96 kb/s for 2.5 ms.
    if effective_bytes < 30 + 5 * lm {
        return offsets;
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
    offsets
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

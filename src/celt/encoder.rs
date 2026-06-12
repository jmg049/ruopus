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
use super::modes::{BETA_COEF, BETA_INTRA, E_MEANS, E_PROB_MODEL, EBANDS, MAX_FINE_BITS, NB_EBANDS, PRED_COEF};
use super::rate::{AllocEc, compute_allocation, init_caps};
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
        let is_transient = lm > 0 && enc.tell() + 3 <= total_bits && transient_analysis(&inputs, in_len, channels);

        // Forward MDCT(s) per channel, then band energies (log domain
        // relative to eMeans) and unit-norm band shapes. `x` is planar.
        let mut x = vec![0.0f32; n * channels];
        let mut band_e = [[0.0f32; NB_EBANDS]; 2];
        let mut band_log_e = [[0.0f32; NB_EBANDS]; 2];
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
                let g = 1.0 / (1e-27 + band_e[c][i]);
                for (xv, &f) in xc[lo..hi].iter_mut().zip(freq[lo..hi].iter()) {
                    *xv = f * g;
                }
            }
        }

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

        // Dynamic allocation: no boosts (one zero flag per band while the
        // budget allows).
        let caps = init_caps(lm, channels);
        {
            let dynalloc_logp = 6u32;
            let total_bits_frac = (total_bits << 3) as i32;
            let mut tell_frac = enc.tell_frac() as i32;
            for &cap in caps.iter().take(end).skip(start) {
                if tell_frac + ((dynalloc_logp << 3) as i32) < total_bits_frac && 0 < cap {
                    enc.encode_bit_logp(false, dynalloc_logp);
                    tell_frac = enc.tell_frac() as i32;
                }
            }
        }

        // Allocation trim: the neutral default.
        let trim = 5usize;
        if enc.tell_frac() + (6 << 3) <= total_bits << 3 {
            enc.encode_icdf(trim, &TRIM_ICDF, 7);
        }

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
        let offsets = [0i32; NB_EBANDS];
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
            trim as i32,
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
/// per channel).
fn transient_analysis(inputs: &[f32], len: usize, channels: usize) -> bool {
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
    mask_metric > 200
}

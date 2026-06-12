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
const TF_SELECT_TABLE: [[i32; 8]; 4] = [
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

        let alloc = compute_allocation(dec, start, end, &offsets, &caps, alloc_trim, bits, c, lm);

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

        // Synthesis: denormalise into the signal domain.
        let mut freq = vec![0.0f32; cc.max(c) * n];
        if silence {
            for ch in &mut self.energy.old_ebands {
                ch.fill(-28.0);
            }
        } else {
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
        }

        // Shift the synthesis history.
        for mem in &mut self.decode_mem {
            mem.copy_within(n..n + (DECODE_BUFFER_SIZE - n + OVERLAP / 2), 0);
        }

        // Zero the uncoded spectrum top.
        let bound = m * EBANDS[eff_end] as usize;
        for ch in 0..c {
            for i in bound..n {
                freq[ch * n + i] = 0.0;
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
                    self.background_log_e[ch][i] =
                        (self.background_log_e[ch][i] + m as f32 * 0.001).min(self.energy.old_ebands[ch][i]);
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
        let mut pcm = vec![0.0f32; n * cc];
        for ch in 0..cc {
            let mut mem = self.preemph_mem[ch];
            let x = &self.decode_mem[ch][out_base..out_base + n];
            for (j, &v) in x.iter().enumerate() {
                let tmp = v + mem + 1e-30;
                mem = PREEMPH_COEF * tmp;
                pcm[j * cc + ch] = tmp * (1.0 / 32768.0);
            }
            self.preemph_mem[ch] = mem;
        }

        if c == 1 {
            self.energy.old_ebands[1] = self.energy.old_ebands[0];
        }

        pcm
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

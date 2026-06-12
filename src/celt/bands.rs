//! Band shape decoding (RFC 6716 §4.3.4; normative `bands.c`, float build,
//! decoder paths only).
//!
//! [`quant_all_bands`] walks the coded bands, converting each band's bit
//! budget into PVQ pulses via recursive splitting: large partitions are halved
//! with an entropy-coded angle `theta` apportioning energy between the
//! halves, until a leaf partition is small enough for one PVQ codeword.
//! Stereo bands split into mid/side the same way ([`quant_band_stereo`]'s
//! `theta`), with intensity and dual-stereo handling from the allocation.
//! Bands with no pulses are *folded*: filled from earlier decoded spectrum
//! (or an LCG noise generator) and renormalised.
//!
//! Several quantities here must be **bit-exact** even in the float build -
//! [`bitexact_cos`], [`bitexact_log2tan`], the theta PDFs, and all budget
//! bookkeeping - because they steer the shared encoder/decoder allocation.
//!
//! # Differences from the reference, none observable
//!
//! - The folding source (`lowband`) is copied per band instead of aliasing
//!   the caller's buffers; all writes to the folding history happen after
//!   every read, so values are identical.
//! - The `i >= effEBands` spill path is omitted: the standard Opus mode has
//!   `effEBands == nbEBands`, so it is unreachable (custom modes are not part
//!   of RFC 6716).

use alloc::vec;
use alloc::vec::Vec;

use crate::range::RangeDecoder;

use super::modes::{EBANDS, LOG_N, NB_EBANDS};
use super::rate::{BITRES, bits2pulses, get_pulses, pulses2bits};
use super::vq::{Spread, alg_unquant, renormalise_vector};

/// `1.0` in the reference's Q15 norm scale; the float build uses plain 1.0.
const Q15_ONE: f32 = 1.0;

/// `NORM_SCALING` in the float build.
const NORM_SCALING: f32 = 1.0;

/// theta resolution offsets (`QTHETA_OFFSET`, `QTHETA_OFFSET_TWOPHASE`).
const QTHETA_OFFSET: i32 = 4;
const QTHETA_OFFSET_TWOPHASE: i32 = 16;

/// The linear congruential generator used for spectral noise filling
/// (`celt_lcg_rand`).
#[must_use]
pub const fn celt_lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

/// `(a*b + 16384) >> 15` on 16-bit values (`FRAC_MUL16`).
const fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + (a as i16 as i32) * (b as i16 as i32)) >> 15
}

/// Bit-exact cosine approximation steering the allocation (`bitexact_cos`).
#[must_use]
pub fn bitexact_cos(x: i16) -> i16 {
    let tmp = (4096 + i32::from(x) * i32::from(x)) >> 13;
    debug_assert!(tmp <= 32767);
    let mut x2 = tmp;
    x2 = (32767 - x2) + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2)));
    debug_assert!(x2 <= 32766);
    (1 + x2) as i16
}

/// Bit-exact `log2(tan)` approximation (`bitexact_log2tan`).
#[must_use]
pub fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = 32 - icos.leading_zeros() as i32;
    let ls = 32 - isin.leading_zeros() as i32;
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// Integer square root (`isqrt32`).
fn isqrt32(mut val: u32) -> u32 {
    if val == 0 {
        return 0;
    }
    let mut g = 0u32;
    let mut bshift = (31 - val.leading_zeros() as i32) >> 1;
    let mut b = 1u32 << bshift;
    while bshift >= 0 {
        let t = ((g << 1) + b) << bshift;
        if t <= val {
            g += b;
            val -= t;
        }
        b >>= 1;
        bshift -= 1;
    }
    g
}

/// In-place Haar transform across `stride`-interleaved pairs (`haar1`).
fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    const INV_SQRT2: f32 = core::f32::consts::FRAC_1_SQRT_2;
    let n0 = n0 >> 1;
    for i in 0..stride {
        for j in 0..n0 {
            let a = stride * 2 * j + i;
            let b = stride * (2 * j + 1) + i;
            let tmp1 = INV_SQRT2 * x[a];
            let tmp2 = INV_SQRT2 * x[b];
            x[a] = tmp1 + tmp2;
            x[b] = tmp1 - tmp2;
        }
    }
}

/// Hadamard reordering table (`ordery_table`), rows for stride 2, 4, 8, 16.
const ORDERY_TABLE: [usize; 30] = [
    1, 0, // stride 2
    3, 0, 2, 1, // stride 4
    7, 0, 4, 3, 6, 1, 5, 2, // stride 8
    15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5, // stride 16
];

fn ordery(stride: usize) -> &'static [usize] {
    let off = stride - 2;
    &ORDERY_TABLE[off..off + stride]
}

/// `deinterleave_hadamard`: frequency order → time order.
fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    if hadamard {
        let ord = ordery(stride);
        for i in 0..stride {
            for j in 0..n0 {
                tmp[ord[i] * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

/// `interleave_hadamard`: time order → frequency order.
fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    if hadamard {
        let ord = ordery(stride);
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ord[i] * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

/// Mid/side → left/right reconstruction (`stereo_merge`).
fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32) {
    let mut xp = 0.0f32;
    let mut side = 0.0f32;
    for (xi, yi) in x.iter().zip(y.iter()) {
        xp += xi * yi;
        side += yi * yi;
    }
    // Compensate for the mid normalisation.
    let xp = mid * xp;
    let mid2 = mid;
    let el = mid2 * mid2 + side - 2.0 * xp;
    let er = mid2 * mid2 + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y.copy_from_slice(x);
        return;
    }

    let lgain = 1.0 / el.sqrt();
    let rgain = 1.0 / er.sqrt();
    for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
        let l = mid * *xi;
        let r = *yi;
        *xi = lgain * (l - r);
        *yi = rgain * (l + r);
    }
}

/// `compute_qn`: the number of theta quantisation levels for a split.
fn compute_qn(n: usize, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    const EXP2_TABLE8: [i16; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = 2 * n as i32 - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let qb = (b + n2 * offset) / n2;
    let qb = qb.min(b - pulse_cap - (4 << BITRES));
    let qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let qn = i32::from(EXP2_TABLE8[(qb & 0x7) as usize]) >> (14 - (qb >> BITRES));
        ((qn + 1) >> 1) << 1
    }
}

/// Per-frame decoding state threaded through the band recursion
/// (`band_ctx`).
pub struct BandCtx {
    /// Current band index.
    i: usize,
    /// First intensity-stereo band.
    intensity: usize,
    /// Spreading decision for the frame.
    spread: Spread,
    /// Time-frequency resolution change for the current band.
    tf_change: i32,
    /// Remaining bits in the frame (1/8 bit units).
    remaining_bits: i32,
    /// Noise-fill LCG state, carried across frames.
    seed: u32,
}

/// Result of theta decoding for a split (`split_ctx`).
struct SplitCtx {
    inv: bool,
    imid: i32,
    iside: i32,
    delta: i32,
    itheta: i32,
    qalloc: i32,
}

/// Decodes the angle splitting a partition in two (`compute_theta`,
/// decoder side).
#[allow(clippy::too_many_arguments, reason = "mirrors the reference compute_theta signature")]
fn compute_theta(
    ctx: &mut BandCtx,
    dec: &mut RangeDecoder,
    n: usize,
    b: &mut i32,
    big_b: usize,
    b0: usize,
    lm: i32,
    stereo: bool,
    fill: &mut u32,
) -> SplitCtx {
    let i = ctx.i;

    // Resolution of the angle: more bits for larger partitions.
    let pulse_cap = i32::from(LOG_N[i]) + lm * (1 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);
    if stereo && i >= ctx.intensity {
        qn = 1;
    }

    let tell = dec.tell_frac() as i32;
    let mut itheta = 0i32;
    let mut inv = false;

    if qn != 1 {
        // Entropy decoding of the angle: a step PDF for stereo, uniform for
        // the time split, triangular otherwise.
        if stereo && n > 2 {
            let p0 = 3i32;
            let x0 = qn / 2;
            let ft = (p0 * (x0 + 1) + x0) as u32;
            let fs = dec.decode(ft) as i32;
            let x = if fs < (x0 + 1) * p0 {
                fs / p0
            } else {
                x0 + 1 + (fs - (x0 + 1) * p0)
            };
            let (fl, fh) = if x <= x0 {
                (p0 * x, p0 * (x + 1))
            } else {
                ((x - 1 - x0) + (x0 + 1) * p0, (x - x0) + (x0 + 1) * p0)
            };
            dec.update(fl as u32, fh as u32, ft);
            itheta = x;
        } else if b0 > 1 || stereo {
            // Uniform PDF.
            itheta = dec.decode_uint(qn as u32 + 1).unwrap_or(0) as i32;
        } else {
            // Triangular PDF.
            let half = qn >> 1;
            let ft = ((half + 1) * (half + 1)) as u32;
            let fm = dec.decode(ft) as i32;
            let (fl, fs);
            if fm < (half * (half + 1)) >> 1 {
                itheta = ((isqrt32(8 * fm as u32 + 1) as i32) - 1) >> 1;
                fs = itheta + 1;
                fl = (itheta * (itheta + 1)) >> 1;
            } else {
                itheta = (2 * (qn + 1) - isqrt32(8 * (ft as i32 - fm - 1) as u32 + 1) as i32) >> 1;
                fs = qn + 1 - itheta;
                fl = ft as i32 - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
            }
            dec.update(fl as u32, (fl + fs) as u32, ft);
        }
        itheta = itheta * 16384 / qn;
    } else if stereo {
        // qn == 1: itheta is 0; an inversion flag may follow.
        if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            inv = dec.decode_bit_logp(2);
        }
        itheta = 0;
    }

    let qalloc = dec.tell_frac() as i32 - tell;
    *b -= qalloc;

    let (imid, iside, delta);
    if itheta == 0 {
        imid = 32767;
        iside = 0;
        *fill &= (1u32 << big_b) - 1;
        delta = -16384;
    } else if itheta == 16384 {
        imid = 0;
        iside = 32767;
        *fill &= ((1u32 << big_b) - 1) << big_b;
        delta = 16384;
    } else {
        imid = i32::from(bitexact_cos(itheta as i16));
        iside = i32::from(bitexact_cos((16384 - itheta) as i16));
        // The mid/side allocation minimising squared error.
        delta = frac_mul16((n as i32 - 1) << 7, bitexact_log2tan(iside, imid));
    }

    SplitCtx {
        inv,
        imid,
        iside,
        delta,
        itheta,
        qalloc,
    }
}

/// Single-sample band: one sign bit per channel (`quant_band_n1`).
fn quant_band_n1(
    ctx: &mut BandCtx,
    dec: &mut RangeDecoder,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    lowband_out: Option<&mut [f32]>,
) -> u32 {
    let mut decode_one = |slot: &mut f32| {
        let mut sign = false;
        if ctx.remaining_bits >= 1 << BITRES {
            sign = dec.decode_raw_bits(1) == 1;
            ctx.remaining_bits -= 1 << BITRES;
        }
        *slot = if sign { -NORM_SCALING } else { NORM_SCALING };
    };
    decode_one(&mut x[0]);
    if let Some(y) = y {
        decode_one(&mut y[0]);
    }
    if let Some(out) = lowband_out {
        out[0] = x[0];
    }
    1
}

/// Recursively decodes a mono partition (`quant_partition`).
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the reference quant_partition signature"
)]
fn quant_partition(
    ctx: &mut BandCtx,
    dec: &mut RangeDecoder,
    x: &mut [f32],
    mut b: i32,
    big_b: usize,
    lowband: Option<&[f32]>,
    lm: i32,
    gain: f32,
    mut fill: u32,
) -> u32 {
    let n = x.len();
    let i = ctx.i;
    let b0 = big_b;

    // Split whenever the leaf would need more bits than one PVQ codeword can
    // hold (cache maximum plus 1.5 bits).
    if lm != -1 && b > super::rate::cache_max_bits(i, lm) + 12 && n > 2 {
        let n = n >> 1;
        let (xs, ys) = x.split_at_mut(n);
        let lm = lm - 1;
        let mut big_b = big_b;
        if big_b == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        big_b = (big_b + 1) >> 1;

        let sctx = compute_theta(ctx, dec, n, &mut b, big_b, b0, lm, false, &mut fill);
        let SplitCtx {
            imid,
            iside,
            mut delta,
            itheta,
            qalloc,
            ..
        } = sctx;
        let mid = imid as f32 / 32768.0;
        let side = iside as f32 / 32768.0;

        // Give more bits to low-energy MDCTs than their fair share.
        if b0 > 1 && (itheta & 0x3fff) != 0 {
            if itheta > 8192 {
                // Rough pre-echo masking approximation.
                delta -= delta >> (4 - lm);
            } else {
                // Forward-masking slope of 1.5 dB per 10 ms.
                delta = 0.min(delta + ((n as i32) << BITRES >> (5 - lm)));
            }
        }
        let mbits = 0.max(b.min((b - delta) / 2));
        let sbits = b - mbits;
        ctx.remaining_bits -= qalloc;

        let (lowband_mid, lowband_side) = match lowband {
            Some(lb) => (Some(&lb[..n]), Some(&lb[n..2 * n])),
            None => (None, None),
        };

        let rebalance = ctx.remaining_bits;
        let mut cm;
        if mbits >= sbits {
            cm = quant_partition(ctx, dec, xs, mbits, big_b, lowband_mid, lm, gain * mid, fill);
            let rebalance = mbits - (rebalance - ctx.remaining_bits);
            let mut sbits = sbits;
            if rebalance > 3 << BITRES && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            cm |=
                quant_partition(ctx, dec, ys, sbits, big_b, lowband_side, lm, gain * side, fill >> big_b) << (b0 >> 1);
        } else {
            cm = quant_partition(ctx, dec, ys, sbits, big_b, lowband_side, lm, gain * side, fill >> big_b) << (b0 >> 1);
            let rebalance = sbits - (rebalance - ctx.remaining_bits);
            let mut mbits = mbits;
            if rebalance > 3 << BITRES && itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            cm |= quant_partition(ctx, dec, xs, mbits, big_b, lowband_mid, lm, gain * mid, fill);
        }
        cm
    } else {
        // Leaf: one PVQ codeword (or folding when no pulses fit).
        let mut q = bits2pulses(i, lm, b);
        let mut curr_bits = pulses2bits(i, lm, q);
        ctx.remaining_bits -= curr_bits;
        // Never bust the budget.
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(i, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            let k = get_pulses(q) as usize;
            alg_unquant(dec, x, k, ctx.spread, big_b, gain).unwrap_or(0)
        } else {
            // No pulses: fill the band anyway.
            let cm_mask = ((1u64 << big_b) - 1) as u32;
            fill &= cm_mask;
            if fill == 0 {
                x.fill(0.0);
                0
            } else {
                let cm;
                match lowband {
                    None => {
                        // Noise-fill from the LCG.
                        for v in x.iter_mut() {
                            ctx.seed = celt_lcg_rand(ctx.seed);
                            *v = (ctx.seed as i32 >> 20) as f32;
                        }
                        cm = cm_mask;
                    },
                    Some(lb) => {
                        // Folded spectrum plus a small dither.
                        for (v, &l) in x.iter_mut().zip(lb) {
                            ctx.seed = celt_lcg_rand(ctx.seed);
                            // About 48 dB below the "normal" folding level.
                            let tmp = if ctx.seed & 0x8000 != 0 {
                                1.0 / 256.0
                            } else {
                                -1.0 / 256.0
                            };
                            *v = l + tmp;
                        }
                        cm = fill;
                    },
                }
                renormalise_vector(x, gain);
                cm
            }
        }
    }
}

/// Bit patterns folding collapse masks through Haar recombination.
const BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
const BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
    0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
];

/// Decodes one mono band (`quant_band`): time/frequency reshaping around the
/// recursive partition decode, plus folding-history output.
#[allow(clippy::too_many_arguments, reason = "mirrors the reference quant_band signature")]
fn quant_band(
    ctx: &mut BandCtx,
    dec: &mut RangeDecoder,
    x: &mut [f32],
    b: i32,
    big_b: usize,
    lowband: Option<&[f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    gain: f32,
    mut fill: u32,
) -> u32 {
    let n = x.len();
    let n0 = n;
    let mut big_b = big_b;
    let b0 = big_b;
    let mut time_divide = 0;
    let mut recombine = 0;
    let long_blocks = b0 == 1;
    let mut tf_change = ctx.tf_change;

    let mut n_b = n / big_b;

    if n == 1 {
        return quant_band_n1(ctx, dec, x, None, lowband_out);
    }

    if tf_change > 0 {
        recombine = tf_change;
    }

    // Work on an owned copy of the folding source so the reshaping below
    // never aliases the caller's history buffer (see module docs).
    let mut lowband_copy: Option<Vec<f32>> = lowband.map(|lb| lb.to_vec());

    for k in 0..recombine {
        if let Some(lb) = lowband_copy.as_mut() {
            haar1(lb, n >> k, 1 << k);
        }
        fill = u32::from(BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize])
            | u32::from(BIT_INTERLEAVE_TABLE[(fill >> 4) as usize]) << 2;
    }
    big_b >>= recombine;
    n_b <<= recombine;

    // Increasing the time resolution.
    while (n_b & 1) == 0 && tf_change < 0 {
        if let Some(lb) = lowband_copy.as_mut() {
            haar1(lb, n_b, big_b);
        }
        fill |= fill << big_b;
        big_b <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tf_change += 1;
    }
    let b0_post = big_b;
    let n_b0 = n_b;

    // Reorganize samples in time order.
    if b0_post > 1
        && let Some(lb) = lowband_copy.as_mut()
    {
        deinterleave_hadamard(lb, n_b >> recombine, b0_post << recombine, long_blocks);
    }

    let mut cm = quant_partition(ctx, dec, x, b, big_b, lowband_copy.as_deref(), lm, gain, fill);

    // Resynthesis: undo the reshaping on the decoded spectrum.
    if b0_post > 1 {
        interleave_hadamard(x, n_b >> recombine, b0_post << recombine, long_blocks);
    }
    let mut n_b = n_b0;
    let mut big_b = b0_post;
    for _ in 0..time_divide {
        big_b >>= 1;
        n_b <<= 1;
        cm |= cm >> big_b;
        haar1(x, n_b, big_b);
    }
    for k in 0..recombine {
        cm = u32::from(BIT_DEINTERLEAVE_TABLE[(cm & 0xF) as usize]);
        haar1(x, n0 >> k, 1 << k);
    }
    let big_b = big_b << recombine;

    // Scale output for later folding.
    if let Some(out) = lowband_out {
        let scale = (n0 as f32).sqrt();
        for (o, &v) in out.iter_mut().zip(x.iter()) {
            *o = scale * v;
        }
    }
    cm & ((1u32 << big_b) - 1)
}

/// Decodes one stereo band (`quant_band_stereo`): theta-coded mid/side with
/// the one-bit side optimisation for N=2, then mid/side merge.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the reference quant_band_stereo signature"
)]
fn quant_band_stereo(
    ctx: &mut BandCtx,
    dec: &mut RangeDecoder,
    x: &mut [f32],
    y: &mut [f32],
    mut b: i32,
    big_b: usize,
    lowband: Option<&[f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    mut fill: u32,
) -> u32 {
    let n = x.len();
    if n == 1 {
        return quant_band_n1(ctx, dec, x, Some(y), lowband_out);
    }

    let orig_fill = fill;
    let sctx = compute_theta(ctx, dec, n, &mut b, big_b, big_b, lm, true, &mut fill);
    let SplitCtx {
        inv,
        imid,
        iside,
        delta,
        itheta,
        qalloc,
    } = sctx;
    let mid = imid as f32 / 32768.0;
    let side = iside as f32 / 32768.0;
    let mut cm;

    if n == 2 {
        // Special two-sample stereo case: the side needs only a sign.
        let sbits = if itheta != 0 && itheta != 16384 { 1 << BITRES } else { 0 };
        let mbits = b - sbits;
        let c = itheta > 8192;
        ctx.remaining_bits -= qalloc + sbits;

        let mut sign = 0i32;
        if sbits != 0 {
            sign = dec.decode_raw_bits(1) as i32;
        }
        let sign = 1 - 2 * sign;

        // x2 is the channel coded with PVQ; y2 is reconstructed orthogonally.
        {
            let (x2, _y2): (&mut [f32], &mut [f32]) = if c { (y, &mut *x) } else { (&mut *x, y) };
            cm = quant_band(ctx, dec, x2, mbits, big_b, lowband, lm, lowband_out, Q15_ONE, orig_fill);
        }
        let (x2, y2): (&[f32; 2], &mut [f32]) = if c { (&[y[0], y[1]], x) } else { (&[x[0], x[1]], y) };
        y2[0] = -(sign as f32) * x2[1];
        y2[1] = (sign as f32) * x2[0];

        // Mix down per the decoded angle.
        let x0 = mid * x[0];
        let x1 = mid * x[1];
        let y0 = side * y[0];
        let y1 = side * y[1];
        x[0] = x0 - y0;
        y[0] = x0 + y0;
        x[1] = x1 - y1;
        y[1] = x1 + y1;
    } else {
        // Normal split.
        let mbits = 0.max(b.min((b - delta) / 2));
        let sbits = b - mbits;
        ctx.remaining_bits -= qalloc;

        let rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            cm = quant_band(ctx, dec, x, mbits, big_b, lowband, lm, lowband_out, Q15_ONE, fill);
            let rebalance = mbits - (rebalance - ctx.remaining_bits);
            let mut sbits = sbits;
            if rebalance > 3 << BITRES && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            cm |= quant_band(ctx, dec, y, sbits, big_b, None, lm, None, side, fill >> big_b);
        } else {
            cm = quant_band(ctx, dec, y, sbits, big_b, None, lm, None, side, fill >> big_b);
            let rebalance = sbits - (rebalance - ctx.remaining_bits);
            let mut mbits = mbits;
            if rebalance > 3 << BITRES && itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            cm |= quant_band(ctx, dec, x, mbits, big_b, lowband, lm, lowband_out, Q15_ONE, fill);
        }
        stereo_merge(x, y, mid);
    }
    if inv {
        for v in y.iter_mut() {
            *v = -*v;
        }
    }
    cm
}

/// Decodes the shapes of all coded bands (`quant_all_bands`, decoder side).
///
/// `x` and `y` are the per-channel normalised spectra (length
/// `EBANDS[NB_EBANDS] << lm`); `shape_bits` is the allocation's per-band PVQ
/// budget, `tf_res` the per-band time/frequency change. Returns the per-band
/// collapse masks (channel-interleaved) for the anti-collapse stage, and the
/// updated noise seed.
///
/// # Panics
///
/// Panics if `dual_stereo` is set without a `y` channel (an allocation for
/// stereo applied to mono input - a caller bug).
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the reference quant_all_bands signature"
)]
pub fn quant_all_bands(
    dec: &mut RangeDecoder,
    start: usize,
    end: usize,
    x: &mut [f32],
    mut y: Option<&mut [f32]>,
    collapse_masks: &mut [u8],
    shape_bits: &[i32; NB_EBANDS],
    short_blocks: bool,
    spread: Spread,
    dual_stereo: bool,
    intensity: usize,
    tf_res: &[i32; NB_EBANDS],
    total_bits: i32,
    mut balance: i32,
    lm: usize,
    coded_bands: usize,
    seed: &mut u32,
) {
    let m = 1usize << lm;
    let big_b = if short_blocks { m } else { 1 };
    let norm_offset = m * EBANDS[start] as usize;
    let channels = 1 + usize::from(y.is_some());
    let mut dual_stereo = dual_stereo;

    // Folding history (per channel), excluding the last band.
    let norm_len = m * EBANDS[NB_EBANDS - 1] as usize - norm_offset;
    let mut norm = vec![0.0f32; norm_len];
    let mut norm2 = vec![0.0f32; if channels == 2 { norm_len } else { 0 }];

    let mut ctx = BandCtx {
        i: start,
        intensity,
        spread,
        tf_change: 0,
        remaining_bits: 0,
        seed: *seed,
    };

    let mut lowband_offset = 0usize;
    let mut update_lowband = true;

    for i in start..end {
        ctx.i = i;
        let last = i == end - 1;
        let band_start = m * EBANDS[i] as usize;
        let band_end = m * EBANDS[i + 1] as usize;
        let n = band_end - band_start;
        let tell = dec.tell_frac() as i32;

        if i != start {
            balance -= tell;
        }
        let remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;

        let b = if i < coded_bands {
            let curr_balance = balance / 3.min(coded_bands as i32 - i as i32);
            0.max(16383.min((remaining_bits + 1).min(shape_bits[i] + curr_balance)))
        } else {
            0
        };

        if band_start as i64 - n as i64 >= (m * EBANDS[start] as usize) as i64
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }

        ctx.tf_change = tf_res[i];

        // Conservative collapse-mask estimate of the folding source.
        let (mut x_cm, mut y_cm);
        let mut effective_lowband = None;
        if lowband_offset != 0 && (spread != Spread::Aggressive || big_b > 1 || ctx.tf_change < 0) {
            let eff = 0.max(m as i32 * i32::from(EBANDS[lowband_offset]) - norm_offset as i32 - n as i32) as usize;
            effective_lowband = Some(eff);
            // Mirrors the reference's pre-decrement/pre-increment scans.
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if m * EBANDS[fold_start] as usize <= eff + norm_offset {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if m * EBANDS[fold_end] as usize >= eff + norm_offset + n {
                    break;
                }
            }
            x_cm = 0u32;
            y_cm = 0u32;
            let mut fold_i = fold_start;
            loop {
                x_cm |= u32::from(collapse_masks[fold_i * channels]);
                y_cm |= u32::from(collapse_masks[fold_i * channels + channels - 1]);
                fold_i += 1;
                if fold_i >= fold_end {
                    break;
                }
            }
        } else {
            x_cm = (1u32 << big_b) - 1;
            y_cm = (1u32 << big_b) - 1;
        }

        if dual_stereo && i == intensity {
            // Switch off dual stereo to do intensity.
            dual_stereo = false;
            for j in 0..(m * EBANDS[i] as usize - norm_offset) {
                norm[j] = 0.5 * (norm[j] + norm2[j]);
            }
        }

        let xb = &mut x[band_start..band_end];
        if dual_stereo {
            let yb = &mut y.as_mut().expect("dual stereo requires Y")[band_start..band_end];
            let lowband1 = effective_lowband.map(|e| norm[e..e + n].to_vec());
            x_cm = quant_band(
                &mut ctx,
                dec,
                xb,
                b / 2,
                big_b,
                lowband1.as_deref(),
                lm as i32,
                if last {
                    None
                } else {
                    Some(&mut norm[band_start - norm_offset..band_end - norm_offset])
                },
                Q15_ONE,
                x_cm,
            );
            let lowband2 = effective_lowband.map(|e| norm2[e..e + n].to_vec());
            y_cm = quant_band(
                &mut ctx,
                dec,
                yb,
                b / 2,
                big_b,
                lowband2.as_deref(),
                lm as i32,
                if last {
                    None
                } else {
                    Some(&mut norm2[band_start - norm_offset..band_end - norm_offset])
                },
                Q15_ONE,
                y_cm,
            );
        } else {
            let lowband = effective_lowband.map(|e| norm[e..e + n].to_vec());
            let lowband_out = if last {
                None
            } else {
                Some(&mut norm[band_start - norm_offset..band_end - norm_offset])
            };
            if let Some(yall) = y.as_mut() {
                let yb = &mut yall[band_start..band_end];
                x_cm = quant_band_stereo(
                    &mut ctx,
                    dec,
                    xb,
                    yb,
                    b,
                    big_b,
                    lowband.as_deref(),
                    lm as i32,
                    lowband_out,
                    x_cm | y_cm,
                );
            } else {
                x_cm = quant_band(
                    &mut ctx,
                    dec,
                    xb,
                    b,
                    big_b,
                    lowband.as_deref(),
                    lm as i32,
                    lowband_out,
                    Q15_ONE,
                    x_cm | y_cm,
                );
            }
            y_cm = x_cm;
        }
        collapse_masks[i * channels] = x_cm as u8;
        collapse_masks[i * channels + channels - 1] = y_cm as u8;
        balance += shape_bits[i] + tell;

        // Update the folding position only while there is ≥ 1 bit/sample.
        update_lowband = b > (n as i32) << BITRES;
    }
    *seed = ctx.seed;
}

/// Anti-collapse for transient frames (`anti_collapse`, RFC 6716 §4.3.5):
/// injects noise into MDCT blocks that received no pulses, at a level tied
/// to the band's recent energy history, then renormalises the band.
///
/// `x` is the per-channel concatenated spectrum (`channels * size`);
/// `log_e`/`prev1`/`prev2` are the current and two previous frames' band
/// energies; `pulses` is the allocation's shape budget (1/8 bits).
#[allow(clippy::too_many_arguments, reason = "mirrors the reference anti_collapse signature")]
pub fn anti_collapse(
    x: &mut [f32],
    collapse_masks: &[u8],
    lm: usize,
    channels: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[[f32; NB_EBANDS]; 2],
    prev1_log_e: &[[f32; NB_EBANDS]; 2],
    prev2_log_e: &[[f32; NB_EBANDS]; 2],
    pulses: &[i32; NB_EBANDS],
    mut seed: u32,
) {
    for i in start..end {
        let n0 = (EBANDS[i + 1] - EBANDS[i]) as usize;
        // Depth in 1/8 bits per sample (integer division, like the reference).
        let depth = (1 + pulses[i]) / ((n0 << lm) as i32);
        let thresh = 0.5 * libm_exp2(-0.125 * depth as f32);
        let sqrt_1 = 1.0 / (((n0 << lm) as f32).sqrt());

        for ch in 0..channels {
            let mut prev1 = prev1_log_e[ch][i];
            let mut prev2 = prev2_log_e[ch][i];
            if channels == 1 {
                prev1 = prev1.max(prev1_log_e[1][i]);
                prev2 = prev2.max(prev2_log_e[1][i]);
            }
            let ediff = (log_e[ch][i] - prev1.min(prev2)).max(0.0);

            // Short blocks don't have the same energy as long blocks.
            let mut r = 2.0 * libm_exp2(-ediff);
            if lm == 3 {
                r *= core::f32::consts::SQRT_2;
            }
            let r = thresh.min(r) * sqrt_1;

            let base = ch * size + ((EBANDS[i] as usize) << lm);
            let band_len = n0 << lm;
            let mut renormalize = false;
            for k in 0..(1usize << lm) {
                // Detect collapse.
                if collapse_masks[i * channels + ch] & (1 << k) == 0 {
                    // Fill with noise.
                    for j in 0..n0 {
                        seed = celt_lcg_rand(seed);
                        x[base + (j << lm) + k] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            // We just added some energy: renormalise.
            if renormalize {
                renormalise_vector(&mut x[base..base + band_len], Q15_ONE);
            }
        }
    }
}

/// `2^x` for the anti-collapse gains (the reference float build's
/// `celt_exp2`).
fn libm_exp2(x: f32) -> f32 {
    (core::f64::consts::LN_2 * f64::from(x)).exp() as f32
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn lcg_matches_reference_constants() {
        assert_eq!(celt_lcg_rand(0), 1_013_904_223);
        assert_eq!(celt_lcg_rand(1), 1_664_525 + 1_013_904_223);
        // A few steps of the sequence, computed independently.
        let mut s = 42u32;
        for _ in 0..3 {
            s = celt_lcg_rand(s);
        }
        assert_eq!(s, {
            let mut t: u64 = 42;
            for _ in 0..3 {
                t = (1_664_525 * t + 1_013_904_223) & 0xFFFF_FFFF;
            }
            t as u32
        });
    }

    #[test]
    fn bitexact_cos_pinned_values_and_unit_relation() {
        // The domain is the itheta grid: multiples of 16384/qn (qn <= 256),
        // i.e. multiples of 64 in [64, 16320] - the endpoints take the
        // explicit itheta == 0/16384 branches and never reach this function.
        // Values pinned against an independent evaluation of the reference
        // polynomial.
        for (x, expected) in [
            (64i16, 32767),
            (1024, 32610),
            (4096, 30274),
            (8192, 23171),
            (12288, 12540),
            (16320, 200),
        ] {
            assert_eq!(bitexact_cos(x), expected, "bitexact_cos({x})");
        }
        // Mid/side from any grid itheta satisfy the approximate unit relation.
        for k in 1..=255i16 {
            let itheta = 64 * k;
            let imid = i32::from(bitexact_cos(itheta));
            let iside = i32::from(bitexact_cos(16384 - itheta));
            let unit = (imid * imid + iside * iside) as f64 / (32768.0 * 32768.0);
            assert!((unit - 1.0).abs() < 0.01, "itheta={itheta}: {unit}");
        }
    }

    #[test]
    fn bitexact_log2tan_is_antisymmetric() {
        assert_eq!(bitexact_log2tan(16384, 16384), 0);
        for &(s, c) in &[(23171, 23171), (30000, 10000), (5000, 25000)] {
            assert_eq!(bitexact_log2tan(s, c), -bitexact_log2tan(c, s));
        }
        // One pinned value: log2(tan) of 30000/10000 ≈ log2(3) = 1.585 in Q11.
        let v = bitexact_log2tan(30000, 10000);
        assert!((f64::from(v) / 2048.0 - 1.585).abs() < 0.01, "{v}");
    }

    #[test]
    fn isqrt32_is_floor_sqrt() {
        for v in [0u32, 1, 2, 3, 4, 8, 9, 15, 16, 17, 1024, 99_980_001, u32::MAX] {
            let r = isqrt32(v);
            assert!(u64::from(r) * u64::from(r) <= u64::from(v));
            assert!((u64::from(r) + 1) * (u64::from(r) + 1) > u64::from(v));
        }
    }

    #[test]
    fn haar_and_hadamard_round_trip() {
        let original: Vec<f32> = (0..32).map(|i| (i as f32 * 0.31).sin()).collect();

        // haar1 is involutive up to the 1/2 scaling pair (applying twice
        // returns the input because (1/sqrt2)^2 * 2 = 1).
        let mut x = original.clone();
        haar1(&mut x, 32, 1);
        haar1(&mut x, 32, 1);
        for (a, b) in original.iter().zip(&x) {
            assert!((a - b).abs() < 1e-5);
        }

        for stride in [2usize, 4, 8, 16] {
            for hadamard in [false, true] {
                let n0 = 32 / stride;
                let mut x = original.clone();
                deinterleave_hadamard(&mut x, n0, stride, hadamard);
                interleave_hadamard(&mut x, n0, stride, hadamard);
                for (a, b) in original.iter().zip(&x) {
                    assert!((a - b).abs() < 1e-6, "stride {stride} hadamard {hadamard}");
                }
            }
        }
    }

    #[test]
    fn stereo_merge_produces_unit_channels() {
        // Orthogonal unit mid and side merge into two unit-norm channels.
        let n = 16usize;
        let mut x: Vec<f32> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let mut y: Vec<f32> = (0..n).map(|i| if i == 1 { 1.0 } else { 0.0 }).collect();
        stereo_merge(&mut x, &mut y, 1.0);
        let nx: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let ny: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((nx - 1.0).abs() < 1e-3, "{nx}");
        assert!((ny - 1.0).abs() < 1e-3, "{ny}");
    }

    /// Smoke test: decoding arbitrary bytes through the full band loop must
    /// terminate, stay within budget bookkeeping, and produce finite output.
    #[test]
    fn quant_all_bands_decodes_arbitrary_streams() {
        use crate::celt::rate::{compute_allocation, init_caps};
        use crate::range::RangeDecoder;

        for lm in 0..4usize {
            for channels in [1usize, 2] {
                for seed_byte in [0x00u8, 0x5A, 0xFF, 0x37] {
                    let m = 1usize << lm;
                    let nsamples = m * EBANDS[NB_EBANDS] as usize;
                    let frame_bytes = 80usize;
                    let data = vec![seed_byte; frame_bytes];
                    let mut dec = RangeDecoder::new(&data);

                    // A plausible allocation for this frame size.
                    let caps = init_caps(lm, channels);
                    let offsets = [0i32; NB_EBANDS];
                    let total = (frame_bytes as i32 * 8) << BITRES;
                    let alloc = compute_allocation(&mut dec, 0, NB_EBANDS, &offsets, &caps, 5, total, channels, lm);

                    let mut x = vec![0.0f32; nsamples];
                    let mut y = vec![0.0f32; nsamples];
                    let mut masks = vec![0u8; NB_EBANDS * channels];
                    let tf_res = [0i32; NB_EBANDS];
                    let mut seed = 0u32;

                    quant_all_bands(
                        &mut dec,
                        0,
                        NB_EBANDS,
                        &mut x,
                        (channels == 2).then_some(&mut y[..]),
                        &mut masks,
                        &alloc.shape_bits,
                        false,
                        Spread::Normal,
                        alloc.dual_stereo,
                        alloc.intensity,
                        &tf_res,
                        total - 1,
                        alloc.balance,
                        lm,
                        alloc.coded_bands,
                        &mut seed,
                    );

                    for (i, v) in x.iter().enumerate() {
                        assert!(v.is_finite(), "lm={lm} C={channels} x[{i}]");
                    }
                    if channels == 2 {
                        for (i, v) in y.iter().enumerate() {
                            assert!(v.is_finite(), "lm={lm} y[{i}]");
                        }
                    }
                }
            }
        }
    }
}

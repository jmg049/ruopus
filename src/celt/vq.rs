//! PVQ vector decoding: spreading rotation, residual normalisation, and the
//! shape decoder (RFC 6716 §4.3.4.3).
//!
//! A band's decoded pulse vector is normalised to unit norm (times the
//! requested gain) and then counter-rotated to undo the encoder's spreading
//! rotation - the psychoacoustic spreading control coded by the `spread`
//! parameter. The collapse mask (one bit per interleaved MDCT block) feeds
//! the anti-collapse logic for transient frames.

// `vec!` is only used by the (std-only) encode helpers in this module.
#[cfg(feature = "std")]
use alloc::vec;

use super::cwrs::decode_pulses;
#[cfg(not(feature = "std"))]
use crate::float::FloatExt;
use crate::range::RangeDecoder;

/// Spreading decision values (RFC 6716 Table 59, `SPREAD_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spread {
    /// No rotation.
    None,
    /// Light spreading (factor 15).
    Light,
    /// Normal spreading (factor 10).
    Normal,
    /// Aggressive spreading (factor 5).
    Aggressive,
}

impl Spread {
    /// Decodes from the 2-bit value used in the bitstream.
    #[must_use]
    pub const fn from_raw(v: u32) -> Self {
        match v & 3 {
            0 => Spread::None,
            1 => Spread::Light,
            2 => Spread::Normal,
            _ => Spread::Aggressive,
        }
    }

    const fn factor(self) -> Option<i32> {
        match self {
            Spread::None => None,
            Spread::Light => Some(15),
            Spread::Normal => Some(10),
            Spread::Aggressive => Some(5),
        }
    }
}

/// One pass of the Givens rotation network (`exp_rotation1`).
fn exp_rotation1(x: &mut [f32], stride: usize, c: f32, s: f32) {
    let len = x.len();
    for i in 0..len - stride {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
    if len > 2 * stride {
        for i in (0..len - 2 * stride).rev() {
            let x1 = x[i];
            let x2 = x[i + stride];
            x[i + stride] = c * x2 + s * x1;
            x[i] = c * x1 - s * x2;
        }
    }
}

/// `(cos(½π·θ), sin(½π·θ))` - the spreading rotation's gains. `θ = ½·gain²`
/// with `gain ≤ 1`, so the angle is in `[0, π/4]`, where these short Taylor
/// polynomials are accurate to far better than `f32` precision. This replaces
/// two libm `cos` calls that were ~25% of CELT encode. It is shared by encode and decode
/// so the rotation stays exactly invertible; the coded pulse indices - and thus
/// the range-coder state - are unaffected, only the spread shaping shifts by a
/// hair (well within the conformance PCM tolerance).
#[inline]
fn rotation_gains(theta: f32) -> (f32, f32) {
    let a = 0.5 * core::f32::consts::PI * theta;
    let a2 = a * a;
    let c = a2
        .mul_add(
            a2.mul_add(a2.mul_add(a2 * (1.0 / 40320.0), -1.0 / 720.0), 1.0 / 24.0),
            -0.5,
        )
        .mul_add(a2, 1.0);
    let s = a * a2.mul_add(a2.mul_add(a2.mul_add(-1.0 / 5040.0, 1.0 / 120.0), -1.0 / 6.0), 1.0);
    (c, s)
}

/// The spreading rotation (`exp_rotation`); `dir` +1 rotates (encoder), -1
/// counter-rotates (decoder). `b` is the number of interleaved blocks.
pub(crate) fn exp_rotation(x: &mut [f32], dir: i32, b: usize, k: usize, spread: Spread) {
    let len = x.len();
    let Some(factor) = spread.factor() else { return };
    if 2 * k >= len {
        return;
    }

    let gain = len as f32 / (len + factor as usize * k) as f32;
    let theta = 0.5 * gain * gain;
    let (c, s) = rotation_gains(theta);

    // An extra rotation pass with a longer stride approximating
    // sqrt(len/stride), for multi-block bands.
    let mut stride2 = 0usize;
    if len >= 8 * b {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * b + (b >> 2) < len {
            stride2 += 1;
        }
    }

    let sub = len / b;
    for i in 0..b {
        let block = &mut x[i * sub..(i + 1) * sub];
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(block, stride2, s, c);
            }
            exp_rotation1(block, 1, c, s);
        } else {
            exp_rotation1(block, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(block, stride2, s, -c);
            }
        }
    }
}

/// Scales the decoded pulse vector to norm `gain` (`normalise_residual`).
fn normalise_residual(iy: &[i32], x: &mut [f32], ryy: f32, gain: f32) {
    let g = gain / ryy.sqrt();
    for (xi, &p) in x.iter_mut().zip(iy) {
        *xi = g * p as f32;
    }
}

/// One bit per block, set when the block received any pulse
/// (`extract_collapse_mask`); feeds the anti-collapse logic.
fn extract_collapse_mask(iy: &[i32], b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = iy.len() / b;
    let mut mask = 0u32;
    for (i, block) in iy.chunks_exact(n0).enumerate().take(b) {
        if block.iter().any(|&v| v != 0) {
            mask |= 1 << i;
        }
    }
    mask
}

/// Decodes one PVQ-coded band shape (`alg_unquant`): pulse vector → unit
/// vector scaled by `gain`, counter-rotated for spreading. Returns the
/// collapse mask, or `None` on a corrupt uniform index.
#[must_use]
pub fn alg_unquant(
    dec: &mut RangeDecoder,
    x: &mut [f32],
    k: usize,
    spread: Spread,
    b: usize,
    gain: f32,
) -> Option<u32> {
    debug_assert!(k > 0, "alg_unquant() needs at least one pulse");
    debug_assert!(x.len() > 1, "alg_unquant() needs at least two dimensions");

    // Reuse a thread-local pulse buffer on `std` (band decode runs dozens of
    // times per frame); on `no_std` (no thread locals) allocate per call.
    #[cfg_attr(feature = "std", allow(unused_mut))]
    let mut run = |iy: &mut alloc::vec::Vec<i32>| -> Option<u32> {
        iy.clear();
        iy.resize(x.len(), 0);
        decode_pulses(dec, iy, k)?;
        let ryy: f32 = iy.iter().map(|&v| (v * v) as f32).sum();
        normalise_residual(iy, x, ryy, gain);
        exp_rotation(x, -1, b, k, spread);
        Some(extract_collapse_mask(iy, b))
    };
    #[cfg(feature = "std")]
    {
        thread_local! {
            static IY: core::cell::RefCell<alloc::vec::Vec<i32>> =
                const { core::cell::RefCell::new(alloc::vec::Vec::new()) };
        }
        IY.with_borrow_mut(run)
    }
    #[cfg(not(feature = "std"))]
    {
        run(&mut alloc::vec::Vec::new())
    }
}

/// Renormalises `x` to norm `gain` (`renormalise_vector`); used for folded
/// (uncoded) band content.
pub fn renormalise_vector(x: &mut [f32], gain: f32) {
    let e: f32 = 1e-15 + x.iter().map(|&v| v * v).sum::<f32>();
    let g = gain / e.sqrt();
    for v in x.iter_mut() {
        *v *= g;
    }
}

/// The float approximation the encoder uses for the theta angle.
#[cfg(feature = "std")]
fn fast_atan2f(y: f32, x: f32) -> f32 {
    const CA: f32 = 0.43157974;
    #[allow(clippy::excessive_precision, reason = "verbatim reference constant")]
    const CB: f32 = 0.67848403;
    const CC: f32 = 0.08595542;
    const CE: f32 = core::f32::consts::FRAC_PI_2;
    let x2 = x * x;
    let y2 = y * y;
    // For very small values the answer doesn't matter.
    if x2 + y2 < 1e-18 {
        return 0.0;
    }
    if x2 < y2 {
        let den = (y2 + CB * x2) * (y2 + CC * x2);
        -x * y * (y2 + CA * x2) / den + if y < 0.0 { -CE } else { CE }
    } else {
        let den = (x2 + CB * y2) * (x2 + CC * y2);
        x * y * (x2 + CA * y2) / den + (if y < 0.0 { -CE } else { CE }) - (if x * y < 0.0 { -CE } else { CE })
    }
}

/// `stereo_itheta`: the quantisation angle between two halves (or
/// mid/side when `stereo`), in Q14.
#[cfg(feature = "std")]
pub(crate) fn stereo_itheta(x: &[f32], y: &[f32], stereo: bool) -> i32 {
    let mut emid = 1e-15f32;
    let mut eside = 1e-15f32;
    if stereo {
        for (&a, &b) in x.iter().zip(y.iter()) {
            let m = 0.5 * a + 0.5 * b;
            let s = 0.5 * a - 0.5 * b;
            emid += m * m;
            eside += s * s;
        }
    } else {
        for &a in x {
            emid += a * a;
        }
        for &b in y {
            eside += b * b;
        }
    }
    let mid = emid.sqrt();
    let side = eside.sqrt();
    // 0.63662 = 2/pi
    #[allow(clippy::approx_constant, reason = "the reference uses this truncated 2/pi")]
    const TWO_OVER_PI: f32 = 0.63662;
    (0.5 + 16384.0 * TWO_OVER_PI * fast_atan2f(side, mid)).floor() as i32
}

/// Finds the K-pulse vector maximising correlation with `x`, writing the
/// signed pulse counts into `iy`. Dispatches to the SSE2 kernel on x86-64
/// (where SSE2 is guaranteed by the baseline ABI) and to the scalar search
/// elsewhere. Both produce a valid (round-trippable) pulse vector; the SIMD
/// path's `rsqrt` approximation may pick a marginally different - still
/// conformant - vector. See [`super::vq_simd`] and `docs/unsafe.md`.
#[cfg(feature = "std")]
#[inline]
fn op_pvq_search(x: &mut [f32], iy: &mut [i32], k: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        super::vq_simd::op_pvq_search(x, iy, k)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        op_pvq_search_scalar(x, iy, k)
    }
}

/// `op_pvq_search` (float build): finds the K-pulse vector maximising
/// correlation with `x` (made non-negative in place).
#[cfg(feature = "std")]
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
fn op_pvq_search_scalar(x: &mut [f32], iy: &mut [i32], k: usize) -> f32 {
    let n = x.len();
    let mut y = vec![0.0f32; n];
    let mut signx = vec![false; n];

    let mut sum = 0.0f32;
    for j in 0..n {
        signx[j] = x[j] < 0.0;
        x[j] = x[j].abs();
        iy[j] = 0;
    }

    let mut xy = 0.0f32;
    let mut yy = 0.0f32;
    let mut pulses_left = k as i32;

    // Pre-search by projecting onto the pyramid.
    if k > (n >> 1) {
        for &v in x.iter() {
            sum += v;
        }
        // Replace tiny or non-finite inputs with a single pulse at 0.
        if !(sum > 1e-15 && sum < 64.0) {
            x[0] = 1.0;
            for v in &mut x[1..] {
                *v = 0.0;
            }
            sum = 1.0;
        }
        // K+e with e < 1 guarantees no more than K pulses.
        let rcp = (k as f32 + 0.8) / sum;
        for j in 0..n {
            // Rounding towards zero is important here.
            iy[j] = (rcp * x[j]).floor() as i32;
            y[j] = iy[j] as f32;
            yy += y[j] * y[j];
            xy += x[j] * y[j];
            y[j] *= 2.0;
            pulses_left -= iy[j];
        }
    }

    // Pathological inputs: dump the remainder into the first bin.
    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as f32;
        yy += tmp * tmp;
        yy += tmp * y[0];
        iy[0] += pulses_left;
        pulses_left = 0;
    }

    for _ in 0..pulses_left {
        let mut best_id = 0usize;
        yy += 1.0;
        let mut rxy = xy + x[0];
        rxy *= rxy;
        let mut best_den = yy + y[0];
        let mut best_num = rxy;
        for j in 1..n {
            let mut rxy = xy + x[j];
            rxy *= rxy;
            let ryy = yy + y[j];
            if best_den * rxy > ryy * best_num {
                best_den = ryy;
                best_num = rxy;
                best_id = j;
            }
        }
        xy += x[best_id];
        yy += y[best_id];
        y[best_id] += 2.0;
        iy[best_id] += 1;
    }

    for j in 0..n {
        if signx[j] {
            iy[j] = -iy[j];
        }
    }
    yy
}

/// Encodes one band's shape with `k` pulses (`alg_quant`, no resynthesis).
#[cfg(feature = "std")]
pub(crate) fn alg_quant(
    enc: &mut crate::range::RangeEncoder,
    x: &mut [f32],
    k: usize,
    spread: Spread,
    b: usize,
) -> u32 {
    let n = x.len();
    debug_assert!(k > 0 && n > 1);
    // Reuse a thread-local pulse buffer on `std`; allocate per call on `no_std`.
    #[cfg_attr(feature = "std", allow(unused_mut))]
    let mut run = |iy: &mut alloc::vec::Vec<i32>| -> u32 {
        iy.clear();
        iy.resize(n, 0);
        exp_rotation(x, 1, b, k, spread);
        let _yy = op_pvq_search(x, iy, k);
        super::cwrs::encode_pulses(enc, iy, k);
        extract_collapse_mask(iy, b)
    };
    #[cfg(feature = "std")]
    {
        thread_local! {
            static IY: core::cell::RefCell<alloc::vec::Vec<i32>> =
                const { core::cell::RefCell::new(alloc::vec::Vec::new()) };
        }
        IY.with_borrow_mut(run)
    }
    #[cfg(not(feature = "std"))]
    {
        run(&mut alloc::vec::Vec::new())
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::celt::cwrs::encode_pulses;
    use crate::range::{RangeDecoder, RangeEncoder};

    fn norm(x: &[f32]) -> f32 {
        x.iter().map(|&v| v * v).sum::<f32>().sqrt()
    }

    #[test]
    fn rotation_is_inverted_by_counter_rotation() {
        for spread in [Spread::Light, Spread::Normal, Spread::Aggressive] {
            for (n, b, k) in [(16usize, 1usize, 3usize), (24, 2, 4), (64, 4, 5), (8, 1, 2)] {
                let original: Vec<f32> = (0..n).map(|i| ((i * 37 + 11) % 19) as f32 / 19.0 - 0.5).collect();
                let mut x = original.clone();
                exp_rotation(&mut x, 1, b, k, spread);
                exp_rotation(&mut x, -1, b, k, spread);
                for (a, b_) in original.iter().zip(&x) {
                    assert!((a - b_).abs() < 1e-5, "spread {spread:?} n={n}");
                }
            }
        }
    }

    #[test]
    fn rotation_preserves_energy() {
        let mut x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.7).sin()).collect();
        let before = norm(&x);
        exp_rotation(&mut x, 1, 2, 4, Spread::Normal);
        assert!((norm(&x) - before).abs() < 1e-4, "rotation is orthonormal");
    }

    #[test]
    fn unquant_returns_unit_vector_times_gain() {
        // Encode a known pulse vector, decode it, check the norm and the
        // direction (up to the rotation, which decode undoes).
        let n = 12usize;
        let k = 5usize;
        let mut enc = RangeEncoder::new(64);
        let y: Vec<i32> = {
            let mut y = vec![0i32; n];
            y[0] = 2;
            y[3] = -1;
            y[7] = 2;
            y
        };
        encode_pulses(&mut enc, &y, k);
        let buf = enc.finalize().expect("fits");

        let mut dec = RangeDecoder::new(&buf);
        let mut x = vec![0.0f32; n];
        let mask = alg_unquant(&mut dec, &mut x, k, Spread::None, 1, 1.0).expect("in range");
        assert_eq!(mask, 1, "B=1 mask is always 1");
        assert!((norm(&x) - 1.0).abs() < 1e-5, "unit norm");
        // With no spreading, the direction is exactly y / ||y||.
        let ryy = y.iter().map(|&v| (v * v) as f32).sum::<f32>().sqrt();
        for (xi, &yi) in x.iter().zip(&y) {
            assert!((xi - yi as f32 / ryy).abs() < 1e-6);
        }
    }

    #[test]
    fn collapse_mask_tracks_blocks_with_pulses() {
        // Two blocks, pulses only in the second.
        let n = 8usize;
        let k = 2usize;
        let mut enc = RangeEncoder::new(64);
        let mut y = vec![0i32; n];
        y[5] = 1;
        y[6] = -1;
        encode_pulses(&mut enc, &y, k);
        let buf = enc.finalize().expect("fits");

        let mut dec = RangeDecoder::new(&buf);
        let mut x = vec![0.0f32; n];
        let mask = alg_unquant(&mut dec, &mut x, k, Spread::None, 2, 1.0).expect("in range");
        assert_eq!(mask, 0b10, "only block 1 has pulses");
    }

    #[test]
    fn renormalise_scales_to_gain() {
        let mut x: Vec<f32> = (0..10).map(|i| i as f32 - 4.5).collect();
        renormalise_vector(&mut x, 0.75);
        assert!((norm(&x) - 0.75).abs() < 1e-5);
    }
}

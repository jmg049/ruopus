//! PVQ vector decoding: spreading rotation, residual normalisation, and the
//! shape decoder (RFC 6716 §4.3.4.3; normative `vq.c`, float build).
//!
//! A band's decoded pulse vector is normalised to unit norm (times the
//! requested gain) and then counter-rotated to undo the encoder's spreading
//! rotation - the psychoacoustic spreading control coded by the `spread`
//! parameter. The collapse mask (one bit per interleaved MDCT block) feeds
//! the anti-collapse logic for transient frames.

use alloc::vec;

use crate::range::RangeDecoder;

use super::cwrs::decode_pulses;

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
    let c = (0.5 * core::f32::consts::PI * theta).cos();
    let s = (0.5 * core::f32::consts::PI * (1.0 - theta)).cos(); // sin(theta)

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

    let mut iy = vec![0i32; x.len()];
    decode_pulses(dec, &mut iy, k)?;
    let ryy: f32 = iy.iter().map(|&v| (v * v) as f32).sum();
    normalise_residual(&iy, x, ryy, gain);
    exp_rotation(x, -1, b, k, spread);
    Some(extract_collapse_mask(&iy, b))
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

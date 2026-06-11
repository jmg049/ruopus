//! The CELT low-overlap MDCT (RFC 6716 §4.3.7; normative `mdct.c`, float
//! build).
//!
//! This is *not* a generic 50%-overlap MDCT: CELT windows only the first and
//! last `overlap` (=120) samples of each block with the Vorbis power window
//! ([`super::tables::WINDOW120`]), passing the middle through unwindowed, and
//! interleaves `B` short transforms for transient frames via the
//! `shift`/`stride` parameters. The backward transform performs its TDAC
//! overlap-add *in the caller's buffer*: the first `overlap` output samples
//! are mixed with the previous block's tail already present there.
//!
//! # The FFT seam
//!
//! Structurally the transform is a pre-rotation, an (un-normalised) N/4-point
//! complex FFT, and a post-rotation - the rotations and windowing are
//! codec-specific and live here; the FFT is generic. It is isolated behind
//! [`fft_forward`]/[`fft_inverse`] so a fast backend can replace the built-in
//! O(n²) evaluation without touching codec logic. The planned accelerated
//! backend is the `spectrograms` crate (this project's existing fast
//! MDCT/FFT work) behind an optional feature; the default build stays
//! dependency-free.
//!
//! Conformance note: the official test vectors compare PCM with a quality
//! threshold (`opus_compare`), not bit-exactly, so the FFT backend may vary;
//! every *bitstream-affecting* computation elsewhere in the crate remains
//! bit-exact.

use alloc::vec;
use alloc::vec::Vec;

/// Precomputed twiddles for one MDCT family (`mdct_lookup`): the standard
/// mode uses `n = 1920` with shifts 0..=3 for 20/10/5/2.5 ms blocks.
#[derive(Debug, Clone)]
pub struct MdctLookup {
    n: usize,
    /// `trig[i] = cos(2*pi*i / n)` for `i in 0..=n/4` (float-build table).
    trig: Vec<f32>,
}

impl MdctLookup {
    /// Builds the lookup for transform family size `n` (`clt_mdct_init`).
    #[must_use]
    pub fn new(n: usize) -> Self {
        let n4 = n >> 2;
        let trig = (0..=n4)
            .map(|i| (2.0 * core::f64::consts::PI * i as f64 / n as f64).cos() as f32)
            .collect();
        MdctLookup { n, trig }
    }

    /// The family size `n`.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.n
    }
}

/// Un-normalised inverse complex FFT: `out[n] = Σ_k in[k]·e^{+2πi kn/N}`.
///
/// The seam for an accelerated backend; the built-in evaluation is O(n²).
fn fft_inverse(input: &[(f32, f32)], output: &mut [(f32, f32)]) {
    let n = input.len();
    let step = 2.0 * core::f64::consts::PI / n as f64;
    for (k, out) in output.iter_mut().enumerate() {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (j, &(xr, xi)) in input.iter().enumerate() {
            let phase = step * (k * j % n) as f64;
            let (s, c) = phase.sin_cos();
            re += f64::from(xr) * c - f64::from(xi) * s;
            im += f64::from(xr) * s + f64::from(xi) * c;
        }
        *out = (re as f32, im as f32);
    }
}

/// Forward complex FFT scaled by `1/N` (kiss_fft's forward convention in
/// Opus): `out[k] = (1/N)·Σ_n in[n]·e^{-2πi kn/N}`.
fn fft_forward(input: &[(f32, f32)], output: &mut [(f32, f32)]) {
    let n = input.len();
    let step = 2.0 * core::f64::consts::PI / n as f64;
    let scale = 1.0 / n as f64;
    for (k, out) in output.iter_mut().enumerate() {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (j, &(xr, xi)) in input.iter().enumerate() {
            let phase = step * (k * j % n) as f64;
            let (s, c) = phase.sin_cos();
            re += f64::from(xr) * c + f64::from(xi) * s;
            im += -f64::from(xr) * s + f64::from(xi) * c;
        }
        *out = ((re * scale) as f32, (im * scale) as f32);
    }
}

impl MdctLookup {
    /// The backward (synthesis) MDCT (`clt_mdct_backward`).
    ///
    /// Decodes `n/2 >> shift` frequency coefficients read at `input[stride·k]`
    /// into time samples. Writes `out[0 .. (overlap/2) + n/2]`; the first
    /// `overlap` samples are TDAC-mixed with the prior contents of `out`
    /// (the previous block's tail or the frame's history), using `window`
    /// on both edges.
    pub fn backward(
        &self,
        input: &[f32],
        out: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let n = self.n >> shift;
        let n2 = n >> 1;
        let n4 = n >> 2;
        debug_assert!(out.len() >= (overlap >> 1) + n2);
        debug_assert_eq!(window.len(), overlap);

        // Small-angle correction: sin(x) ~= x for the trig table's offset.
        let sine = (2.0 * core::f64::consts::PI * 0.125 / n as f64) as f32;

        // Pre-rotate the strided spectral coefficients into N/4 complex bins.
        let mut f2 = vec![(0.0f32, 0.0f32); n4];
        {
            let t = &self.trig;
            for (i, y) in f2.iter_mut().enumerate() {
                let x1 = input[stride * 2 * i]; // even coefficients, ascending
                let x2 = input[stride * (n2 - 1 - 2 * i)]; // odd, descending
                let yr = -x2 * t[i << shift] + x1 * t[(n4 - i) << shift];
                let yi = -x2 * t[(n4 - i) << shift] - x1 * t[i << shift];
                // Works because the cos is nearly one.
                *y = (yr - yi * sine, yi + yr * sine);
            }
        }

        // Inverse N/4 complex FFT (un-normalised) into the output buffer.
        let mut time = vec![(0.0f32, 0.0f32); n4];
        fft_inverse(&f2, &mut time);
        for (i, &(re, im)) in time.iter().enumerate() {
            out[(overlap >> 1) + 2 * i] = re;
            out[(overlap >> 1) + 2 * i + 1] = im;
        }

        // Post-rotate and de-shuffle from both ends at once, in place.
        {
            let base = overlap >> 1;
            let t = &self.trig;
            // When N4 is odd the middle pair is computed twice, matching the
            // reference exactly.
            for i in 0..((n4 + 1) >> 1) {
                let p0 = base + 2 * i;
                let p1 = base + n2 - 2 - 2 * i;

                let re = out[p0];
                let im = out[p0 + 1];
                let t0 = t[i << shift];
                let t1 = t[(n4 - i) << shift];
                // The scale-by-2 happens when mixing the windows below.
                let yr = re * t0 - im * t1;
                let yi = im * t0 + re * t1;
                let re2 = out[p1];
                let im2 = out[p1 + 1];
                out[p0] = -(yr - yi * sine);
                out[p1 + 1] = yi + yr * sine;

                let t0 = t[(n4 - i - 1) << shift];
                let t1 = t[(i + 1) << shift];
                let yr = re2 * t0 - im2 * t1;
                let yi = im2 * t0 + re2 * t1;
                out[p1] = -(yr - yi * sine);
                out[p0 + 1] = yi + yr * sine;
            }
        }

        // Mirror both edges for TDAC, mixing with the existing buffer tail.
        {
            for i in 0..overlap / 2 {
                let a = i;
                let b = overlap - 1 - i;
                let x1 = out[b];
                let x2 = out[a];
                let w1 = window[i];
                let w2 = window[overlap - 1 - i];
                out[a] = w2 * x2 - w1 * x1;
                out[b] = w1 * x2 + w2 * x1;
            }
        }
    }

    /// The forward (analysis) MDCT (`clt_mdct_forward`); needed by the
    /// encoder and by the round-trip tests that validate [`backward`].
    ///
    /// Reads `n/2 + overlap` samples (the block plus both windowed edges)
    /// from `input` and writes `n/2 >> shift... ` - precisely, `n2` strided
    /// coefficients to `out[stride·k]` where `n2 = (self.n >> shift) / 2`.
    ///
    /// [`backward`]: Self::backward
    pub fn forward(&self, input: &[f32], out: &mut [f32], window: &[f32], overlap: usize, shift: usize, stride: usize) {
        let n = self.n >> shift;
        let n2 = n >> 1;
        let n4 = n >> 2;
        debug_assert!(input.len() >= n2 + overlap);
        debug_assert_eq!(window.len(), overlap);

        let sine = (2.0 * core::f64::consts::PI * 0.125 / n as f64) as f32;

        // Window, shuffle, fold the four conceptual blocks [a, b, c, d].
        let mut f = vec![0.0f32; n2];
        {
            let half = overlap >> 1;
            let quarter = (overlap + 3) >> 2;
            let mut yp = 0usize;
            // Edge region: both window tails involved.
            for i in 0..quarter {
                let xp1 = half + 2 * i;
                let xp2 = n2 - 1 + half - 2 * i;
                let w1 = window[half + 2 * i];
                let w2 = window[half - 1 - 2 * i];
                f[yp] = w2 * input[xp1 + n2] + w1 * input[xp2];
                f[yp + 1] = w1 * input[xp1] - w2 * input[xp2 - n2];
                yp += 2;
            }
            // Middle region: unwindowed pass-through.
            for i in quarter..(n4 - quarter) {
                let xp1 = half + 2 * i;
                let xp2 = n2 - 1 + half - 2 * i;
                f[yp] = input[xp2];
                f[yp + 1] = input[xp1];
                yp += 2;
            }
            // Other edge.
            for i in (n4 - quarter)..n4 {
                let k = i - (n4 - quarter);
                let xp1 = half + 2 * i;
                let xp2 = n2 - 1 + half - 2 * i;
                let w1 = window[2 * k];
                let w2 = window[overlap - 1 - 2 * k];
                f[yp] = -w1 * input[xp1 - n2] + w2 * input[xp2];
                f[yp + 1] = w2 * input[xp1] + w1 * input[xp2 + n2];
                yp += 2;
            }
        }

        // Pre-rotation.
        let mut fc = vec![(0.0f32, 0.0f32); n4];
        {
            let t = &self.trig;
            for (i, c) in fc.iter_mut().enumerate() {
                let re = f[2 * i];
                let im = f[2 * i + 1];
                let yr = -re * t[i << shift] - im * t[(n4 - i) << shift];
                let yi = -im * t[i << shift] + re * t[(n4 - i) << shift];
                *c = (yr + yi * sine, yi - yr * sine);
            }
        }

        // N/4 complex FFT (downscales by 4/N).
        let mut f2 = vec![(0.0f32, 0.0f32); n4];
        fft_forward(&fc, &mut f2);

        // Post-rotate and interleave to the strided output.
        {
            let t = &self.trig;
            for (i, &(re, im)) in f2.iter().enumerate() {
                let yr = im * t[(n4 - i) << shift] + re * t[i << shift];
                let yi = re * t[(n4 - i) << shift] - im * t[i << shift];
                out[stride * 2 * i] = yr - yi * sine;
                out[stride * (n2 - 1 - 2 * i)] = yi + yr * sine;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::celt::tables::WINDOW120;

    /// The Vorbis power window satisfies the TDAC condition
    /// `w[i]^2 + w[overlap-1-i]^2 == 1`.
    #[test]
    fn window_satisfies_tdac() {
        for (i, &w) in WINDOW120.iter().enumerate() {
            let w1 = f64::from(w);
            let w2 = f64::from(WINDOW120[119 - i]);
            assert!((w1 * w1 + w2 * w2 - 1.0).abs() < 1e-6, "i={i}");
            // And each sample matches the defining formula.
            let t = core::f64::consts::FRAC_PI_2 * (i as f64 + 0.5) / 120.0;
            let expected = (core::f64::consts::FRAC_PI_2 * t.sin().powi(2)).sin();
            assert!((w1 - expected).abs() < 1e-5, "i={i}");
        }
    }

    /// Forward → backward over consecutive overlapping blocks reconstructs
    /// the signal exactly (TDAC perfect reconstruction) - the joint
    /// correctness check for both transforms, the window, and the scaling.
    #[test]
    fn tdac_perfect_reconstruction() {
        // 5 ms blocks of the standard family: N = 1920 >> 2 = 480, N2 = 240.
        let lookup = MdctLookup::new(1920);
        let shift = 2usize;
        let n2 = (1920 >> shift) / 2;
        let overlap = 120usize;
        let frames = 6usize;

        // A deterministic full-band test signal.
        let total = n2 * frames + overlap + n2;
        let signal: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32;
                (t * 0.1).sin() + 0.5 * (t * 0.037).cos() + 0.25 * (t * 0.41).sin()
            })
            .collect();

        // Forward each block (hop N2), then synthesize with TDAC mixing.
        let mut synth = vec![0.0f32; total];
        for f in 0..frames {
            let mut freq = vec![0.0f32; n2];
            lookup.forward(&signal[f * n2..], &mut freq, &WINDOW120, overlap, shift, 1);
            let out = &mut synth[f * n2..];
            lookup.backward(&freq, out, &WINDOW120, overlap, shift, 1);
        }

        // The interior - past the first block's unmixed head and before the
        // last (incomplete) block - reconstructs the signal exactly, with
        // zero delay: the TDAC mixing in `backward` already folds the
        // half-overlap offset away.
        for i in 2 * overlap..(frames - 1) * n2 {
            let got = synth[i];
            let want = signal[i];
            assert!(
                (got - want).abs() < 1e-3,
                "sample {i}: got {got}, want {want} (err {})",
                (got - want).abs()
            );
        }
    }

    /// Short-block interleaving: B transforms with stride B must round-trip
    /// the same way through matching forward/backward calls.
    #[test]
    fn interleaved_short_blocks_round_trip() {
        let lookup = MdctLookup::new(1920);
        let shift = 3usize; // 2.5 ms blocks: N2 = 120
        let n2 = (1920 >> shift) / 2;
        let b = 2usize;
        let overlap = 120usize;
        let frames = 8usize;

        let total = n2 * frames + overlap + n2;
        let signal: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.21).sin() - 0.3 * ((i as f32) * 0.05).cos())
            .collect();

        // Interleave two block streams exactly as a 2-short-block frame does:
        // block k of a frame uses input offset k*N2 and coefficient stride B
        // with offset k.
        let mut synth = vec![0.0f32; total];
        for f in 0..frames / b {
            let base = f * b * n2;
            let mut freq = vec![0.0f32; n2 * b];
            for k in 0..b {
                lookup.forward(&signal[base + k * n2..], &mut freq[k..], &WINDOW120, overlap, shift, b);
            }
            for k in 0..b {
                lookup.backward(&freq[k..], &mut synth[base + k * n2..], &WINDOW120, overlap, shift, b);
            }
        }

        for i in 2 * overlap..(frames - 2) * n2 {
            let got = synth[i];
            let want = signal[i];
            assert!((got - want).abs() < 1e-3, "sample {i}: got {got}, want {want}");
        }
    }
}

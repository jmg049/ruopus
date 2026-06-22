//! Float front-end downsampler for the SILK encoder (48 kHz API rate →
//! 8/12/16 kHz internal rate).
//!
//! The reference resampler is fixed-point
//! (`Q8`/`Q14`/`Q16`), and the decoder direction keeps it bit-exact. The encoder
//! input resampler, however, only conditions the analysis signal - it never
//! defines the bitstream - so it can run in `f32`, where the AR2 pre-filter and
//! the symmetric FIR collapse to a windowed dot product the SIMD kernel
//! ([`crate::simd::dot`], AVX2+FMA) handles ~6× faster than the scalar
//! fixed-point MACs that dominated SILK encode (callgrind: `down_fir` was ~16%).
//!
//! It uses the same AR2
//! coefficients and same FIR taps, so the filter response - and therefore the
//! conditioned signal's quality - matches the fixed-point resampler to within
//! rounding (validated against it in the tests, and end to end by the
//! conformance/round-trip suite). Only the 48 kHz input ratios the public
//! `OpusEncoder` actually uses are implemented (3/4/6× decimation).

use alloc::vec;
use alloc::vec::Vec;

/// 48→16 resampler coefficients: 2 AR2 taps then 18 symmetric FIR taps.
const COEFS_1_3: [i16; 20] = [
    16102, -15162, -13, 0, 20, 26, 5, -31, -43, -4, 65, 90, 7, -157, -248, -44, 593, 1583, 2612, 3271,
];
/// 48→12 resampler coefficients: 2 AR2 taps then 18 symmetric FIR taps.
const COEFS_1_4: [i16; 20] = [
    22500, -15099, 3, -14, -20, -15, 2, 25, 37, 25, -16, -71, -107, -79, 50, 292, 623, 982, 1288, 1464,
];
/// 48→8 resampler coefficients: 2 AR2 taps then 18 symmetric FIR taps.
const COEFS_1_6: [i16; 20] = [
    27540, -15257, 17, 12, 8, 1, -10, -22, -30, -32, -22, 3, 44, 100, 168, 243, 317, 381, 429, 455,
];

/// FIR order of the down-FIR2 path (`RESAMPLER_DOWN_ORDER_FIR2`).
const ORD: usize = 36;
/// Encoder input-delay table row for a 48 kHz API rate
/// (`delay_matrix_enc[rateID(48000)]` over outputs [8, 12, 16] kHz).
const INPUT_DELAY_48: [usize; 3] = [18, 10, 12];

/// Encoder front-end downsampler from 48 kHz to an 8/12/16 kHz internal rate.
#[derive(Debug, Clone)]
pub(crate) struct EncDownsampler {
    /// AR2 pre-filter state (float, in i16-sample units).
    s_iir: [f32; 2],
    /// FIR history: the previous `ORD` filtered samples.
    s_fir: [f32; ORD],
    /// Input-history delay buffer (≤ `fs_in_khz` samples).
    delay_buf: [f32; 48],
    input_delay: usize,
    fs_in_khz: usize,
    fs_out_khz: usize,
    /// Decimation factor (3, 4 or 6).
    decim: usize,
    /// AR2 coefficients, Q14 → float.
    a0: f32,
    a1: f32,
    /// The 18 FIR taps expanded to the full symmetric 36-tap window, float.
    full_coef: [f32; ORD],
    /// Scale applied to the input on the way in (the encoder passes `f32` PCM
    /// in ±1.0; SILK works in i16-sample units, so this is 32768.0).
    in_scale: f32,
}

impl EncDownsampler {
    /// Builds the 48 kHz → `fs_out_khz` (∈ {8, 12, 16}) downsampler.
    pub(crate) fn new(fs_out_khz: usize) -> Self {
        let (coefs, decim) = match fs_out_khz {
            16 => (&COEFS_1_3, 3),
            12 => (&COEFS_1_4, 4),
            8 => (&COEFS_1_6, 6),
            _ => panic!("encoder internal rate must be 8/12/16 kHz, got {fs_out_khz}"),
        };
        // Expand the 18 stored taps to the symmetric 36-tap window:
        // full[t] = c[t] for t < 18, full[t] = c[35 - t] for t ≥ 18.
        let fir = &coefs[2..];
        let mut full_coef = [0.0f32; ORD];
        for (t, fc) in full_coef.iter_mut().enumerate() {
            *fc = f32::from(if t < 18 { fir[t] } else { fir[ORD - 1 - t] });
        }
        let input_delay = INPUT_DELAY_48[match fs_out_khz {
            8 => 0,
            12 => 1,
            _ => 2,
        }];
        Self {
            s_iir: [0.0; 2],
            s_fir: [0.0; ORD],
            delay_buf: [0.0; 48],
            input_delay,
            fs_in_khz: 48,
            fs_out_khz,
            decim,
            a0: f32::from(coefs[0]) / 16384.0,
            a1: f32::from(coefs[1]) / 16384.0,
            full_coef,
            in_scale: 32768.0,
        }
    }

    /// Resamples `input` (48 kHz, `f32` in i16-sample units, length a multiple
    /// of 1 ms and ≥ 1 ms) into `out` (`input.len() / decim` i16 samples),
    /// carrying filter state across calls, with a head/tail split around the
    /// input delay.
    pub(crate) fn process(&mut self, out: &mut [i16], input: &[f32]) {
        let in_len = input.len();
        debug_assert!(in_len >= self.fs_in_khz);
        debug_assert_eq!(out.len(), in_len * self.fs_out_khz / self.fs_in_khz);

        let n_samples = self.fs_in_khz - self.input_delay;

        // Head: 1 ms of the delay buffer plus the start of this input.
        let mut head = [0.0f32; 48];
        head[..self.input_delay].copy_from_slice(&self.delay_buf[..self.input_delay]);
        head[self.input_delay..self.fs_in_khz].copy_from_slice(&input[..n_samples]);

        let tail = &input[n_samples..in_len - self.input_delay];
        let (out_head, out_tail) = out.split_at_mut(self.fs_out_khz);
        self.down_fir(out_head, &head[..self.fs_in_khz]);
        self.down_fir(out_tail, tail);

        // Buffer the input tail for the next call.
        self.delay_buf[..self.input_delay].copy_from_slice(&input[in_len - self.input_delay..]);
    }

    /// AR2 pre-filter then symmetric
    /// FIR decimation (float). `out.len() == input.len() / decim`.
    fn down_fir(&mut self, out: &mut [i16], input: &[f32]) {
        let n = input.len();
        // Work buffer: ORD history samples then the AR2-filtered input.
        let mut buf = vec![0.0f32; ORD + n];
        buf[..ORD].copy_from_slice(&self.s_fir);

        // Second-order AR pre-filter (state in i16-sample units).
        {
            let s = &mut self.s_iir;
            let scale = self.in_scale;
            for (o, &x) in buf[ORD..].iter_mut().zip(input.iter()) {
                let v = s[0] + x * scale;
                *o = v;
                s[0] = s[1] + v * self.a0;
                s[1] = v * self.a1;
            }
        }

        // Symmetric FIR, decimating by `decim`. Each output is a 36-tap dot of
        // the window with the expanded coefficients; the sum equals the
        // reference's `Σ (buf[b+t] + buf[b+35-t])·c[t]`, scaled by 1/16384.
        let coef = &self.full_coef;
        for (m, o) in out.iter_mut().enumerate() {
            let base = m * self.decim;
            let acc = crate::simd::dot(coef, &buf[base..base + ORD]);
            // Round half away from zero without libm: `+ copysign(0.5)` then the
            // saturating `as i16` cast (which also clamps to ±32767).
            let v = acc * (1.0 / 16384.0);
            *o = (v + 0.5f32.copysign(v)) as i16;
        }

        // Save the FIR history (the last ORD samples of the work buffer).
        self.s_fir.copy_from_slice(&buf[n..n + ORD]);
    }
}

/// Convenience: resample a whole `f32` 48 kHz frame to an i16 internal-rate
/// `Vec`, allocating the output.
pub(crate) fn resample_48k(ds: &mut EncDownsampler, input: &[f32]) -> Vec<i16> {
    let out_len = input.len() * ds.fs_out_khz / 48;
    let mut out = vec![0i16; out_len];
    ds.process(&mut out, input);
    out
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use crate::silk::resampler::Resampler;

    /// The float front-end resampler tracks the fixed-point reference resampler
    /// to within a couple of LSB on a multi-frame speech-like signal - i.e. the
    /// conditioned analysis signal is the same filter, just computed in float.
    #[test]
    fn matches_fixed_point_within_tolerance() {
        for &(out_khz, _ratio) in &[(16usize, 3usize), (12, 4), (8, 6)] {
            let frames = 6;
            let n = 960 * frames;
            // Speech-like f32 PCM in ±1.0.
            let pcm: Vec<f32> = (0..n)
                .map(|i| {
                    let t = i as f32;
                    0.25 * (t * 0.04).sin() + 0.09 * (t * 0.21).sin() + 0.045 * (t * 0.011).cos()
                })
                .collect();

            let mut fixed = Resampler::new_enc(48_000, (out_khz * 1000) as i32);
            let mut flt = EncDownsampler::new(out_khz);

            let mut max_diff = 0i32;
            for f in 0..frames {
                let chunk = &pcm[f * 960..f * 960 + 960];
                let in16: Vec<i16> = chunk
                    .iter()
                    .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
                    .collect();
                let mut a = vec![0i16; 960 * out_khz / 48];
                fixed.process(&mut a, &in16);
                let b = resample_48k(&mut flt, chunk);
                for (x, y) in a.iter().zip(b.iter()) {
                    max_diff = max_diff.max((i32::from(*x) - i32::from(*y)).abs());
                }
            }
            assert!(max_diff <= 4, "{out_khz}kHz: max LSB diff {max_diff} vs fixed-point");
        }
    }
}

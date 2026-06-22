//! SIMD dot-product primitives shared by the hot analysis loops (CELT pitch
//! search, SILK LPC/pitch correlation). The kernels are the dominant encoder
//! cost - `celt_pitch_xcorr` alone is ~two thirds of CELT encode - and map
//! directly to fused multiply-add, so this is where the encoder closes the gap
//! to libopus's hand-written SIMD.
//!
//! All entry points are safe wrappers that pick the widest available kernel at
//! runtime (AVX2+FMA → SSE2 → scalar) and fall back to scalar off x86-64. They
//! feed encoder *decisions* (pitch lag/gain, analysis coefficients), which the
//! reference float build does not require to be bit-exact, so using FMA - which
//! rounds the product and sum together - is sound. See `docs/unsafe.md`.

/// `Σ x[i]·y[i]` over `x.len()` lanes. `y` must be at least as long as `x`.
#[must_use]
#[cfg_attr(target_arch = "x86_64", allow(unsafe_code))]
pub(crate) fn dot(x: &[f32], y: &[f32]) -> f32 {
    debug_assert!(y.len() >= x.len());
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: both slices expose ≥ `x.len()` contiguous `f32`s; the kernels
        // only read those lanes (vector loads are masked to the SIMD width with
        // a scalar tail). AVX2 is gated behind runtime detection; SSE2 is part
        // of the x86-64 baseline ABI so it is always available.
        unsafe { if has_avx2() { dot_avx2(x, y) } else { dot_sse2(x, y) } }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        dot_scalar(x, y)
    }
}

/// Two dot products sharing the first operand: `(Σ x·y1, Σ x·y2)`. Both `y1`
/// and `y2` must be at least as long as `x`.
#[must_use]
#[cfg_attr(target_arch = "x86_64", allow(unsafe_code))]
pub(crate) fn dual_dot(x: &[f32], y1: &[f32], y2: &[f32]) -> (f32, f32) {
    debug_assert!(y1.len() >= x.len() && y2.len() >= x.len());
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: as `dot`; all three slices expose ≥ `x.len()` lanes.
        unsafe {
            if has_avx2() {
                dual_dot_avx2(x, y1, y2)
            } else {
                dual_dot_sse2(x, y1, y2)
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (dot_scalar(x, y1), dot_scalar(x, y2))
    }
}

/// Cross-correlation `out[i] = Σ_j x[j]·y[i+j]` for `i` in `0..out.len()`
/// (`celt_pitch_xcorr`). Computes four lags per pass so each `x` block is
/// loaded once and shared across them (libopus's `xcorr_kernel`), instead of an
/// independent dot per lag. `y` must be at least `out.len() + len - 1` long.
#[cfg_attr(target_arch = "x86_64", allow(unsafe_code))]
pub(crate) fn pitch_xcorr(x: &[f32], y: &[f32], out: &mut [f32], len: usize) {
    debug_assert!(x.len() >= len && y.len() + 1 >= out.len() + len);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: AVX2 gated; the kernel reads `x[..len]` and `y[i..i+len]`
            // for `i < out.len()`, all within the asserted bounds.
            unsafe { pitch_xcorr_avx2(x, y, out, len) };
            return;
        }
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = dot(&x[..len], &y[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2,fma")]
unsafe fn pitch_xcorr_avx2(x: &[f32], y: &[f32], out: &mut [f32], len: usize) {
    use core::arch::x86_64::*;
    let max_pitch = out.len();
    let (xp, yp) = (x.as_ptr(), y.as_ptr());
    // SAFETY: 8-wide loads of `x[j..]` (j+8≤len via the tail split) and
    // `y[i+j..]` (i+3+j+8 ≤ out.len()+len ≤ y.len()+1) stay in bounds.
    unsafe {
        let mut i = 0;
        while i + 4 <= max_pitch {
            let (mut s0, mut s1, mut s2, mut s3) = (
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            );
            let mut j = 0;
            while j + 8 <= len {
                let xv = _mm256_loadu_ps(xp.add(j));
                s0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(yp.add(i + j)), s0);
                s1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(yp.add(i + 1 + j)), s1);
                s2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(yp.add(i + 2 + j)), s2);
                s3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(yp.add(i + 3 + j)), s3);
                j += 8;
            }
            let (mut r0, mut r1, mut r2, mut r3) = (hsum256(s0), hsum256(s1), hsum256(s2), hsum256(s3));
            while j < len {
                let xv = *xp.add(j);
                r0 += xv * *yp.add(i + j);
                r1 += xv * *yp.add(i + 1 + j);
                r2 += xv * *yp.add(i + 2 + j);
                r3 += xv * *yp.add(i + 3 + j);
                j += 1;
            }
            *out.get_unchecked_mut(i) = r0;
            *out.get_unchecked_mut(i + 1) = r1;
            *out.get_unchecked_mut(i + 2) = r2;
            *out.get_unchecked_mut(i + 3) = r3;
            i += 4;
        }
        while i < max_pitch {
            out[i] = dot(&x[..len], &y[i..]);
            i += 1;
        }
    }
}

/// 6-tap FIR: `out[j] = Σ_{k=0..6} inp[j+k]·c[k]` for `j` in `0..out.len()`.
/// `inp` must be at least `out.len() + 5` long. Used by the CELT pitch
/// downsampler's whitening filter (`celt_fir5`, expressed with an explicit
/// 5-sample input history so it is a forward convolution).
#[cfg_attr(target_arch = "x86_64", allow(unsafe_code))]
pub(crate) fn fir6(out: &mut [f32], inp: &[f32], c: [f32; 6]) {
    debug_assert!(inp.len() >= out.len() + 5);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: AVX2 gated by runtime detection; the kernel reads
            // `inp[j..j+8+5]` only where `j + 8 ≤ out.len()` and writes
            // `out[j..j+8]`, with a scalar tail - all in bounds given the
            // `inp.len() ≥ out.len()+5` precondition.
            unsafe { fir6_avx2(out, inp, c) };
            return;
        }
    }
    for (j, o) in out.iter_mut().enumerate() {
        *o = c[0] * inp[j]
            + c[1] * inp[j + 1]
            + c[2] * inp[j + 2]
            + c[3] * inp[j + 3]
            + c[4] * inp[j + 4]
            + c[5] * inp[j + 5];
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2,fma")]
unsafe fn fir6_avx2(out: &mut [f32], inp: &[f32], c: [f32; 6]) {
    use core::arch::x86_64::*;
    let n = out.len();
    let (ip, op) = (inp.as_ptr(), out.as_mut_ptr());
    // SAFETY: 8-wide loads read `inp[j..j+13]` (j+8+5 ≤ n+5 ≤ inp.len()); stores
    // write `out[j..j+8]`; the scalar tail covers the remainder.
    unsafe {
        let c0 = _mm256_set1_ps(c[0]);
        let c1 = _mm256_set1_ps(c[1]);
        let c2 = _mm256_set1_ps(c[2]);
        let c3 = _mm256_set1_ps(c[3]);
        let c4 = _mm256_set1_ps(c[4]);
        let c5 = _mm256_set1_ps(c[5]);
        let mut j = 0;
        while j + 8 <= n {
            let mut acc = _mm256_mul_ps(_mm256_loadu_ps(ip.add(j)), c0);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(ip.add(j + 1)), c1, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(ip.add(j + 2)), c2, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(ip.add(j + 3)), c3, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(ip.add(j + 4)), c4, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(ip.add(j + 5)), c5, acc);
            _mm256_storeu_ps(op.add(j), acc);
            j += 8;
        }
        while j < n {
            *op.add(j) = c[0] * *ip.add(j)
                + c[1] * *ip.add(j + 1)
                + c[2] * *ip.add(j + 2)
                + c[3] * *ip.add(j + 3)
                + c[4] * *ip.add(j + 4)
                + c[5] * *ip.add(j + 5);
            j += 1;
        }
    }
}

/// `out[i] = (x[i]·scale).round().clamp(-32768, 32767) as i16`, matching
/// `f32::round` (round half away from zero) bit-for-bit. Used for the SILK
/// float→i16 conversions (pitch analysis frames, encoder input), which feed
/// reference-pinned analysis, so the rounding must match exactly.
#[cfg_attr(target_arch = "x86_64", allow(unsafe_code))]
pub(crate) fn scale_round_to_i16(out: &mut [i16], x: &[f32], scale: f32) {
    debug_assert!(out.len() == x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: AVX2 gated by runtime detection; lengths are equal and
            // the kernel processes 8-wide with a scalar tail, so all accesses
            // are in bounds.
            unsafe { scale_round_to_i16_avx2(out, x, scale) };
            return;
        }
    }
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = (v * scale).round().clamp(-32768.0, 32767.0) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn scale_round_to_i16_avx2(out: &mut [i16], x: &[f32], scale: f32) {
    use core::arch::x86_64::*;
    let n = x.len();
    let (xp, op) = (x.as_ptr(), out.as_mut_ptr());
    // SAFETY: 8-wide loads/stores start at `i` with `i + 8 ≤ n`; the scalar
    // tail handles the remainder. Round-half-away is `trunc(v + copysign(0.5,v))`.
    unsafe {
        let vscale = _mm256_set1_ps(scale);
        let half = _mm256_set1_ps(0.5);
        let signmask = _mm256_set1_ps(-0.0);
        let lo = _mm256_set1_ps(-32768.0);
        let hi = _mm256_set1_ps(32767.0);
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_mul_ps(_mm256_loadu_ps(xp.add(i)), vscale);
            let bias = _mm256_or_ps(_mm256_and_ps(v, signmask), half);
            let r = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(_mm256_add_ps(v, bias));
            let r = _mm256_max_ps(lo, _mm256_min_ps(hi, r));
            // f32 -> i32 (truncation; already integral), then saturating pack to i16.
            let i32s = _mm256_cvttps_epi32(r);
            // Pack the two 128-bit halves of i32 lanes into one 128-bit of i16.
            let packed = _mm_packs_epi32(_mm256_castsi256_si128(i32s), _mm256_extracti128_si256::<1>(i32s));
            _mm_storeu_si128(op.add(i).cast(), packed);
            i += 8;
        }
        while i < n {
            *op.add(i) = (*xp.add(i) * scale).round().clamp(-32768.0, 32767.0) as i16;
            i += 1;
        }
    }
}

#[inline]
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
fn dot_scalar(x: &[f32], y: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..x.len() {
        s += x[i] * y[i];
    }
    s
}

/// `Σ x[i]·y[i]` accumulated in `f64` (inputs are `f32`). For the SILK pitch
/// analysis, whose reference helpers accumulate in double precision and whose
/// outputs are pinned against the reference - using a `f64` accumulator keeps
/// the result within the pin tolerance where an `f32` accumulator might drift.
/// `y` must be at least as long as `x`.
#[must_use]
#[cfg_attr(target_arch = "x86_64", allow(unsafe_code))]
pub(crate) fn dot_f64(x: &[f32], y: &[f32]) -> f64 {
    debug_assert!(y.len() >= x.len());
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: both slices expose ≥ `x.len()` lanes; loads are width-masked
        // with a scalar tail. AVX2 gated by runtime detection, SSE2 baseline.
        unsafe {
            if has_avx2() {
                dot_f64_avx2(x, y)
            } else {
                dot_f64_sse2(x, y)
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut s = 0.0f64;
        for i in 0..x.len() {
            s += f64::from(x[i]) * f64::from(y[i]);
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
use core::sync::atomic::{AtomicU8, Ordering};

#[cfg(target_arch = "x86_64")]
static AVX2: AtomicU8 = AtomicU8::new(0); // 0 = unknown, 1 = no, 2 = yes

/// Cached `is_x86_feature_detected!("avx2","fma")`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2() -> bool {
    match AVX2.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            #[cfg(feature = "std")]
            let ok = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
            #[cfg(not(feature = "std"))]
            let ok = false; // runtime detection needs std; SSE2 baseline still applies
            AVX2.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
            ok
        },
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = x.len();
    let (xp, yp) = (x.as_ptr(), y.as_ptr());
    // SAFETY: every load below starts at `i` with `i + width ≤ n ≤ len`. Four
    // accumulators hide the FMA latency for long dots; short dots fall straight
    // to the 8-wide tail.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i)), _mm256_loadu_ps(yp.add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i + 8)), _mm256_loadu_ps(yp.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i + 16)), _mm256_loadu_ps(yp.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i + 24)), _mm256_loadu_ps(yp.add(i + 24)), a3);
            i += 32;
        }
        let mut a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        while i + 8 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i)), _mm256_loadu_ps(yp.add(i)), a0);
            i += 8;
        }
        let mut s = hsum256(a0);
        while i < n {
            s += *xp.add(i) * *yp.add(i);
            i += 1;
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2,fma")]
unsafe fn dual_dot_avx2(x: &[f32], y1: &[f32], y2: &[f32]) -> (f32, f32) {
    use core::arch::x86_64::*;
    let n = x.len();
    let (xp, y1p, y2p) = (x.as_ptr(), y1.as_ptr(), y2.as_ptr());
    // SAFETY: every load starts at `i` with `i + 8 ≤ n ≤ len` for all slices.
    unsafe {
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let xv = _mm256_loadu_ps(xp.add(i));
            a1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(y1p.add(i)), a1);
            a2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(y2p.add(i)), a2);
            i += 8;
        }
        let (mut s1, mut s2) = (hsum256(a1), hsum256(a2));
        while i < n {
            let xv = *xp.add(i);
            s1 += xv * *y1p.add(i);
            s2 += xv * *y2p.add(i);
            i += 1;
        }
        (s1, s2)
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
unsafe fn dot_sse2(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = x.len();
    let (xp, yp) = (x.as_ptr(), y.as_ptr());
    // SAFETY: every load starts at `i` with `i + 4 ≤ n ≤ len`; SSE2 is baseline.
    unsafe {
        let mut a0 = _mm_setzero_ps();
        let mut a1 = _mm_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(xp.add(i)), _mm_loadu_ps(yp.add(i))));
            a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(xp.add(i + 4)), _mm_loadu_ps(yp.add(i + 4))));
            i += 8;
        }
        if i + 4 <= n {
            a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(xp.add(i)), _mm_loadu_ps(yp.add(i))));
            i += 4;
        }
        let mut s = hsum128(_mm_add_ps(a0, a1));
        while i < n {
            s += *xp.add(i) * *yp.add(i);
            i += 1;
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
unsafe fn dual_dot_sse2(x: &[f32], y1: &[f32], y2: &[f32]) -> (f32, f32) {
    use core::arch::x86_64::*;
    let n = x.len();
    let (xp, y1p, y2p) = (x.as_ptr(), y1.as_ptr(), y2.as_ptr());
    // SAFETY: every load starts at `i` with `i + 4 ≤ n ≤ len` for all slices.
    unsafe {
        let mut a1 = _mm_setzero_ps();
        let mut a2 = _mm_setzero_ps();
        let mut i = 0;
        while i + 4 <= n {
            let xv = _mm_loadu_ps(xp.add(i));
            a1 = _mm_add_ps(a1, _mm_mul_ps(xv, _mm_loadu_ps(y1p.add(i))));
            a2 = _mm_add_ps(a2, _mm_mul_ps(xv, _mm_loadu_ps(y2p.add(i))));
            i += 4;
        }
        let (mut s1, mut s2) = (hsum128(a1), hsum128(a2));
        while i < n {
            let xv = *xp.add(i);
            s1 += xv * *y1p.add(i);
            s2 += xv * *y2p.add(i);
            i += 1;
        }
        (s1, s2)
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[inline]
unsafe fn hsum128(v: core::arch::x86_64::__m128) -> f32 {
    use core::arch::x86_64::*;
    // SAFETY: pure register shuffles/adds, no memory access.
    unsafe {
        let shuf = _mm_add_ps(v, _mm_shuffle_ps::<0x4E>(v, v));
        let sums = _mm_add_ss(shuf, _mm_shuffle_ps::<0xB1>(shuf, shuf));
        _mm_cvtss_f32(sums)
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx")]
unsafe fn hsum256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    // SAFETY: pure register ops; extracts the two 128-bit halves and folds.
    unsafe {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps::<1>(v);
        hsum128(_mm_add_ps(lo, hi))
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f64_avx2(x: &[f32], y: &[f32]) -> f64 {
    use core::arch::x86_64::*;
    let n = x.len();
    let (xp, yp) = (x.as_ptr(), y.as_ptr());
    // SAFETY: each 4-wide f32 load starts at `i` with `i + 4 ≤ n ≤ len`; the
    // products are widened to f64. Four independent accumulators hide the FMA
    // latency (the loop would otherwise be latency-bound on a single chain).
    unsafe {
        let mut a0 = _mm256_setzero_pd();
        let mut a1 = _mm256_setzero_pd();
        let mut a2 = _mm256_setzero_pd();
        let mut a3 = _mm256_setzero_pd();
        let mut i = 0;
        while i + 16 <= n {
            let p = |o: usize| _mm256_cvtps_pd(_mm_loadu_ps(xp.add(o)));
            let q = |o: usize| _mm256_cvtps_pd(_mm_loadu_ps(yp.add(o)));
            a0 = _mm256_fmadd_pd(p(i), q(i), a0);
            a1 = _mm256_fmadd_pd(p(i + 4), q(i + 4), a1);
            a2 = _mm256_fmadd_pd(p(i + 8), q(i + 8), a2);
            a3 = _mm256_fmadd_pd(p(i + 12), q(i + 12), a3);
            i += 16;
        }
        let mut acc = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
        while i + 4 <= n {
            acc = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i))),
                _mm256_cvtps_pd(_mm_loadu_ps(yp.add(i))),
                acc,
            );
            i += 4;
        }
        // Horizontal sum of the 4 f64 lanes.
        let lo = _mm256_castpd256_pd128(acc);
        let hi = _mm256_extractf128_pd::<1>(acc);
        let s2 = _mm_add_pd(lo, hi);
        let mut s = _mm_cvtsd_f64(_mm_add_pd(s2, _mm_unpackhi_pd(s2, s2)));
        while i < n {
            s += f64::from(*xp.add(i)) * f64::from(*yp.add(i));
            i += 1;
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
unsafe fn dot_f64_sse2(x: &[f32], y: &[f32]) -> f64 {
    use core::arch::x86_64::*;
    let n = x.len();
    let (xp, yp) = (x.as_ptr(), y.as_ptr());
    // SAFETY: each 2-wide f64 widening reads f32 lanes `i..i+2 ≤ n ≤ len`;
    // SSE2 (`cvtps2pd` widens the low two f32) is baseline on x86-64.
    unsafe {
        let mut acc = _mm_setzero_pd();
        let mut i = 0;
        while i + 2 <= n {
            // Load 2 f32 and widen the low pair to 2 f64.
            let xf = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(xp.add(i).cast())));
            let yf = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(yp.add(i).cast())));
            acc = _mm_add_pd(acc, _mm_mul_pd(xf, yf));
            i += 2;
        }
        let mut s = _mm_cvtsd_f64(_mm_add_pd(acc, _mm_unpackhi_pd(acc, acc)));
        while i < n {
            s += f64::from(*xp.add(i)) * f64::from(*yp.add(i));
            i += 1;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    fn seq(n: usize, off: usize) -> Vec<f32> {
        (0..n).map(|i| (((i + off) * 7 % 19) as f32 / 19.0) - 0.5).collect()
    }
    fn ref_dot(x: &[f32], y: &[f32]) -> f64 {
        x.iter().zip(y).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum()
    }
    fn close(got: f64, want: f64, tag: &str) {
        assert!((got - want).abs() <= 1e-3 + want.abs() * 1e-4, "{tag}: {got} vs {want}");
    }

    /// Sizes that straddle the SSE2 (4) and AVX2 (8/16) lane widths, so every
    /// kernel runs both its vector body and its scalar tail. Under Miri this is
    /// the soundness check for unsafe site #2 (the `simd.rs` dot/FIR/convert
    /// kernels): each load/store must stay within the asserted slice bounds.
    const SIZES: [usize; 16] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48];

    #[test]
    fn dot_dual_f64_match_reference() {
        for &n in &SIZES {
            // `y` is longer than `x` to exercise the `y.len() >= x.len()` contract.
            let x = seq(n, 0);
            let y = seq(n + 3, 5);
            let y2 = seq(n + 3, 11);
            let want = ref_dot(&x, &y);
            close(f64::from(dot(&x, &y)), want, "dot");
            close(dot_f64(&x, &y), want, "dot_f64");
            let (a, b) = dual_dot(&x, &y, &y2);
            close(f64::from(a), want, "dual.0");
            close(f64::from(b), ref_dot(&x, &y2), "dual.1");
        }
    }

    #[test]
    fn pitch_xcorr_matches_reference() {
        for &len in &[1usize, 3, 4, 7, 8, 9, 16, 17, 32] {
            for &max in &[1usize, 4, 5, 8] {
                let x = seq(len, 0);
                let y = seq(max + len, 3); // y.len() = max+len ≥ out.len()+len-1
                let mut out = vec![0.0f32; max];
                pitch_xcorr(&x, &y, &mut out, len);
                for (i, &o) in out.iter().enumerate() {
                    close(f64::from(o), ref_dot(&x, &y[i..i + len]), "xcorr");
                }
            }
        }
    }

    #[test]
    fn fir6_matches_reference() {
        let c = [0.2f32, -0.1, 0.3, 0.05, -0.25, 0.15];
        for &m in &[1usize, 3, 4, 7, 8, 9, 16, 17, 32] {
            let inp = seq(m + 5, 2);
            let mut out = vec![0.0f32; m];
            fir6(&mut out, &inp, c);
            for (j, &o) in out.iter().enumerate() {
                let want: f64 = (0..6).map(|k| f64::from(inp[j + k]) * f64::from(c[k])).sum();
                close(f64::from(o), want, "fir6");
            }
        }
    }

    #[test]
    fn scale_round_matches_f32_round_exactly() {
        for &n in &SIZES {
            let x = seq(n, 4);
            let mut out = vec![0i16; n];
            scale_round_to_i16(&mut out, &x, 32768.0);
            for (i, &o) in out.iter().enumerate() {
                let want = (x[i] * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
                assert_eq!(o, want, "scale_round n={n} i={i}");
            }
        }
    }
}

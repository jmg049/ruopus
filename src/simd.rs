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
    // SAFETY: every load below starts at `i` with `i + width ≤ n ≤ len`.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i)), _mm256_loadu_ps(yp.add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i + 8)), _mm256_loadu_ps(yp.add(i + 8)), a1);
            i += 16;
        }
        if i + 8 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i)), _mm256_loadu_ps(yp.add(i)), a0);
            i += 8;
        }
        let mut s = hsum256(_mm256_add_ps(a0, a1));
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
    // products are widened to f64 and accumulated in four f64 lanes.
    unsafe {
        let mut acc = _mm256_setzero_pd();
        let mut i = 0;
        while i + 4 <= n {
            let xf = _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i)));
            let yf = _mm256_cvtps_pd(_mm_loadu_ps(yp.add(i)));
            acc = _mm256_fmadd_pd(xf, yf, acc);
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

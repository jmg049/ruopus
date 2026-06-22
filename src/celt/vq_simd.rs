//! SIMD acceleration for the PVQ pulse search (`op_pvq_search`) - the dominant
//! cost of CELT band encoding (the O(K·N) inner argmax). x86-64 guarantees
//! SSE2 in its baseline ABI, so the SSE2 path runs unconditionally there; other
//! targets fall back to the scalar search in `vq.rs`.
//!
//! Ports `celt/x86/vq_sse2.c`. Per pulse the search maximises
//! `r = (xy + X[j]) · rsqrt(yy + y[j])` - `rsqrt` is a fast 12-bit
//! approximation, so the chosen pulse vector can differ from the scalar
//! cross-multiply search by a hair. That is fine: the result is still a valid
//! Opus bitstream that the decoder follows exactly (the entropy oracle is the
//! range coder, not the encoder's pulse choice), and every round-trip /
//! conformance test passes on either path. See `docs/unsafe.md`.

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(super) fn op_pvq_search(x: &[f32], iy: &mut [i32], k: usize) -> f32 {
    // SAFETY: AVX2 (when detected) and SSE2 (x86-64 baseline) intrinsics are
    // available on this target. Both kernels operate on local buffers padded to
    // a multiple of the lane width ≥ N, so each vector access stays in bounds;
    // results are copied back into `iy[..N]`.
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 confirmed available; the kernel only touches its own
        // `cap`-padded local buffers.
        unsafe { op_pvq_search_avx2(x, iy, k) }
    } else {
        // SAFETY: SSE2 is part of the x86-64 baseline ABI.
        unsafe { op_pvq_search_sse2(x, iy, k) }
    }
}

#[cfg(target_arch = "x86_64")]
std::thread_local! {
    /// Reused work buffers for the pulse search (`xs`, `y`, `signy`, `iy`),
    /// sized to the band's padded width. The search runs ~once per band per
    /// frame, so reusing these avoids four heap allocations (and their zeroing)
    /// on every call. They are re-initialised each call (sentinels in the
    /// padding, cleared `y`/`iy`), so no state carries between calls.
    static PVQ_XS: core::cell::RefCell<alloc::vec::Vec<f32>> = const { core::cell::RefCell::new(alloc::vec::Vec::new()) };
    static PVQ_Y: core::cell::RefCell<alloc::vec::Vec<f32>> = const { core::cell::RefCell::new(alloc::vec::Vec::new()) };
    static PVQ_SIGNY: core::cell::RefCell<alloc::vec::Vec<f32>> = const { core::cell::RefCell::new(alloc::vec::Vec::new()) };
    static PVQ_IY: core::cell::RefCell<alloc::vec::Vec<i32>> = const { core::cell::RefCell::new(alloc::vec::Vec::new()) };
}

/// AVX2 port of [`op_pvq_search_sse2`]: identical algorithm, 8 lanes wide, using
/// the native 32-bit integer max (`_mm256_max_epi32`) for index tracking. The
/// chosen pulse vector is still a valid PVQ codeword (only the per-pulse argmax
/// tie-break may differ), so the bitstream round-trips exactly.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn op_pvq_search_avx2(input: &[f32], iy_out: &mut [i32], k: usize) -> f32 {
    use core::arch::x86_64::*;

    let n = input.len();
    let cap = (n + 7).next_multiple_of(8);
    let mut xs = PVQ_XS.with(|s| core::mem::take(&mut *s.borrow_mut()));
    let mut y = PVQ_Y.with(|s| core::mem::take(&mut *s.borrow_mut()));
    let mut signy = PVQ_SIGNY.with(|s| core::mem::take(&mut *s.borrow_mut()));
    let mut iy = PVQ_IY.with(|s| core::mem::take(&mut *s.borrow_mut()));
    xs.resize(cap, 0.0);
    y.resize(cap, 0.0);
    signy.resize(cap, 0.0);
    iy.resize(cap, 0);
    xs[..n].copy_from_slice(input);
    xs[n..cap].fill(0.0);

    // Broadcast the horizontal max of an 8-lane vector to all lanes.
    #[allow(unsafe_code)]
    #[target_feature(enable = "avx2")]
    unsafe fn hmax(v: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256 {
        use core::arch::x86_64::*;
        let v = _mm256_max_ps(v, _mm256_permute2f128_ps::<0x01>(v, v));
        let v = _mm256_max_ps(v, _mm256_shuffle_ps::<0x4E>(v, v));
        _mm256_max_ps(v, _mm256_shuffle_ps::<0xB1>(v, v))
    }
    // Scalar max of an 8-lane i32 vector.
    #[allow(unsafe_code)]
    #[target_feature(enable = "avx2")]
    unsafe fn hmax_i32(p: core::arch::x86_64::__m256i) -> i32 {
        use core::arch::x86_64::*;
        let m = _mm_max_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256::<1>(p));
        let m = _mm_max_epi32(m, _mm_shuffle_epi32::<0x4E>(m));
        let m = _mm_max_epi32(m, _mm_shuffle_epi32::<0xB1>(m));
        _mm_cvtsi128_si32(m)
    }

    // SAFETY: every vector access below starts at `j < n ≤ cap - 7`, within the
    // `cap`-sized buffers; the padding lanes carry search-losing sentinels.
    unsafe {
        let signmask = _mm256_set1_ps(-0.0);
        let eights = _mm256_set1_epi32(8);
        let xp = xs.as_mut_ptr();
        let yp = y.as_mut_ptr();
        let iyp = iy.as_mut_ptr();
        let sp = signy.as_mut_ptr();

        // Strip signs, accumulate Σ|x|, clear y/iy.
        let mut sums = _mm256_setzero_ps();
        let mut j = 0;
        while j < n {
            let x8 = _mm256_loadu_ps(xp.add(j));
            let s8 = _mm256_cmp_ps::<_CMP_LT_OQ>(x8, _mm256_setzero_ps());
            let x8 = _mm256_andnot_ps(signmask, x8);
            sums = _mm256_add_ps(sums, x8);
            _mm256_storeu_ps(yp.add(j), _mm256_setzero_ps());
            _mm256_storeu_si256(iyp.add(j).cast(), _mm256_setzero_si256());
            _mm256_storeu_ps(xp.add(j), x8);
            _mm256_storeu_ps(sp.add(j), s8);
            j += 8;
        }
        let sum_all = {
            let h = _mm256_add_ps(sums, _mm256_permute2f128_ps::<0x01>(sums, sums));
            let h = _mm256_add_ps(h, _mm256_shuffle_ps::<0x4E>(h, h));
            let h = _mm256_add_ps(h, _mm256_shuffle_ps::<0xB1>(h, h));
            _mm256_cvtss_f32(h)
        };

        let mut xy = 0.0f32;
        let mut yy = 0.0f32;
        let mut pulses_left = k as i32;

        // Pre-search: project onto the pyramid.
        if k > (n >> 1) {
            let mut rcp_sum = sum_all;
            if !(sum_all > 1e-15 && sum_all < 64.0) {
                xs[0] = 1.0;
                for v in &mut xs[1..n] {
                    *v = 0.0;
                }
                rcp_sum = 1.0;
            }
            let rcp8 = _mm256_set1_ps((k as f32 + 0.8) / rcp_sum);
            let mut xy8 = _mm256_setzero_ps();
            let mut yy8 = _mm256_setzero_ps();
            let mut psum = _mm256_setzero_si256();
            let mut j = 0;
            while j < n {
                let x8 = _mm256_loadu_ps(xp.add(j));
                let iy8 = _mm256_cvttps_epi32(_mm256_mul_ps(x8, rcp8));
                psum = _mm256_add_epi32(psum, iy8);
                _mm256_storeu_si256(iyp.add(j).cast(), iy8);
                let y8 = _mm256_cvtepi32_ps(iy8);
                xy8 = _mm256_add_ps(xy8, _mm256_mul_ps(x8, y8));
                yy8 = _mm256_add_ps(yy8, _mm256_mul_ps(y8, y8));
                _mm256_storeu_ps(yp.add(j), _mm256_add_ps(y8, y8));
                j += 8;
            }
            let ph = _mm_add_epi32(_mm256_castsi256_si128(psum), _mm256_extracti128_si256::<1>(psum));
            let ph = _mm_add_epi32(ph, _mm_shuffle_epi32::<0x4E>(ph));
            let ph = _mm_add_epi32(ph, _mm_shuffle_epi32::<0xB1>(ph));
            pulses_left -= _mm_cvtsi128_si32(ph);
            let hx = {
                let h = _mm256_add_ps(xy8, _mm256_permute2f128_ps::<0x01>(xy8, xy8));
                let h = _mm256_add_ps(h, _mm256_shuffle_ps::<0x4E>(h, h));
                _mm256_cvtss_f32(_mm256_add_ps(h, _mm256_shuffle_ps::<0xB1>(h, h)))
            };
            xy = hx;
            let hy = {
                let h = _mm256_add_ps(yy8, _mm256_permute2f128_ps::<0x01>(yy8, yy8));
                let h = _mm256_add_ps(h, _mm256_shuffle_ps::<0x4E>(h, h));
                _mm256_cvtss_f32(_mm256_add_ps(h, _mm256_shuffle_ps::<0xB1>(h, h)))
            };
            yy = hy;
        }

        // Sentinels in the padding so those lanes never win the search.
        for p in n..cap {
            xs[p] = -100.0;
            y[p] = 100.0;
        }

        if pulses_left > n as i32 + 3 {
            let tmp = pulses_left as f32;
            yy += tmp * tmp;
            yy += tmp * y[0];
            iy[0] += pulses_left;
            pulses_left = 0;
        }

        for _ in 0..pulses_left {
            yy += 1.0;
            let xy8 = _mm256_set1_ps(xy);
            let yy8 = _mm256_set1_ps(yy);
            let mut max = _mm256_set1_ps(-f32::INFINITY);
            let mut pos = _mm256_setzero_si256();
            let mut count = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            let mut j = 0;
            while j < n {
                let x8 = _mm256_add_ps(_mm256_loadu_ps(xp.add(j)), xy8);
                let y8 = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_loadu_ps(yp.add(j)), yy8));
                let r8 = _mm256_mul_ps(x8, y8);
                let gt = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(r8, max));
                pos = _mm256_max_epi32(pos, _mm256_and_si256(count, gt));
                max = _mm256_max_ps(max, r8);
                count = _mm256_add_epi32(count, eights);
                j += 8;
            }
            // Recover the index of a max lane.
            let maxb = hmax(max);
            pos = _mm256_and_si256(pos, _mm256_castps_si256(_mm256_cmp_ps::<_CMP_EQ_OQ>(max, maxb)));
            let best_id = hmax_i32(pos) as usize;

            xy += xs[best_id];
            yy += y[best_id];
            y[best_id] += 2.0;
            iy[best_id] += 1;
        }

        // Restore the signs.
        let mut j = 0;
        while j < n {
            let y8 = _mm256_loadu_si256(iyp.add(j).cast());
            let s8 = _mm256_castps_si256(_mm256_loadu_ps(sp.add(j)));
            let y8 = _mm256_xor_si256(_mm256_add_epi32(y8, s8), s8);
            _mm256_storeu_si256(iyp.add(j).cast(), y8);
            j += 8;
        }

        iy_out.copy_from_slice(&iy[..n]);
        PVQ_XS.with(|s| *s.borrow_mut() = core::mem::take(&mut xs));
        PVQ_Y.with(|s| *s.borrow_mut() = core::mem::take(&mut y));
        PVQ_SIGNY.with(|s| *s.borrow_mut() = core::mem::take(&mut signy));
        PVQ_IY.with(|s| *s.borrow_mut() = core::mem::take(&mut iy));
        yy
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
unsafe fn op_pvq_search_sse2(input: &[f32], iy_out: &mut [i32], k: usize) -> f32 {
    use core::arch::x86_64::*;

    let n = input.len();
    let cap = (n + 3).next_multiple_of(4);
    let mut xs = PVQ_XS.with(|s| core::mem::take(&mut *s.borrow_mut()));
    let mut y = PVQ_Y.with(|s| core::mem::take(&mut *s.borrow_mut()));
    let mut signy = PVQ_SIGNY.with(|s| core::mem::take(&mut *s.borrow_mut()));
    let mut iy = PVQ_IY.with(|s| core::mem::take(&mut *s.borrow_mut()));
    xs.resize(cap, 0.0);
    y.resize(cap, 0.0);
    signy.resize(cap, 0.0);
    iy.resize(cap, 0);
    xs[..n].copy_from_slice(input);
    xs[n..cap].fill(0.0);

    // SAFETY: see the wrapper; all indices below are < `cap` and 16-byte SIMD
    // ops read/write 4 lanes starting at `j < n ≤ cap - 3`.
    unsafe {
        let signmask = _mm_set1_ps(-0.0);
        let fours = _mm_set1_epi32(4);
        let xp = xs.as_mut_ptr();
        let yp = y.as_mut_ptr();
        let iyp = iy.as_mut_ptr();
        let sp = signy.as_mut_ptr();

        // Strip signs, accumulate Σ|x|, clear y/iy.
        let mut sums = _mm_setzero_ps();
        let mut j = 0;
        while j < n {
            let x4 = _mm_loadu_ps(xp.add(j));
            let s4 = _mm_cmplt_ps(x4, _mm_setzero_ps());
            let x4 = _mm_andnot_ps(signmask, x4);
            sums = _mm_add_ps(sums, x4);
            _mm_storeu_ps(yp.add(j), _mm_setzero_ps());
            _mm_storeu_si128(iyp.add(j).cast(), _mm_setzero_si128());
            _mm_storeu_ps(xp.add(j), x4);
            _mm_storeu_ps(sp.add(j), s4);
            j += 4;
        }
        sums = _mm_add_ps(sums, _mm_shuffle_ps::<0x4E>(sums, sums));
        sums = _mm_add_ps(sums, _mm_shuffle_ps::<0xB1>(sums, sums));

        let mut xy = 0.0f32;
        let mut yy = 0.0f32;
        let mut pulses_left = k as i32;

        // Pre-search: project onto the pyramid.
        if k > (n >> 1) {
            let sum = _mm_cvtss_f32(sums);
            let mut sums = sums;
            if !(sum > 1e-15 && sum < 64.0) {
                xs[0] = 1.0;
                for v in &mut xs[1..n] {
                    *v = 0.0;
                }
                sums = _mm_set1_ps(1.0);
            }
            let rcp4 = _mm_mul_ps(_mm_set1_ps(k as f32 + 0.8), _mm_rcp_ps(sums));
            let mut xy4 = _mm_setzero_ps();
            let mut yy4 = _mm_setzero_ps();
            let mut psum = _mm_setzero_si128();
            let mut j = 0;
            while j < n {
                let x4 = _mm_loadu_ps(xp.add(j));
                let rx4 = _mm_mul_ps(x4, rcp4);
                let iy4 = _mm_cvttps_epi32(rx4);
                psum = _mm_add_epi32(psum, iy4);
                _mm_storeu_si128(iyp.add(j).cast(), iy4);
                let y4 = _mm_cvtepi32_ps(iy4);
                xy4 = _mm_add_ps(xy4, _mm_mul_ps(x4, y4));
                yy4 = _mm_add_ps(yy4, _mm_mul_ps(y4, y4));
                _mm_storeu_ps(yp.add(j), _mm_add_ps(y4, y4));
                j += 4;
            }
            psum = _mm_add_epi32(psum, _mm_shuffle_epi32::<0x4E>(psum));
            psum = _mm_add_epi32(psum, _mm_shuffle_epi32::<0xB1>(psum));
            pulses_left -= _mm_cvtsi128_si32(psum);
            xy4 = _mm_add_ps(xy4, _mm_shuffle_ps::<0x4E>(xy4, xy4));
            xy4 = _mm_add_ps(xy4, _mm_shuffle_ps::<0xB1>(xy4, xy4));
            xy = _mm_cvtss_f32(xy4);
            yy4 = _mm_add_ps(yy4, _mm_shuffle_ps::<0x4E>(yy4, yy4));
            yy4 = _mm_add_ps(yy4, _mm_shuffle_ps::<0xB1>(yy4, yy4));
            yy = _mm_cvtss_f32(yy4);
        }

        // Sentinels in the padding so those lanes never win the search.
        for p in n..cap {
            xs[p] = -100.0;
            y[p] = 100.0;
        }

        if pulses_left > n as i32 + 3 {
            let tmp = pulses_left as f32;
            yy += tmp * tmp;
            yy += tmp * y[0];
            iy[0] += pulses_left;
            pulses_left = 0;
        }

        for _ in 0..pulses_left {
            yy += 1.0;
            let xy4 = _mm_set1_ps(xy);
            let yy4 = _mm_set1_ps(yy);
            let mut max = _mm_setzero_ps();
            let mut pos = _mm_setzero_si128();
            let mut count = _mm_set_epi32(3, 2, 1, 0);
            let mut j = 0;
            while j < n {
                let x4 = _mm_add_ps(_mm_loadu_ps(xp.add(j)), xy4);
                let y4 = _mm_rsqrt_ps(_mm_add_ps(_mm_loadu_ps(yp.add(j)), yy4));
                let r4 = _mm_mul_ps(x4, y4);
                pos = _mm_max_epi16(pos, _mm_and_si128(count, _mm_castps_si128(_mm_cmpgt_ps(r4, max))));
                max = _mm_max_ps(max, r4);
                count = _mm_add_epi32(count, fours);
                j += 4;
            }
            // Horizontal max, then recover the index of a max lane.
            let mut max2 = _mm_max_ps(max, _mm_shuffle_ps::<0x4E>(max, max));
            max2 = _mm_max_ps(max2, _mm_shuffle_ps::<0xB1>(max2, max2));
            pos = _mm_and_si128(pos, _mm_castps_si128(_mm_cmpeq_ps(max, max2)));
            pos = _mm_max_epi16(pos, _mm_unpackhi_epi64(pos, pos));
            pos = _mm_max_epi16(pos, _mm_shufflelo_epi16::<0x4E>(pos));
            let best_id = _mm_cvtsi128_si32(pos) as usize;

            xy += xs[best_id];
            yy += y[best_id];
            y[best_id] += 2.0;
            iy[best_id] += 1;
        }

        // Restore the signs.
        let mut j = 0;
        while j < n {
            let y4 = _mm_loadu_si128(iyp.add(j).cast());
            let s4 = _mm_castps_si128(_mm_loadu_ps(sp.add(j)));
            let y4 = _mm_xor_si128(_mm_add_epi32(y4, s4), s4);
            _mm_storeu_si128(iyp.add(j).cast(), y4);
            j += 4;
        }

        iy_out.copy_from_slice(&iy[..n]);
        PVQ_XS.with(|s| *s.borrow_mut() = core::mem::take(&mut xs));
        PVQ_Y.with(|s| *s.borrow_mut() = core::mem::take(&mut y));
        PVQ_SIGNY.with(|s| *s.borrow_mut() = core::mem::take(&mut signy));
        PVQ_IY.with(|s| *s.borrow_mut() = core::mem::take(&mut iy));
        yy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    /// A deterministic, non-trivial band of `n` samples.
    fn band(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5).collect()
    }

    /// Exercises both SIMD pulse-search kernels across a sweep of band widths
    /// that straddle the SSE2 (4-lane) and AVX2 (8-lane) boundaries - including
    /// the padded-tail cases - and asserts the PVQ invariant (`Σ|iy| == k`).
    ///
    /// Under Miri this is the soundness check for unsafe site #1: every load and
    /// store in the kernels (over the `cap`-padded local buffers) must stay in
    /// bounds, and the padding lanes must never win a pulse. It is deliberately
    /// small so it completes quickly in the interpreter.
    #[test]
    #[allow(unsafe_code, reason = "calls the SIMD kernels directly to validate them under Miri")]
    fn pvq_search_kernels_place_k_pulses_in_bounds() {
        for &n in &[2usize, 3, 4, 5, 7, 8, 9, 11, 15, 16, 17, 23, 24, 25] {
            for &k in &[1usize, 2, 5, n.saturating_sub(1).max(1), n + 3] {
                let x = band(n);

                // The public dispatcher (whichever kernel this CPU/Miri selects).
                let mut iy = vec![0i32; n];
                let _ = op_pvq_search(&x, &mut iy, k);
                let sum: i32 = iy.iter().map(|v| v.abs()).sum();
                assert_eq!(sum, k as i32, "dispatch: n={n} k={k} sum|iy|={sum}");

                // The SSE2 kernel directly (always valid on x86-64).
                let mut iy_sse = vec![0i32; n];
                // SAFETY: SSE2 is part of the x86-64 baseline ABI; `iy_sse` is `n` long.
                let _ = unsafe { op_pvq_search_sse2(&x, &mut iy_sse, k) };
                let sum_sse: i32 = iy_sse.iter().map(|v| v.abs()).sum();
                assert_eq!(sum_sse, k as i32, "sse2: n={n} k={k}");

                // The AVX2 kernel directly, when this target/Miri reports it.
                if std::is_x86_feature_detected!("avx2") {
                    let mut iy_avx = vec![0i32; n];
                    // SAFETY: AVX2 confirmed available; `iy_avx` is `n` long.
                    let _ = unsafe { op_pvq_search_avx2(&x, &mut iy_avx, k) };
                    let sum_avx: i32 = iy_avx.iter().map(|v| v.abs()).sum();
                    assert_eq!(sum_avx, k as i32, "avx2: n={n} k={k}");
                }
            }
        }
    }
}

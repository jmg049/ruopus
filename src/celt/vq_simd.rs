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
    // SAFETY: SSE2 is part of the x86-64 baseline (always enabled for this
    // target), so every intrinsic below is available. All loads/stores operate
    // on local buffers sized up to a multiple of 4 ≥ N + 3, so each 128-bit
    // (4-lane) access stays in bounds; results are copied back into `iy[..N]`.
    unsafe { op_pvq_search_sse2(x, iy, k) }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
unsafe fn op_pvq_search_sse2(input: &[f32], iy_out: &mut [i32], k: usize) -> f32 {
    use core::arch::x86_64::*;

    let n = input.len();
    let cap = (n + 3).next_multiple_of(4);
    let mut xs = alloc::vec![0.0f32; cap];
    let mut y = alloc::vec![0.0f32; cap];
    let mut signy = alloc::vec![0.0f32; cap];
    let mut iy = alloc::vec![0i32; cap];
    xs[..n].copy_from_slice(input);

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
        yy
    }
}

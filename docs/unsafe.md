# Unsafe code

`opus_native` is overwhelmingly safe Rust. The crate-level lint is
`unsafe_code = "deny"` (not `forbid`), so `unsafe` is rejected *everywhere by
default* and only permitted at sites that opt in with an explicit
`#[allow(unsafe_code)]` and a `// SAFETY:` justification. Clippy's
`undocumented_unsafe_blocks = "deny"` additionally forces every `unsafe` block
to carry that justification. This file is the authoritative list of those
sites; if you add one, add it here.

The only reason `unsafe` exists in the crate is hand-written SIMD: a handful of
performance-critical hot loops use `core::arch` intrinsics, which are `unsafe`
to call because the compiler cannot prove the target CPU supports the
instruction. We use only `std::arch` intrinsics behind compile-time or runtime
CPU-feature checks - never `portable_simd` (nightly) and never raw inline asm.

## Why SIMD here is also conformance-safe

These kernels live on the **encoder** path. The Opus bitstream is defined by the
range coder, not by any particular encoder pulse choice: as long as the encoder
and decoder agree on the range state symbol-for-symbol (which they do - that is
tested), any *valid* pulse vector round-trips exactly. The SIMD pulse search
uses a fast `rsqrt` approximation that can pick a marginally different - still
valid - vector than the scalar search. Every round-trip and conformance test
passes on either path, so the approximation changes nothing observable.

## Sites

| # | Location | Intrinsics | Why it is sound |
|---|----------|------------|-----------------|
| 1 | `src/celt/vq_simd.rs` - `op_pvq_search` and its `_sse2`/`_avx2` kernels | SSE2 baseline + AVX2 (`core::arch::x86_64`) | Gated `#[cfg(target_arch = "x86_64")]`. SSE2 is part of the x86-64 baseline ABI (always available); the AVX2 kernel is reached only behind `is_x86_feature_detected!("avx2")`. All loads/stores operate on local heap buffers padded to a multiple of the lane width ≥ N, so each vector access starts at `j < N ≤ cap - (lanes-1)` and stays in bounds; results are copied back into `iy[..N]`. The padding lanes carry search-losing sentinels so they can never be selected. |
| 2 | `src/simd.rs` - `dot` / `dual_dot` / `dot_f64` and their `_avx2`/`_sse2` kernels | SSE2 baseline + AVX2/FMA (`core::arch::x86_64`) | Gated `#[cfg(target_arch = "x86_64")]`. Dot products over `&[f32]` slices (`dot_f64` widens to an `f64` accumulator for the SILK pitch analysis); the public wrappers assert `y.len() >= x.len()`, and every vector load starts at `i` with `i + width ≤ x.len()`, with a scalar tail for the remainder - so all reads stay in bounds. The AVX2+FMA kernels are reached only behind a cached `is_x86_feature_detected!("avx2","fma")` (needs `std`; without it the SSE2 baseline path is used). Results feed encoder *decisions* only (pitch lag/gain, analysis correlations), which the reference float build does not require to be bit-exact, so FMA's fused rounding is acceptable. |
| 3 | `src/celt/mdct.rs` - `prerotate` | SSE2 (`core::arch::x86_64`) | Gated `#[cfg(target_arch = "x86_64")]`; SSE2 is the x86-64 baseline. The forward MDCT pre-rotation as a complex multiply over `f`/`tw`/`fc`: each iteration reads `f[2i..2i+8]` and `tw[2i..2i+8]` and writes `fc[2i..2i+8]` (as `f32`) with `i + 4 ≤ n4`, all within asserted lengths (`f.len() ≥ 2·n4`, `tw.len() = 2·n4`, `fc.len() ≥ n4`); a scalar tail covers the remainder. The `(f32, f32)`→two-`f32` store layout is guarded by the MDCT TDAC round-trip test. |

### Adding a site

1. Keep the `unsafe` surface as small as possible - one `unsafe fn` per kernel,
   called from a thin safe wrapper.
2. Put a `// SAFETY:` comment on the wrapper *and* on the inner `unsafe` block
   explaining the invariant that makes every intrinsic/access valid.
3. If the instruction set is not part of the target's baseline (e.g. AVX2),
   gate the call on `is_x86_feature_detected!` and keep a scalar fallback.
4. Add a row to the table above.
5. Add a focused soundness test that drives the new kernel directly over
   boundary-straddling sizes (so Miri can check it - see below), and make sure
   `tools/miri.sh` reaches it.

## Verifying soundness with Miri

Every `unsafe` block above is checked under [Miri](https://github.com/rust-lang/miri),
Rust's interpreter for detecting undefined behaviour (out-of-bounds accesses,
misalignment, invalid values, aliasing violations). Miri *executes* the SSE2
and AVX2 intrinsics, so it validates the actual vector loads and stores in each
kernel - not a scalar stand-in.

```
rustup +nightly component add miri   # one-time
tools/miri.sh
```

The runner exercises every unsafe site twice - once on the x86-64 baseline
(SSE2, which runs unconditionally) and once with `-C target-feature=+avx2,+fma`
so the AVX2 paths are taken too - through these focused tests:

| Test | Covers |
|------|--------|
| `simd::tests::*` | site #2 (`dot`/`dual_dot`/`dot_f64`/`pitch_xcorr`/`fir6`/`scale_round_to_i16`) and site #1 (`op_pvq_search` SSE2 + AVX2 kernels) |
| `celt::mdct::tests::*` | site #3 (forward-MDCT pre-rotation) |

Each test sweeps band/vector lengths that straddle the 4-lane (SSE2) and 8-lane
(AVX2) widths - including the padded-tail cases - and checks the result against
a scalar reference, so Miri sees the full range of in-bounds and remainder
accesses. The full encode/decode round-trips are **not** run under Miri (the
interpreter is far too slow for them); their large-scale correctness is covered
by the native test suite and the conformance vectors, while these kernel tests
isolate exactly the memory-safety-critical `unsafe` code.

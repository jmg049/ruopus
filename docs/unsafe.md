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
| 1 | `src/celt/vq_simd.rs` - `op_pvq_search` / `op_pvq_search_sse2` | SSE2 (`core::arch::x86_64`) | Gated `#[cfg(target_arch = "x86_64")]`. SSE2 is part of the x86-64 baseline ABI, so every intrinsic is always available on this target - no runtime detection needed. All loads/stores operate on local heap buffers sized to a multiple of 4 ≥ N + 3, so each 128-bit (4-lane) access starts at `j < N ≤ cap - 3` and stays in bounds; results are copied back into `iy[..N]`. The padding lanes carry search-losing sentinels so they can never be selected. |

### Adding a site

1. Keep the `unsafe` surface as small as possible - one `unsafe fn` per kernel,
   called from a thin safe wrapper.
2. Put a `// SAFETY:` comment on the wrapper *and* on the inner `unsafe` block
   explaining the invariant that makes every intrinsic/access valid.
3. If the instruction set is not part of the target's baseline (e.g. AVX2),
   gate the call on `is_x86_feature_detected!` and keep a scalar fallback.
4. Add a row to the table above.

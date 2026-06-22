#!/usr/bin/env bash
# Validate the crate's `unsafe` SIMD kernels under Miri (Rust's UB detector).
#
# Every `unsafe` site in this crate is a hand-written SIMD kernel (see
# docs/unsafe.md). This script runs the focused kernel soundness tests under
# Miri so that every load/store in those kernels is checked for out-of-bounds,
# misalignment, and other UB - once on the x86-64 baseline (SSE2, which runs
# unconditionally) and once with AVX2+FMA forced on, so *both* code paths of
# each runtime-dispatched site are exercised.
#
# Miri cannot run the full encode/decode round-trips at a useful speed (the
# interpreter is orders of magnitude slower), so these tests exercise the
# kernels directly with boundary-straddling sizes instead. Correctness of the
# kernels at large scale is covered by the normal (native) test suite and the
# conformance vectors.
#
# Requires the nightly toolchain with the `miri` component:
#   rustup +nightly component add miri
#
# Usage:  tools/miri.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# The unsafe-kernel soundness tests:
#   simd::tests        -> sites #1 (op_pvq_search) and #2 (dot/FIR/convert)
#   celt::mdct::tests  -> site #3 (forward-MDCT pre-rotation)
FILTERS=("simd::tests" "celt::mdct::tests")
ARGS=(--no-default-features --features std --lib)

run_config() {
    local label="$1" rustflags="$2"
    echo "================================================================"
    echo "Miri: $label"
    echo "================================================================"
    for f in "${FILTERS[@]}"; do
        MIRIFLAGS="-Zmiri-disable-isolation" RUSTFLAGS="$rustflags" \
            cargo +nightly miri test "${ARGS[@]}" "$f"
    done
}

run_config "x86-64 baseline (SSE2)" ""
run_config "AVX2 + FMA" "-C target-feature=+avx2,+fma"

echo
echo "All Miri SIMD soundness checks passed."

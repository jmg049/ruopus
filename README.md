# opus_native

A pure-Rust implementation of the [Opus audio codec](https://opus-codec.org/)
([RFC 6716](https://www.rfc-editor.org/rfc/rfc6716)).

**No FFI. No dependencies. Unsafe only in documented SIMD kernels.**

> **Status: pre-release, under active development.** The decoder is
> feature-complete and passes the official conformance criterion: all twelve
> test vectors at 99.2-100% `opus_compare` quality, bit-exact final ranges on
> every packet, sample-identical SILK PCM. Loss concealment, in-band FEC, and
> DTX are validated sample-exact against libopus; output rates 8-48 kHz;
> surround (multistream) files decode end to end. The encoder is next.
> Nothing here is API-stable yet.

## Why

Every Rust project that touches Opus today links `libopus` through FFI. There is
no complete, conformant, pure-Rust implementation - this crate exists to be that:
a decoder (first) and encoder (second) that pass the official Opus test vectors,
compile to any target Rust reaches (including `wasm32` and embedded `no_std`),
and can be embedded by higher-level audio crates without a C toolchain.

`opus_native` is deliberately standalone: it depends on nothing and exposes
plain `&[u8]`/`&[i16]`/`&[f32]` interfaces, so it can sit underneath any audio
framework.

## Design principles

- **Bit-exact**: all entropy-coder arithmetic follows RFC 6716 exactly; the
  encoder's range state is verified to match the decoder's symbol-for-symbol.
- **Decode-first**: the decoder is the normative half of the spec and the half
  that conformance vectors exercise. Encoder work follows decoder work at every
  layer.
- **`unsafe` denied by default**: the crate lint is `unsafe_code = "deny"`, so
  `unsafe` is rejected everywhere except a few explicitly-annotated SIMD hot
  loops, each with a `// SAFETY:` justification and listed in
  [`docs/unsafe.md`](docs/unsafe.md). The only `unsafe` is hand-written
  `std::arch` SIMD (no nightly `portable_simd`, no inline asm); it sits on the
  encoder path where the range coder - not the pulse choice - defines the
  bitstream, so every round-trip and conformance test passes either way.
- **`no_std` + `alloc`**: the `std` feature (on by default) only adds
  `std::error::Error` impls and conveniences.
- **Fast by default, zero-dep by choice**: the codec targets near-real-time
  streaming, so the default build routes the MDCT's inner FFT through the
  [`spectrograms`](https://crates.io/crates/spectrograms) crate's planned
  FFTs (~10× faster decode). The FFT sits behind a seam: disabling the
  `spectrograms` feature leaves a dependency-free build (the built-in
  evaluation) for embedded/wasm-minimal targets. Everything else in the
  crate is dependency-free either way.

## Layout

| Module | RFC 6716 | Status |
|--------|----------|--------|
| `range` | §4.1, §5.1 | Range decoder + encoder, raw bits, uniform ints, `tell`/`tell_frac` |
| `packet` | §3 | TOC parsing, frame packing codes 0-3, padding, R1-R7 validation |
| `lpc` | §4.2/§5.2 groundwork | Levinson-Durbin, LP analysis/synthesis, pitch estimation, LTP |
| `experimental` | - | pre-conformance frame codec, mode detection, crossover, mid/side (feature `experimental-codec`) |
| `decoder` | §4 | the Opus decoder: TOC dispatch, hybrid, redundancy, transitions, PLC/FEC/DTX, 8-48 kHz output - all twelve official vectors bit-exact on the final-range oracle |
| `multistream` | RFC 7845 §5.1.1 | surround layouts: self-delimited demux, N decoders, channel mapping |
| `celt` | §4.3 | complete decoder (RFC 8251 updates included) |
| `silk` | §4.2 | complete decoder for the normal path (PLC/CNG pending); pure SILK vectors decode sample-identically |
| `ogg` | RFC 3533 + RFC 7845 | Ogg pages (CRC, lacing, resync), packet reassembly, `OpusHead`/`OpusTags`, granule/pre-skip/end-trim timing, stream reader + writer |

## Performance

The default build decodes far beyond realtime (one core, release build,
official conformance vectors):

| Stream | `spectrograms` FFT (default) | built-in FFT (zero-dep) |
|--------|------------------------------|-------------------------|
| testvector01 (CELT stereo SWB/FB) | ~410× realtime | ~10× realtime |
| testvector07 (CELT stereo, all bandwidths) | ~730× realtime | - |

Reproduce with
`cargo run --release --example decode_throughput tests/vectors/testvector01.bit`.

### vs libopus

Measured in-process against **libopus 1.6.1** (the C reference, SIMD-enabled,
via its `opus` Rust FFI binding) on the same data - `cargo bench --bench
vs_libopus --features std` (needs a system libopus; a dev-dependency only).
This is pure scalar Rust against hand-optimised SIMD C. There is no other
complete pure-Rust Opus codec to compare against; the FFI binding *is*
libopus, so binding ≈ native C.

**Decode** (× realtime, one core; ratio = how many times faster than libopus):

| Mode | `opus_native` | libopus 1.6 | speedup |
|------|---------------|-------------|---------|
| SILK wideband 16 kb/s | **2090×** | 1160× | **1.8× faster** |
| hybrid fullband 32 kb/s | **1180×** | 800× | **1.5× faster** |
| CELT fullband 64 kb/s | 1320× | 1540× | 0.87× |

We decode speech (SILK/hybrid) faster than SIMD libopus. CELT is closest to
libopus's SIMD (0.87×) after the table-driven CWRS rewrite; the residual gap is
the MDCT, where libopus's SIMD still wins.

**Encode** (× realtime). Both encoders have a complexity knob (0-10,
`set_complexity`); the only fair comparison is at *matched* complexity, so each
column runs `opus_native` and libopus at the **same** setting:

Measured pinned to a single performance core (the test machine is a hybrid
P/E-core CPU; unpinned runs migrate between core types and are not reproducible):

| Mode | ours c0 | libopus c0 | ours / lib | ours c10 | libopus c10 | ours / lib |
|------|---------|-----------|-----------|----------|-------------|-----------|
| SILK wideband 16 kb/s | 800× | 740× | **1.08×** | 750× | 218× | **3.44×** |
| hybrid fullband 32 kb/s | 575× | 560× | **1.03×** | 442× | 193× | **2.29×** |
| CELT fullband 64 kb/s | 1079× | 1087× | **0.99×** | 689× | 432× | **1.59×** |

At matched complexity-0 SILK and hybrid now **beat** libopus (1.08× / 1.03×) and
CELT is at parity (0.99×); at matched complexity-10 we're 1.6-3.4× faster. (libopus's c10 also
enables delayed-decision NSQ and warped noise shaping, which `opus_native` does
not yet implement, so its c10 buys quality we don't spend cycles on - the c0
column is the cleaner like-for-like.) Taking SILK/hybrid from ~0.87-0.90× to a
lead came from profiling `opus_native` against libopus 1.6.1 *function for
function* (perf with libopus's debuginfo, pinned to one core), then matching or
beating its hot paths, every change bit-exact: a table-driven rate-distortion
choice in the NSQ inner loop (a port of `silk_NSQ_sse4_1`'s `table[64][4]`);
compiler-unrolled per-order LPC analysis filters (libopus dispatches to
`silk_LPC_analysis_filterN_FLP`; a generic SIMD dot per output sample lost to the
per-call horizontal fold); an in-place fused NSQ noise-shape feedback loop (no
per-sample scratch array); and stack-allocating the shell-coder pulse tree and
the Burg work vectors (libopus does both on the stack; we were allocating ~80
Vecs per frame in the shell coder alone). Earlier work did SIMD (AVX2+FMA / SSE2)
of the hot loops plus general tuning - latency-hiding in the dot kernels, and
reverting SIMD where a scalar loop measured faster *in cycles* (not instruction
counts):

- **CELT**: the PVQ pulse search (SSE2 *and* an AVX2 path libopus doesn't ship);
  the pre-filter pitch analysis (`celt_pitch_xcorr` + downsampler whitening,
  two-thirds of CELT encode); the forward MDCT pre-rotation (folded into a
  precomputed-twiddle complex multiply); reused per-frame and per-band scratch
  buffers.
- **SILK**: the pitch analysis (cross-/autocorrelation, LPC whitening) and Burg
  LPC; the front-end 48→16 kHz resampler, reworked from a scalar fixed-point
  FIR to a float SIMD decimator (it was the second-largest SILK cost); and the
  NSQ prediction filters, with a bit-exact fixed-point SIMD dot so the bitstream
  is unchanged.

At matched complexity the two are now neck and neck: several stages
(`opus_native`'s input resampler, the pitch decimation, the range coder) already
run faster than libopus's, and the NSQ and pitch analysis are at parity. What is
left is genuinely serial, fixed-point work the reference also runs scalar - the
NSQ rate-distortion recurrence, pre-emphasis, the transient detector's IIR
filters - plus the float-correlation dot products, where both use the same
4-wide widening FMA kernel. Every mode encodes and decodes at hundreds of ×
realtime.

## Conformance

The decoder **passes the official conformance criterion**: every one of
the twelve vectors scores 99.2-100% on the `opus_compare` quality metric
(pass bar: ≥ 0%), with per-packet final ranges bit-exact across the whole
suite. It is built against the official
[Opus test vectors](https://opus-codec.org/testvectors/) (RFC 8251 set).
Fetch them with `tools/fetch-testvectors.sh` (~121 MB, not committed); the
conformance tests in `tests/conformance.rs` skip cleanly when absent. The
packet layer validates against every packet of all twelve vectors. The
CELT-only vectors (testvector01/07/11) decode with per-packet final-range
equality - the bit-exactness oracle - and the synthesized PCM scores
83-104 dB SNR against the reference decode, far beyond the official
`opus_compare` criterion. The harness grows the remaining vectors as the
SILK decoder lands.

## License

MIT, see [LICENSE](LICENSE).

The Opus codec itself is royalty-free; see the
[Opus IPR statements](https://datatracker.ietf.org/ipr/search/?rfc=6716&submit=rfc).

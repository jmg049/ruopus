# opus_rs

A pure-Rust implementation of the [Opus audio codec](https://opus-codec.org/)
([RFC 6716](https://www.rfc-editor.org/rfc/rfc6716)): decoder and encoder, with
no C and no FFI.

**Pure Rust. `unsafe` only in a few SIMD kernels, every one checked under
[Miri](https://github.com/rust-lang/miri). Runs on stable Rust**

> The decoder passes the official Opus conformance vectors, and the encoder produces standard Opus that libopus and ffmpeg decode.

## Overview

`opus_rs` is a from-scratch Rust implementation of Opus. It links no
`libopus`, needs no C toolchain, and exposes plain `&[u8]`/`&[i16]`/`&[f32]`
interfaces, so it embeds under any audio stack.

- Pure Rust, no FFI. Builds on `wasm32`. The decoder is `no_std` + `alloc`
  (build with `default-features = false, features = ["libm"]`); the encoder
  currently needs `std`.
- `unsafe` is denied by default. The only exceptions are a few `std::arch` SIMD
  hot loops, each carrying a `// SAFETY:` note ([`docs/unsafe.md`](docs/unsafe.md))
  and checked for undefined behaviour by Miri on both the SSE2 and AVX2 paths
  (`tools/miri.sh`). No `portable_simd`, no inline asm.
- Zero required dependencies. The default build adds one optional FFT crate for
  faster decoding; `default-features = false` is fully dependency-free.

## Use

```toml
[dependencies]
opus_rs = "0.1"
```

```rust
use opus_rs::{OpusDecoder, OpusEncoder};

// Decode Opus packets to interleaved f32 PCM.
let mut dec = OpusDecoder::new(2); // channels
let pcm = dec.decode_packet(&packet)?;

// Encode 48 kHz PCM (one 20 ms frame is 960 samples per channel, interleaved).
let mut enc = OpusEncoder::new(1);
enc.set_bitrate(Some(24_000));
let packet = enc.encode_auto(&pcm_960, 4000)?;
```

```rust
// Whole Ogg Opus files.
let (pcm, head) = opus_rs::decode_ogg_opus(&bytes)?;
let ogg = opus_rs::encode_ogg_opus(&pcm, 2, 96_000);
```

## Performance

Measured against libopus 1.6.1 (SIMD-enabled C) on identical data, one core,
pinned to a single performance core: `cargo bench --bench vs_libopus --features
std`. Figures are x realtime; "ratio" is opus_rs divided by libopus.

**Decode**

| Mode | opus_rs | libopus | ratio |
|------|-------------|---------|-------|
| SILK wideband 16 kb/s | 2095x | 1171x | 1.79x |
| hybrid fullband 32 kb/s | 1199x | 850x | 1.41x |
| CELT fullband 64 kb/s | 1389x | 1566x | 0.89x |

Speech decode (SILK, hybrid) is faster than SIMD libopus. CELT trails on the
MDCT, where libopus's SIMD wins.

**Encode** (matched complexity)

| Mode | opus_rs | libopus | ratio |
|------|-------------|---------|-------|
| SILK wideband 16 kb/s | 734x | 740x | 0.99x |
| hybrid fullband 32 kb/s | 560x | 562x | 1.00x |
| CELT fullband 64 kb/s | 1088x | 1092x | 1.00x |

At matched complexity, encode is at parity with libopus across all modes.
Against libopus at its default complexity it runs 1.6 to 3.2x faster (it does
not yet spend cycles on delayed-decision NSQ or warped noise shaping).

## Conformance

Passes the official Opus conformance criterion: all twelve
[RFC 8251 test vectors](https://opus-codec.org/testvectors/) score 99.2 to 100%
on opus_compare, with per-packet final ranges bit-exact. Fetch the vectors with
`tools/fetch-testvectors.sh` (about 121 MB, not committed); the conformance
tests skip cleanly without them.

## License

MIT, see [LICENSE](LICENSE). The Opus format is royalty-free; see the
[Opus IPR statements](https://datatracker.ietf.org/ipr/search/?rfc=6716&submit=rfc).

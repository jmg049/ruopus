# opus_native

A pure-Rust implementation of the [Opus audio codec](https://opus-codec.org/)
([RFC 6716](https://www.rfc-editor.org/rfc/rfc6716)).

**No FFI. No unsafe code. No dependencies.**

> **Status: pre-release, under active development.** The entropy-coding and
> packet layers are complete and tested; the SILK and CELT decoders are in
> progress. Nothing here is API-stable yet.

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
- **`forbid(unsafe_code)`**: enforced at the crate level.
- **`no_std` + `alloc`**: the `std` feature (on by default) only adds
  `std::error::Error` impls and conveniences.
- **Zero dependencies**: nothing in the dependency tree but this crate.

## Layout

| Module | RFC 6716 | Status |
|--------|----------|--------|
| `range` | §4.1, §5.1 | Range decoder + encoder, raw bits, uniform ints, `tell`/`tell_frac` |
| `packet` | §3 | TOC parsing, frame packing codes 0-3, padding, R1-R7 validation |
| `lpc` | §4.2/§5.2 groundwork | Levinson-Durbin, LP analysis/synthesis, pitch estimation, LTP |
| `experimental` | - | pre-conformance frame codec, mode detection, crossover, mid/side (feature `experimental-codec`) |
| `silk` | §4.2 | planned (conformant decoder) |
| `celt` | §4.3 | planned (conformant decoder) |
| `ogg` | RFC 7845 | planned (likely behind a feature flag) |

## License

MIT, see [LICENSE](LICENSE).

The Opus codec itself is royalty-free; see the
[Opus IPR statements](https://datatracker.ietf.org/ipr/search/?rfc=6716&submit=rfc).

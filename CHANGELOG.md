# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Conformance harness against the official Opus test vectors (RFC 8251 set, fetched by `tools/fetch-testvectors.sh`): `opus_demo` bitstream parsing, packet-level validation of all 20,075 packets across the twelve vectors, TOC-duration agreement with the reference PCM, full configuration coverage; skips cleanly when vectors are absent
- Ogg container (RFC 3533) and Ogg Opus mapping (RFC 7845): CRC-verified page parsing with capture-pattern resync, cross-page packet reassembly with the RFC 7845 continuity rules, page writing, `OpusHead`/`OpusTags` headers (all channel-mapping families), per-packet granule resolution (pre-skip, end trimming), and a conformant stream writer - interop-tested against an ffmpeg/libopus file
- `lpc` module: Levinson-Durbin, LP analysis/synthesis filters (stateless and cross-frame), pitch estimation, and single-tap LTP - ported from `audio_samples` and decoupled to plain slices
- `experimental` module (feature `experimental-codec`, on by default): the pre-conformance SILK-style frame codec, spectral-flatness mode detection, hybrid crossover, and mid/side helpers, with documented divergences from RFC 6716
- Packet framing layer (RFC 6716 §3): TOC byte introspection (mode/bandwidth/frame size per Table 2), frame packing codes 0-3, padding, and full [R1]-[R7] malformed-packet validation
- Range decoder and encoder (RFC 6716 §4.1/§5.1): symbol, binary, ICDF, raw-bits, and uniform-integer coding with `tell`/`tell_frac`, verified by encoder/decoder `rng`-agreement round-trips

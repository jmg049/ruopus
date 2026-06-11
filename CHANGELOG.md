# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- CELT band shape decoding (§4.3.4): PVQ shape decode with spreading rotation (`vq`), and the full band loop (`bands`) - recursive theta splits with the step/uniform/triangular PDFs, bit-exact cos/log2tan steering, stereo (theta, intensity, dual, N=2 special case), time/frequency reshaping (Haar + Hadamard), spectral folding with LCG noise fill, and collapse-mask tracking
- CELT bit allocation (§4.3.3): the full decoder-side `compute_allocation`/`interp_bits2pulses` derivation - quality-vector interpolation, explicit band skipping, intensity/dual-stereo parameters, fine/shape budget split with cap rebalancing - plus the pulse cache (`bits2pulses`/`pulses2bits`) and allocation tables extracted mechanically from the reference sources
- CELT energy envelope decoding (§4.3.2): coarse energy with time/frequency prediction and the Laplace probability model, budget-starved fallback codes, fine refinement, and final-bit distribution; plus the standard 48 kHz mode tables (band layout, energy means, prediction coefficients)
- CELT decoder kernels (`celt` module): the Laplace coder for coarse energy deltas (§4.3.2.1) and the PVQ codeword enumeration (§4.3.4.2, table-free CWRS) - exhaustively tested for index bijection against the reference V(N,K) table and through range-coder round trips
- Conformance harness against the official Opus test vectors (RFC 8251 set, fetched by `tools/fetch-testvectors.sh`): `opus_demo` bitstream parsing, packet-level validation of all 20,075 packets across the twelve vectors, TOC-duration agreement with the reference PCM, full configuration coverage; skips cleanly when vectors are absent
- Ogg container (RFC 3533) and Ogg Opus mapping (RFC 7845): CRC-verified page parsing with capture-pattern resync, cross-page packet reassembly with the RFC 7845 continuity rules, page writing, `OpusHead`/`OpusTags` headers (all channel-mapping families), per-packet granule resolution (pre-skip, end trimming), and a conformant stream writer - interop-tested against an ffmpeg/libopus file
- `lpc` module: Levinson-Durbin, LP analysis/synthesis filters (stateless and cross-frame), pitch estimation, and single-tap LTP - ported from `audio_samples` and decoupled to plain slices
- `experimental` module (feature `experimental-codec`, on by default): the pre-conformance SILK-style frame codec, spectral-flatness mode detection, hybrid crossover, and mid/side helpers, with documented divergences from RFC 6716
- Packet framing layer (RFC 6716 §3): TOC byte introspection (mode/bandwidth/frame size per Table 2), frame packing codes 0-3, padding, and full [R1]-[R7] malformed-packet validation
- Range decoder and encoder (RFC 6716 §4.1/§5.1): symbol, binary, ICDF, raw-bits, and uniform-integer coding with `tell`/`tell_frac`, verified by encoder/decoder `rng`-agreement round-trips

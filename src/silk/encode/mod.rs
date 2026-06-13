//! The SILK encoder (RFC 6716 §5.2; normative `silk/` and `silk/float/`).
//!
//! Work in progress. SILK is the speech/low-bitrate half of Opus; this
//! module is being built up kernel by kernel, starting with the linear
//! prediction analysis, each validated in isolation before the full encode
//! pipeline (LPC/LTP/gain/NLSF quantisation, the noise-shaping quantiser,
//! and the index/pulse bitstream) is assembled.
//!
//! The analysis uses the reference float build; the quantisation indices it
//! emits are read back by the existing bit-exact fixed-point decoder, so the
//! round trip is the conformance oracle.
#![allow(
    dead_code,
    reason = "encoder kernels are landing incrementally; wired into the pipeline as it assembles"
)]

pub(crate) mod lpc;

//! A pure-Rust implementation of the Opus audio codec ([RFC 6716]).
//!
//! No FFI, no unsafe code, no dependencies. The crate core is `no_std` +
//! `alloc`; the default `std` feature only adds [`std::error::Error`] impls.
//!
//! # Status
//!
//! Pre-release. The layers currently implemented, bottom-up:
//!
//! | Module | RFC 6716 | Contents |
//! |--------|----------|----------|
//! | [`range`] | §4.1, §5.1 | range decoder + encoder: symbols, binary/ICDF contexts, raw bits, uniform integers, `tell`/`tell_frac` |
//! | [`packet`] | §3 | TOC byte, frame packing codes 0-3, padding, R1-R7 validation |
//! | [`lpc`] | analysis groundwork for §4.2/§5.2 | Levinson-Durbin, LP analysis/synthesis filters, pitch estimation, single-tap LTP |
//! | [`experimental`] | - | the pre-conformance frame codec, mode detection, hybrid crossover, and mid/side helpers ported from `audio_samples` |
//!
//! The conformant SILK (§4.2) and CELT (§4.3) decoders are under construction
//! on top of these layers; the [`experimental`] module documents exactly how
//! it differs from real Opus in the meantime.
//!
//! # Bit-exactness
//!
//! Every arithmetic operation in the entropy coder follows the RFC text
//! exactly; the encoder is verified against the decoder symbol-for-symbol
//! (their `rng` states must agree after every operation - see RFC 6716 §5.1).
//! All multi-byte values, state update rules, and rounding behaviours are
//! documented at their definition with the RFC section they implement.
//!
//! [RFC 6716]: https://www.rfc-editor.org/rfc/rfc6716

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "experimental-codec")]
pub mod experimental;
#[cfg(feature = "experimental-codec")]
pub mod lpc;
pub mod packet;
pub mod range;

pub use packet::{Bandwidth, FrameSize, Mode, Packet, PacketError, Toc};
pub use range::{RangeDecoder, RangeEncoder, RangeEncoderError};

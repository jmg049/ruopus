//! A pure-Rust implementation of the Opus audio codec ([RFC 6716]).
//!
//! No FFI. `unsafe` is confined to a few documented SIMD kernels, all checked
//! under Miri. The decoder, the range coder and the packet layer build for
//! `no_std` + `alloc` (enable the `libm` feature); the encoder currently
//! requires `std`.
//!
//! | Module | RFC 6716 | Contents |
//! |--------|----------|----------|
//! | [`range`] | §4.1, §5.1 | range decoder + encoder: symbols, binary/ICDF contexts, raw bits, uniform integers, `tell`/`tell_frac` |
//! | [`packet`] | §3 | TOC byte, frame packing codes 0-3, padding, R1-R7 validation |
//! | [`silk`] | §4.2 | SILK decoder and encoder |
//! | [`celt`] | §4.3 | CELT decoder and encoder |
//! | [`ogg`] | RFC 3533 + RFC 7845 | Ogg pages, packet reassembly, `OpusHead`/`OpusTags`, granule/pre-skip timing, stream reader/writer |
//!
//! The decoder passes the official RFC 8251 conformance vectors; the encoder
//! produces standard Opus that libopus and ffmpeg decode.
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

// Float transcendentals need `std` (inherent `f32`/`f64` methods) or `libm`.
#[cfg(all(not(feature = "std"), not(feature = "libm")))]
compile_error!(
    "ruopus needs floating-point transcendentals: enable the default `std` feature, or for \
     `no_std` build with `default-features = false, features = [\"libm\"]`"
);

// On `no_std`, this trait re-supplies the std-only float methods (`x.sin()`,
// ...) via `libm`; on `std` it is not compiled and the inherent methods are used.
#[cfg(not(feature = "std"))]
mod float;

pub mod celt;
mod decoder;
// The encoder (and its analysis) is still std-only for now; no_std currently
// targets the decoder. The SIMD kernels are encoder-only too (the decode path
// is SIMD-free), so they stay behind `std`.
#[cfg(feature = "std")]
mod encoder;
#[cfg(feature = "std")]
mod encoder_analysis;
mod multistream;
pub use decoder::{OggDecodeError, OpusDecoder, decode_ogg_opus};
#[cfg(feature = "std")]
pub use encoder::{Application, EncodeError, OpusEncoder, Signal, encode_ogg_opus};
pub use multistream::MultistreamDecoder;
#[cfg(feature = "std")]
pub use silk::encode::api::{SilkEncoder, SilkStereoEncoder};
pub mod ogg;
pub mod packet;
pub mod range;
pub mod silk;
#[cfg(feature = "std")]
mod simd;

#[cfg(feature = "python")]
mod python;

pub use packet::{Bandwidth, FrameSize, Mode, Packet, PacketError, Toc};
pub use range::{RangeDecoder, RangeEncoder, RangeEncoderError};

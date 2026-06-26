//! Python bindings for `opus_native`, built with [PyO3].
//!
//! Compiled only under the `python` feature (a plain `cargo build`/`cargo test`
//! stays zero-dependency and pure Rust); `maturin build` links it into the
//! `opus_native` extension module. Every item is declared so PyO3's
//! `experimental-inspect` pass can emit a complete, docstring-carrying `.pyi`.
//!
//! [PyO3]: https://pyo3.rs

use pyo3::prelude::*;

mod decoder;
mod encoder;
mod enums;
mod errors;
mod lowlevel;
mod multistream;
mod numpy_io;
mod ogg;
mod packet;

/// Returns the `opus_native` crate version (the Rust ``CARGO_PKG_VERSION``).
///
/// Examples
/// --------
/// >>> import opus_native
/// >>> opus_native.version()
/// '0.1.0'
#[pyfunction]
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// A pure-Rust implementation of the Opus audio codec (RFC 6716).
///
/// The Python API mirrors the crate's Rust public interface: a stateful
/// :class:`OpusDecoder`, the packet enums (:class:`Mode`, :class:`Bandwidth`,
/// :class:`FrameSize`), and an exception hierarchy rooted at
/// :class:`OpusError`. PCM crosses the boundary as NumPy ``float32``/``int16``
/// arrays shaped ``(frames, channels)``, moved (not copied) out of Rust.
#[pymodule]
mod opus_native {
    #[pymodule_export]
    use super::decoder::OpusDecoder;
    #[pymodule_export]
    use super::encoder::OpusEncoder;
    #[pymodule_export]
    use super::enums::{Application, Bandwidth, FrameSize, Mode, Signal};
    #[pymodule_export]
    use super::errors::{EncodeError, OggError, OpusError, PacketError};
    // The `lowlevel` submodule (declared as a sibling pymodule below, the form
    // PyO3's declarative macros and introspection support).
    #[pymodule_export]
    use super::lowlevel_module;
    #[pymodule_export]
    use super::multistream::MultistreamDecoder;
    #[pymodule_export]
    use super::ogg::{OpusHead, decode_ogg_opus, encode_ogg_opus};
    #[pymodule_export]
    use super::packet::{Packet, Toc};
    #[pymodule_export]
    use super::version;
}

/// Direct access to the SILK, LPC, and CELT layers beneath the Opus packet
/// codec. Advanced building blocks; ordinary use should prefer the top-level
/// :class:`OpusEncoder` / :class:`OpusDecoder`.
#[pymodule(name = "lowlevel")]
mod lowlevel_module {
    #[pymodule_export]
    use super::lowlevel::celt::{CeltDecoder, CeltEncoder};
    #[pymodule_export]
    use super::lowlevel::lpc::{
        LpcCoefficients, compute_autocorrelation, estimate_pitch, levinson_durbin, lpc_analysis, lpc_residual,
        lpc_residual_stateful, lpc_synthesis, lpc_synthesis_stateful, ltp_residual, ltp_synthesis,
    };
    #[pymodule_export]
    use super::lowlevel::silk::{DecControl, SilkDecoder, SilkEncoder, SilkStereoEncoder};
}

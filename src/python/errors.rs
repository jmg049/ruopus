//! Python exception hierarchy mirroring the crate's Rust error enums.
//!
//! Rooted at [`OpusError`] so Python callers can catch every codec failure with
//! a single `except ruopus.OpusError`. Each Rust error type converts into
//! the matching exception via a `From<…> for PyErr` impl, so the bindings use
//! `?` directly and the original `Display` message is preserved.
//!
//! These use [`create_exception!`], which builds genuine Python exception types
//! (raisable, with working `args`/`str`). They are added to the module via
//! `#[pymodule_export]` in `mod.rs`, so no `#[pymodule_init]` is needed and the
//! introspected stub stays free of a catch-all `__getattr__`.

use pyo3::create_exception;
use pyo3::exceptions::PyException;

create_exception!(
    ruopus,
    OpusError,
    PyException,
    "Base class for every error raised by `ruopus`.\n\n\
     Catch this to handle any codec failure regardless of its specific kind."
);

create_exception!(
    ruopus,
    PacketError,
    OpusError,
    "Raised when an Opus packet is malformed (RFC 6716 §3.4 rules R1-R7).\n\n\
     Corresponds to the Rust `ruopus::PacketError`."
);

create_exception!(
    ruopus,
    EncodeError,
    OpusError,
    "Raised when encoding fails: an unsupported frame size, or an output budget \
     outside 3..=1275 bytes that the packet could not be made to fit.\n\n\
     Corresponds to the Rust `ruopus::EncodeError`."
);

create_exception!(
    ruopus,
    OggError,
    OpusError,
    "Raised when decoding an Ogg Opus stream fails (bad container, bad packet, \
     or an unsupported channel-mapping family).\n\n\
     Corresponds to the Rust `ruopus::OggDecodeError`."
);

impl From<crate::packet::PacketError> for pyo3::PyErr {
    fn from(e: crate::packet::PacketError) -> Self {
        PacketError::new_err(e.to_string())
    }
}

impl From<crate::encoder::EncodeError> for pyo3::PyErr {
    fn from(e: crate::encoder::EncodeError) -> Self {
        EncodeError::new_err(e.to_string())
    }
}

impl From<crate::decoder::OggDecodeError> for pyo3::PyErr {
    fn from(e: crate::decoder::OggDecodeError) -> Self {
        OggError::new_err(e.to_string())
    }
}

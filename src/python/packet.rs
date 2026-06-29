//! Packet introspection types: `Toc` and `Packet`.
//!
//! The core `Packet<'a>` borrows its input; Python can't safely hold that
//! borrow, so the Python `Packet` is an owned snapshot - it copies each frame's
//! bytes at parse time and exposes them as `bytes`.

use std::time::Duration;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::enums::{Bandwidth, FrameSize, Mode};

/// The table-of-contents byte heading every Opus packet (RFC 6716 §3.1).
///
/// A value type wrapping the raw byte; all 256 values are valid TOCs.
#[pyclass(module = "ruopus", name = "Toc", eq, hash, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Toc {
    inner: crate::packet::Toc,
}

#[pymethods]
impl Toc {
    /// Interpret ``byte`` as a TOC byte.
    #[new]
    fn new(byte: u8) -> Self {
        Self {
            inner: crate::packet::Toc::new(byte),
        }
    }

    /// Build a TOC byte from its three fields.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``config > 31`` or ``frame_count_code > 3``.
    #[staticmethod]
    fn from_parts(config: u8, stereo: bool, frame_count_code: u8) -> PyResult<Self> {
        if config > 31 {
            return Err(PyValueError::new_err("config must be 0..=31"));
        }
        if frame_count_code > 3 {
            return Err(PyValueError::new_err("frame_count_code must be 0..=3"));
        }
        Ok(Self {
            inner: crate::packet::Toc::from_parts(config, stereo, frame_count_code),
        })
    }

    /// The raw TOC byte.
    #[getter]
    fn byte(&self) -> u8 {
        self.inner.byte()
    }

    /// The configuration number (0..=31): the top five bits.
    #[getter]
    fn config(&self) -> u8 {
        self.inner.config()
    }

    /// ``True`` for stereo, ``False`` for mono.
    #[getter]
    fn stereo(&self) -> bool {
        self.inner.stereo()
    }

    /// The number of channels (1 or 2).
    #[getter]
    fn channels(&self) -> u8 {
        self.inner.channels()
    }

    /// The frame-count code (0..=3): the bottom two bits.
    #[getter]
    fn frame_count_code(&self) -> u8 {
        self.inner.frame_count_code()
    }

    /// The operating mode for this configuration.
    #[getter]
    fn mode(&self) -> Mode {
        self.inner.mode().into()
    }

    /// The audio bandwidth for this configuration.
    #[getter]
    fn bandwidth(&self) -> Bandwidth {
        self.inner.bandwidth().into()
    }

    /// The frame size for this configuration.
    #[getter]
    fn frame_size(&self) -> FrameSize {
        self.inner.frame_size().into()
    }

    fn __int__(&self) -> u8 {
        self.inner.byte()
    }

    fn __repr__(&self) -> String {
        format!(
            "Toc(byte={}, config={}, channels={}, frame_count_code={})",
            self.inner.byte(),
            self.inner.config(),
            self.inner.channels(),
            self.inner.frame_count_code(),
        )
    }
}

/// A parsed Opus packet: its TOC plus the compressed frames (RFC 6716 §3).
///
/// Behaves as a read-only sequence of frames: ``len(packet)`` is the frame
/// count and ``packet[i]`` is frame ``i`` as ``bytes`` (a frame may be empty,
/// signalling DTX). Construct with ``Packet(data)`` to parse a standard packet,
/// or :meth:`parse_self_delimited` for the self-delimiting framing used by
/// multistream payloads.
#[pyclass(module = "ruopus", name = "Packet", frozen, sequence)]
pub struct Packet {
    toc: Toc,
    frames: Vec<Vec<u8>>,
    padding: usize,
    duration: Duration,
}

impl Packet {
    fn from_parsed(p: &crate::packet::Packet) -> Self {
        Self {
            toc: Toc { inner: p.toc() },
            frames: p.frames().iter().map(|f| f.to_vec()).collect(),
            padding: p.padding(),
            duration: p.duration(),
        }
    }
}

#[pymethods]
impl Packet {
    /// Parse one Opus packet (RFC 6716 §3.2), validating R1-R7.
    ///
    /// Raises
    /// ------
    /// PacketError
    ///     If the packet is malformed.
    #[new]
    fn parse(data: &[u8]) -> PyResult<Self> {
        Ok(Self::from_parsed(&crate::packet::Packet::parse(data)?))
    }

    /// Parse one self-delimited Opus packet (RFC 6716 Appendix B).
    ///
    /// Returns the packet and the number of bytes it consumed, so the caller
    /// can continue parsing the next stream in a multistream payload.
    ///
    /// Raises
    /// ------
    /// PacketError
    ///     If the packet is malformed.
    #[staticmethod]
    #[pyo3(signature = (data) -> "tuple[Packet, int]")]
    fn parse_self_delimited(data: &[u8]) -> PyResult<(Self, usize)> {
        let (p, used) = crate::packet::Packet::parse_self_delimited(data)?;
        Ok((Self::from_parsed(&p), used))
    }

    /// The table-of-contents byte.
    #[getter]
    fn toc(&self) -> Toc {
        self.toc
    }

    /// The compressed frames, in order, as ``bytes`` (some may be empty).
    #[getter]
    fn frames<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyBytes>> {
        self.frames.iter().map(|f| PyBytes::new(py, f)).collect()
    }

    /// Bytes of Opus padding the packet carried (code 3 only).
    #[getter]
    fn padding(&self) -> usize {
        self.padding
    }

    /// Total audio duration of the packet, as a ``datetime.timedelta``.
    #[getter]
    fn duration(&self) -> Duration {
        self.duration
    }

    fn __len__(&self) -> usize {
        self.frames.len()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyBytes>> {
        let n = self.frames.len() as isize;
        let i = if index < 0 { index + n } else { index };
        if i < 0 || i >= n {
            return Err(PyIndexError::new_err("frame index out of range"));
        }
        Ok(PyBytes::new(py, &self.frames[i as usize]))
    }

    fn __repr__(&self) -> String {
        format!(
            "Packet(frames={}, channels={}, duration={:?})",
            self.frames.len(),
            self.toc.channels(),
            self.duration,
        )
    }
}

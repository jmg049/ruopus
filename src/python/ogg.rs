//! Ogg Opus bindings: the `OpusHead` identification header and the
//! whole-file `encode_ogg_opus` / `decode_ogg_opus` convenience functions.

use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::numpy_io::{borrow_interleaved_f32, interleaved_f32_to_numpy};

/// The identification header of an Ogg Opus stream (RFC 7845 §5.1).
///
/// The channel-mapping table is flattened onto this object: family-0 streams
/// report :attr:`mapping_family` ``0`` with the optional fields ``None``;
/// other families expose their stream/coupled counts and per-channel table.
#[pyclass(module = "opus_rs", name = "OpusHead", frozen)]
pub struct OpusHead {
    inner: crate::ogg::OpusHead,
}

#[pymethods]
impl OpusHead {
    /// Encapsulation version (``1`` for this specification).
    #[getter]
    fn version(&self) -> u8 {
        self.inner.version
    }

    /// Output channel count (never zero).
    #[getter]
    fn channel_count(&self) -> u8 {
        self.inner.channel_count
    }

    /// Samples at 48 kHz to discard from the start of decoder output.
    #[getter]
    fn pre_skip(&self) -> u16 {
        self.inner.pre_skip
    }

    /// Sample rate of the original input in Hz (metadata only; 0 = unspecified).
    #[getter]
    fn input_sample_rate(&self) -> u32 {
        self.inner.input_sample_rate
    }

    /// Output gain in Q7.8 dB, applied by players on top of decoder output.
    #[getter]
    fn output_gain_q8(&self) -> i16 {
        self.inner.output_gain_q8
    }

    /// Channel-mapping family: ``0`` for mono/stereo, otherwise the table family.
    #[getter]
    fn mapping_family(&self) -> u8 {
        match &self.inner.channel_mapping {
            crate::ogg::ChannelMapping::Family0 => 0,
            crate::ogg::ChannelMapping::Table { family, .. } => *family,
        }
    }

    /// Number of encoded streams (``None`` for family 0).
    #[getter]
    fn stream_count(&self) -> Option<u8> {
        match &self.inner.channel_mapping {
            crate::ogg::ChannelMapping::Family0 => None,
            crate::ogg::ChannelMapping::Table { stream_count, .. } => Some(*stream_count),
        }
    }

    /// Number of coupled (stereo) streams (``None`` for family 0).
    #[getter]
    fn coupled_count(&self) -> Option<u8> {
        match &self.inner.channel_mapping {
            crate::ogg::ChannelMapping::Family0 => None,
            crate::ogg::ChannelMapping::Table { coupled_count, .. } => Some(*coupled_count),
        }
    }

    /// The per-channel mapping table (``None`` for family 0).
    #[getter]
    fn channel_mapping(&self) -> Option<Vec<u8>> {
        match &self.inner.channel_mapping {
            crate::ogg::ChannelMapping::Family0 => None,
            crate::ogg::ChannelMapping::Table { mapping, .. } => Some(mapping.clone()),
        }
    }

    /// Serialise this header to an ``OpusHead`` packet.
    #[pyo3(signature = () -> "bytes")]
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    fn __repr__(&self) -> String {
        format!(
            "OpusHead(version={}, channel_count={}, pre_skip={}, input_sample_rate={}, mapping_family={})",
            self.inner.version,
            self.inner.channel_count,
            self.inner.pre_skip,
            self.inner.input_sample_rate,
            self.mapping_family(),
        )
    }
}

impl From<crate::ogg::OpusHead> for OpusHead {
    fn from(inner: crate::ogg::OpusHead) -> Self {
        Self { inner }
    }
}

/// Encode interleaved 48 kHz ``float32`` PCM to a complete in-memory Ogg Opus
/// file.
///
/// Parameters
/// ----------
/// pcm : numpy.ndarray
///     Interleaved 48 kHz ``float32`` PCM (1-D, or 2-D ``(frames, channels)``).
/// channels : int
///     1 (mono) or 2 (stereo).
/// bitrate : int
///     Target bitrate in bits/s.
///
/// Returns
/// -------
/// bytes
///     A complete Ogg Opus file.
#[pyfunction]
#[pyo3(signature = (pcm: "numpy.typing.NDArray[numpy.float32]", channels, bitrate))]
pub fn encode_ogg_opus<'py>(
    py: Python<'py>,
    pcm: numpy::PyReadonlyArrayDyn<'_, f32>,
    channels: usize,
    bitrate: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let pcm = borrow_interleaved_f32(&pcm, channels)?;
    let out = py.detach(|| crate::encoder::encode_ogg_opus(&pcm, channels, bitrate));
    Ok(PyBytes::new(py, &out))
}

/// Decode a complete in-memory Ogg Opus file (RFC 7845 §4): pre-skip removal,
/// end trimming, and the ``OpusHead`` output gain applied.
///
/// Only channel mapping family 0 (mono/stereo) is supported until a
/// multistream Ogg decoder exists.
///
/// Parameters
/// ----------
/// data : bytes
///     A complete Ogg Opus file.
///
/// Returns
/// -------
/// tuple[numpy.ndarray, OpusHead]
///     Interleaved 48 kHz ``float32`` PCM shaped ``(frames, channels)`` and the
///     parsed identification header.
///
/// Raises
/// ------
/// OggError
///     For a malformed container, a bad packet, or an unsupported mapping.
#[pyfunction]
#[pyo3(signature = (data) -> "tuple[numpy.typing.NDArray[numpy.float32], OpusHead]")]
pub fn decode_ogg_opus<'py>(py: Python<'py>, data: &[u8]) -> PyResult<(Bound<'py, PyArray2<f32>>, OpusHead)> {
    let owned = data.to_vec();
    let (pcm, head) = py.detach(|| crate::decoder::decode_ogg_opus(&owned))?;
    let channels = head.channel_count as usize;
    let arr = interleaved_f32_to_numpy(py, pcm, channels)?;
    Ok((arr, OpusHead::from(head)))
}

//! The multistream (surround) decoder, exposed as
//! `opus_rs.MultistreamDecoder`.

use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::numpy_io::interleaved_f32_to_numpy;

/// A multistream Opus decoder (``OpusMSDecoder``, RFC 7845 §5.1.1).
///
/// Decodes ``streams`` elementary Opus streams per packet - the first
/// ``coupled`` decoded as stereo, the rest mono - and routes their channels to
/// the output layout through ``mapping``. Output is interleaved PCM as a NumPy
/// array shaped ``(frames, channels)``.
///
/// Parameters
/// ----------
/// streams : int
///     Number of elementary streams (>= 1).
/// coupled : int
///     Number of those streams that are stereo-coupled (<= ``streams``).
/// mapping : Sequence[int]
///     Per-output-channel source index; ``255`` means a silent channel. The
///     output channel count is ``len(mapping)``.
/// sample_rate : int, optional
///     Output sample rate in Hz; one of 48000, 24000, 16000, 12000, 8000.
///     Defaults to 48000.
#[pyclass(module = "opus_rs", name = "MultistreamDecoder")]
pub struct MultistreamDecoder {
    inner: crate::multistream::MultistreamDecoder,
    channels: usize,
    sample_rate: u32,
}

#[pymethods]
impl MultistreamDecoder {
    #[new]
    #[pyo3(signature = (streams, coupled, mapping, *, sample_rate = 48_000))]
    fn new(streams: usize, coupled: usize, mapping: Vec<u8>, sample_rate: u32) -> PyResult<Self> {
        if streams < 1 || coupled > streams || streams + coupled > 255 {
            return Err(PyValueError::new_err(
                "require streams >= 1, coupled <= streams, streams + coupled <= 255",
            ));
        }
        if !matches!(sample_rate, 48_000 | 24_000 | 16_000 | 12_000 | 8_000) {
            return Err(PyValueError::new_err(
                "sample_rate must be one of 48000, 24000, 16000, 12000, 8000",
            ));
        }
        let decoded_channels = (streams + coupled) as u8;
        if !mapping.iter().all(|&m| m == 255 || m < decoded_channels) {
            return Err(PyValueError::new_err(
                "every mapping entry must be 255 or less than streams + coupled",
            ));
        }
        if mapping.is_empty() {
            return Err(PyValueError::new_err("mapping must have at least one channel"));
        }
        Ok(Self {
            channels: mapping.len(),
            inner: crate::multistream::MultistreamDecoder::with_rate(sample_rate, streams, coupled, &mapping),
            sample_rate,
        })
    }

    /// Output channel count (``len(mapping)``).
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Output sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Decode one multistream packet to interleaved ``float32`` PCM.
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     One multistream Opus packet (self-delimited streams followed by a
    ///     standard final stream).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Shape ``(frames, channels)``, dtype ``float32``.
    ///
    /// Raises
    /// ------
    /// PacketError
    ///     If any elementary stream is malformed or they disagree on duration.
    #[pyo3(signature = (data) -> "numpy.typing.NDArray[numpy.float32]")]
    fn decode_packet<'py>(&mut self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let owned = data.to_vec();
        let pcm = py.detach(|| self.inner.decode_packet(&owned))?;
        interleaved_f32_to_numpy(py, pcm, self.channels)
    }

    fn __repr__(&self) -> String {
        format!(
            "MultistreamDecoder(channels={}, sample_rate={})",
            self.channels, self.sample_rate
        )
    }
}

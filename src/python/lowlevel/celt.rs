//! Low-level CELT bindings: the top-level CELT encoder and decoder.
//!
//! These code/decode a raw CELT frame body (no Opus TOC). For ordinary use
//! prefer :class:`opus_rs.OpusEncoder` / :class:`opus_rs.OpusDecoder`,
//! which wrap CELT in the Opus packet layer. Decoding a coded CELT frame body
//! requires the range decoder, which is not exposed; :class:`CeltDecoder` here
//! covers construction, state, and packet-loss concealment.

use numpy::{PyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::python::numpy_io::{borrow_interleaved_f32, interleaved_f32_to_numpy};

/// The CELT encoder (`celt_encoder`).
///
/// Parameters
/// ----------
/// channels : int, optional
///     1 (mono) or 2 (stereo). Defaults to 1.
/// complexity : int, optional
///     Encode complexity 0-10. Defaults to 10.
/// bitrate : int or None, optional
///     Target VBR bitrate in bits/s, or ``None`` for CBR. Defaults to ``None``.
#[pyclass(module = "opus_rs.lowlevel", name = "CeltEncoder")]
pub struct CeltEncoder {
    inner: crate::celt::encoder::CeltEncoder,
    channels: usize,
    complexity: u8,
    bitrate: Option<u32>,
}

#[pymethods]
impl CeltEncoder {
    #[new]
    #[pyo3(signature = (channels = 1, *, complexity = 10, bitrate = None))]
    fn new(channels: usize, complexity: u8, bitrate: Option<u32>) -> PyResult<Self> {
        if channels != 1 && channels != 2 {
            return Err(PyValueError::new_err("channels must be 1 or 2"));
        }
        let mut inner = crate::celt::encoder::CeltEncoder::with_channels(channels);
        inner.set_complexity(complexity);
        inner.set_target_bitrate(bitrate);
        Ok(Self {
            inner,
            channels,
            complexity: complexity.min(10),
            bitrate,
        })
    }

    /// Number of channels (1 or 2).
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Encode complexity 0-10.
    #[getter]
    fn get_complexity(&self) -> u8 {
        self.complexity
    }

    #[setter]
    fn set_complexity(&mut self, complexity: u8) {
        self.inner.set_complexity(complexity);
        self.complexity = complexity.min(10);
    }

    /// Target VBR bitrate in bits/s, or ``None`` for CBR.
    #[getter]
    fn get_bitrate(&self) -> Option<u32> {
        self.bitrate
    }

    #[setter]
    fn set_bitrate(&mut self, bitrate: Option<u32>) {
        self.inner.set_target_bitrate(bitrate);
        self.bitrate = bitrate;
    }

    /// The range coder state after the last encode (``OPUS_GET_FINAL_RANGE``).
    #[getter]
    fn final_range(&self) -> u32 {
        self.inner.final_range()
    }

    /// Encode one CELT frame to a raw frame body (no Opus TOC).
    ///
    /// Parameters
    /// ----------
    /// pcm : numpy.ndarray
    ///     Interleaved 48 kHz ``float32`` PCM (1-D, or 2-D ``(frames, channels)``);
    ///     120/240/480/960 samples per channel.
    /// nb_bytes : int
    ///     Output budget in bytes.
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The coded CELT frame body.
    #[pyo3(signature = (pcm: "numpy.typing.NDArray[numpy.float32]", nb_bytes) -> "bytes")]
    fn encode_frame<'py>(
        &mut self,
        py: Python<'py>,
        pcm: PyReadonlyArrayDyn<'_, f32>,
        nb_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let pcm = borrow_interleaved_f32(&pcm, self.channels)?;
        let payload = py.detach(|| self.inner.encode_frame(&pcm, nb_bytes));
        Ok(PyBytes::new(py, &payload))
    }

    /// Encode one CELT frame, coding only the first ``end`` bands.
    ///
    /// Parameters
    /// ----------
    /// pcm : numpy.ndarray
    ///     Interleaved 48 kHz ``float32`` PCM (1-D, or 2-D ``(frames, channels)``).
    /// nb_bytes : int
    ///     Output budget in bytes.
    /// end : int
    ///     Number of coded CELT bands.
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The coded CELT frame body.
    #[pyo3(signature = (pcm: "numpy.typing.NDArray[numpy.float32]", nb_bytes, end) -> "bytes")]
    fn encode_frame_bw<'py>(
        &mut self,
        py: Python<'py>,
        pcm: PyReadonlyArrayDyn<'_, f32>,
        nb_bytes: usize,
        end: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let pcm = borrow_interleaved_f32(&pcm, self.channels)?;
        let payload = py.detach(|| self.inner.encode_frame_bw(&pcm, nb_bytes, end));
        Ok(PyBytes::new(py, &payload))
    }

    fn __repr__(&self) -> String {
        format!(
            "CeltEncoder(channels={}, complexity={}, bitrate={})",
            self.channels,
            self.complexity,
            self.bitrate.map_or_else(|| "None".to_string(), |b| b.to_string()),
        )
    }
}

/// The CELT decoder (`celt_decoder`).
///
/// Parameters
/// ----------
/// channels : int
///     1 (mono) or 2 (stereo).
/// sample_rate : int, optional
///     Output sample rate in Hz; one of 48000, 24000, 16000, 12000, 8000.
///     Defaults to 48000.
#[pyclass(module = "opus_rs.lowlevel", name = "CeltDecoder")]
pub struct CeltDecoder {
    inner: crate::celt::decoder::CeltDecoder,
    channels: usize,
    sample_rate: u32,
}

#[pymethods]
impl CeltDecoder {
    #[new]
    #[pyo3(signature = (channels, *, sample_rate = 48_000))]
    fn new(channels: usize, sample_rate: u32) -> PyResult<Self> {
        if channels != 1 && channels != 2 {
            return Err(PyValueError::new_err("channels must be 1 or 2"));
        }
        if !matches!(sample_rate, 48_000 | 24_000 | 16_000 | 12_000 | 8_000) {
            return Err(PyValueError::new_err(
                "sample_rate must be one of 48000, 24000, 16000, 12000, 8000",
            ));
        }
        Ok(Self {
            inner: crate::celt::decoder::CeltDecoder::with_rate(channels, sample_rate),
            channels,
            sample_rate,
        })
    }

    /// Number of channels (1 or 2).
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Output sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The range coder state after the last decode (``OPUS_GET_FINAL_RANGE``).
    #[getter]
    fn final_range(&self) -> u32 {
        self.inner.final_range()
    }

    /// Conceal one lost CELT frame of ``frame_size`` samples per channel.
    ///
    /// Parameters
    /// ----------
    /// frame_size : int
    ///     Samples per channel to conceal (at 48 kHz).
    /// start : int
    ///     First coded band.
    /// end : int
    ///     One past the last coded band.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Shape ``(frames, channels)``, dtype ``float32``.
    #[pyo3(signature = (frame_size, start, end) -> "numpy.typing.NDArray[numpy.float32]")]
    fn decode_lost<'py>(
        &mut self,
        py: Python<'py>,
        frame_size: usize,
        start: usize,
        end: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let pcm = py.detach(|| self.inner.decode_lost(frame_size, start, end));
        interleaved_f32_to_numpy(py, pcm, self.channels)
    }

    fn __repr__(&self) -> String {
        format!(
            "CeltDecoder(channels={}, sample_rate={})",
            self.channels, self.sample_rate
        )
    }
}

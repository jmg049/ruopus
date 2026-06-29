//! Low-level SILK bindings: the SILK-layer encoders and decoder.
//!
//! These operate on `int16` PCM at the SILK *internal* rate (8/12/16 kHz), not
//! the 48 kHz Opus rate - they are the raw SILK codec, below the Opus packet
//! layer. For ordinary use prefer :class:`ruopus.OpusEncoder` /
//! :class:`ruopus.OpusDecoder`.

use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::python::numpy_io::{borrow_1d, interleaved_i16_to_numpy};

fn check_silk_config(fs_khz: i32, nb_subfr: usize) -> PyResult<()> {
    if !matches!(fs_khz, 8 | 12 | 16) {
        return Err(PyValueError::new_err("fs_khz must be 8, 12, or 16"));
    }
    if !matches!(nb_subfr, 2 | 4) {
        return Err(PyValueError::new_err("nb_subfr must be 2 (10 ms) or 4 (20 ms)"));
    }
    Ok(())
}

/// The SILK encoder for one mono stream (`silk_encoder`).
///
/// Parameters
/// ----------
/// fs_khz : int
///     Internal sample rate in kHz: 8, 12, or 16.
/// nb_subfr : int
///     Subframes per frame: 4 for 20 ms, 2 for 10 ms.
/// bitrate : int, optional
///     Target bitrate in bits/s. Defaults to 25000.
/// complexity : int, optional
///     Encode complexity 0-10. Defaults to 10.
#[pyclass(module = "ruopus.lowlevel", name = "SilkEncoder")]
pub struct SilkEncoder {
    inner: crate::silk::encode::api::SilkEncoder,
    fs_khz: i32,
    nb_subfr: usize,
    bitrate: i32,
    complexity: u8,
}

impl SilkEncoder {
    fn frame_length(&self) -> usize {
        self.nb_subfr * 5 * self.fs_khz as usize
    }

    fn check_input(&self, len: usize) -> PyResult<()> {
        let fl = self.frame_length();
        if len == 0 || len % fl != 0 {
            return Err(PyValueError::new_err(format!(
                "input length must be a non-zero multiple of one frame ({fl} samples)"
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl SilkEncoder {
    #[new]
    #[pyo3(signature = (fs_khz, nb_subfr, *, bitrate = 25_000, complexity = 10))]
    fn new(fs_khz: i32, nb_subfr: usize, bitrate: i32, complexity: u8) -> PyResult<Self> {
        check_silk_config(fs_khz, nb_subfr)?;
        let mut inner = crate::silk::encode::api::SilkEncoder::new(fs_khz, nb_subfr);
        inner.set_bitrate(bitrate);
        inner.set_complexity(complexity);
        Ok(Self {
            inner,
            fs_khz,
            nb_subfr,
            bitrate,
            complexity: complexity.min(10),
        })
    }

    /// Target bitrate in bits/s.
    #[getter]
    fn get_bitrate(&self) -> i32 {
        self.bitrate
    }

    #[setter]
    fn set_bitrate(&mut self, bps: i32) {
        self.inner.set_bitrate(bps);
        self.bitrate = bps;
    }

    /// Encode complexity 0-10 (pitch-search depth).
    #[getter]
    fn get_complexity(&self) -> u8 {
        self.complexity
    }

    #[setter]
    fn set_complexity(&mut self, complexity: u8) {
        self.inner.set_complexity(complexity);
        self.complexity = complexity.min(10);
    }

    /// The range coder state after the last :meth:`encode` (``OPUS_GET_FINAL_RANGE``).
    #[getter]
    fn final_range(&self) -> u32 {
        self.inner.final_range()
    }

    /// Encode ``int16`` PCM (a whole number of frames) into a SILK payload.
    ///
    /// Parameters
    /// ----------
    /// pcm : numpy.ndarray
    ///     1-D ``int16`` PCM at the internal rate; length a multiple of one
    ///     frame (``nb_subfr * 5 * fs_khz`` samples).
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The SILK payload.
    #[pyo3(signature = (pcm: "numpy.typing.NDArray[numpy.int16]") -> "bytes")]
    fn encode<'py>(&mut self, py: Python<'py>, pcm: PyReadonlyArray1<'_, i16>) -> PyResult<Bound<'py, PyBytes>> {
        let pcm = borrow_1d(&pcm);
        self.check_input(pcm.len())?;
        let payload = py.detach(|| self.inner.encode(&pcm));
        Ok(PyBytes::new(py, &payload))
    }

    /// Encode into at most ``max_payload`` bytes, lowering the rate to fit.
    ///
    /// Parameters
    /// ----------
    /// pcm : numpy.ndarray
    ///     1-D ``int16`` PCM (a whole number of frames).
    /// max_payload : int
    ///     Byte ceiling for the payload.
    ///
    /// Returns
    /// -------
    /// bytes or None
    ///     The payload, or ``None`` if even the minimum bitrate cannot fit.
    #[pyo3(signature = (pcm: "numpy.typing.NDArray[numpy.int16]", max_payload) -> "bytes | None")]
    fn encode_capped<'py>(
        &mut self,
        py: Python<'py>,
        pcm: PyReadonlyArray1<'_, i16>,
        max_payload: usize,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let pcm = borrow_1d(&pcm);
        self.check_input(pcm.len())?;
        let payload = py.detach(|| self.inner.encode_capped(&pcm, max_payload));
        Ok(payload.map(|p| PyBytes::new(py, &p)))
    }

    fn __repr__(&self) -> String {
        format!(
            "SilkEncoder(fs_khz={}, nb_subfr={}, bitrate={}, complexity={})",
            self.fs_khz, self.nb_subfr, self.bitrate, self.complexity
        )
    }
}

/// The SILK encoder for one stereo stream (`silk` stereo encoder).
///
/// Parameters
/// ----------
/// fs_khz : int
///     Internal sample rate in kHz: 8, 12, or 16.
/// nb_subfr : int
///     Subframes per frame: 4 for 20 ms, 2 for 10 ms.
/// bitrate : int, optional
///     Target bitrate in bits/s. Defaults to 25000.
/// complexity : int, optional
///     Encode complexity 0-10. Defaults to 10.
#[pyclass(module = "ruopus.lowlevel", name = "SilkStereoEncoder")]
pub struct SilkStereoEncoder {
    inner: crate::silk::encode::api::SilkStereoEncoder,
    fs_khz: i32,
    nb_subfr: usize,
    bitrate: i32,
    complexity: u8,
}

#[pymethods]
impl SilkStereoEncoder {
    #[new]
    #[pyo3(signature = (fs_khz, nb_subfr, *, bitrate = 25_000, complexity = 10))]
    fn new(fs_khz: i32, nb_subfr: usize, bitrate: i32, complexity: u8) -> PyResult<Self> {
        check_silk_config(fs_khz, nb_subfr)?;
        let mut inner = crate::silk::encode::api::SilkStereoEncoder::new(fs_khz, nb_subfr);
        inner.set_bitrate(bitrate);
        inner.set_complexity(complexity);
        Ok(Self {
            inner,
            fs_khz,
            nb_subfr,
            bitrate,
            complexity: complexity.min(10),
        })
    }

    /// Target bitrate in bits/s.
    #[getter]
    fn get_bitrate(&self) -> i32 {
        self.bitrate
    }

    #[setter]
    fn set_bitrate(&mut self, bps: i32) {
        self.inner.set_bitrate(bps);
        self.bitrate = bps;
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

    /// Encode left/right ``int16`` channels into a SILK stereo payload.
    ///
    /// Parameters
    /// ----------
    /// left, right : numpy.ndarray
    ///     1-D ``int16`` PCM at the internal rate, equal length, a whole number
    ///     of frames.
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The SILK stereo payload.
    #[pyo3(signature = (left: "numpy.typing.NDArray[numpy.int16]", right: "numpy.typing.NDArray[numpy.int16]") -> "bytes")]
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        left: PyReadonlyArray1<'_, i16>,
        right: PyReadonlyArray1<'_, i16>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let left = borrow_1d(&left);
        let right = borrow_1d(&right);
        let fl = self.nb_subfr * 5 * self.fs_khz as usize;
        if left.len() != right.len() {
            return Err(PyValueError::new_err("left and right must have equal length"));
        }
        if left.is_empty() || left.len() % fl != 0 {
            return Err(PyValueError::new_err(format!(
                "channel length must be a non-zero multiple of one frame ({fl} samples)"
            )));
        }
        let payload = py.detach(|| self.inner.encode(&left, &right));
        Ok(PyBytes::new(py, &payload))
    }

    fn __repr__(&self) -> String {
        format!(
            "SilkStereoEncoder(fs_khz={}, nb_subfr={}, bitrate={}, complexity={})",
            self.fs_khz, self.nb_subfr, self.bitrate, self.complexity
        )
    }
}

/// Decoder control parameters for the low-level SILK decoder (`silk_DecControlStruct`).
///
/// Parameters
/// ----------
/// channels_internal : int
///     Channels coded in the bitstream (1 or 2).
/// channels_api : int
///     Channels to produce (1 or 2).
/// internal_sample_rate : int
///     SILK internal rate in Hz (8000, 12000, or 16000).
/// api_sample_rate : int
///     Output rate in Hz.
/// payload_size_ms : int
///     Packet duration in ms (10, 20, 40, or 60).
#[pyclass(module = "ruopus.lowlevel", name = "DecControl", from_py_object)]
#[derive(Clone, Copy)]
pub struct DecControl {
    pub(crate) inner: crate::silk::api::DecControl,
}

#[pymethods]
impl DecControl {
    #[new]
    #[pyo3(signature = (channels_internal, channels_api, internal_sample_rate, api_sample_rate, payload_size_ms))]
    fn new(
        channels_internal: usize,
        channels_api: usize,
        internal_sample_rate: i32,
        api_sample_rate: i32,
        payload_size_ms: usize,
    ) -> Self {
        Self {
            inner: crate::silk::api::DecControl {
                channels_internal,
                channels_api,
                internal_sample_rate,
                api_sample_rate,
                payload_size_ms,
            },
        }
    }

    #[getter]
    fn get_channels_internal(&self) -> usize {
        self.inner.channels_internal
    }
    #[setter]
    fn set_channels_internal(&mut self, v: usize) {
        self.inner.channels_internal = v;
    }
    #[getter]
    fn get_channels_api(&self) -> usize {
        self.inner.channels_api
    }
    #[setter]
    fn set_channels_api(&mut self, v: usize) {
        self.inner.channels_api = v;
    }
    #[getter]
    fn get_internal_sample_rate(&self) -> i32 {
        self.inner.internal_sample_rate
    }
    #[setter]
    fn set_internal_sample_rate(&mut self, v: i32) {
        self.inner.internal_sample_rate = v;
    }
    #[getter]
    fn get_api_sample_rate(&self) -> i32 {
        self.inner.api_sample_rate
    }
    #[setter]
    fn set_api_sample_rate(&mut self, v: i32) {
        self.inner.api_sample_rate = v;
    }
    #[getter]
    fn get_payload_size_ms(&self) -> usize {
        self.inner.payload_size_ms
    }
    #[setter]
    fn set_payload_size_ms(&mut self, v: usize) {
        self.inner.payload_size_ms = v;
    }

    fn __repr__(&self) -> String {
        format!(
            "DecControl(channels_internal={}, channels_api={}, internal_sample_rate={}, api_sample_rate={}, payload_size_ms={})",
            self.inner.channels_internal,
            self.inner.channels_api,
            self.inner.internal_sample_rate,
            self.inner.api_sample_rate,
            self.inner.payload_size_ms,
        )
    }
}

/// The SILK decoder for one Opus stream (`silk_decoder`).
///
/// The range-coder plumbing is handled internally: :meth:`decode` takes the
/// raw payload ``bytes`` directly.
#[pyclass(module = "ruopus.lowlevel", name = "SilkDecoder")]
pub struct SilkDecoder {
    inner: crate::silk::api::SilkDecoder,
}

#[pymethods]
impl SilkDecoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: crate::silk::api::SilkDecoder::new(),
        }
    }

    /// Decode a SILK payload to ``int16`` PCM shaped ``(frames, channels_api)``.
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     The SILK payload.
    /// ctl : DecControl
    ///     Decoder control parameters.
    /// new_packet : bool, optional
    ///     Whether this begins a new packet. Defaults to ``True``.
    #[pyo3(signature = (data, ctl, new_packet = true) -> "numpy.typing.NDArray[numpy.int16]")]
    fn decode<'py>(
        &mut self,
        py: Python<'py>,
        data: &[u8],
        ctl: DecControl,
        new_packet: bool,
    ) -> PyResult<Bound<'py, PyArray2<i16>>> {
        let owned = data.to_vec();
        let channels = ctl.inner.channels_api.max(1);
        let mut out = Vec::new();
        py.detach(|| {
            let mut dec = crate::range::RangeDecoder::new(&owned);
            self.inner.decode(&mut dec, &ctl.inner, new_packet, &mut out);
        });
        interleaved_i16_to_numpy(py, out, channels)
    }

    /// Decode in-band FEC (LBRR) from a SILK payload.
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     The payload whose FEC data is decoded.
    /// ctl : DecControl
    ///     Decoder control parameters.
    /// new_packet : bool, optional
    ///     Whether this begins a new packet. Defaults to ``True``.
    #[pyo3(signature = (data, ctl, new_packet = true) -> "numpy.typing.NDArray[numpy.int16]")]
    fn decode_fec<'py>(
        &mut self,
        py: Python<'py>,
        data: &[u8],
        ctl: DecControl,
        new_packet: bool,
    ) -> PyResult<Bound<'py, PyArray2<i16>>> {
        let owned = data.to_vec();
        let channels = ctl.inner.channels_api.max(1);
        let mut out = Vec::new();
        py.detach(|| {
            let mut dec = crate::range::RangeDecoder::new(&owned);
            self.inner.decode_fec(&mut dec, &ctl.inner, new_packet, &mut out);
        });
        interleaved_i16_to_numpy(py, out, channels)
    }

    /// Conceal one lost SILK frame.
    ///
    /// Parameters
    /// ----------
    /// ctl : DecControl
    ///     Decoder control parameters.
    #[pyo3(signature = (ctl) -> "numpy.typing.NDArray[numpy.int16]")]
    fn decode_lost<'py>(&mut self, py: Python<'py>, ctl: DecControl) -> PyResult<Bound<'py, PyArray2<i16>>> {
        let channels = ctl.inner.channels_api.max(1);
        let mut out = Vec::new();
        py.detach(|| self.inner.decode_lost(&ctl.inner, &mut out));
        interleaved_i16_to_numpy(py, out, channels)
    }

    fn __repr__(&self) -> String {
        "SilkDecoder()".to_string()
    }
}

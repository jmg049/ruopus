//! The Opus encoder, exposed to Python as `opus_native.OpusEncoder`.

use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::enums::Bandwidth;
use super::numpy_io::borrow_interleaved_f32;

/// An Opus encoder for a single stream at 48 kHz.
///
/// Encodes one frame of interleaved 48 kHz ``float32`` PCM (a 1-D interleaved
/// array, or a 2-D ``(frames, channels)`` array) into an Opus packet returned
/// as ``bytes``. Configuration is exposed as properties (:attr:`complexity`,
/// :attr:`bitrate`, :attr:`dtx`, :attr:`bandwidth`); the encoder is stateful,
/// so feed it consecutive frames from one stream.
///
/// Parameters
/// ----------
/// channels : int
///     1 (mono) or 2 (stereo).
/// complexity : int, optional
///     Encode complexity 0-10 (higher is better quality and slower).
///     Defaults to 10.
/// bitrate : int or None, optional
///     Target bitrate in bits/s, or ``None`` for the per-mode default
///     (CBR filling ``max_bytes``). Defaults to ``None``.
/// dtx : bool, optional
///     Enable discontinuous transmission. Defaults to ``False``.
/// bandwidth : Bandwidth, optional
///     Coded audio bandwidth. Defaults to :attr:`Bandwidth.FullBand`.
///
/// Examples
/// --------
/// >>> import numpy as np, opus_native
/// >>> enc = opus_native.OpusEncoder(2, bitrate=64000)
/// >>> frame = np.zeros((960, 2), dtype=np.float32)   # 20 ms stereo at 48 kHz
/// >>> packet = enc.encode(frame)
#[pyclass(module = "opus_native", name = "OpusEncoder")]
pub struct OpusEncoder {
    inner: crate::encoder::OpusEncoder,
    channels: usize,
    complexity: u8,
    bitrate: Option<u32>,
    dtx: bool,
    bandwidth: Bandwidth,
    vbr: bool,
    inband_fec: bool,
    packet_loss_perc: u8,
}

impl OpusEncoder {
    fn encode_with<'py>(
        &mut self,
        py: Python<'py>,
        pcm: &PyReadonlyArrayDyn<'_, f32>,
        max_bytes: usize,
        f: fn(&mut crate::encoder::OpusEncoder, &[f32], usize) -> Result<Vec<u8>, crate::encoder::EncodeError>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let pcm = borrow_interleaved_f32(pcm, self.channels)?;
        let packet = py.detach(|| f(&mut self.inner, &pcm, max_bytes))?;
        Ok(PyBytes::new(py, &packet))
    }
}

#[pymethods]
impl OpusEncoder {
    #[new]
    #[pyo3(signature = (channels, *, complexity = 10, bitrate = None, dtx = false, bandwidth = Bandwidth::FullBand, inband_fec = false, packet_loss_perc = 0))]
    fn new(
        channels: usize,
        complexity: u8,
        bitrate: Option<u32>,
        dtx: bool,
        bandwidth: Bandwidth,
        inband_fec: bool,
        packet_loss_perc: u8,
    ) -> PyResult<Self> {
        if channels != 1 && channels != 2 {
            return Err(PyValueError::new_err("channels must be 1 or 2"));
        }
        let mut inner = crate::encoder::OpusEncoder::new(channels);
        inner.set_complexity(complexity);
        inner.set_bitrate(bitrate);
        inner.set_dtx(dtx);
        inner.set_bandwidth(bandwidth.into());
        inner.set_inband_fec(inband_fec);
        inner.set_packet_loss_perc(packet_loss_perc);
        Ok(Self {
            inner,
            channels,
            complexity: complexity.min(10),
            bitrate,
            dtx,
            bandwidth,
            vbr: true,
            inband_fec,
            packet_loss_perc: packet_loss_perc.min(100),
        })
    }

    /// Number of channels (1 or 2).
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Encode complexity 0-10 (``OPUS_SET_COMPLEXITY``); higher is better
    /// quality and slower. Values above 10 are clamped.
    #[getter]
    fn get_complexity(&self) -> u8 {
        self.complexity
    }

    #[setter]
    fn set_complexity(&mut self, complexity: u8) {
        self.inner.set_complexity(complexity);
        self.complexity = complexity.min(10);
    }

    /// Target bitrate in bits/s, or ``None`` for the per-mode default
    /// (``OPUS_SET_BITRATE``). For CELT this selects VBR with ``max_bytes`` as a
    /// ceiling; ``None`` restores CBR.
    #[getter]
    fn get_bitrate(&self) -> Option<u32> {
        self.bitrate
    }

    #[setter]
    fn set_bitrate(&mut self, bitrate: Option<u32>) {
        self.inner.set_bitrate(bitrate);
        self.bitrate = bitrate;
    }

    /// Whether discontinuous transmission is enabled (``OPUS_SET_DTX``).
    #[getter]
    fn get_dtx(&self) -> bool {
        self.dtx
    }

    #[setter]
    fn set_dtx(&mut self, on: bool) {
        self.inner.set_dtx(on);
        self.dtx = on;
    }

    /// In-band forward error correction (``OPUS_SET_INBAND_FEC``). When enabled,
    /// SILK-mode packets carry a redundant LBRR copy of the previous packet's
    /// frame(s), so a lost packet can be recovered from its successor via
    /// :meth:`OpusDecoder.decode_fec`.
    #[getter]
    fn get_inband_fec(&self) -> bool {
        self.inband_fec
    }

    #[setter]
    fn set_inband_fec(&mut self, on: bool) {
        self.inner.set_inband_fec(on);
        self.inband_fec = on;
    }

    /// Expected packet-loss percentage 0-100 (``OPUS_SET_PACKET_LOSS_PERC``),
    /// clamped. Biases the encoder toward loss-robust coding.
    #[getter]
    fn get_packet_loss_perc(&self) -> u8 {
        self.packet_loss_perc
    }

    #[setter]
    fn set_packet_loss_perc(&mut self, perc: u8) {
        self.inner.set_packet_loss_perc(perc);
        self.packet_loss_perc = perc.min(100);
    }

    /// Variable bitrate (``OPUS_SET_VBR``). ``True`` (default) makes a set
    /// bitrate a VBR target; ``False`` codes constant bitrate (a fixed byte
    /// count per CELT frame at the target rate).
    #[getter]
    fn get_vbr(&self) -> bool {
        self.vbr
    }

    #[setter]
    fn set_vbr(&mut self, vbr: bool) {
        self.inner.set_vbr(vbr);
        self.vbr = vbr;
    }

    /// Coded audio bandwidth (``OPUS_SET_BANDWIDTH``).
    #[getter]
    fn get_bandwidth(&self) -> Bandwidth {
        self.bandwidth
    }

    #[setter]
    fn set_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.inner.set_bandwidth(bandwidth.into());
        self.bandwidth = bandwidth;
    }

    /// The range coder state after the last packet (``OPUS_GET_FINAL_RANGE``).
    ///
    /// A conformant decoder finishes the same packet with this exact value.
    #[getter]
    fn final_range(&self) -> u32 {
        self.inner.final_range()
    }

    /// Reset the encoder to its freshly-created state (``OPUS_RESET_STATE``).
    ///
    /// Keeps the configuration (channels, complexity, bitrate, bandwidth, DTX)
    /// but drops all cross-frame history, so the next packet is coded as if it
    /// were the first.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Encode one frame, automatically choosing SILK, hybrid, or CELT.
    ///
    /// A simplified mode decision based on frame size, bandwidth, and target
    /// bitrate (not libopus's full hysteresis).
    ///
    /// Parameters
    /// ----------
    /// pcm : numpy.ndarray
    ///     Interleaved 48 kHz ``float32`` PCM in ``[-1, 1]``: a 1-D array or a
    ///     2-D ``(frames, channels)`` array. Frames per channel must be one of
    ///     120, 240, 480, 960, 1920, 2880 (2.5-60 ms).
    /// max_bytes : int, optional
    ///     Output budget in bytes, ``3..=1275``. Defaults to 1275.
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The encoded Opus packet (including its TOC byte).
    ///
    /// Raises
    /// ------
    /// EncodeError
    ///     For an unsupported frame size or an unusable budget.
    #[pyo3(signature = (pcm: "numpy.typing.NDArray[numpy.float32]", max_bytes = 1275))]
    fn encode_auto<'py>(
        &mut self,
        py: Python<'py>,
        pcm: PyReadonlyArrayDyn<'_, f32>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.encode_with(py, &pcm, max_bytes, crate::encoder::OpusEncoder::encode_auto)
    }

    /// Encode one frame as CELT-only.
    ///
    /// Parameters
    /// ----------
    /// pcm : numpy.ndarray
    ///     Interleaved 48 kHz ``float32`` PCM; 120/240/480/960 samples per
    ///     channel (2.5/5/10/20 ms).
    /// max_bytes : int, optional
    ///     Output budget in bytes, ``3..=1275``. Defaults to 1275.
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The encoded Opus packet.
    ///
    /// Raises
    /// ------
    /// EncodeError
    ///     For an unsupported frame size or an unusable budget.
    #[pyo3(signature = (pcm: "numpy.typing.NDArray[numpy.float32]", max_bytes = 1275))]
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        pcm: PyReadonlyArrayDyn<'_, f32>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.encode_with(py, &pcm, max_bytes, crate::encoder::OpusEncoder::encode)
    }

    /// Encode one frame as SILK-only (speech).
    ///
    /// Parameters
    /// ----------
    /// pcm : numpy.ndarray
    ///     Interleaved 48 kHz ``float32`` PCM; 480/960/1920/2880 samples per
    ///     channel (10/20/40/60 ms).
    /// max_bytes : int, optional
    ///     Output budget in bytes, ``3..=1275``. Defaults to 1275.
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The encoded Opus packet.
    ///
    /// Raises
    /// ------
    /// EncodeError
    ///     For an unsupported frame size or an unusable budget.
    #[pyo3(signature = (pcm: "numpy.typing.NDArray[numpy.float32]", max_bytes = 1275))]
    fn encode_silk<'py>(
        &mut self,
        py: Python<'py>,
        pcm: PyReadonlyArrayDyn<'_, f32>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.encode_with(py, &pcm, max_bytes, crate::encoder::OpusEncoder::encode_silk)
    }

    /// Encode one frame as hybrid (SILK low band + CELT high band).
    ///
    /// Parameters
    /// ----------
    /// pcm : numpy.ndarray
    ///     Interleaved 48 kHz ``float32`` PCM; 480/960 samples per channel
    ///     (10/20 ms).
    /// max_bytes : int, optional
    ///     Output budget in bytes, ``3..=1275``. Defaults to 1275.
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The encoded Opus packet.
    ///
    /// Raises
    /// ------
    /// EncodeError
    ///     For an unsupported frame size or an unusable budget.
    #[pyo3(signature = (pcm: "numpy.typing.NDArray[numpy.float32]", max_bytes = 1275))]
    fn encode_hybrid<'py>(
        &mut self,
        py: Python<'py>,
        pcm: PyReadonlyArrayDyn<'_, f32>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.encode_with(py, &pcm, max_bytes, crate::encoder::OpusEncoder::encode_hybrid)
    }

    fn __repr__(&self) -> String {
        format!(
            "OpusEncoder(channels={}, complexity={}, bitrate={}, dtx={})",
            self.channels,
            self.complexity,
            self.bitrate.map_or_else(|| "None".to_string(), |b| b.to_string()),
            if self.dtx { "True" } else { "False" },
        )
    }
}

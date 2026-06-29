//! The Opus decoder, exposed to Python as `ruopus.OpusDecoder`.

use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::numpy_io::{interleaved_f32_to_numpy, interleaved_i16_to_numpy};

/// An Opus decoder for a single stream.
///
/// Decodes Opus packets to interleaved PCM as a NumPy array shaped
/// ``(frames, channels)`` (mono is ``(frames, 1)``). The decoder is stateful:
/// feed it consecutive packets from one stream so inter-frame state (overlap,
/// concealment, mode transitions) stays continuous.
///
/// Parameters
/// ----------
/// channels : int
///     Output channel count, 1 (mono) or 2 (stereo).
/// sample_rate : int, optional
///     Output sample rate in Hz; one of 48000, 24000, 16000, 12000, 8000.
///     Defaults to 48000.
///
/// Examples
/// --------
/// >>> import ruopus
/// >>> dec = ruopus.OpusDecoder(2, sample_rate=48000)
/// >>> pcm = dec.decode_packet(packet)        # (frames, 2) float32 in [-1, 1]
#[pyclass(module = "ruopus", name = "OpusDecoder")]
pub struct OpusDecoder {
    inner: crate::decoder::OpusDecoder,
    channels: usize,
    sample_rate: u32,
}

#[pymethods]
impl OpusDecoder {
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
            inner: crate::decoder::OpusDecoder::with_rate(sample_rate, channels),
            channels,
            sample_rate,
        })
    }

    /// Output channel count (1 or 2).
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Output sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The range coder state after the last packet (``OPUS_GET_FINAL_RANGE``).
    ///
    /// A conformant encoder finishes the same packet with this exact value;
    /// it is the bit-exactness oracle. Zero after a concealed packet.
    #[getter]
    fn final_range(&self) -> u32 {
        self.inner.final_range()
    }

    /// Decode one Opus packet to interleaved float32 PCM in ``[-1, 1]``.
    ///
    /// A 0- or 1-byte payload (TOC only) is treated as DTX and concealed as one
    /// frame of the last good packet's duration.
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     One Opus packet (including its TOC byte).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Shape ``(frames, channels)``, dtype ``float32``.
    ///
    /// Raises
    /// ------
    /// PacketError
    ///     If the packet violates RFC 6716 framing.
    #[pyo3(signature = (data) -> "numpy.typing.NDArray[numpy.float32]")]
    fn decode_packet<'py>(&mut self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyArray2<f32>>> {
        // Packets are tiny (<=1275 bytes); copy out so the decode kernel can run
        // with the GIL released without holding a borrow into Python memory.
        let owned = data.to_vec();
        let pcm = py.detach(|| self.inner.decode_packet(&owned))?;
        interleaved_f32_to_numpy(py, pcm, self.channels)
    }

    /// Decode one Opus packet to interleaved int16 PCM.
    ///
    /// Converts exactly as ``opus_demo`` does: scale by 32768, saturate, round
    /// ties to even.
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     One Opus packet (including its TOC byte).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Shape ``(frames, channels)``, dtype ``int16``.
    ///
    /// Raises
    /// ------
    /// PacketError
    ///     If the packet violates RFC 6716 framing.
    #[pyo3(signature = (data) -> "numpy.typing.NDArray[numpy.int16]")]
    fn decode_packet_i16<'py>(&mut self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyArray2<i16>>> {
        let owned = data.to_vec();
        let pcm = py.detach(|| self.inner.decode_packet_i16(&owned))?;
        interleaved_i16_to_numpy(py, pcm, self.channels)
    }

    /// Conceal one lost packet of ``frame_size`` samples per channel.
    ///
    /// Like ``opus_decode(NULL)``: CELT concealment extrapolates the last pitch
    /// period; frames following SILK/hybrid packets fade to silence (SILK PLC is
    /// not yet ported). The final range of a concealed packet is 0.
    ///
    /// Parameters
    /// ----------
    /// frame_size : int
    ///     Samples per channel to conceal, corresponding to 2.5-60 ms at the
    ///     output sample rate.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Shape ``(frames, channels)``, dtype ``float32``.
    #[pyo3(signature = (frame_size) -> "numpy.typing.NDArray[numpy.float32]")]
    fn decode_lost<'py>(&mut self, py: Python<'py>, frame_size: usize) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let pcm = py.detach(|| self.inner.decode_lost(frame_size));
        interleaved_f32_to_numpy(py, pcm, self.channels)
    }

    /// Decode the in-band FEC (LBRR) data of ``data`` to recover a lost packet.
    ///
    /// Like ``opus_decode(..., decode_fec=1)``: everything except the FEC'd
    /// duration is concealed, then the recovered low-bitrate redundancy frame
    /// completes it. Falls back to plain concealment when the packet carries no
    /// usable FEC (CELT-only modes, or a shorter request).
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     The *next* received packet, whose FEC data reconstructs the lost one.
    /// frame_size : int
    ///     Samples per channel of the lost frame to recover.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Shape ``(frames, channels)``, dtype ``float32``.
    ///
    /// Raises
    /// ------
    /// PacketError
    ///     If the packet violates RFC 6716 framing.
    #[pyo3(signature = (data, frame_size) -> "numpy.typing.NDArray[numpy.float32]")]
    fn decode_fec<'py>(
        &mut self,
        py: Python<'py>,
        data: &[u8],
        frame_size: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let owned = data.to_vec();
        let pcm = py.detach(|| self.inner.decode_fec(&owned, frame_size))?;
        interleaved_f32_to_numpy(py, pcm, self.channels)
    }

    fn __repr__(&self) -> String {
        format!(
            "OpusDecoder(channels={}, sample_rate={})",
            self.channels, self.sample_rate
        )
    }
}

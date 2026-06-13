//! The public Opus encoder (RFC 6716 §5).
//!
//! This first iteration produces **CELT-only fullband** packets at 48 kHz -
//! a single CELT frame (TOC code 0) per call, across the four CELT frame
//! sizes (2.5/5/10/20 ms), mono or stereo. The heavy lifting is in
//! [`crate::celt::encoder::CeltEncoder`]; this layer chooses the TOC byte
//! and frames the payload into a conformant Opus packet. SILK and hybrid
//! modes will extend the mode selection here without changing the API.

use alloc::vec::Vec;

use crate::celt::encoder::CeltEncoder;

/// Errors returned by [`OpusEncoder::encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// The frame had an unsupported number of samples per channel. At
    /// 48 kHz the encoder accepts 120, 240, 480 or 960 (2.5/5/10/20 ms).
    InvalidFrameSize,
    /// The output budget is outside the usable range: at least 3 bytes
    /// (1 TOC + 2 payload) and at most 1275 (the Opus per-frame limit).
    InvalidBudget,
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            EncodeError::InvalidFrameSize => "frame size must be 120/240/480/960 samples per channel at 48 kHz",
            EncodeError::InvalidBudget => "output budget must be between 3 and 1275 bytes",
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EncodeError {}

/// A pure-Rust Opus encoder producing CELT-only fullband packets at 48 kHz.
///
/// ```
/// use opus_native::{OpusEncoder, OpusDecoder};
/// let mut enc = OpusEncoder::new(1);
/// let mut dec = OpusDecoder::new(1);
/// let frame = vec![0.0f32; 960]; // 20 ms of mono silence
/// let packet = enc.encode(&frame, 64).unwrap();
/// let pcm = dec.decode_packet(&packet).unwrap();
/// assert_eq!(pcm.len(), 960);
/// // The decoder finishes on the encoder's range state - bit exactness.
/// assert_eq!(dec.final_range(), enc.final_range());
/// ```
pub struct OpusEncoder {
    celt: CeltEncoder,
    channels: usize,
}

impl OpusEncoder {
    /// Creates an encoder for `channels` (1 or 2) at 48 kHz, fullband.
    ///
    /// # Panics
    ///
    /// Panics unless `channels` is 1 or 2.
    #[must_use]
    pub fn new(channels: usize) -> Self {
        assert!(channels == 1 || channels == 2, "channels must be 1 or 2");
        OpusEncoder {
            celt: CeltEncoder::with_channels(channels),
            channels,
        }
    }

    /// The range state after the last encoded packet (`OPUS_GET_FINAL_RANGE`).
    /// A conformant decoder finishes the packet with this exact value.
    #[must_use]
    pub const fn final_range(&self) -> u32 {
        self.celt.final_range()
    }

    /// Encodes one frame of interleaved 48 kHz f32 PCM in `[-1, 1]` into an
    /// Opus packet of at most `max_bytes` bytes (including the TOC).
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidFrameSize`] if `pcm` is not 120/240/480/960
    /// samples per channel; [`EncodeError::InvalidBudget`] if `max_bytes`
    /// is outside `3..=1275`.
    pub fn encode(&mut self, pcm: &[f32], max_bytes: usize) -> Result<Vec<u8>, EncodeError> {
        if self.channels == 0 || pcm.len() % self.channels != 0 {
            return Err(EncodeError::InvalidFrameSize);
        }
        let n = pcm.len() / self.channels;
        let lm = match n {
            120 => 0u8,
            240 => 1,
            480 => 2,
            960 => 3,
            _ => return Err(EncodeError::InvalidFrameSize),
        };
        if !(3..=1275).contains(&max_bytes) {
            return Err(EncodeError::InvalidBudget);
        }

        // TOC: CELT-only fullband configs 28..=31, stereo flag, code 0
        // (one frame per packet).
        let config = 28 + lm;
        let toc = (config << 3) | (u8::from(self.channels == 2) << 2);

        let payload = self.celt.encode_frame(pcm, max_bytes - 1);
        let mut packet = Vec::with_capacity(payload.len() + 1);
        packet.push(toc);
        packet.extend_from_slice(&payload);
        Ok(packet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpusDecoder;
    use alloc::vec::Vec;

    #[test]
    fn round_trips_through_the_decoder_at_every_frame_size() {
        for &(spf, channels) in &[(120usize, 1usize), (240, 1), (480, 1), (960, 1), (960, 2), (240, 2)] {
            let mut enc = OpusEncoder::new(channels);
            let mut dec = OpusDecoder::new(channels);
            for f in 0..30 {
                let mut pcm = Vec::with_capacity(spf * channels);
                for i in 0..spf {
                    let t = (f * spf + i) as f32 / 48_000.0;
                    let s = 0.5 * (2.0 * core::f32::consts::PI * 440.0 * t).sin();
                    for _ in 0..channels {
                        pcm.push(s);
                    }
                }
                let packet = enc.encode(&pcm, 96).expect("encode");
                let pcm_out = dec.decode_packet(&packet).expect("decode");
                assert_eq!(pcm_out.len(), spf * channels);
                assert_eq!(
                    dec.final_range(),
                    enc.final_range(),
                    "range mismatch at spf={spf} ch={channels} frame {f}"
                );
            }
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut enc = OpusEncoder::new(1);
        assert_eq!(enc.encode(&[0.0; 100], 64), Err(EncodeError::InvalidFrameSize));
        assert_eq!(enc.encode(&[0.0; 960], 2), Err(EncodeError::InvalidBudget));
        assert_eq!(enc.encode(&[0.0; 960], 2000), Err(EncodeError::InvalidBudget));
    }
}

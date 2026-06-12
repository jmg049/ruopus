//! The Opus decoder (RFC 6716 §4; normative `opus_decoder.c`): TOC
//! dispatch over the SILK and CELT decoders, hybrid operation, embedded
//! redundancy, and mode-transition smoothing.
//!
//! Output is 48 kHz interleaved f32 (the float build's native form);
//! [`OpusDecoder::decode_packet_i16`] converts exactly as `opus_demo`
//! does. Packet-loss concealment is not yet ported: mode transitions that
//! the reference smooths with a 5 ms concealment frame fade from silence
//! instead, which only affects 2.5 ms of audio at rare mode switches and
//! never the entropy stream.

use alloc::vec;
use alloc::vec::Vec;

use crate::celt::decoder::CeltDecoder;
use crate::celt::tables::WINDOW120;
use crate::packet::{Bandwidth, Mode, Packet, PacketError};
use crate::range::RangeDecoder;
use crate::silk::api::{DecControl, SilkDecoder};

/// Frame sizes at 48 kHz.
const F2_5: usize = 120;
const F5: usize = 240;
const F20: usize = 960;

/// The CELT end band per bandwidth (`opus_decoder.c`).
const fn end_band(bw: Bandwidth) -> usize {
    match bw {
        Bandwidth::NarrowBand => 13,
        Bandwidth::MediumBand | Bandwidth::WideBand => 17,
        Bandwidth::SuperWideBand => 19,
        Bandwidth::FullBand => 21,
    }
}

/// `smooth_fade`: crossfade `in1` → `in2` over 2.5 ms using the squared
/// CELT window.
fn smooth_fade(in1: &[f32], in2: &[f32], out: &mut [f32], overlap: usize, channels: usize) {
    for c in 0..channels {
        for i in 0..overlap {
            let w = WINDOW120[i] * WINDOW120[i];
            out[i * channels + c] = w * in2[i * channels + c] + (1.0 - w) * in1[i * channels + c];
        }
    }
}

/// The Opus decoder for one stream at 48 kHz output.
pub struct OpusDecoder {
    channels: usize,
    silk: SilkDecoder,
    celt: CeltDecoder,
    stream_channels: usize,
    prev_mode: Option<Mode>,
    prev_redundancy: bool,
    /// The CELT end band of the previous frame (for concealment).
    prev_end: usize,
    /// Frame duration of the last good packet (`st->frame_size`), capping
    /// each concealment chunk.
    last_frame_size: usize,
    /// The range state of the most recent packet
    /// (`OPUS_GET_FINAL_RANGE`): main coder XOR redundant coder.
    final_range: u32,
}

impl OpusDecoder {
    /// Creates a decoder producing `channels` (1 or 2) at 48 kHz.
    ///
    /// # Panics
    ///
    /// Panics unless `channels` is 1 or 2.
    #[must_use]
    pub fn new(channels: usize) -> Self {
        assert!(channels == 1 || channels == 2);
        OpusDecoder {
            channels,
            silk: SilkDecoder::new(),
            celt: CeltDecoder::new(channels),
            stream_channels: channels,
            prev_mode: None,
            prev_redundancy: false,
            prev_end: 21,
            last_frame_size: 120,
            final_range: 0,
        }
    }

    /// The bit-exactness oracle (`OPUS_GET_FINAL_RANGE`).
    #[must_use]
    pub const fn final_range(&self) -> u32 {
        self.final_range
    }

    /// Decodes one Opus packet to interleaved 48 kHz f32
    /// (`channels * duration` samples).
    ///
    /// # Errors
    ///
    /// Returns the packet-layer error for malformed packets.
    pub fn decode_packet(&mut self, data: &[u8]) -> Result<Vec<f32>, PacketError> {
        let packet = Packet::parse(data)?;
        let toc = packet.toc();
        self.stream_channels = usize::from(toc.channels());

        let mut out = Vec::new();
        for frame in packet.frames() {
            let pcm = self.decode_frame(
                frame,
                toc.mode(),
                toc.bandwidth(),
                toc.frame_size().samples_per_channel_48k(),
            );
            out.extend_from_slice(&pcm);
        }
        Ok(out)
    }

    /// Like [`decode_packet`](Self::decode_packet) but converting to s16
    /// exactly as `opus_demo` (scale, saturate, round ties to even).
    ///
    /// # Errors
    ///
    /// Returns the packet-layer error for malformed packets.
    pub fn decode_packet_i16(&mut self, data: &[u8]) -> Result<Vec<i16>, PacketError> {
        Ok(self
            .decode_packet(data)?
            .into_iter()
            .map(|x| (x * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16)
            .collect())
    }

    /// Conceals one lost packet of `frame_size` samples per channel
    /// (10-60 ms), like `opus_decode(NULL)`. CELT concealment extrapolates
    /// the last pitch period; SILK concealment is not yet ported, so
    /// frames following SILK or hybrid packets fade to silence. The final
    /// range of a concealed packet is 0.
    ///
    /// # Panics
    ///
    /// Panics if `frame_size` does not correspond to 2.5-60 ms at 48 kHz.
    #[must_use]
    pub fn decode_lost(&mut self, frame_size: usize) -> Vec<f32> {
        let channels = self.channels;
        let mut out = vec![0.0f32; frame_size * channels];
        if self.prev_mode == Some(Mode::CeltOnly) {
            let mut done = 0usize;
            while done < frame_size {
                // Each chunk is capped by the last good packet's frame
                // duration, then quantised to a runnable size
                // (opus_decode_frame's PLC sizing).
                let mut n = (frame_size - done).min(self.last_frame_size);
                if n > F20 {
                    n = F20;
                } else if n < F20 {
                    if n > 480 {
                        n = 480;
                    } else if n > F5 && n < 480 {
                        n = F5;
                    }
                }
                let pcm = self.celt.decode_lost(n, 0, self.prev_end);
                out[done * channels..(done + n) * channels].copy_from_slice(&pcm);
                done += n;
            }
        }
        self.final_range = 0;
        out
    }

    /// `opus_decode_frame`, normal path (no FEC, no loss).
    #[allow(clippy::too_many_lines, reason = "mirrors the reference sequence")]
    fn decode_frame(&mut self, data: &[u8], mode: Mode, bandwidth: Bandwidth, frame_size: usize) -> Vec<f32> {
        let channels = self.channels;
        let mut len = data.len();
        let audiosize = frame_size;
        let mut pcm = vec![0.0f32; frame_size * channels];

        // Transition detection (mode switch involving CELT-only).
        let transition = self.prev_mode.is_some_and(|prev| {
            (mode == Mode::CeltOnly && prev != Mode::CeltOnly && !self.prev_redundancy)
                || (mode != Mode::CeltOnly && prev == Mode::CeltOnly)
        });
        // Transition audio comes from concealment in the previous mode.
        // CELT concealment is ported; SILK/hybrid concealment still fades
        // from silence.
        let mut pcm_transition = vec![0.0f32; F5 * channels];

        let mut dec = RangeDecoder::new(data);

        // SILK half (SILK-only and hybrid).
        let mut pcm_silk = vec![0i16; frame_size.max(480) * channels];
        if mode != Mode::CeltOnly {
            if self.prev_mode == Some(Mode::CeltOnly) {
                self.silk = SilkDecoder::new();
            }
            let payload_size_ms = 10.max(1000 * audiosize / 48000);
            let ctl = DecControl {
                channels_internal: self.stream_channels,
                channels_api: channels,
                internal_sample_rate: if mode == Mode::SilkOnly {
                    match bandwidth {
                        Bandwidth::NarrowBand => 8000,
                        Bandwidth::MediumBand => 12000,
                        _ => 16000,
                    }
                } else {
                    16000
                },
                api_sample_rate: 48000,
                payload_size_ms,
            };
            let mut silk_out: Vec<i16> = Vec::new();
            let n_calls = payload_size_ms.div_ceil(20).max(1);
            for call in 0..n_calls {
                self.silk.decode(&mut dec, &ctl, call == 0, &mut silk_out);
            }
            debug_assert_eq!(silk_out.len(), frame_size * channels);
            pcm_silk[..silk_out.len()].copy_from_slice(&silk_out);
        }

        // Embedded redundancy.
        let mut redundancy = false;
        let mut celt_to_silk = false;
        let mut redundancy_bytes = 0usize;
        if mode != Mode::CeltOnly && dec.tell() as usize + 17 + 20 * usize::from(mode == Mode::Hybrid) <= 8 * len {
            redundancy = if mode == Mode::Hybrid {
                dec.decode_bit_logp(12)
            } else {
                true
            };
            if redundancy {
                celt_to_silk = dec.decode_bit_logp(1);
                redundancy_bytes = if mode == Mode::Hybrid {
                    dec.decode_uint(256).unwrap_or(0) as usize + 2
                } else {
                    len - ((dec.tell() as usize + 7) >> 3)
                };
                len -= redundancy_bytes;
                // Sanity check (non-normative behaviour for bad packets).
                if len * 8 < dec.tell() as usize {
                    len = 0;
                    redundancy_bytes = 0;
                    redundancy = false;
                } else {
                    // Keep CELT's raw bits out of the redundant tail.
                    dec.shrink_storage(len);
                }
            }
        }
        // Redundancy supersedes the transition fade - the redundant frame
        // provides the smoothing (`if (redundancy) transition = 0`).
        let transition = transition && !redundancy;
        if transition && mode != Mode::CeltOnly && self.prev_mode == Some(Mode::CeltOnly) {
            let n = F5.min(frame_size);
            let pcm = self.celt.decode_lost(n, 0, self.prev_end);
            pcm_transition[..n * channels].copy_from_slice(&pcm);
        }
        let start_band = if mode == Mode::CeltOnly { 0 } else { 17 };

        let celt_end = end_band(bandwidth);
        let mut redundant_audio = vec![0.0f32; F5 * channels];
        let mut redundant_rng = 0u32;

        // 5 ms redundant frame for CELT → SILK (decoded with the carried
        // CELT state, before the main frame).
        if redundancy && celt_to_silk {
            let tail = &data[data.len() - redundancy_bytes..];
            let mut rdec = RangeDecoder::new(tail);
            redundant_audio =
                self.celt
                    .decode_frame(&mut rdec, redundancy_bytes, F5, self.stream_channels, 0, celt_end);
            redundant_rng = rdec.range_size();
        }

        if mode != Mode::SilkOnly {
            let celt_frame_size = F20.min(frame_size);
            // Discard stale CELT state on a mode change.
            if self.prev_mode.is_some_and(|prev| prev != mode) && !self.prev_redundancy {
                self.celt = CeltDecoder::new(channels);
            }
            pcm = self.celt.decode_frame(
                &mut dec,
                len,
                celt_frame_size,
                self.stream_channels,
                start_band,
                celt_end,
            );
            if celt_frame_size < frame_size {
                pcm.resize(frame_size * channels, 0.0);
            }
        } else {
            // For hybrid → SILK transitions the CELT MDCT fades out by
            // decoding a silence frame.
            if self.prev_mode == Some(Mode::Hybrid) && !(redundancy && celt_to_silk && self.prev_redundancy) {
                let silence = [0xFF, 0xFF];
                let mut sdec = RangeDecoder::new(&silence);
                let fade = self
                    .celt
                    .decode_frame(&mut sdec, 2, F2_5, self.stream_channels, 0, celt_end);
                pcm[..F2_5 * channels].copy_from_slice(&fade);
            }
        }

        // Add the SILK contribution.
        if mode != Mode::CeltOnly {
            for (p, &s) in pcm.iter_mut().zip(pcm_silk.iter()) {
                *p += f32::from(s) / 32768.0;
            }
        }

        // 5 ms redundant frame for SILK → CELT (fresh CELT state), faded
        // in over the last 2.5 ms of the frame.
        if redundancy && !celt_to_silk {
            self.celt = CeltDecoder::new(channels);
            let tail = &data[data.len() - redundancy_bytes..];
            let mut rdec = RangeDecoder::new(tail);
            redundant_audio =
                self.celt
                    .decode_frame(&mut rdec, redundancy_bytes, F5, self.stream_channels, 0, celt_end);
            redundant_rng = rdec.range_size();
            let off = channels * (frame_size - F2_5);
            let faded: Vec<f32> = {
                let in1 = &pcm[off..];
                let in2 = &redundant_audio[channels * F2_5..];
                let mut out = vec![0.0f32; F2_5 * channels];
                smooth_fade(in1, in2, &mut out, F2_5, channels);
                out
            };
            pcm[off..].copy_from_slice(&faded);
        }

        // CELT → SILK redundancy: the first 2.5 ms is the redundant audio,
        // fading into the SILK output (skipped if the CELT state was stale).
        if redundancy && celt_to_silk && (self.prev_mode != Some(Mode::SilkOnly) || self.prev_redundancy) {
            pcm[..F2_5 * channels].copy_from_slice(&redundant_audio[..F2_5 * channels]);
            let faded: Vec<f32> = {
                let in1 = &redundant_audio[channels * F2_5..];
                let in2 = &pcm[channels * F2_5..];
                let mut out = vec![0.0f32; F2_5 * channels];
                smooth_fade(in1, in2, &mut out, F2_5, channels);
                out
            };
            pcm[channels * F2_5..channels * 2 * F2_5].copy_from_slice(&faded);
        }

        // Mode-transition fade (from the concealment placeholder).
        if transition {
            if audiosize >= F5 {
                pcm[..channels * F2_5].copy_from_slice(&pcm_transition[..channels * F2_5]);
                let faded: Vec<f32> = {
                    let in1 = &pcm_transition[channels * F2_5..];
                    let in2 = &pcm[channels * F2_5..];
                    let mut out = vec![0.0f32; F2_5 * channels];
                    smooth_fade(in1, in2, &mut out, F2_5, channels);
                    out
                };
                pcm[channels * F2_5..channels * 2 * F2_5].copy_from_slice(&faded);
            } else {
                let faded: Vec<f32> = {
                    let mut out = vec![0.0f32; F2_5 * channels];
                    smooth_fade(&pcm_transition, &pcm, &mut out, F2_5, channels);
                    out
                };
                pcm[..channels * F2_5].copy_from_slice(&faded);
            }
        }

        self.final_range = dec.range_size() ^ redundant_rng;
        self.prev_mode = Some(mode);
        self.prev_end = celt_end;
        self.last_frame_size = frame_size;
        self.prev_redundancy = redundancy && !celt_to_silk;
        pcm
    }
}

/// Errors from [`decode_ogg_opus`].
#[derive(Debug)]
pub enum OggDecodeError {
    /// The container or headers are malformed.
    Container(crate::ogg::OggOpusError),
    /// An audio packet violates RFC 6716 framing.
    Packet(PacketError),
    /// Channel mapping families other than 0 need a multistream decoder
    /// (not yet implemented).
    UnsupportedMapping,
}

impl core::fmt::Display for OggDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OggDecodeError::Container(e) => write!(f, "Ogg Opus container error: {e}"),
            OggDecodeError::Packet(e) => write!(f, "Opus packet error: {e}"),
            OggDecodeError::UnsupportedMapping => {
                write!(f, "channel mapping families other than 0 are not supported yet")
            },
        }
    }
}

impl std::error::Error for OggDecodeError {}

/// Decodes an in-memory Ogg Opus file end to end (RFC 7845 §4):
/// pre-skip removal, end trimming from the final granule position, and the
/// `OpusHead` output gain. Returns interleaved 48 kHz f32 and the parsed
/// header.
///
/// # Errors
///
/// See [`OggDecodeError`]. Only channel mapping family 0 (mono/stereo) is
/// supported until a multistream decoder exists.
pub fn decode_ogg_opus(data: &[u8]) -> Result<(Vec<f32>, crate::ogg::OpusHead), OggDecodeError> {
    use crate::ogg::{ChannelMapping, OggOpusReader};

    let mut reader = OggOpusReader::new(data).map_err(OggDecodeError::Container)?;
    let head = reader.head().clone();
    if head.channel_mapping != ChannelMapping::Family0 {
        return Err(OggDecodeError::UnsupportedMapping);
    }
    let channels = usize::from(head.channel_count);
    let mut decoder = OpusDecoder::new(channels);

    let mut pcm: Vec<f32> = Vec::new();
    let mut final_granule = 0u64;
    while let Some(pkt) = reader.next() {
        pcm.extend(decoder.decode_packet(&pkt.data).map_err(OggDecodeError::Packet)?);
        final_granule = pkt.granule_position;
    }

    // Pre-skip at the front; end trimming against the final granule.
    let pre_skip = usize::from(head.pre_skip);
    let total = (final_granule.saturating_sub(u64::from(head.pre_skip))) as usize;
    let mut pcm: Vec<f32> = pcm.into_iter().skip(pre_skip * channels).collect();
    pcm.truncate(total * channels);

    // Output gain, Q7.8 dB.
    if head.output_gain_q8 != 0 {
        let gain = libm_exp10(f64::from(head.output_gain_q8) / (20.0 * 256.0)) as f32;
        for v in &mut pcm {
            *v *= gain;
        }
    }
    Ok((pcm, head))
}

/// `10^x`.
fn libm_exp10(x: f64) -> f64 {
    (x * core::f64::consts::LN_10).exp()
}

//! The SILK decoder API layer (RFC 6716 §4.2.3-4.2.4).
//!
//! One call decodes one 10 or 20 ms frame of a packet (packets carry up to
//! three frames): the first call of a packet reads the per-frame VAD bits
//! and the LBRR flags and skips any LBRR data; stereo packets then carry
//! the mid/side predictor weights and an optional mid-only flag. Each
//! channel decodes through [`SilkChannelDecoder`], stereo unmixes to
//! left/right, and the resampler produces output at the API rate.

#![allow(dead_code, reason = "wired into the Opus-level decoder next")]
#![allow(missing_docs, reason = "crate-internal API surfaced for conformance testing")]

use alloc::vec;
use alloc::vec::Vec;

use super::decoder::SilkChannelDecoder;
use super::indices::{CondCoding, decode_indices};
use super::pulses::decode_pulses;
use super::resampler::Resampler;
use super::stereo::{StereoDecState, stereo_decode_mid_only, stereo_decode_pred, stereo_ms_to_lr};
use super::tables::{LBRR_FLAGS_2_ICDF, LBRR_FLAGS_3_ICDF};
use crate::range::RangeDecoder;

/// Maximum number of frames per packet.
const MAX_FRAMES_PER_PACKET: usize = 3;

/// One channel: frame decoder, resampler, and per-packet flags.
struct ChannelState {
    dec: SilkChannelDecoder,
    resampler: Resampler,
    n_frames_decoded: usize,
    n_frames_per_packet: usize,
    vad_flags: [bool; MAX_FRAMES_PER_PACKET],
    lbrr_flags: [bool; MAX_FRAMES_PER_PACKET],
}

/// Per-call control (decode-relevant fields).
#[derive(Debug, Clone, Copy)]
pub struct DecControl {
    /// Channels coded in the bitstream (1 or 2).
    pub channels_internal: usize,
    /// Channels to produce (1 or 2).
    pub channels_api: usize,
    /// SILK internal rate in Hz (8000, 12000 or 16000).
    pub internal_sample_rate: i32,
    /// Output rate in Hz.
    pub api_sample_rate: i32,
    /// Packet duration in ms (10, 20, 40 or 60).
    pub payload_size_ms: usize,
}

/// The SILK decoder for one Opus stream.
pub struct SilkDecoder {
    channels: [Option<ChannelState>; 2],
    stereo: StereoDecState,
    prev_decode_only_middle: bool,
    n_channels_internal: usize,
    n_channels_api: usize,
}

impl Default for SilkDecoder {
    fn default() -> Self {
        SilkDecoder::new()
    }
}

impl SilkDecoder {
    /// Create a new decoder.
    #[must_use]
    pub fn new() -> Self {
        SilkDecoder {
            channels: [None, None],
            stereo: StereoDecState::default(),
            prev_decode_only_middle: false,
            n_channels_internal: 0,
            n_channels_api: 0,
        }
    }

    /// Normal decode: decodes the next frame of the packet from `dec`,
    /// appending interleaved output at the API rate to `out`. `new_packet`
    /// must be true on the first call for each packet.
    ///
    /// # Panics
    ///
    /// Panics on invalid control parameters (rates/channels/duration
    /// outside what RFC 6716 allows) - the Opus layer validates these from
    /// the TOC.
    pub fn decode(&mut self, dec: &mut RangeDecoder, ctl: &DecControl, new_packet: bool, out: &mut Vec<i16>) {
        self.decode_inner(dec, ctl, new_packet, false, out);
    }

    /// Decodes the in-band FEC (LBRR) data of this packet for the frames
    /// that carry it, concealing the rest.
    ///
    /// # Panics
    ///
    /// Panics on invalid control parameters.
    pub fn decode_fec(&mut self, dec: &mut RangeDecoder, ctl: &DecControl, new_packet: bool, out: &mut Vec<i16>) {
        self.decode_inner(dec, ctl, new_packet, true, out);
    }

    fn decode_inner(
        &mut self,
        dec: &mut RangeDecoder,
        ctl: &DecControl,
        new_packet: bool,
        fec: bool,
        out: &mut Vec<i16>,
    ) {
        assert!(ctl.channels_internal == 1 || ctl.channels_internal == 2);
        let fs_khz = (ctl.internal_sample_rate >> 10) + 1;
        debug_assert!(fs_khz == 8 || fs_khz == 12 || fs_khz == 16);
        let (n_frames_per_packet, nb_subfr) = match ctl.payload_size_ms {
            10 => (1, 2),
            20 => (1, 4),
            40 => (2, 4),
            60 => (3, 4),
            _ => panic!("invalid SILK payload duration"),
        };

        if new_packet {
            for ch in self.channels.iter_mut().flatten() {
                ch.n_frames_decoded = 0;
            }
        }

        // Mono → stereo transition: reset the second channel.
        if ctl.channels_internal > self.n_channels_internal {
            self.channels[1] = None;
        }

        // (Re)configure channels on the first frame of a packet.
        let first_frame = self.channels[0].as_ref().is_none_or(|ch| ch.n_frames_decoded == 0);
        if first_frame {
            for n in 0..ctl.channels_internal {
                let needs_new = self.channels[n]
                    .as_ref()
                    .is_none_or(|ch| ch.dec.fs_khz != fs_khz || ch.resampler_rate() != ctl.api_sample_rate);
                if needs_new {
                    self.channels[n] = Some(ChannelState {
                        dec: SilkChannelDecoder::new(fs_khz, nb_subfr),
                        resampler: Resampler::new(fs_khz * 1000, ctl.api_sample_rate),
                        n_frames_decoded: 0,
                        n_frames_per_packet,
                        vad_flags: [false; MAX_FRAMES_PER_PACKET],
                        lbrr_flags: [false; MAX_FRAMES_PER_PACKET],
                    });
                } else if let Some(ch) = self.channels[n].as_mut() {
                    // Same rates; the frame duration may still change.
                    ch.n_frames_per_packet = n_frames_per_packet;
                    ch.dec.set_frame_duration(nb_subfr);
                }
            }
        }

        // Stereo output of a newly stereo stream: reset prediction memory
        // and clone the resampler state into the side channel.
        if ctl.channels_api == 2
            && ctl.channels_internal == 2
            && (self.n_channels_api == 1 || self.n_channels_internal == 1)
        {
            self.stereo.pred_prev_q13 = [0; 2];
            self.stereo.s_side = [0; 2];
            if let [Some(ch0), Some(ch1)] = &mut self.channels {
                ch1.resampler = ch0.resampler.clone();
            }
        }
        self.n_channels_api = ctl.channels_api;
        self.n_channels_internal = ctl.channels_internal;

        let frame_length;
        {
            let ch0 = self.channels[0].as_ref().expect("configured above");
            frame_length = ch0.dec.frame_length;
        }

        // First call of the packet: VAD and LBRR flags, then skip LBRR.
        if first_frame {
            for n in 0..ctl.channels_internal {
                let ch = self.channels[n].as_mut().expect("configured");
                for i in 0..ch.n_frames_per_packet {
                    ch.vad_flags[i] = dec.decode_bit_logp(1);
                }
                let lbrr_flag = dec.decode_bit_logp(1);
                ch.lbrr_flags = [false; MAX_FRAMES_PER_PACKET];
                if lbrr_flag {
                    if ch.n_frames_per_packet == 1 {
                        ch.lbrr_flags[0] = true;
                    } else {
                        let table: &[u8] = if ch.n_frames_per_packet == 2 {
                            &LBRR_FLAGS_2_ICDF
                        } else {
                            &LBRR_FLAGS_3_ICDF
                        };
                        let symbol = dec.decode_icdf(table, 8) + 1;
                        for i in 0..ch.n_frames_per_packet {
                            ch.lbrr_flags[i] = (symbol >> i) & 1 == 1;
                        }
                    }
                }
            }

            // Regular decoding skips all LBRR data; FEC decoding consumes
            // it in the per-frame loop instead.
            for i in 0..if fec { 0 } else { n_frames_per_packet } {
                for n in 0..ctl.channels_internal {
                    if !self.channels[n].as_ref().expect("configured").lbrr_flags[i] {
                        continue;
                    }
                    if ctl.channels_internal == 2 && n == 0 {
                        let _ = stereo_decode_pred(dec);
                        if !self.channels[1].as_ref().expect("configured").lbrr_flags[i] {
                            let _ = stereo_decode_mid_only(dec);
                        }
                    }
                    let ch = self.channels[n].as_mut().expect("configured");
                    let cond = if i > 0 && ch.lbrr_flags[i - 1] {
                        CondCoding::Conditionally
                    } else {
                        CondCoding::Independently
                    };
                    let indices = decode_indices(
                        dec,
                        ch.dec.fs_khz,
                        ch.dec.nb_subfr,
                        true,
                        true,
                        cond,
                        &mut ch.dec.ec_prev,
                    );
                    let _ = decode_pulses(
                        dec,
                        i32::from(indices.signal_type),
                        i32::from(indices.quant_offset_type),
                        ch.dec.frame_length,
                    );
                }
            }
        }

        // Stereo predictor weights and the mid-only flag.
        let mut ms_pred_q13 = [0i32; 2];
        let mut decode_only_middle = false;
        if ctl.channels_internal == 2 {
            let frame_index = self.channels[0].as_ref().expect("configured").n_frames_decoded;
            let coded = !fec || self.channels[0].as_ref().expect("configured").lbrr_flags[frame_index];
            if coded {
                ms_pred_q13 = stereo_decode_pred(dec);
                let side = self.channels[1].as_ref().expect("configured");
                let side_coded = if fec {
                    side.lbrr_flags[frame_index]
                } else {
                    side.vad_flags[frame_index]
                };
                if !side_coded {
                    decode_only_middle = stereo_decode_mid_only(dec);
                }
            } else {
                ms_pred_q13 = [
                    i32::from(self.stereo.pred_prev_q13[0]),
                    i32::from(self.stereo.pred_prev_q13[1]),
                ];
            }
        }

        // First side frame after mid-only: reset side prediction memory.
        if ctl.channels_internal == 2 && !decode_only_middle && self.prev_decode_only_middle {
            let ch1 = self.channels[1].as_mut().expect("configured");
            ch1.dec.reset_side_prediction();
        }

        // Decode each channel (frame_length + 2 with the stereo history).
        let mut mid = vec![0i16; frame_length + 2];
        let mut side = vec![0i16; frame_length + 2];
        let frames_decoded0 = self.channels[0].as_ref().expect("configured").n_frames_decoded;
        let has_side = if fec {
            !self.prev_decode_only_middle
                || (ctl.channels_internal == 2
                    && self.channels[1].as_ref().expect("configured").lbrr_flags[frames_decoded0])
        } else {
            !decode_only_middle
        };
        for n in 0..ctl.channels_internal {
            if n == 0 || has_side {
                // Independent coding when no previous frame is available.
                // Both channels effectively see the pre-loop frame count.
                let frame_index = frames_decoded0 as i32;
                let ch_lbrr = self.channels[n].as_ref().expect("configured").lbrr_flags;
                let cond = if frame_index <= 0 {
                    CondCoding::Independently
                } else if fec {
                    if ch_lbrr[frames_decoded0 - 1] {
                        CondCoding::Conditionally
                    } else {
                        CondCoding::Independently
                    }
                } else if n > 0 && self.prev_decode_only_middle {
                    // A skipped side frame leaves the LTP state defined.
                    CondCoding::IndependentlyNoLtpScaling
                } else {
                    CondCoding::Conditionally
                };
                let buf = if n == 0 { &mut mid } else { &mut side };
                let ch = self.channels[n].as_mut().expect("configured");
                if fec && !ch.lbrr_flags[ch.n_frames_decoded] {
                    // No FEC for this frame: conceal it.
                    ch.dec.decode_frame_lost(&mut buf[2..]);
                } else {
                    let vad_flag = ch.vad_flags[ch.n_frames_decoded];
                    ch.dec.decode_frame(dec, &mut buf[2..], vad_flag, fec, cond);
                }
            } else {
                side[2..].fill(0);
            }
            self.channels[n].as_mut().expect("configured").n_frames_decoded += 1;
        }

        if ctl.channels_api == 2 && ctl.channels_internal == 2 {
            // Mid/side → left/right.
            stereo_ms_to_lr(
                &mut self.stereo,
                &mut mid,
                &mut side,
                &ms_pred_q13,
                fs_khz,
                frame_length,
            );
        } else {
            // Mono: buffer the two-sample history.
            mid[..2].copy_from_slice(&self.stereo.s_mid);
            self.stereo.s_mid.copy_from_slice(&mid[frame_length..frame_length + 2]);
        }

        let n_samples_out = frame_length * ctl.api_sample_rate as usize / (fs_khz as usize * 1000);

        // Resample each output channel (input offset 1: one sample of the
        // stereo history) and interleave.
        let base = out.len();
        out.resize(base + n_samples_out * ctl.channels_api, 0);
        let mut resampled = vec![0i16; n_samples_out];
        for n in 0..ctl.channels_api.min(ctl.channels_internal) {
            let src = if n == 0 { &mid } else { &side };
            let ch = self.channels[n].as_mut().expect("configured");
            ch.resampler.process(&mut resampled, &src[1..=frame_length]);
            if ctl.channels_api == 2 {
                for (i, &s) in resampled.iter().enumerate() {
                    out[base + n + 2 * i] = s;
                }
            } else {
                out[base..].copy_from_slice(&resampled);
            }
        }

        // Stereo output from a mono stream: duplicate the channel.
        if ctl.channels_api == 2 && ctl.channels_internal == 1 {
            for i in 0..n_samples_out {
                out[base + 1 + 2 * i] = out[base + 2 * i];
            }
        }

        self.prev_decode_only_middle = decode_only_middle;
    }
}

impl SilkDecoder {
    /// Conceals one frame per call, appending interleaved output at the API
    /// rate to `out`. The decoder must have decoded at least one good
    /// packet.
    ///
    /// # Panics
    ///
    /// Panics on invalid control parameters or if no packet was decoded
    /// yet.
    pub fn decode_lost(&mut self, ctl: &DecControl, out: &mut Vec<i16>) {
        let fs_khz = (ctl.internal_sample_rate >> 10) + 1;
        // A lost call reconfigures the frame duration exactly like a good
        // one (the payload size says how much to conceal).
        let nb_subfr = if ctl.payload_size_ms == 10 { 2 } else { 4 };
        for n in 0..ctl.channels_internal {
            let ch = self.channels[n].as_mut().expect("a good packet first");
            debug_assert_eq!(ch.dec.fs_khz, fs_khz);
            ch.dec.set_frame_duration(nb_subfr);
        }
        let frame_length = self.channels[0].as_ref().expect("configured").dec.frame_length;

        // Concealed frames reuse the previous stereo prediction.
        let ms_pred_q13 = [
            i32::from(self.stereo.pred_prev_q13[0]),
            i32::from(self.stereo.pred_prev_q13[1]),
        ];

        let mut mid = vec![0i16; frame_length + 2];
        let mut side = vec![0i16; frame_length + 2];
        let has_side = !self.prev_decode_only_middle;
        for n in 0..ctl.channels_internal {
            if n == 0 || has_side {
                let buf = if n == 0 { &mut mid } else { &mut side };
                let ch = self.channels[n].as_mut().expect("configured");
                ch.dec.decode_frame_lost(&mut buf[2..]);
            } else {
                side[2..].fill(0);
            }
            let ch = self.channels[n].as_mut().expect("configured");
            ch.n_frames_decoded += 1;
        }

        if ctl.channels_api == 2 && ctl.channels_internal == 2 {
            stereo_ms_to_lr(
                &mut self.stereo,
                &mut mid,
                &mut side,
                &ms_pred_q13,
                fs_khz,
                frame_length,
            );
        } else {
            mid[..2].copy_from_slice(&self.stereo.s_mid);
            self.stereo.s_mid.copy_from_slice(&mid[frame_length..frame_length + 2]);
        }

        let n_samples_out = frame_length * ctl.api_sample_rate as usize / (fs_khz as usize * 1000);
        let base = out.len();
        out.resize(base + n_samples_out * ctl.channels_api, 0);
        let mut resampled = vec![0i16; n_samples_out];
        for n in 0..ctl.channels_api.min(ctl.channels_internal) {
            let src = if n == 0 { &mid } else { &side };
            let ch = self.channels[n].as_mut().expect("configured");
            ch.resampler.process(&mut resampled, &src[1..=frame_length]);
            if ctl.channels_api == 2 {
                for (i, &s) in resampled.iter().enumerate() {
                    out[base + n + 2 * i] = s;
                }
            } else {
                out[base..].copy_from_slice(&resampled);
            }
        }
        if ctl.channels_api == 2 && ctl.channels_internal == 1 {
            for i in 0..n_samples_out {
                out[base + 1 + 2 * i] = out[base + 2 * i];
            }
        }

        // Remove the gain clamping so energy doesn't bounce back after
        // losses while it is decaying.
        for n in 0..ctl.channels_internal {
            if let Some(ch) = self.channels[n].as_mut() {
                ch.dec.params.last_gain_index = 10;
            }
        }
    }
}

impl ChannelState {
    fn resampler_rate(&self) -> i32 {
        self.resampler.output_rate_hz()
    }
}

//! Ogg encapsulation for Opus (RFC 7845): the `OpusHead`/`OpusTags` headers
//! and stream-level reading/writing.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::page::{NO_GRANULE, OggPacket, PacketReader, PageReader, PageWriter};
use crate::packet::Toc;

/// Magic signature of the identification header (RFC 7845 §5.1).
pub const OPUS_HEAD_MAGIC: [u8; 8] = *b"OpusHead";

/// Magic signature of the comment header (RFC 7845 §5.2).
pub const OPUS_TAGS_MAGIC: [u8; 8] = *b"OpusTags";

/// Why an Ogg Opus stream failed to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OggOpusError {
    /// No logical bitstream beginning with an `OpusHead` packet was found.
    NoOpusStream,
    /// The identification header is malformed or truncated.
    InvalidIdHeader,
    /// The encapsulation major version is unsupported (RFC 7845 §5.1 item 2:
    /// versions 16 and up are incompatible).
    UnsupportedVersion(u8),
    /// The comment header is missing, malformed, or truncated.
    InvalidCommentHeader,
}

impl fmt::Display for OggOpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OggOpusError::NoOpusStream => f.write_str("no Opus logical bitstream found"),
            OggOpusError::InvalidIdHeader => f.write_str("malformed OpusHead identification header"),
            OggOpusError::UnsupportedVersion(v) => {
                write!(f, "unsupported Ogg Opus encapsulation version {v}")
            },
            OggOpusError::InvalidCommentHeader => f.write_str("malformed OpusTags comment header"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OggOpusError {}

/// The channel mapping of an Ogg Opus stream (RFC 7845 §5.1.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelMapping {
    /// Family 0: mono or stereo, one Opus stream, no table coded.
    Family0,
    /// Families 1 (Vorbis surround order), 255 (undefined), and any other
    /// family carrying an explicit mapping table.
    Table {
        /// The channel mapping family octet.
        family: u8,
        /// Total Opus streams per Ogg packet, `N` (≥ 1).
        stream_count: u8,
        /// Streams decoded as stereo, `M` (≤ `N`).
        coupled_count: u8,
        /// One entry per output channel: decoded-channel index or 255 for
        /// silence.
        mapping: Vec<u8>,
    },
}

impl ChannelMapping {
    /// Total number of Opus streams in each Ogg packet.
    #[must_use]
    pub fn stream_count(&self) -> u8 {
        match self {
            ChannelMapping::Family0 => 1,
            ChannelMapping::Table { stream_count, .. } => *stream_count,
        }
    }
}

/// The identification header of an Ogg Opus stream (RFC 7845 §5.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusHead {
    /// Encapsulation version; `1` for this specification, with the upper four
    /// bits as a compatible major version.
    pub version: u8,
    /// Output channel count `C`; never zero.
    pub channel_count: u8,
    /// Samples (at 48 kHz) to discard from decoder output at startup, and the
    /// offset subtracted from granule positions to obtain PCM positions.
    pub pre_skip: u16,
    /// Sample rate of the *original* input in Hz - metadata only, not the
    /// playback rate. Zero means unspecified.
    pub input_sample_rate: u32,
    /// Output gain in Q7.8 dB, applied by players on top of decoder output.
    pub output_gain_q8: i16,
    /// Stream-to-channel mapping.
    pub channel_mapping: ChannelMapping,
}

impl OpusHead {
    /// Parses an `OpusHead` packet.
    ///
    /// # Errors
    ///
    /// [`OggOpusError::InvalidIdHeader`] for structural problems;
    /// [`OggOpusError::UnsupportedVersion`] when the major version is not 0
    /// (i.e., the version octet is 16 or greater).
    pub fn parse(data: &[u8]) -> Result<Self, OggOpusError> {
        if data.len() < 19 || data[0..8] != OPUS_HEAD_MAGIC {
            return Err(OggOpusError::InvalidIdHeader);
        }
        let version = data[8];
        if version >> 4 != 0 {
            return Err(OggOpusError::UnsupportedVersion(version));
        }
        let channel_count = data[9];
        if channel_count == 0 {
            return Err(OggOpusError::InvalidIdHeader);
        }
        let pre_skip = u16::from_le_bytes([data[10], data[11]]);
        let input_sample_rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let output_gain_q8 = i16::from_le_bytes([data[16], data[17]]);
        let family = data[18];

        let channel_mapping = if family == 0 {
            if channel_count > 2 {
                return Err(OggOpusError::InvalidIdHeader);
            }
            ChannelMapping::Family0
        } else {
            // Families other than 0 carry an explicit table (§5.1 item 8).
            let table = &data[19..];
            if table.len() < 2 + usize::from(channel_count) {
                return Err(OggOpusError::InvalidIdHeader);
            }
            let stream_count = table[0];
            let coupled_count = table[1];
            if stream_count == 0
                || coupled_count > stream_count
                || usize::from(stream_count) + usize::from(coupled_count) > 255
            {
                return Err(OggOpusError::InvalidIdHeader);
            }
            let mapping = table[2..2 + usize::from(channel_count)].to_vec();
            let decoded_channels = stream_count + coupled_count;
            if mapping.iter().any(|&idx| idx != 255 && idx >= decoded_channels) {
                return Err(OggOpusError::InvalidIdHeader);
            }
            ChannelMapping::Table {
                family,
                stream_count,
                coupled_count,
                mapping,
            }
        };

        Ok(OpusHead {
            version,
            channel_count,
            pre_skip,
            input_sample_rate,
            output_gain_q8,
            channel_mapping,
        })
    }

    /// Serialises this header into an `OpusHead` packet.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(19);
        out.extend_from_slice(&OPUS_HEAD_MAGIC);
        out.push(self.version);
        out.push(self.channel_count);
        out.extend_from_slice(&self.pre_skip.to_le_bytes());
        out.extend_from_slice(&self.input_sample_rate.to_le_bytes());
        out.extend_from_slice(&self.output_gain_q8.to_le_bytes());
        match &self.channel_mapping {
            ChannelMapping::Family0 => out.push(0),
            ChannelMapping::Table {
                family,
                stream_count,
                coupled_count,
                mapping,
            } => {
                out.push(*family);
                out.push(*stream_count);
                out.push(*coupled_count);
                out.extend_from_slice(mapping);
            },
        }
        out
    }

    /// A standard family-0 header for mono or stereo.
    ///
    /// # Panics
    ///
    /// Panics if `channels` is not 1 or 2.
    #[must_use]
    pub fn family0(channels: u8, pre_skip: u16, input_sample_rate: u32) -> Self {
        assert!(channels == 1 || channels == 2, "family 0 allows 1 or 2 channels");
        OpusHead {
            version: 1,
            channel_count: channels,
            pre_skip,
            input_sample_rate,
            output_gain_q8: 0,
            channel_mapping: ChannelMapping::Family0,
        }
    }
}

/// The comment header of an Ogg Opus stream (RFC 7845 §5.2): a vendor string
/// plus `TAG=value` user comments, both UTF-8.
///
/// Comments are stored as raw bytes to allow lossless round-tripping of
/// streams whose tags are not valid UTF-8; [`OpusTags::comments_lossy`]
/// provides convenient string access.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OpusTags {
    /// Encoder/muxer identification string.
    pub vendor: Vec<u8>,
    /// User comments, conventionally of the form `TAG=value`.
    pub comments: Vec<Vec<u8>>,
}

impl OpusTags {
    /// Parses an `OpusTags` packet.
    ///
    /// Length fields are validated against the remaining packet size before
    /// any allocation, as the RFC requires, so hostile headers cannot demand
    /// unbounded memory.
    ///
    /// # Errors
    ///
    /// [`OggOpusError::InvalidCommentHeader`] for any structural problem.
    pub fn parse(data: &[u8]) -> Result<Self, OggOpusError> {
        let mut rest = data
            .strip_prefix(&OPUS_TAGS_MAGIC)
            .ok_or(OggOpusError::InvalidCommentHeader)?;

        let vendor = read_len_prefixed(&mut rest)?.to_vec();

        let count = read_u32(&mut rest)? as usize;
        // Each comment needs at least its 4-byte length field.
        if count > rest.len() / 4 {
            return Err(OggOpusError::InvalidCommentHeader);
        }
        let mut comments = Vec::with_capacity(count);
        for _ in 0..count {
            comments.push(read_len_prefixed(&mut rest)?.to_vec());
        }

        Ok(OpusTags { vendor, comments })
    }

    /// Serialises this header into an `OpusTags` packet.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&OPUS_TAGS_MAGIC);
        out.extend_from_slice(&(self.vendor.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.vendor);
        out.extend_from_slice(&(self.comments.len() as u32).to_le_bytes());
        for c in &self.comments {
            out.extend_from_slice(&(c.len() as u32).to_le_bytes());
            out.extend_from_slice(c);
        }
        out
    }

    /// The comments as (lossily decoded) strings.
    pub fn comments_lossy(&self) -> impl Iterator<Item = String> + '_ {
        self.comments.iter().map(|c| String::from_utf8_lossy(c).into_owned())
    }

    /// Looks up the value of `tag` (case-insensitive, per Vorbis-comment
    /// convention), lossily decoded.
    #[must_use]
    pub fn get(&self, tag: &str) -> Option<String> {
        self.comments.iter().find_map(|c| {
            let s = String::from_utf8_lossy(c);
            let (name, value) = s.split_once('=')?;
            name.eq_ignore_ascii_case(tag).then(|| String::from(value))
        })
    }

    /// Appends a `TAG=value` comment.
    pub fn push(&mut self, tag: &str, value: &str) {
        let mut c = Vec::with_capacity(tag.len() + 1 + value.len());
        c.extend_from_slice(tag.as_bytes());
        c.push(b'=');
        c.extend_from_slice(value.as_bytes());
        self.comments.push(c);
    }
}

/// Reads a little-endian u32 from the front of `rest`.
fn read_u32(rest: &mut &[u8]) -> Result<u32, OggOpusError> {
    let (head, tail) = rest.split_at_checked(4).ok_or(OggOpusError::InvalidCommentHeader)?;
    *rest = tail;
    Ok(u32::from_le_bytes(head.try_into().expect("4-byte slice")))
}

/// Reads a u32-length-prefixed field from the front of `rest`.
fn read_len_prefixed<'a>(rest: &mut &'a [u8]) -> Result<&'a [u8], OggOpusError> {
    let len = read_u32(rest)? as usize;
    let (head, tail) = rest.split_at_checked(len).ok_or(OggOpusError::InvalidCommentHeader)?;
    *rest = tail;
    Ok(head)
}

/// One audio data packet from an Ogg Opus stream, with its resolved timing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioPacket {
    /// The raw Opus packet (RFC 6716 framing; parse with
    /// [`crate::packet::Packet::parse`]).
    pub data: Vec<u8>,
    /// Granule position at the *end* of this packet: the count of 48 kHz
    /// samples decoded so far including this packet, before pre-skip
    /// adjustment. Derived from page granules and per-packet TOC durations.
    pub granule_position: u64,
    /// This packet completed on the stream's final page.
    pub eos: bool,
}

/// A reader for one Ogg Opus logical bitstream held in memory.
///
/// Locates the first logical bitstream whose BOS packet is an `OpusHead`,
/// parses both mandatory headers, and iterates audio packets with granule
/// positions resolved per packet (working forwards from page granules using
/// each packet's TOC duration, RFC 7845 §4).
#[derive(Debug, Clone)]
pub struct OggOpusReader<'a> {
    head: OpusHead,
    tags: OpusTags,
    serial: u32,
    data: &'a [u8],
    packets: PacketReader<'a>,
    /// Packets with resolved granule positions awaiting delivery.
    queue: VecDeque<AudioPacket>,
    /// Granule position of the most recently resolved packet, used as a
    /// forward fallback for a truncated tail with no final page anchor.
    last_position: u64,
}

impl<'a> OggOpusReader<'a> {
    /// Parses the headers of the first Opus logical bitstream in `data`.
    ///
    /// # Errors
    ///
    /// [`OggOpusError::NoOpusStream`] when no BOS page carries an `OpusHead`
    /// packet; header errors are propagated from [`OpusHead::parse`] and
    /// [`OpusTags::parse`].
    pub fn new(data: &'a [u8]) -> Result<Self, OggOpusError> {
        // Find the Opus BOS page (grouped streams put all BOS pages first,
        // but scan the whole stream to also survive mild corruption).
        let serial = PageReader::new(data)
            .filter(|p| p.bos)
            .find(|p| p.body.starts_with(&OPUS_HEAD_MAGIC))
            .map(|p| p.serial)
            .ok_or(OggOpusError::NoOpusStream)?;

        let mut packets = PacketReader::new(data, serial);
        let head_pkt = packets.next().ok_or(OggOpusError::NoOpusStream)?;
        let head = OpusHead::parse(&head_pkt.data)?;
        let tags_pkt = packets.next().ok_or(OggOpusError::InvalidCommentHeader)?;
        let tags = OpusTags::parse(&tags_pkt.data)?;

        Ok(OggOpusReader {
            head,
            tags,
            serial,
            data,
            packets,
            queue: VecDeque::new(),
            last_position: 0,
        })
    }

    /// The identification header.
    #[must_use]
    pub const fn head(&self) -> &OpusHead {
        &self.head
    }

    /// The comment header.
    #[must_use]
    pub const fn tags(&self) -> &OpusTags {
        &self.tags
    }

    /// The serial number of the Opus logical bitstream.
    #[must_use]
    pub const fn serial(&self) -> u32 {
        self.serial
    }

    /// Total PCM duration in 48 kHz samples, after pre-skip removal: the
    /// final page granule minus the pre-skip.
    ///
    /// Scans for the last page of the logical stream; `None` when the stream
    /// has no audio page with a granule position.
    #[must_use]
    pub fn pcm_duration_48k(&self) -> Option<u64> {
        let last = PageReader::new(self.data)
            .filter(|p| p.serial == self.serial && p.granule_position != NO_GRANULE)
            .map(|p| p.granule_position)
            .last()?;
        Some(last.saturating_sub(u64::from(self.head.pre_skip)))
    }

    /// Advances to the next audio data packet.
    ///
    /// Granule positions are resolved per packet by working *backward* from
    /// each page's authoritative granule using the TOC duration of every
    /// packet in the group (RFC 7845 §4). This automatically accounts for the
    /// initial pre-skip offset (§4.5) and for end trimming on the final page
    /// (§4.4): the last packet's position is the page granule even when that
    /// is less than a full packet's worth of new samples.
    #[expect(
        clippy::should_implement_trait,
        reason = "fallible, stateful iteration; a plain method reads better than a fused Iterator"
    )]
    pub fn next(&mut self) -> Option<AudioPacket> {
        loop {
            if let Some(pkt) = self.queue.pop_front() {
                self.last_position = pkt.granule_position;
                return Some(pkt);
            }

            // Gather packets up to (and including) the next page anchor.
            let mut group: alloc::vec::Vec<OggPacket> = alloc::vec::Vec::new();
            for pkt in self.packets.by_ref() {
                let anchored = pkt.granule_position != NO_GRANULE;
                group.push(pkt);
                if anchored {
                    break;
                }
            }
            let anchor = group.last()?.granule_position;
            if anchor != NO_GRANULE {
                // Walk backward from the anchor assigning positions.
                let mut pos = anchor;
                let mut positions: alloc::vec::Vec<u64> = group
                    .iter()
                    .rev()
                    .map(|p| {
                        let this = pos;
                        pos = pos.saturating_sub(packet_samples_48k(&p.data).unwrap_or(0));
                        this
                    })
                    .collect();
                positions.reverse();
                for (pkt, granule_position) in group.into_iter().zip(positions) {
                    self.queue.push_back(AudioPacket {
                        data: pkt.data,
                        granule_position,
                        eos: pkt.eos,
                    });
                }
            } else {
                // Truncated stream with no final anchor: best-effort forward
                // continuation from the last resolved position.
                let mut pos = self.last_position;
                for pkt in group {
                    pos += packet_samples_48k(&pkt.data).unwrap_or(0);
                    self.queue.push_back(AudioPacket {
                        data: pkt.data,
                        granule_position: pos,
                        eos: pkt.eos,
                    });
                }
            }
        }
    }
}

/// The number of 48 kHz samples an Opus packet decodes to, from its TOC.
fn packet_samples_48k(packet: &[u8]) -> Option<u64> {
    let toc = Toc::new(*packet.first()?);
    let per_frame = toc.frame_size().samples_per_channel_48k() as u64;
    let frames = match toc.frame_count_code() {
        0 => 1,
        1 | 2 => 2,
        _ => u64::from(*packet.get(1)? & 0x3F),
    };
    Some(per_frame * frames)
}

/// Writes a single-logical-stream Ogg Opus file (RFC 7845 §3 packet
/// organization): ID header alone on the BOS page, comment header completing
/// its own page, then audio packets with granule positions accumulated from
/// their TOC durations.
#[derive(Debug, Clone)]
pub struct OggOpusWriter {
    out: Vec<u8>,
    pages: PageWriter,
    granule: u64,
    finished: bool,
}

impl OggOpusWriter {
    /// Begins a stream: writes the ID and comment header pages.
    #[must_use]
    pub fn new(head: &OpusHead, tags: &OpusTags, serial: u32) -> Self {
        let mut out = Vec::new();
        let mut pages = PageWriter::new(serial);
        // Granule position must be zero on both header pages (RFC 7845 §4).
        pages.push(&mut out, &head.to_bytes(), 0, false);
        pages.flush(&mut out);
        pages.push(&mut out, &tags.to_bytes(), 0, false);
        pages.flush(&mut out);
        OggOpusWriter {
            out,
            pages,
            granule: u64::from(head.pre_skip),
            finished: false,
        }
    }

    /// Appends one Opus packet. Granule positions advance by the packet's TOC
    /// duration; the first packet's position additionally covers the
    /// pre-skip, per RFC 7845 §4.5.
    ///
    /// `last` marks the final packet and closes the stream with the EOS flag.
    pub fn push(&mut self, packet: &[u8], last: bool) {
        debug_assert!(!self.finished, "stream already finished");
        self.granule += packet_samples_48k(packet).unwrap_or(0);
        self.pages.push(&mut self.out, packet, self.granule, last);
        if last {
            self.finished = true;
        }
    }

    /// Finishes the stream (emitting an empty EOS page if no packet was
    /// marked `last`) and returns the serialized file.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if !self.finished {
            self.pages.push(&mut self.out, &[], self.granule, true);
        }
        self.out
    }
}

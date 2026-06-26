//! The Ogg container (RFC 3533) and the Ogg encapsulation of Opus
//! (RFC 7845).
//!
//! Two layers:
//!
//! - [`page`]-level: [`Page`]/[`PageReader`] parse a physical Ogg bitstream with CRC verification and capture-pattern
//!   resynchronization; [`PacketReader`] reassembles the packets of one logical bitstream (including packets spanning
//!   pages); [`PageWriter`] does the reverse. This layer is codec-agnostic - it will serve Vorbis and FLAC-in-Ogg as
//!   well as Opus.
//! - Opus mapping: [`OpusHead`]/[`OpusTags`] header packets, [`OggOpusReader`] (header parsing, per-packet granule
//!   reconstruction, duration) and [`OggOpusWriter`] (conformant page layout: ID header alone on the BOS page, comment
//!   header finishing its page, granule positions accumulated from packet TOC durations).
//!
//! Both operate on in-memory byte slices, keeping the module `no_std` +
//! `alloc`; std I/O adapters belong to higher-level crates.

mod crc;
mod opus;
mod page;

pub use opus::{AudioPacket, ChannelMapping, OggOpusError, OggOpusReader, OggOpusWriter, OpusHead, OpusTags};
pub use page::{CAPTURE_PATTERN, NO_GRANULE, OggError, OggPacket, PacketReader, Page, PageReader, PageWriter};

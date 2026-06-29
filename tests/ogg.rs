//! Ogg container tests: page round-trips, reassembly edge cases mandated by
//! RFC 3533/7845, and interop against a real libopus/ffmpeg-produced file.

use opus_rs::ogg::{
    NO_GRANULE, OggOpusReader, OggOpusWriter, OpusHead, OpusTags, PacketReader, Page, PageReader, PageWriter,
};

/// A real Ogg Opus file produced by ffmpeg/libopus: 0.5 s of 440 Hz mono.
const FFMPEG_FILE: &[u8] = include_bytes!("fixtures/sine_mono.opus");

// ---- Interop with a real-world file -----------------------------------------

#[test]
fn parses_ffmpeg_file_headers() {
    let reader = OggOpusReader::new(FFMPEG_FILE).expect("valid Ogg Opus");

    let head = reader.head();
    assert_eq!(head.version, 1);
    assert_eq!(head.channel_count, 1);
    assert_eq!(head.input_sample_rate, 48_000);
    assert_eq!(head.output_gain_q8, 0);
    // libopus mono delay: 312 samples at 48 kHz (6.5 ms).
    assert_eq!(head.pre_skip, 312);

    let vendor = String::from_utf8_lossy(&reader.tags().vendor).into_owned();
    assert!(vendor.contains("Lavf"), "ffmpeg vendor string, got: {vendor}");
    assert!(reader.tags().get("encoder").is_some(), "ffmpeg writes an encoder tag");
}

#[test]
fn every_page_of_ffmpeg_file_passes_crc() {
    let pages: Vec<_> = PageReader::new(FFMPEG_FILE).collect();
    assert!(pages.len() >= 3, "ID page + tags page + audio pages");
    assert!(pages[0].bos);
    assert!(pages.last().expect("non-empty").eos);
    // PageReader verifies CRCs; reaching the byte count proves full coverage.
    let mut reader = PageReader::new(FFMPEG_FILE);
    while reader.next().is_some() {}
    assert_eq!(reader.position(), FFMPEG_FILE.len(), "every byte accounted for");
}

#[test]
fn ffmpeg_file_duration_and_packets() {
    let mut reader = OggOpusReader::new(FFMPEG_FILE).expect("valid");

    // 0.5 s at 48 kHz after pre-skip removal.
    assert_eq!(reader.pcm_duration_48k(), Some(24_000));

    // Default libopus frame size is 20 ms: 26 packets cover the pre-skip plus
    // 0.5 s of audio (26*960 = 24 960 decodable; granule ends at 24 312 =
    // 312 pre-skip + 24 000 audio, so 648 samples are trimmed). The fixture
    // puts all packets on a single EOS page, and the backward granule walk
    // resolves the first packet to position 312 - the trim and pre-skip
    // overlap at the very front, exactly as libopus's own demuxer computes.
    // Every packet must parse under the RFC 6716 framing rules.
    let mut granules = Vec::new();
    let mut saw_eos = false;
    while let Some(pkt) = reader.next() {
        let parsed = opus_rs::Packet::parse(&pkt.data).expect("valid Opus packet");
        assert_eq!(parsed.toc().channels(), 1);
        granules.push(pkt.granule_position);
        saw_eos = pkt.eos;
    }
    assert_eq!(granules.len(), 26, "26 x 20 ms packets cover 0.5 s plus pre-skip");
    assert!(saw_eos, "final packet flagged EOS");
    let expected: Vec<u64> = (0..26).map(|i| 312 + i * 960).collect();
    assert_eq!(granules, expected, "positions resolved backward from the page anchor");
}

// ---- Writer/reader round-trip ------------------------------------------------

/// A valid one-byte Opus packet body: config 1 (SILK NB 20 ms), mono, code 0,
/// followed by `len` payload bytes.
fn fake_packet(len: usize, fill: u8) -> Vec<u8> {
    let mut p = vec![0x08u8];
    p.extend(std::iter::repeat_n(fill, len));
    p
}

#[test]
fn ogg_opus_write_read_round_trip() {
    let head = OpusHead::family0(1, 312, 48_000);
    let mut tags = OpusTags {
        vendor: b"opus_rs test".to_vec(),
        comments: Vec::new(),
    };
    tags.push("TITLE", "Round Trip");
    tags.push("artist", "opus_rs");

    let packets: Vec<Vec<u8>> = (0..7).map(|i| fake_packet(40 + i * 13, i as u8)).collect();

    let mut writer = OggOpusWriter::new(&head, &tags, 0x00DD_BA11);
    for (i, p) in packets.iter().enumerate() {
        writer.push(p, i == packets.len() - 1);
    }
    let file = writer.finish();

    let mut reader = OggOpusReader::new(&file).expect("readable");
    assert_eq!(reader.head(), &head);
    assert_eq!(reader.tags().get("title").as_deref(), Some("Round Trip"));
    assert_eq!(reader.tags().get("ARTIST").as_deref(), Some("opus_rs"));

    let mut got = Vec::new();
    let mut granules = Vec::new();
    while let Some(pkt) = reader.next() {
        got.push(pkt.data);
        granules.push(pkt.granule_position);
    }
    assert_eq!(got, packets, "packets survive byte-identically");

    // Each 20 ms packet advances the granule by 960, starting above pre-skip.
    let expected: Vec<u64> = (1..=7).map(|i| 312 + i * 960).collect();
    assert_eq!(granules, expected);
    assert_eq!(reader.pcm_duration_48k(), Some(7 * 960));
}

#[test]
fn header_pages_are_laid_out_per_rfc7845() {
    let head = OpusHead::family0(2, 0, 44_100);
    let tags = OpusTags::default();
    let mut writer = OggOpusWriter::new(&head, &tags, 7);
    writer.push(&fake_packet(10, 0xAB), true);
    let file = writer.finish();

    let pages: Vec<_> = PageReader::new(&file).collect();
    // ID header alone on the BOS page; comment header finishing its own page;
    // both with granule position zero (RFC 7845 §3, §4).
    assert!(pages[0].bos);
    assert_eq!(pages[0].granule_position, 0);
    assert_eq!(pages[0].segments.len(), 1, "ID header alone on page 0");
    assert_eq!(pages[1].granule_position, 0);
    assert!(!pages[1].bos);
    assert!(pages.last().expect("pages").eos);
    // Sequence numbers count up from zero.
    for (i, p) in pages.iter().enumerate() {
        assert_eq!(p.sequence, i as u32);
        assert_eq!(p.serial, 7);
    }
}

// ---- Page/packet layer edge cases -------------------------------------------

#[test]
fn packet_spanning_multiple_pages_reassembles() {
    // 200 000 bytes needs four pages (65 025 max body per page).
    let big: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    let mut out = Vec::new();
    let mut writer = PageWriter::new(42);
    writer.push(&mut out, &big, 1234, false);
    writer.flush(&mut out);

    let pages: Vec<_> = PageReader::new(&out).collect();
    assert!(pages.len() >= 4, "got {} pages", pages.len());
    assert!(pages[1].continued, "later pages flagged continued");
    assert_eq!(
        pages[..pages.len() - 1]
            .iter()
            .filter(|p| p.granule_position == NO_GRANULE)
            .count(),
        pages.len() - 1,
        "no packet completes on the spanned pages"
    );

    let packets: Vec<_> = PacketReader::new(&out, 42).collect();
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].data, big);
    assert_eq!(packets[0].granule_position, 1234);
}

#[test]
fn packet_of_exactly_255_bytes_terminates_with_zero_lacing() {
    let exact = vec![0x5Au8; 255];
    let mut out = Vec::new();
    let mut writer = PageWriter::new(1);
    writer.push(&mut out, &exact, 99, false);
    writer.push(&mut out, &[], 100, false); // zero-length packet too
    writer.flush(&mut out);

    let (page, _) = Page::parse(&out).expect("valid page");
    assert_eq!(page.segments, &[255, 0, 0], "255-lacing + 0 terminator + nil packet");

    let packets: Vec<_> = PacketReader::new(&out, 1).collect();
    assert_eq!(packets.len(), 2);
    assert_eq!(packets[0].data.len(), 255);
    assert_eq!(packets[1].data.len(), 0);
}

#[test]
fn corrupt_page_is_skipped_and_reader_resyncs() {
    let mut out = Vec::new();
    let mut writer = PageWriter::new(5);
    writer.push(&mut out, b"first", 1, false);
    writer.flush(&mut out);
    let second_page_at = out.len();
    writer.push(&mut out, b"second", 2, false);
    writer.flush(&mut out);
    writer.push(&mut out, b"third", 3, false);
    writer.flush(&mut out);

    // Corrupt one body byte of the second page: its CRC must now fail.
    let mut corrupted = out.clone();
    *corrupted.last_mut().expect("non-empty") ^= 0xFF; // corrupt LAST page instead
    let pages: Vec<_> = PageReader::new(&corrupted).collect();
    assert_eq!(pages.len(), 2, "corrupted final page dropped");

    let mut mid_corrupt = out;
    mid_corrupt[second_page_at + 30] ^= 0xFF;
    let packets: Vec<_> = PacketReader::new(&mid_corrupt, 5).collect();
    let datas: Vec<&[u8]> = packets.iter().map(|p| p.data.as_slice()).collect();
    assert_eq!(
        datas,
        [b"first".as_slice(), b"third".as_slice()],
        "middle page skipped, resynced"
    );
}

#[test]
fn garbage_prefix_is_skipped() {
    let mut data = vec![0xDEu8; 1000];
    let mut writer = PageWriter::new(9);
    writer.push(&mut data, b"payload", 7, true);

    let pages: Vec<_> = PageReader::new(&data).collect();
    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].body, b"payload");
}

#[test]
fn hostile_tags_header_is_rejected_without_allocation_blowup() {
    // Claims a 4 GiB vendor string in a 30-byte packet.
    let mut evil = b"OpusTags".to_vec();
    evil.extend_from_slice(&u32::MAX.to_le_bytes());
    evil.extend_from_slice(&[0u8; 18]);
    assert!(OpusTags::parse(&evil).is_err());

    // Claims more comments than the packet could possibly hold.
    let mut evil = b"OpusTags".to_vec();
    evil.extend_from_slice(&0u32.to_le_bytes()); // empty vendor
    evil.extend_from_slice(&u32::MAX.to_le_bytes()); // comment count
    assert!(OpusTags::parse(&evil).is_err());
}

#[test]
fn opus_head_round_trips_all_families() {
    let family0 = OpusHead::family0(2, 3840, 44_100);
    assert_eq!(OpusHead::parse(&family0.to_bytes()).expect("valid"), family0);

    let surround = OpusHead {
        version: 1,
        channel_count: 6,
        pre_skip: 312,
        input_sample_rate: 48_000,
        output_gain_q8: -256, // -1 dB
        channel_mapping: opus_rs::ogg::ChannelMapping::Table {
            family: 1,
            stream_count: 4,
            coupled_count: 2,
            mapping: vec![0, 4, 1, 2, 3, 5],
        },
    };
    let parsed = OpusHead::parse(&surround.to_bytes()).expect("valid");
    assert_eq!(parsed, surround);
    assert_eq!(parsed.channel_mapping.stream_count(), 4);

    // Family 0 with more than 2 channels is invalid.
    let mut bad = OpusHead::family0(2, 0, 0).to_bytes();
    bad[9] = 3;
    assert!(OpusHead::parse(&bad).is_err());
}

/// End-to-end file decode: the ffmpeg/libopus-encoded 500 ms A440 fixture
/// through container parsing, packet decode, pre-skip, and end trimming.
///
/// The duration must be exact, every packet's final range must match the
/// values recorded from libopus decoding the same packets, and the audio
/// must fit a 440 Hz sine at the encode's own quality (~25 dB for this
/// low-bitrate hybrid fixture).
#[cfg(feature = "std")]
#[test]
fn decodes_a_real_ogg_opus_file_end_to_end() {
    let data = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sine_mono.opus")).expect("fixture");
    let (pcm, head) = opus_rs::decode_ogg_opus(&data).expect("decode");

    assert_eq!(head.channel_count, 1);
    assert_eq!(head.pre_skip, 312);
    assert_eq!(pcm.len(), 24_000, "500 ms at 48 kHz after pre-skip and trim");

    // Per-packet final ranges recorded from libopus 1.5.2 decoding these
    // exact packets (the PCM also matched at 123 dB SNR when recorded).
    const FINAL_RANGES: [u32; 26] = [
        234180096, 566826752, 743515392, 786692608, 16007633, 51959808, 18082048, 544261376, 60771840, 29612544,
        1676097792, 477368832, 234494208, 180129280, 35754240, 857939456, 146146560, 1461526, 132782592, 1759044352,
        446119424, 1346046976, 41955584, 104922624, 1739180032, 19610880,
    ];
    {
        use opus_rs::OpusDecoder;
        use opus_rs::ogg::OggOpusReader;
        let mut reader = OggOpusReader::new(&data).expect("container");
        let mut decoder = OpusDecoder::new(1);
        let mut i = 0;
        while let Some(pkt) = reader.next() {
            let _ = decoder.decode_packet(&pkt.data).expect("packet");
            assert_eq!(decoder.final_range(), FINAL_RANGES[i], "packet {i}");
            i += 1;
        }
        assert_eq!(i, 26);
    }

    // Project the interior onto a 440 Hz sine and measure the residual.
    let seg = &pcm[4_000..20_000];
    let w = 2.0 * core::f64::consts::PI * 440.0 / 48_000.0;
    let (mut cs, mut sn) = (0.0f64, 0.0f64);
    for (i, &x) in seg.iter().enumerate() {
        cs += f64::from(x) * (w * i as f64).cos();
        sn += f64::from(x) * (w * i as f64).sin();
    }
    let m = seg.len() as f64;
    let (a, b) = (2.0 * cs / m, 2.0 * sn / m);
    let amp = (a * a + b * b).sqrt();
    assert!((0.11..0.14).contains(&amp), "sine amplitude {amp}");

    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for (i, &x) in seg.iter().enumerate() {
        let fit = a * (w * i as f64).cos() + b * (w * i as f64).sin();
        sig += fit * fit;
        noise += (f64::from(x) - fit) * (f64::from(x) - fit);
    }
    let snr_db = 10.0 * (sig / noise).log10();
    assert!(snr_db > 20.0, "440 Hz fit SNR {snr_db:.1} dB");
}

/// End-to-end surround decode: a 5.1 (channel mapping family 1) file of
/// six per-channel sines through the multistream decoder. Each output
/// channel must be dominated by one of the six encoded frequencies, all
/// six must appear, and the duration must be exact. (Cross-checked
/// against libopus' multistream decoder at 115.8 dB when recorded;
/// `examples/surround_check.rs`.)
#[cfg(feature = "std")]
#[test]
fn decodes_a_surround_ogg_opus_file() {
    let data = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/surround_51.opus")).expect("fixture");
    let (pcm, head) = opus_rs::decode_ogg_opus(&data).expect("decode");
    assert_eq!(head.channel_count, 6);
    assert_eq!(pcm.len(), 48_000 * 6, "one second of 5.1 at 48 kHz");

    let mut found = Vec::new();
    for ch in 0..6 {
        // Dominant frequency by projection over the interior.
        let seg: Vec<f32> = pcm[8_000 * 6..40_000 * 6].iter().skip(ch).step_by(6).copied().collect();
        let (mut best_f, mut best_p) = (0u32, 0.0f64);
        for f in [60u32, 300, 500, 700, 1100, 1300] {
            let w = 2.0 * core::f64::consts::PI * f64::from(f) / 48_000.0;
            let (mut c, mut s) = (0.0f64, 0.0f64);
            for (i, &x) in seg.iter().enumerate() {
                c += f64::from(x) * (w * i as f64).cos();
                s += f64::from(x) * (w * i as f64).sin();
            }
            let p = c * c + s * s;
            if p > best_p {
                best_p = p;
                best_f = f;
            }
        }
        found.push(best_f);
    }
    found.sort_unstable();
    assert_eq!(found, [60, 300, 500, 700, 1100, 1300], "per-channel sines");
}

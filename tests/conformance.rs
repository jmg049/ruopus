//! Conformance harness against the official Opus decoder test vectors
//! (RFC 8251 update of the RFC 6716 vectors).
//!
//! The vectors are ~121 MB and are not committed; fetch them with
//! `tools/fetch-testvectors.sh`. Every test in this file skips (passing,
//! with a note) when `tests/vectors/` is absent, so the suite stays green in
//! a fresh checkout.
//!
//! Each `testvectorNN.bit` file is in `opus_demo` framing: per packet, a
//! 4-byte big-endian payload length, a 4-byte big-endian "encoder final
//! range" word (the range coder's `rng` after encoding - the bit-exactness
//! oracle the decoder must reproduce), then the payload. `testvectorNN.dec`
//! is the reference decoder output: interleaved stereo s16le at 48 kHz
//! (`testvectorNNm.dec` is the mono-downmix decode of the same stream).
//!
//! What runs today: the packet layer is validated against every packet of
//! all twelve vectors. As the SILK/CELT decoders land, this file grows the
//! full decode comparison: final-range equality per packet, then PCM quality
//! scoring against the `.dec` references (the `opus_compare` criterion).

use std::path::{Path, PathBuf};

use opus_native::{Mode, Packet};

/// One packet from an `opus_demo` bitstream file.
struct DemoPacket {
    data: Vec<u8>,
    /// The encoder's final range-coder `rng` value; a conformant decoder's
    /// range decoder finishes each packet with this exact value.
    #[expect(dead_code, reason = "the oracle for the decoder once SILK/CELT land")]
    final_range: u32,
}

/// Parses an `opus_demo`-format bitstream file.
fn parse_bit_file(data: &[u8]) -> Vec<DemoPacket> {
    let mut packets = Vec::new();
    let mut off = 0usize;
    while off + 8 <= data.len() {
        let len = u32::from_be_bytes(data[off..off + 4].try_into().expect("4 bytes")) as usize;
        let final_range = u32::from_be_bytes(data[off + 4..off + 8].try_into().expect("4 bytes"));
        off += 8;
        assert!(off + len <= data.len(), "truncated .bit file");
        packets.push(DemoPacket {
            data: data[off..off + len].to_vec(),
            final_range,
        });
        off += len;
    }
    assert_eq!(off, data.len(), "trailing garbage in .bit file");
    packets
}

fn vectors_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors");
    if dir.join("testvector01.bit").exists() {
        Some(dir)
    } else {
        eprintln!("skipping: official test vectors not present; run tools/fetch-testvectors.sh");
        None
    }
}

/// `(name, packet count)` for all twelve vectors - counts pinned so a parser
/// regression that silently drops packets cannot pass.
const VECTORS: [(&str, usize); 12] = [
    ("testvector01", 2147), // CELT-only SWB/FB
    ("testvector02", 1185), // SILK-only NB
    ("testvector03", 998),  // SILK-only MB
    ("testvector04", 1265), // SILK-only WB
    ("testvector05", 2037), // Hybrid SWB
    ("testvector06", 1876), // Hybrid FB
    ("testvector07", 4186), // CELT-only, every bandwidth
    ("testvector08", 1247), // mode switching
    ("testvector09", 1337), // mode switching
    ("testvector10", 1912), // Hybrid/CELT FB
    ("testvector11", 553),  // CELT FB code 3
    ("testvector12", 1332), // SILK code 3, all bandwidths
];

#[test]
fn every_official_vector_packet_parses() {
    let Some(dir) = vectors_dir() else { return };

    let mut total = 0usize;
    for (name, expected_count) in VECTORS {
        let bits = std::fs::read(dir.join(format!("{name}.bit"))).expect("read .bit");
        let packets = parse_bit_file(&bits);
        assert_eq!(packets.len(), expected_count, "{name}: packet count");

        for (i, pkt) in packets.iter().enumerate() {
            let parsed = Packet::parse(&pkt.data)
                .unwrap_or_else(|e| panic!("{name} packet {i}: rejected valid packet: {e}"));
            // Frame data must be non-empty for these vectors (no DTX), and
            // mode/bandwidth/frame-size must be resolvable for every config.
            assert!(!parsed.frames().is_empty(), "{name} packet {i}: no frames");
            let toc = parsed.toc();
            let _ = (toc.mode(), toc.bandwidth(), toc.frame_size());
        }
        total += packets.len();
    }
    assert_eq!(total, 20_075, "total packets across the official suite");
}

#[test]
fn toc_durations_sum_to_reference_pcm_length() {
    let Some(dir) = vectors_dir() else { return };

    for (name, _) in VECTORS {
        let bits = std::fs::read(dir.join(format!("{name}.bit"))).expect("read .bit");
        let samples_48k: u64 = parse_bit_file(&bits)
            .iter()
            .map(|pkt| {
                let parsed = Packet::parse(&pkt.data).expect("valid");
                parsed.frames().len() as u64
                    * parsed.toc().frame_size().samples_per_channel_48k() as u64
            })
            .sum();

        // The reference decode is interleaved stereo s16le at 48 kHz.
        let dec_bytes = std::fs::metadata(dir.join(format!("{name}.dec"))).expect("dec").len();
        let dec_frames = dec_bytes / 4;
        assert_eq!(
            samples_48k, dec_frames,
            "{name}: TOC duration sum vs reference PCM frames"
        );
    }
}

#[test]
fn vector_suite_exercises_every_configuration_class() {
    let Some(dir) = vectors_dir() else { return };

    let mut configs_seen = [false; 32];
    let mut modes = [false; 3];
    for (name, _) in VECTORS {
        let bits = std::fs::read(dir.join(format!("{name}.bit"))).expect("read .bit");
        for pkt in parse_bit_file(&bits) {
            let toc = Packet::parse(&pkt.data).expect("valid").toc();
            configs_seen[usize::from(toc.config())] = true;
            modes[match toc.mode() {
                Mode::SilkOnly => 0,
                Mode::Hybrid => 1,
                Mode::CeltOnly => 2,
            }] = true;
        }
    }
    assert_eq!(configs_seen, [true; 32], "all 32 TOC configurations appear in the suite");
    assert_eq!(modes, [true; 3], "all three modes appear in the suite");
}

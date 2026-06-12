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
    #[cfg_attr(not(feature = "std"), allow(dead_code, reason = "the CELT decode test needs std"))]
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
            let parsed =
                Packet::parse(&pkt.data).unwrap_or_else(|e| panic!("{name} packet {i}: rejected valid packet: {e}"));
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
                parsed.frames().len() as u64 * parsed.toc().frame_size().samples_per_channel_48k() as u64
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
    assert_eq!(
        configs_seen, [true; 32],
        "all 32 TOC configurations appear in the suite"
    );
    assert_eq!(modes, [true; 3], "all three modes appear in the suite");
}

/// The CELT end band per Opus bandwidth (`opus_decoder.c` endband mapping).
#[cfg(feature = "std")]
fn celt_end_band(bw: opus_native::Bandwidth) -> usize {
    use opus_native::Bandwidth;
    match bw {
        Bandwidth::NarrowBand => 13,
        Bandwidth::MediumBand | Bandwidth::WideBand => 17,
        Bandwidth::SuperWideBand => 19,
        Bandwidth::FullBand => 21,
    }
}

/// Decodes every CELT-only vector packet and checks two oracles per vector:
///
/// 1. **Final range** (per packet): the decoder's range-coder `rng` after
///    the last frame must equal the encoder's recorded value - it matches
///    only if every entropy-coded symbol was consumed exactly as produced.
/// 2. **PCM output** (whole vector): the synthesized audio against the
///    reference decoder's `.dec` output. The synthesis chain (denormalise,
///    inverse MDCT, post-filter, de-emphasis) never touches the range
///    coder, so only this catches bugs there. The reference is a float
///    build like ours, so the SNR demanded here is far above the official
///    `opus_compare` bar; FFT and float-ordering differences are all that
///    remain.
#[cfg(feature = "std")]
#[test]
fn celt_only_vectors_final_range_is_bit_exact_and_pcm_matches() {
    use opus_native::RangeDecoder;
    use opus_native::celt::decoder::CeltDecoder;

    let Some(dir) = vectors_dir() else { return };

    // The CELT-only vectors: testvector01 (SWB/FB), 07 (every bandwidth),
    // 11 (FB code 3).
    for name in ["testvector01", "testvector07", "testvector11"] {
        let bits = std::fs::read(dir.join(format!("{name}.bit"))).expect("read .bit");
        let packets = parse_bit_file(&bits);
        let reference = std::fs::read(dir.join(format!("{name}.dec"))).expect("read .dec");

        let mut decoder = CeltDecoder::new(2);
        let mut pcm = Vec::new();
        for (pi, pkt) in packets.iter().enumerate() {
            let parsed = Packet::parse(&pkt.data).expect("valid");
            let toc = parsed.toc();
            assert_eq!(toc.mode(), Mode::CeltOnly, "{name} packet {pi}");
            let frame_size = toc.frame_size().samples_per_channel_48k();
            let channels = usize::from(toc.channels());
            let end = celt_end_band(toc.bandwidth());

            let mut final_range = 0u32;
            for frame in parsed.frames() {
                let mut dec = RangeDecoder::new(frame);
                pcm.extend(decoder.decode_frame(&mut dec, frame.len(), frame_size, channels, 0, end));
                final_range = dec.range_size();
            }
            assert_eq!(
                final_range, pkt.final_range,
                "{name} packet {pi}: final range mismatch (decoder desynchronized)"
            );
        }

        // The reference decode is interleaved stereo s16le at 48 kHz,
        // converted from float exactly as `opus_demo` does (scale 32768,
        // saturate, round to nearest with ties to even).
        assert_eq!(pcm.len(), reference.len() / 2, "{name}: PCM length");
        let mut signal = 0.0f64;
        let mut noise = 0.0f64;
        for (ours, theirs) in pcm.iter().zip(reference.chunks_exact(2)) {
            let theirs = i16::from_le_bytes([theirs[0], theirs[1]]);
            let ours = (ours * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16;
            signal += f64::from(theirs) * f64::from(theirs);
            noise += f64::from(ours - theirs) * f64::from(ours - theirs);
        }
        let snr_db = 10.0 * (signal / noise.max(f64::MIN_POSITIVE)).log10();
        eprintln!("{name}: {} packets bit-exact, PCM SNR {snr_db:.1} dB", packets.len());
        assert!(snr_db > 45.0, "{name}: PCM SNR {snr_db:.1} dB vs reference decode");
    }
}

/// Decodes every packet of the pure SILK vectors through the SILK decoder
/// and checks the same two oracles as the CELT test: per-packet final
/// range and PCM quality against the reference decode. (testvector12 also
/// carries SILK data but includes hybrid packets, so it joins the suite
/// with the Opus-level decoder.)
#[cfg(feature = "std")]
#[test]
fn silk_only_vectors_final_range_is_bit_exact_and_pcm_matches() {
    use opus_native::RangeDecoder;
    use opus_native::silk::api::{DecControl, SilkDecoder};

    let Some(dir) = vectors_dir() else { return };

    for name in ["testvector02", "testvector03", "testvector04"] {
        let bits = std::fs::read(dir.join(format!("{name}.bit"))).expect("read .bit");
        let packets = parse_bit_file(&bits);
        let reference = std::fs::read(dir.join(format!("{name}.dec"))).expect("read .dec");

        let mut decoder = SilkDecoder::new();
        let mut pcm: Vec<i16> = Vec::new();
        for (pi, pkt) in packets.iter().enumerate() {
            let parsed = Packet::parse(&pkt.data).expect("valid");
            let toc = parsed.toc();
            assert_eq!(toc.mode(), Mode::SilkOnly, "{name} packet {pi}");
            let frame_ms = match toc.frame_size().samples_per_channel_48k() {
                480 => 10,
                960 => 20,
                1920 => 40,
                2880 => 60,
                other => panic!("{name} packet {pi}: SILK frame size {other}"),
            };
            let ctl = DecControl {
                channels_internal: usize::from(toc.channels()),
                channels_api: 2,
                internal_sample_rate: match toc.bandwidth() {
                    opus_native::Bandwidth::NarrowBand => 8000,
                    opus_native::Bandwidth::MediumBand => 12000,
                    _ => 16000,
                },
                api_sample_rate: 48000,
                payload_size_ms: frame_ms,
            };

            let mut final_range = 0u32;
            for frame in parsed.frames() {
                let mut dec = RangeDecoder::new(frame);
                let n_calls = frame_ms.div_ceil(20).max(1);
                for call in 0..n_calls {
                    decoder.decode(&mut dec, &ctl, call == 0, &mut pcm);
                }
                final_range = dec.range_size();
                // A SILK-only frame with at least 17 bits to spare carries
                // a redundant 5 ms CELT frame in its tail; the recorded
                // final range XORs both coders (opus_decoder.c).
                if dec.tell() + 17 <= 8 * frame.len() as u32 {
                    let _celt_to_silk = dec.decode_bit_logp(1);
                    let redundancy_bytes = frame.len() - ((dec.tell() as usize + 7) >> 3);
                    let tail = &frame[frame.len() - redundancy_bytes..];
                    let mut rdec = RangeDecoder::new(tail);
                    let mut celt = opus_native::celt::decoder::CeltDecoder::new(2);
                    let _ = celt.decode_frame(
                        &mut rdec,
                        redundancy_bytes,
                        240,
                        ctl.channels_internal,
                        0,
                        // The end band follows the packet bandwidth.
                        celt_end_band(toc.bandwidth()),
                    );
                    final_range = dec.range_size() ^ rdec.range_size();
                }
            }
            assert_eq!(
                final_range, pkt.final_range,
                "{name} packet {pi}: final range mismatch (decoder desynchronized)"
            );
        }

        assert_eq!(pcm.len(), reference.len() / 2, "{name}: PCM length");
        let mut signal = 0.0f64;
        let mut noise = 0.0f64;
        for (ours, theirs) in pcm.iter().zip(reference.chunks_exact(2)) {
            let theirs = i16::from_le_bytes([theirs[0], theirs[1]]);
            signal += f64::from(theirs) * f64::from(theirs);
            noise += f64::from(ours - theirs) * f64::from(ours - theirs);
        }
        let snr_db = 10.0 * (signal / noise.max(f64::MIN_POSITIVE)).log10();
        eprintln!("{name}: {} packets, PCM SNR {snr_db:.1} dB", packets.len());
        assert!(snr_db > 40.0, "{name}: PCM SNR {snr_db:.1} dB vs reference decode");
    }
}

/// Decodes every packet of all twelve vectors through the Opus-level
/// decoder (TOC dispatch, SILK, CELT, hybrid, redundancy) and checks the
/// final-range oracle per packet plus PCM quality per vector.
///
/// Mode-transition concealment is not yet ported (fades come from
/// silence), so vectors with mode switches allow a lower SNR; the entropy
/// stream is unaffected and final ranges must match exactly everywhere.
#[cfg(feature = "std")]
#[test]
fn all_vectors_decode_through_the_opus_decoder() {
    use opus_native::OpusDecoder;

    let Some(dir) = vectors_dir() else { return };

    for (name, _) in VECTORS {
        let bits = std::fs::read(dir.join(format!("{name}.bit"))).expect("read .bit");
        let packets = parse_bit_file(&bits);
        let reference = std::fs::read(dir.join(format!("{name}.dec"))).expect("read .dec");

        let mut decoder = OpusDecoder::new(2);
        let mut pcm: Vec<i16> = Vec::new();
        for (pi, pkt) in packets.iter().enumerate() {
            pcm.extend(decoder.decode_packet_i16(&pkt.data).expect("valid packet"));
            assert_eq!(
                decoder.final_range(),
                pkt.final_range,
                "{name} packet {pi}: final range mismatch"
            );
        }

        assert_eq!(pcm.len(), reference.len() / 2, "{name}: PCM length");
        let mut signal = 0.0f64;
        let mut noise = 0.0f64;
        for (ours, theirs) in pcm.iter().zip(reference.chunks_exact(2)) {
            let theirs = i16::from_le_bytes([theirs[0], theirs[1]]);
            signal += f64::from(theirs) * f64::from(theirs);
            noise += f64::from(ours - theirs) * f64::from(ours - theirs);
        }
        let snr_db = 10.0 * (signal / noise.max(f64::MIN_POSITIVE)).log10();
        eprintln!("{name}: {} packets, PCM SNR {snr_db:.1} dB", packets.len());
        // Far above the official opus_compare bar; measured floor is
        // ~82.5 dB (testvector09).
        assert!(snr_db > 75.0, "{name}: PCM SNR {snr_db:.1} dB vs reference decode");
    }
}

/// Decodes testvector01 with a 10% simulated loss pattern: concealment
/// must produce the right duration and finite, bounded audio. (Concealment
/// is not normative; its decisions were verified against libopus
/// separately - all pitch picks and mode choices matched.)
#[cfg(feature = "std")]
#[test]
fn concealment_survives_simulated_loss() {
    use opus_native::OpusDecoder;

    let Some(dir) = vectors_dir() else { return };
    let bits = std::fs::read(dir.join("testvector01.bit")).expect("read .bit");
    let packets = parse_bit_file(&bits);

    let mut decoder = OpusDecoder::new(2);
    let mut total = 0usize;
    for (idx, pkt) in packets.iter().enumerate() {
        let pcm = if idx % 10 == 7 {
            let parsed = Packet::parse(&pkt.data).expect("valid");
            let dur = parsed.frames().len() * parsed.toc().frame_size().samples_per_channel_48k();
            decoder.decode_lost(dur)
        } else {
            decoder.decode_packet(&pkt.data).expect("valid")
        };
        assert!(pcm.iter().all(|v| v.is_finite() && v.abs() < 2.0));
        total += pcm.len();
    }
    // Matches libopus' sample count for this loss pattern exactly
    // (verified against the reference in a differential run).
    assert_eq!(total, 2_830_080, "total stereo samples");
}

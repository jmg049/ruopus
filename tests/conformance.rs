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
    // (vector, expected total stereo samples - matching libopus' output
    // for the same loss pattern exactly, verified differentially.)
    for (name, want_total) in [
        ("testvector01", 2_830_080usize),
        ("testvector02", 2_402_880),
        ("testvector04", 2_556_480),
        ("testvector05", 2_608_320),
    ] {
        let bits = std::fs::read(dir.join(format!("{name}.bit"))).expect("read .bit");
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
        assert_eq!(total, want_total, "{name}: total stereo samples");
    }
}

/// Decoding at reduced API rates must produce the right duration and
/// bounded audio at every rate. (Correctness is verified differentially
/// against libopus: SILK and hybrid output is sample-exact at 8-16 kHz,
/// CELT at 24 kHz matches at 117 dB; `examples/rate_check.rs`.)
#[cfg(feature = "std")]
#[test]
fn decodes_at_every_api_rate() {
    use opus_native::OpusDecoder;

    let Some(dir) = vectors_dir() else { return };
    let bits = std::fs::read(dir.join("testvector02.bit")).expect("read .bit");
    let packets = parse_bit_file(&bits);

    for fs in [8_000u32, 12_000, 16_000, 24_000, 48_000] {
        let mut decoder = OpusDecoder::with_rate(fs, 2);
        let mut total = 0usize;
        for pkt in &packets {
            let pcm = decoder.decode_packet(&pkt.data).expect("valid");
            assert!(pcm.iter().all(|v| v.is_finite() && v.abs() < 2.0));
            total += pcm.len();
        }
        assert_eq!(total, 2 * (1_201_440 * fs as usize / 48_000), "duration at {fs} Hz");
    }
}

/// The official `opus_compare` quality metric (48 kHz stereo form):
/// windowed per-frequency energies, psychoacoustic masking from the
/// reference, two-frame averaging, and the log-spectral distortion pooling.
/// Returns (internal weighted error, quality percent); conformance is
/// quality ≥ 0.
#[cfg(feature = "std")]
#[allow(clippy::needless_range_loop, reason = "mirrors the reference triple loops")]
fn opus_compare_48k_stereo(x: &[f32], y: &[f32]) -> (f64, f64) {
    const NBANDS: usize = 21;
    const NFREQS: usize = 240;
    const BANDS: [usize; NBANDS + 1] = [
        0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 68, 80, 96, 120, 156, 200,
    ];
    const WIN: usize = 480;
    const STEP: usize = 120;
    const NCH: usize = 2;

    assert_eq!(x.len(), y.len());
    let nframes = (x.len() / NCH - WIN + STEP) / STEP;

    // Hann window and DFT twiddles.
    let window: Vec<f32> = (0..WIN)
        .map(|j| 0.5 - 0.5 * (2.0 * core::f32::consts::PI / (WIN - 1) as f32 * j as f32).cos())
        .collect();
    let c: Vec<f32> = (0..WIN)
        .map(|j| (2.0 * core::f32::consts::PI / WIN as f32 * j as f32).cos())
        .collect();
    let s: Vec<f32> = (0..WIN)
        .map(|j| (2.0 * core::f32::consts::PI / WIN as f32 * j as f32).sin())
        .collect();

    let band_energy = |input: &[f32], out_bands: bool| -> (Vec<f32>, Vec<f32>) {
        let mut bands = vec![0.0f32; if out_bands { nframes * NBANDS * NCH } else { 0 }];
        let mut ps = vec![0.0f32; nframes * NFREQS * NCH];
        let mut xw = vec![0.0f32; NCH * WIN];
        for xi in 0..nframes {
            for ci in 0..NCH {
                for xk in 0..WIN {
                    xw[ci * WIN + xk] = window[xk] * input[(xi * STEP + xk) * NCH + ci];
                }
            }
            let mut xj = 0usize;
            for bi in 0..NBANDS {
                let mut p = [0.0f32; 2];
                while xj < BANDS[bi + 1] {
                    for ci in 0..NCH {
                        let mut re = 0.0f32;
                        let mut im = 0.0f32;
                        let mut ti = 0usize;
                        for xk in 0..WIN {
                            re += c[ti] * xw[ci * WIN + xk];
                            im -= s[ti] * xw[ci * WIN + xk];
                            ti += xj;
                            if ti >= WIN {
                                ti -= WIN;
                            }
                        }
                        let pwr = re * re + im * im + 100_000.0;
                        ps[(xi * NFREQS + xj) * NCH + ci] = pwr;
                        p[ci] += pwr;
                    }
                    xj += 1;
                }
                if out_bands {
                    let width = (BANDS[bi + 1] - BANDS[bi]) as f32;
                    bands[(xi * NBANDS + bi) * NCH] = p[0] / width;
                    bands[(xi * NBANDS + bi) * NCH + 1] = p[1] / width;
                }
            }
        }
        (bands, ps)
    };

    let (mut xb, mut xps) = band_energy(x, true);
    let (_, mut yps) = band_energy(y, false);

    for xi in 0..nframes {
        // Frequency masking: 10 dB/Bark up, 15 dB/Bark down.
        for bi in 1..NBANDS {
            for ci in 0..NCH {
                xb[(xi * NBANDS + bi) * NCH + ci] += 0.1 * xb[(xi * NBANDS + bi - 1) * NCH + ci];
            }
        }
        for bi in (0..NBANDS - 1).rev() {
            for ci in 0..NCH {
                xb[(xi * NBANDS + bi) * NCH + ci] += 0.03 * xb[(xi * NBANDS + bi + 1) * NCH + ci];
            }
        }
        // Temporal masking: -3 dB / 2.5 ms.
        if xi > 0 {
            for bi in 0..NBANDS {
                for ci in 0..NCH {
                    xb[(xi * NBANDS + bi) * NCH + ci] += 0.5 * xb[((xi - 1) * NBANDS + bi) * NCH + ci];
                }
            }
        }
        // Cross-talk allowance.
        for bi in 0..NBANDS {
            let l = xb[(xi * NBANDS + bi) * NCH];
            let r = xb[(xi * NBANDS + bi) * NCH + 1];
            xb[(xi * NBANDS + bi) * NCH] += 0.01 * r;
            xb[(xi * NBANDS + bi) * NCH + 1] += 0.01 * l;
        }
        // Apply masking to both spectra.
        for bi in 0..NBANDS {
            for xj in BANDS[bi]..BANDS[bi + 1] {
                for ci in 0..NCH {
                    let m = 0.1 * xb[(xi * NBANDS + bi) * NCH + ci];
                    xps[(xi * NFREQS + xj) * NCH + ci] += m;
                    yps[(xi * NFREQS + xj) * NCH + ci] += m;
                }
            }
        }
    }

    // Two-frame averaging.
    for xj in 0..BANDS[NBANDS] {
        for ci in 0..NCH {
            let mut xtmp = xps[xj * NCH + ci];
            let mut ytmp = yps[xj * NCH + ci];
            for xi in 1..nframes {
                let x2 = xps[(xi * NFREQS + xj) * NCH + ci];
                let y2 = yps[(xi * NFREQS + xj) * NCH + ci];
                xps[(xi * NFREQS + xj) * NCH + ci] += xtmp;
                yps[(xi * NFREQS + xj) * NCH + ci] += ytmp;
                xtmp = x2;
                ytmp = y2;
            }
        }
    }

    // Pool the log-spectral distortion.
    let mut err = 0.0f64;
    for xi in 0..nframes {
        let mut ef = 0.0f64;
        for bi in 0..NBANDS {
            let mut eb = 0.0f64;
            for xj in BANDS[bi]..BANDS[bi + 1] {
                for ci in 0..NCH {
                    let re =
                        f64::from(yps[(xi * NFREQS + xj) * NCH + ci]) / f64::from(xps[(xi * NFREQS + xj) * NCH + ci]);
                    let mut im = re - re.ln() - 1.0;
                    // Less sensitive around the SILK/CELT crossover.
                    if (79..=81).contains(&xj) {
                        im *= 0.1;
                    }
                    if xj == 80 {
                        im *= 0.1;
                    }
                    eb += im;
                }
            }
            eb /= ((BANDS[bi + 1] - BANDS[bi]) * NCH) as f64;
            ef += eb * eb;
        }
        ef /= NBANDS as f64;
        let ef2 = ef * ef;
        err += ef2 * ef2;
    }
    let err = (err / nframes as f64).powf(1.0 / 16.0);
    let q = 100.0 * (1.0 - 0.5 * (1.0 + err).ln() / 1.13f64.ln());
    (err, q)
}

/// The official conformance criterion: `opus_compare` quality ≥ 0 against
/// the reference decode, for every vector. Slow (naive DFT over the whole
/// suite) - run explicitly with `cargo test --release -- --ignored`.
#[cfg(feature = "std")]
#[test]
#[ignore = "minutes of DFT; run with --release -- --ignored"]
fn official_quality_metric_passes_every_vector() {
    use opus_native::OpusDecoder;

    let Some(dir) = vectors_dir() else { return };
    for (name, _) in VECTORS {
        let bits = std::fs::read(dir.join(format!("{name}.bit"))).expect("read .bit");
        let reference = std::fs::read(dir.join(format!("{name}.dec"))).expect("read .dec");
        let refpcm: Vec<f32> = reference
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])))
            .collect();

        let mut decoder = OpusDecoder::new(2);
        let mut pcm: Vec<f32> = Vec::new();
        for pkt in parse_bit_file(&bits) {
            pcm.extend(
                decoder
                    .decode_packet_i16(&pkt.data)
                    .expect("valid")
                    .into_iter()
                    .map(f32::from),
            );
        }
        let (err, q) = opus_compare_48k_stereo(&refpcm, &pcm);
        eprintln!("{name}: opus_compare quality {q:.1}% (err {err:.6})");
        assert!(q >= 0.0, "{name}: FAILS the official metric (err {err})");
    }
}

/// The CELT encoder round trip: encoded packets must decode through our
/// conformant decoder with bit-identical range states (the same oracle
/// libopus's opus_demo enforces - it accepted this encoder's streams with
/// zero mismatches when cross-checked), reconstructing the signal at the
/// codec's one-window delay.
#[cfg(feature = "std")]
#[test]
fn celt_encoder_round_trips_through_the_decoder() {
    use opus_native::OpusDecoder;
    use opus_native::celt::encoder::CeltEncoder;

    let mut enc = CeltEncoder::new();
    let mut dec = OpusDecoder::new(1);
    let mut input = Vec::new();
    let mut output = Vec::new();
    for f in 0..50 {
        let pcm: Vec<f32> = (0..960)
            .map(|i| {
                let t = (f * 960 + i) as f32 / 48_000.0;
                // Percussive bursts exercise the transient/short-block path.
                let burst = if f % 7 == 3 && (480..600).contains(&i) {
                    0.4 * (2.0 * core::f32::consts::PI * 3100.0 * t).sin() * (-(i as f32 - 480.0) / 30.0).exp()
                } else {
                    0.0
                };
                0.5 * (2.0 * core::f32::consts::PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * core::f32::consts::PI * 1800.0 * t).sin()
                    + burst
            })
            .collect();
        input.extend_from_slice(&pcm);
        let payload = enc.encode_frame(&pcm, 159);
        // TOC: config 31 (CELT-only fullband 20 ms), mono, code 0.
        let mut packet = vec![0xF8u8];
        packet.extend_from_slice(&payload);
        output.extend(dec.decode_packet(&packet).expect("valid packet"));
        assert_eq!(
            dec.final_range(),
            enc.final_range(),
            "frame {f}: encoder/decoder range states diverged"
        );
    }
    // One MDCT window of algorithmic delay.
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for i in 4_800..input.len() - 120 {
        let a = f64::from(input[i]);
        let b = f64::from(output[i + 120]);
        sig += a * a;
        noise += (a - b) * (a - b);
    }
    let snr = 10.0 * (sig / noise.max(1e-30)).log10();
    // Dynalloc + trim analysis lifts this well above the original 20 dB
    // floor (~30 dB on this tonal signal at 159 bytes/frame).
    assert!(snr > 25.0, "round-trip SNR {snr:.1} dB");
}

/// The stereo CELT encoder round trip: decorrelated channels through the
/// mid/side machinery, with the same range-state oracle (libopus's
/// opus_demo also accepted this encoder's stereo streams with zero
/// mismatches when cross-checked).
#[cfg(feature = "std")]
#[test]
fn celt_stereo_encoder_round_trips_through_the_decoder() {
    use opus_native::OpusDecoder;
    use opus_native::celt::encoder::CeltEncoder;

    let mut enc = CeltEncoder::with_channels(2);
    let mut dec = OpusDecoder::new(2);
    let mut input = Vec::new();
    let mut output = Vec::new();
    for f in 0..50 {
        let mut pcm = Vec::with_capacity(960 * 2);
        for i in 0..960 {
            let t = (f * 960 + i) as f32 / 48_000.0;
            // Percussive bursts exercise the transient/short-block path.
            let burst = if f % 7 == 3 && (480..600).contains(&i) {
                0.4 * (2.0 * core::f32::consts::PI * 3100.0 * t).sin() * (-(i as f32 - 480.0) / 30.0).exp()
            } else {
                0.0
            };
            pcm.push(
                0.5 * (2.0 * core::f32::consts::PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * core::f32::consts::PI * 1800.0 * t).sin()
                    + burst,
            );
            pcm.push(
                0.4 * (2.0 * core::f32::consts::PI * 660.0 * t).sin()
                    + 0.2 * (2.0 * core::f32::consts::PI * 2500.0 * t + 0.7).sin()
                    + burst,
            );
        }
        input.extend_from_slice(&pcm);
        let payload = enc.encode_frame(&pcm, 159);
        // TOC: config 31 (CELT-only fullband 20 ms), stereo, code 0.
        let mut packet = vec![0xFCu8];
        packet.extend_from_slice(&payload);
        output.extend(dec.decode_packet(&packet).expect("valid packet"));
        assert_eq!(
            dec.final_range(),
            enc.final_range(),
            "frame {f}: encoder/decoder range states diverged"
        );
    }
    // One MDCT window of algorithmic delay (interleaved: 2 * 120).
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for i in 9_600..input.len() - 240 {
        let a = f64::from(input[i]);
        let b = f64::from(output[i + 240]);
        sig += a * a;
        noise += (a - b) * (a - b);
    }
    let snr = 10.0 * (sig / noise.max(1e-30)).log10();
    assert!(snr > 10.0, "stereo round-trip SNR {snr:.1} dB");
}

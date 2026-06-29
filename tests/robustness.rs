//! Decoder robustness against untrusted/malformed input: it must never panic
//! (a DoS surface) regardless of the bytes it is handed. Bounded versions of
//! the `fuzz_decode`/`fuzz_encode` examples, plus the exact packets that once
//! panicked (regression guards).

#![cfg(feature = "std")]

use ruopus::{Bandwidth, OpusDecoder, OpusEncoder};

/// Mutated hybrid packets once underflowed `RangeDecoder::force_tell`
/// (`assert bits >= current`) and then `decoder.rs` `len -= redundancy_bytes`.
/// These exact bytes must decode without panicking (cleanly, as garbage).
#[test]
fn malformed_hybrid_packets_do_not_panic() {
    let cases: [&[u8]; 2] = [
        &[
            0x71, 0xa3, 0xa8, 0x78, 0x62, 0x25, 0xbb, 0x39, 0xb8, 0x9d, 0xd7, 0x49, 0x3f, 0x7f, 0xde, 0x69, 0xe8, 0x92,
            0x65, 0xed, 0x1e, 0x80, 0x56, 0x42, 0x88, 0x0d, 0xad, 0xcd, 0x4a, 0x95, 0x93, 0x9d, 0x3f, 0x73, 0x96, 0x29,
            0x8a, 0xc8, 0xee, 0xed, 0xa4, 0xd3, 0x6a, 0xca, 0xbf, 0x7d, 0x94,
        ],
        &[
            0x68, 0xba, 0x1a, 0x99, 0x0c, 0x74, 0x21, 0xfe, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
            0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        ],
    ];
    for (i, pkt) in cases.iter().enumerate() {
        let mut dec = OpusDecoder::new(1);
        // Must not panic; output (if any) must be finite.
        if let Ok(out) = dec.decode_packet(pkt) {
            assert!(out.iter().all(|v| v.is_finite()), "case {i} produced non-finite output");
        }
    }
}

struct Rng(u64);
impl Rng {
    fn n(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Random bytes and bit-flipped valid packets never panic the decoder and
/// never yield non-finite samples.
#[test]
fn decoder_survives_random_and_mutated_packets() {
    // Seed packets across modes to mutate (reach the SILK/CELT internals).
    let seeds: Vec<(usize, Vec<u8>)> = [
        (1usize, Bandwidth::WideBand, 16_000u32),
        (1, Bandwidth::SuperWideBand, 32_000),
        (1, Bandwidth::FullBand, 64_000),
        (2, Bandwidth::FullBand, 96_000),
    ]
    .iter()
    .filter_map(|&(ch, bw, br)| {
        let mut enc = OpusEncoder::new(ch);
        enc.set_bandwidth(bw);
        enc.set_bitrate(Some(br));
        let pcm: Vec<f32> = (0..960 * ch).map(|i| 0.3 * (i as f32 / 13.0).sin()).collect();
        enc.encode_auto(&pcm, 1275).ok().map(|p| (ch, p))
    })
    .collect();

    let mut rng = Rng(0xF0F0_1234);
    for it in 0..30_000u64 {
        let (ch, pkt) = if it % 2 == 0 {
            let ch = 1 + (rng.n() % 2) as usize;
            let len = (rng.n() % 64) as usize;
            (ch, (0..len).map(|_| (rng.n() & 0xff) as u8).collect::<Vec<u8>>())
        } else {
            let (ch, base) = &seeds[(rng.n() as usize) % seeds.len()];
            let mut p = base.clone();
            for _ in 0..1 + rng.n() % 4 {
                let idx = (rng.n() as usize) % p.len();
                p[idx] ^= 1 << (rng.n() % 8);
            }
            (*ch, p)
        };
        let mut dec = OpusDecoder::new(ch);
        if let Ok(out) = dec.decode_packet(&pkt) {
            assert!(out.iter().all(|v| v.is_finite()), "non-finite output at it={it}");
        }
    }
}

/// The encoder never panics across the config space on diverse signals, and
/// every packet it produces decodes to finite output.
#[test]
fn encoder_survives_diverse_signals() {
    const SIZES: [usize; 6] = [120, 240, 480, 960, 1920, 2880];
    const BWS: [Bandwidth; 5] = [
        Bandwidth::NarrowBand,
        Bandwidth::MediumBand,
        Bandwidth::WideBand,
        Bandwidth::SuperWideBand,
        Bandwidth::FullBand,
    ];
    let mut rng = Rng(0x0BAD_F00D);
    for _ in 0..8_000u64 {
        let ch = 1 + (rng.n() % 2) as usize;
        let mut enc = OpusEncoder::new(ch);
        enc.set_bandwidth(BWS[(rng.n() % 5) as usize]);
        enc.set_bitrate(if rng.n() % 4 == 0 {
            None
        } else {
            Some(6_000 + (rng.n() % 200_000) as u32)
        });
        // DTX is exercised separately (dtx_* unit tests); here frame sizes vary
        // per frame, which is incompatible with DTX's fixed-size concealment.
        let mut dec = OpusDecoder::new(ch);
        for _ in 0..1 + rng.n() % 4 {
            let spf = SIZES[(rng.n() % 6) as usize];
            let amp = [0.0f32, 1e-4, 0.3, 1.0, 1.9][(rng.n() % 5) as usize];
            let kind = rng.n() % 4;
            let pcm: Vec<f32> = (0..spf * ch)
                .map(|i| {
                    let t = i as f32 / 48_000.0;
                    match kind {
                        0 => 0.0,
                        1 => amp,
                        2 => amp * (2.0 * std::f32::consts::PI * 440.0 * t).sin(),
                        _ => amp * ((((i as u64).wrapping_mul(2654435761) >> 16) & 0xffff) as f32 / 32768.0 - 1.0),
                    }
                })
                .collect();
            if let Ok(pkt) = enc.encode_auto(&pcm, 1275) {
                let out = dec.decode_packet(&pkt).expect("decode our own packet");
                assert_eq!(out.len(), spf * ch);
                assert!(out.iter().all(|v| v.is_finite()));
            }
        }
    }
}

//! Robustness fuzz for the decoder against untrusted input - random bytes and
//! bit-flipped *valid* packets (which reach the SILK/CELT decode internals).
//! A decoder must never panic on malformed data (a DoS surface). Asserts no
//! panic and finite output; cleanly-rejected packets (`Err`) are fine.
//!
//!   cargo run --release --example fuzz_decode --features std -- 500000

use opus_native::{Bandwidth, OpusDecoder, OpusEncoder};

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

/// A small library of real packets to mutate, one per mode/bandwidth.
fn seed_packets() -> Vec<(usize, Vec<u8>)> {
    let mut out = Vec::new();
    for &(ch, bw, br, spf) in &[
        (1usize, Bandwidth::NarrowBand, 16_000u32, 960usize),
        (1, Bandwidth::WideBand, 16_000, 960),
        (1, Bandwidth::SuperWideBand, 32_000, 960),
        (1, Bandwidth::FullBand, 64_000, 960),
        (2, Bandwidth::FullBand, 96_000, 960),
        (1, Bandwidth::FullBand, 64_000, 240),
    ] {
        let mut enc = OpusEncoder::new(ch);
        enc.set_bandwidth(bw);
        enc.set_bitrate(Some(br));
        let pcm: Vec<f32> = (0..spf * ch)
            .map(|i| 0.3 * (i as f32 / 13.0).sin() + 0.1 * (i as f32 / 3.1).sin())
            .collect();
        if let Ok(p) = enc.encode_auto(&pcm, 1275) {
            out.push((ch, p));
        }
    }
    out
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let iters: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(500_000);
    let mut rng = Rng(0x1234_5678);
    let seeds = seed_packets();
    let mut bad = 0u64;

    for it in 0..iters {
        // Half random bytes, half bit-flipped valid packets.
        let (ch, pkt) = if it % 2 == 0 || seeds.is_empty() {
            let ch = 1 + (rng.n() % 2) as usize;
            let len = (rng.n() % 64) as usize;
            let mut p = vec![0u8; len];
            for b in p.iter_mut() {
                *b = (rng.n() & 0xff) as u8;
            }
            (ch, p)
        } else {
            let (ch, base) = &seeds[(rng.n() as usize) % seeds.len()];
            let mut p = base.clone();
            // Flip 1..4 random bits.
            for _ in 0..1 + rng.n() % 4 {
                if !p.is_empty() {
                    let idx = (rng.n() as usize) % p.len();
                    p[idx] ^= 1 << (rng.n() % 8);
                }
            }
            (*ch, p)
        };

        let mut dec = OpusDecoder::new(ch);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dec.decode_packet(&pkt)));
        match r {
            Err(_) => {
                bad += 1;
                if bad <= 10 {
                    eprintln!(
                        "PANIC it={it} ch={ch} len={} bytes={:02x?}",
                        pkt.len(),
                        &pkt[..pkt.len().min(8)]
                    );
                }
            },
            Ok(Ok(out)) => {
                if out.iter().any(|v| !v.is_finite()) {
                    bad += 1;
                    if bad <= 10 {
                        eprintln!("NON-FINITE it={it} ch={ch} len={}", pkt.len());
                    }
                }
            },
            Ok(Err(_)) => {}, // cleanly rejected - good
        }
        if it % 100_000 == 0 && it > 0 {
            eprintln!("  ... {it}, {bad} bad");
        }
    }
    println!("fuzz_decode: {iters} iterations, {bad} panics/non-finite");
    if bad > 0 {
        std::process::exit(1);
    }
}

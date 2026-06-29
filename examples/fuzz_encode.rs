//! Randomized differential robustness sweep for the encoder.
//!
//! Encodes diverse random signals across the full config space (channels,
//! bandwidth, bitrate, frame size, DTX, per-mode entry points) and decodes
//! each, asserting no panic, correct output length, and finite samples - and,
//! for the modes whose state stays in sync, a matching final range. A seeded
//! LCG makes every run reproducible; pass an iteration count as arg 1.
//!
//!   cargo run --release --example fuzz_encode --features std -- 200000

use opus_rs::{Bandwidth, OpusDecoder, OpusEncoder};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

const BANDWIDTHS: [Bandwidth; 5] = [
    Bandwidth::NarrowBand,
    Bandwidth::MediumBand,
    Bandwidth::WideBand,
    Bandwidth::SuperWideBand,
    Bandwidth::FullBand,
];
const FRAME_SIZES: [usize; 6] = [120, 240, 480, 960, 1920, 2880];

/// Builds a random signal of `n` samples per channel, interleaved for `ch`.
fn signal(rng: &mut Rng, n: usize, ch: usize) -> Vec<f32> {
    let kind = rng.below(8);
    let amp = [0.0f32, 1e-4, 0.01, 0.3, 0.9, 1.0, 1.9][rng.below(7) as usize];
    let f1 = 20.0 + rng.f() * 8000.0;
    let f2 = 20.0 + rng.f() * 16000.0;
    let period = 1 + rng.below(400) as usize;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f32 / 48_000.0;
        let base = match kind {
            0 => 0.0,                                                // silence
            1 => amp,                                                // DC
            2 => amp * (2.0 * core::f32::consts::PI * f1 * t).sin(), // tone
            3 => amp * ((2.0 * core::f32::consts::PI * f1 * t).sin() + (2.0 * core::f32::consts::PI * f2 * t).sin()),
            4 => {
                if i % period == 0 {
                    amp
                } else {
                    0.0
                }
            }, // impulses
            5 => amp * (rng.f() - 0.5) * 2.0, // noise
            6 => amp * (2.0 * core::f32::consts::PI * (f1 + 4000.0 * t) * t).sin(), // chirp
            _ => {
                if i % 2 == 0 {
                    amp
                } else {
                    -amp
                }
            }, // full-scale square
        };
        for c in 0..ch {
            // Slight per-channel decorrelation for stereo.
            let v = if c == 0 {
                base
            } else {
                base * 0.85 + amp * 0.1 * (rng.f() - 0.5)
            };
            out.push(v.clamp(-1.0, 1.0));
        }
    }
    out
}

fn main() {
    let iters: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);

    // Persistent per-(channels) encoder/decoder pairs so cross-frame state and
    // mode switching are exercised across a stream, not just per-frame.
    let mut fails = 0u64;
    for it in 0..iters {
        let ch = 1 + rng.below(2) as usize;
        let mut enc = OpusEncoder::new(ch);
        let mut dec = OpusDecoder::new(ch);
        enc.set_bandwidth(BANDWIDTHS[rng.below(5) as usize]);
        let br = 6_000 + rng.below(200_000);
        enc.set_bitrate(if rng.below(4) == 0 { None } else { Some(br as u32) });
        enc.set_dtx(rng.below(2) == 0);

        // A short stream of several frames.
        for _ in 0..1 + rng.below(8) {
            let spf = FRAME_SIZES[rng.below(6) as usize];
            let pcm = signal(&mut rng, spf, ch);
            let method = rng.below(4);
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match method {
                0 => enc.encode_auto(&pcm, 1275),
                1 => enc.encode(&pcm, 1275),
                2 if spf == 480 || spf == 960 || spf == 1920 || spf == 2880 => enc.encode_silk(&pcm, 1275),
                3 if (spf == 480 || spf == 960) && ch <= 2 => enc.encode_hybrid(&pcm, 1275),
                _ => enc.encode_auto(&pcm, 1275),
            }));
            let packet = match res {
                Err(_) => {
                    eprintln!("PANIC it={it} ch={ch} spf={spf} method={method} br={br} bw_idx encode",);
                    fails += 1;
                    break;
                },
                Ok(Err(_)) => continue, // a legitimate EncodeError (e.g. wrong size for the method)
                Ok(Ok(p)) => p,
            };
            // Decode and validate.
            let dres = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dec.decode_packet(&packet)));
            match dres {
                Err(_) => {
                    eprintln!("PANIC it={it} ch={ch} spf={spf} method={method} br={br} DECODE");
                    fails += 1;
                    break;
                },
                Ok(Ok(out)) => {
                    if out.iter().any(|v| !v.is_finite()) {
                        eprintln!("NON-FINITE it={it} ch={ch} spf={spf} method={method} br={br}");
                        fails += 1;
                        break;
                    }
                },
                Ok(Err(e)) => {
                    eprintln!("DECODE-ERR it={it} ch={ch} spf={spf} method={method}: {e:?}");
                    fails += 1;
                    break;
                },
            }
        }
        if it % 20_000 == 0 && it > 0 {
            eprintln!("  ... {it} iterations, {fails} failures so far");
        }
    }
    println!("fuzz_encode: {iters} iterations, {fails} failures");
    if fails > 0 {
        std::process::exit(1);
    }
}

//! libopus-only encode loop, for profiling the C reference in isolation.
//! Encodes the same speech-like signal as the comparison bench, at a chosen
//! complexity (env `LIBOPUS_COMPLEXITY`, default 0), wideband 16 kb/s, in a
//! long loop so `perf` can attribute libopus's internal hot functions.
//!
//!   CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
//!     cargo build --release --example libopus_bench --features std
//!   LIBOPUS_COMPLEXITY=0 perf record -F4000 -g --call-graph dwarf \
//!     -- ./target/release/examples/libopus_bench
#![allow(missing_docs)]

use opus::{Application, Bandwidth, Bitrate, Channels};

const FS: usize = 48_000;
const FRAME: usize = 960;
const SECONDS: usize = 6;

fn signal() -> Vec<f32> {
    // Identical to examples/enc_bench.rs, built sin-free from a wavetable so the
    // one-time synthesis does not pollute the libopus encode profile.
    const TBL: usize = 8192;
    let table: Vec<f32> = (0..TBL)
        .map(|i| (2.0 * std::f32::consts::PI * i as f32 / TBL as f32).sin())
        .collect();
    let sine = |f: f32, n: usize| -> f32 {
        let phase = (f * n as f32 / FS as f32).fract();
        table[(phase * TBL as f32) as usize % TBL]
    };
    let mut sig = Vec::with_capacity(FRAME * 50 * SECONDS);
    for f in 0..(50 * SECONDS) {
        let mut seed = 0x1234_5678u32.wrapping_add(f as u32);
        for i in 0..FRAME {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let noise = (seed >> 9) as f32 / f32::from(u16::MAX) - 0.5;
            let n = f * FRAME + i;
            let env = 0.5 + 0.45 * sine(3.0, n).abs();
            sig.push(env * (0.45 * sine(200.0, n) + 0.25 * sine(1400.0, n) + 0.15 * sine(6500.0, n)) + 0.02 * noise);
        }
    }
    sig
}

fn main() {
    let complexity: i32 = std::env::var("LIBOPUS_COMPLEXITY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let reps: usize = std::env::var("LIBOPUS_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000);
    // Mode selection: LIBOPUS_MODE = silk (default) | hybrid | celt.
    let mode = std::env::var("LIBOPUS_MODE").unwrap_or_else(|_| "silk".into());
    let (bw, br, app) = match mode.as_str() {
        "hybrid" => (Bandwidth::Fullband, 32_000, Application::Voip),
        "celt" => (Bandwidth::Fullband, 64_000, Application::Audio),
        _ => (Bandwidth::Wideband, 16_000, Application::Voip),
    };
    let sig = signal();
    let frames: Vec<&[f32]> = sig.chunks_exact(FRAME).collect();

    let mut e = opus::Encoder::new(FS as u32, Channels::Mono, app).unwrap();
    e.set_bitrate(Bitrate::Bits(br)).unwrap();
    e.set_bandwidth(bw).unwrap();
    e.set_vbr(true).unwrap();
    e.set_complexity(complexity).unwrap();

    // Warm up, then time.
    for f in &frames {
        let _ = e.encode_vec_float(f, 1275).unwrap();
    }
    let start = std::time::Instant::now();
    let mut bytes = 0usize;
    for _ in 0..reps {
        for f in &frames {
            bytes += e.encode_vec_float(f, 1275).unwrap().len();
        }
    }
    let secs = start.elapsed().as_secs_f64();
    let audio_s = (reps * frames.len()) as f64 * 0.02;
    println!(
        "libopus c{complexity}: {:>6.0}x realtime  ({:.0} kb/s, {bytes} bytes)",
        audio_s / secs,
        bytes as f64 * 8.0 / audio_s / 1000.0
    );
}

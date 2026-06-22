//! Encoder throughput benchmark (× realtime), per mode. The counterpart to
//! `decode_throughput`. Encodes a fixed speech-like signal and reports how
//! much faster than realtime each mode runs.
//!
//!   cargo run --release --example enc_bench --features std

use std::time::Instant;

use opus_native::{Bandwidth, OpusEncoder};

fn bench(label: &str, ch: usize, bw: Bandwidth, br: u32, frames: usize) {
    // The SAME reproducible speech/music-like signal as the libopus profiling
    // bench (examples/libopus_bench.rs) and benches/vs_libopus.rs: voiced tones
    // under a syllabic envelope, plus light noise and a high partial. Built from
    // a precomputed sine wavetable so signal synthesis costs no libm `sinf`
    // (huge phase arguments would otherwise hit sinf's slow range-reduction and
    // pollute the encode profile - `perf` samples the whole process).
    const TBL: usize = 8192;
    let table: Vec<f32> = (0..TBL)
        .map(|i| (2.0 * std::f32::consts::PI * i as f32 / TBL as f32).sin())
        .collect();
    let sine = |f: f32, n: usize| -> f32 {
        let phase = (f * n as f32 / 48_000.0).fract();
        table[(phase * TBL as f32) as usize % TBL]
    };
    let make = |f: usize| -> Vec<f32> {
        let mut seed = 0x1234_5678u32.wrapping_add(f as u32);
        (0..960 * ch)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let noise = (seed >> 9) as f32 / f32::from(u16::MAX) - 0.5;
                let n = f * 960 + i / ch;
                let env = 0.5 + 0.45 * sine(3.0, n).abs();
                env * (0.45 * sine(200.0, n) + 0.25 * sine(1400.0, n) + 0.15 * sine(6500.0, n)) + 0.02 * noise
            })
            .collect()
    };
    let sigs: Vec<Vec<f32>> = (0..200).map(make).collect();

    let mut enc = OpusEncoder::new(ch);
    enc.set_bandwidth(bw);
    enc.set_bitrate(Some(br));
    if let Some(c) = std::env::var("OPUS_BENCH_COMPLEXITY").ok().and_then(|v| v.parse().ok()) {
        enc.set_complexity(c);
    }
    for s in &sigs {
        let _ = enc.encode_auto(s, 1275); // warm up lazily-created state
    }

    let start = Instant::now();
    let mut bytes = 0usize;
    for _ in 0..frames / 200 {
        for s in &sigs {
            bytes += enc.encode_auto(s, 1275).map_or(0, |p| p.len());
        }
    }
    let secs = start.elapsed().as_secs_f64();
    let audio_s = frames as f64 * 0.02;
    println!(
        "{label:<22} {:>6.0}× realtime  ({:.0} kb/s)",
        audio_s / secs,
        bytes as f64 * 8.0 / audio_s / 1000.0
    );
    // With OPUS_PROF=1 set, print this mode's per-stage breakdown.
    if std::env::var_os("OPUS_PROF").is_some() {
        eprintln!("[{label}]");
        opus_native::prof::dump();
    }
}

fn main() {
    // Frame count per mode; reduce for callgrind/valgrind via OPUS_BENCH_FRAMES.
    // A single mode can be selected with OPUS_BENCH_MODE=silk|hybrid|celt.
    let f: usize = std::env::var("OPUS_BENCH_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let mode = std::env::var("OPUS_BENCH_MODE").unwrap_or_default();
    let want = |m: &str| mode.is_empty() || mode == m;
    println!("encoder throughput (release):");
    if want("silk") {
        bench("SILK WB 16k mono", 1, Bandwidth::WideBand, 16_000, f);
    }
    if want("hybrid") {
        bench("hybrid FB 32k mono", 1, Bandwidth::FullBand, 32_000, f);
    }
    if want("celt") {
        bench("CELT FB 96k mono", 1, Bandwidth::FullBand, 96_000, f);
    }
    if want("hybrid") {
        bench("hybrid FB 48k stereo", 2, Bandwidth::FullBand, 48_000, f);
    }
}

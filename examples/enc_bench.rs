//! Encoder throughput benchmark (× realtime), per mode. The counterpart to
//! `decode_throughput`. Encodes a fixed speech-like signal and reports how
//! much faster than realtime each mode runs.
//!
//!   cargo run --release --example enc_bench --features std

use std::time::Instant;

use opus_native::{Bandwidth, OpusEncoder};

fn bench(label: &str, ch: usize, bw: Bandwidth, br: u32, frames: usize) {
    // A speech/music-like signal: low tone + a high partial, per channel.
    let make = |f: usize| -> Vec<f32> {
        (0..960 * ch)
            .map(|i| {
                let t = (f * 960 + i / ch) as f32 / 48_000.0;
                0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.15 * (2.0 * std::f32::consts::PI * 3000.0 * t).sin()
            })
            .collect()
    };
    let sigs: Vec<Vec<f32>> = (0..200).map(make).collect();

    let mut enc = OpusEncoder::new(ch);
    enc.set_bandwidth(bw);
    enc.set_bitrate(Some(br));
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

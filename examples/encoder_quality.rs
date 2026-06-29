//! Encoder quality differential harness: ours vs libopus, same input.
//!
//! For each (mode, bandwidth, bitrate) config it encodes the input with our
//! `OpusEncoder`, decodes it with our `OpusDecoder`, and measures the
//! delay-aligned signal-to-noise ratio and the achieved bitrate. It then does
//! the same through the reference (`opus_demo -e` then `opus_demo -d`) and
//! prints both side by side, so a change to our encoder can be judged on real
//! audio - higher SNR at a comparable bitrate is better - rather than on
//! synthetic tones, which mislead (pure tones overfit per-subframe LPC).
//!
//! Usage:
//!   cargo run --example encoder_quality --features std -- <input.pcm>
//!
//! `<input.pcm>` is raw 48 kHz mono signed-16-bit little-endian PCM. Produce
//! one with, e.g.:
//!   ffmpeg -i speech.wav -ac 1 -ar 48000 -f s16le out.pcm
//!
//! The reference encoder is found via `$OPUS_DEMO` or
//! `/tmp/opus-ref/build/opus_demo`. If it is missing, only our side is shown.

use std::path::Path;
use std::process::Command;

use ruopus::{Bandwidth, OpusDecoder, OpusEncoder};

const FRAME: usize = 960; // 20 ms at 48 kHz

#[derive(Clone, Copy)]
enum Mode {
    Silk,
    Hybrid,
    Celt,
}

struct Config {
    label: &'static str,
    mode: Mode,
    bw: Bandwidth,
    bitrate: u32,
    /// `opus_demo` application and `-bandwidth` token for the reference run.
    demo_app: &'static str,
    demo_bw: &'static str,
}

fn read_pcm_s16le(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(2)
        .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
        .collect()
}

fn write_pcm_s16le(path: &str, pcm: &[f32]) {
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for &v in pcm {
        let s = (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, &bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

/// Delay-aligned global SNR in dB: searches a lag window for the alignment
/// that maximises SNR (the codec output lags the input by its algorithmic
/// delay), skipping a warm-up margin. Returns `(best_lag, snr_db)`.
fn aligned_snr(reference: &[f32], degraded: &[f32]) -> (usize, f64) {
    const WARMUP: usize = 960;
    const MAX_LAG: usize = 2000;
    let mut best = (0usize, f64::NEG_INFINITY);
    for lag in 0..=MAX_LAG {
        if degraded.len() <= lag + WARMUP {
            break;
        }
        let n = (reference.len() - WARMUP).min(degraded.len() - WARMUP - lag);
        if n < FRAME {
            break;
        }
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for i in 0..n {
            let r = f64::from(reference[WARMUP + i]);
            let d = f64::from(degraded[WARMUP + lag + i]);
            sig += r * r;
            err += (r - d) * (r - d);
        }
        let snr = 10.0 * (sig / err.max(1e-12)).log10();
        if snr > best.1 {
            best = (lag, snr);
        }
    }
    best
}

/// Encodes and decodes `pcm` through our codec at the given config, returning
/// the decoded signal, total coded byte count, and the number of frames that
/// fell back to CELT (a hybrid frame whose SILK low band could not be squeezed
/// under its byte share falls back to CELT-only so the stream stays complete).
fn our_roundtrip(pcm: &[f32], cfg: &Config) -> (Vec<f32>, usize, usize) {
    let mut enc = OpusEncoder::new(1);
    enc.set_bandwidth(cfg.bw);
    enc.set_bitrate(Some(cfg.bitrate));
    let mut dec = OpusDecoder::new(1);
    // Generous budget: the rate is governed by the target bitrate (VBR), and we
    // report the achieved rate - measuring quality at a target, not CBR fill.
    let max_bytes = 1275;

    let mut out = Vec::with_capacity(pcm.len());
    let mut total = 0usize;
    let mut fallbacks = 0usize;
    for frame in pcm.chunks_exact(FRAME) {
        let packet = match cfg.mode {
            Mode::Silk => enc.encode_silk(frame, max_bytes),
            Mode::Hybrid => enc.encode_hybrid(frame, max_bytes),
            Mode::Celt => enc.encode(frame, max_bytes),
        }
        .or_else(|_| {
            fallbacks += 1;
            enc.encode(frame, max_bytes)
        })
        .expect("our encode (incl. CELT fallback)");
        total += packet.len();
        out.extend_from_slice(&dec.decode_packet(&packet).expect("our decode"));
    }
    (out, total, fallbacks)
}

/// Runs the reference encoder+decoder via `opus_demo`, returning the decoded
/// signal and total coded byte count, or `None` if `opus_demo` is unavailable.
fn libopus_roundtrip(input_path: &str, cfg: &Config, demo: &str) -> Option<(Vec<f32>, usize)> {
    if !Path::new(demo).exists() {
        return None;
    }
    let bit = format!("/tmp/eq_{}.bit", cfg.label);
    let out = format!("/tmp/eq_{}.out.pcm", cfg.label);
    let enc = Command::new(demo)
        .args([
            "-e",
            cfg.demo_app,
            "48000",
            "1",
            &cfg.bitrate.to_string(),
            "-bandwidth",
            cfg.demo_bw,
            "-framesize",
            "20",
            input_path,
            &bit,
        ])
        .output()
        .ok()?;
    if !enc.status.success() {
        eprintln!("opus_demo -e failed for {}", cfg.label);
        return None;
    }
    let dec = Command::new(demo)
        .args(["-d", "48000", "1", &bit, &out])
        .output()
        .ok()?;
    if !dec.status.success() {
        eprintln!("opus_demo -d failed for {}", cfg.label);
        return None;
    }
    // Sum coded packet sizes from the .bit container (4-byte BE length, 4-byte
    // BE range, then the packet - repeated per frame).
    let raw = std::fs::read(&bit).ok()?;
    let mut total = 0usize;
    let mut i = 0;
    while i + 8 <= raw.len() {
        let len = u32::from_be_bytes([raw[i], raw[i + 1], raw[i + 2], raw[i + 3]]) as usize;
        i += 8 + len;
        total += len;
    }
    Some((read_pcm_s16le(&out), total))
}

fn kbps(total_bytes: usize, n_samples: usize) -> f64 {
    let seconds = n_samples as f64 / 48_000.0;
    (total_bytes as f64 * 8.0) / seconds / 1000.0
}

fn main() {
    let input = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: encoder_quality <input.pcm>  (48 kHz mono s16le)");
        std::process::exit(2);
    });
    let pcm = read_pcm_s16le(&input);
    let n = pcm.len() / FRAME * FRAME;
    let demo = std::env::var("OPUS_DEMO").unwrap_or_else(|_| "/tmp/opus-ref/build/opus_demo".to_string());

    let configs = [
        Config {
            label: "silk-wb-12k",
            mode: Mode::Silk,
            bw: Bandwidth::WideBand,
            bitrate: 12_000,
            demo_app: "voip",
            demo_bw: "WB",
        },
        Config {
            label: "silk-wb-16k",
            mode: Mode::Silk,
            bw: Bandwidth::WideBand,
            bitrate: 16_000,
            demo_app: "voip",
            demo_bw: "WB",
        },
        Config {
            label: "silk-wb-24k",
            mode: Mode::Silk,
            bw: Bandwidth::WideBand,
            bitrate: 24_000,
            demo_app: "voip",
            demo_bw: "WB",
        },
        Config {
            label: "hyb-swb-24k",
            mode: Mode::Hybrid,
            bw: Bandwidth::SuperWideBand,
            bitrate: 24_000,
            demo_app: "voip",
            demo_bw: "SWB",
        },
        Config {
            label: "hyb-fb-32k",
            mode: Mode::Hybrid,
            bw: Bandwidth::FullBand,
            bitrate: 32_000,
            demo_app: "voip",
            demo_bw: "FB",
        },
        Config {
            label: "celt-fb-64k",
            mode: Mode::Celt,
            bw: Bandwidth::FullBand,
            bitrate: 64_000,
            demo_app: "restricted-lowdelay",
            demo_bw: "FB",
        },
    ];

    println!("input: {input}  ({:.2} s, {} frames)", n as f64 / 48_000.0, n / FRAME);
    println!(
        "{:<14} {:>10} {:>9} {:>12} {:>9}   {:>8}",
        "config", "ours kbps", "ours SNR", "libopus kbps", "lib SNR", "ΔSNR"
    );
    println!("{}", "-".repeat(72));

    for cfg in &configs {
        let (our_out, our_bytes, fallbacks) = our_roundtrip(&pcm[..n], cfg);
        let (_, our_snr) = aligned_snr(&pcm[..n], &our_out);
        let our_kbps = kbps(our_bytes, n);
        write_pcm_s16le(&format!("/tmp/eq_{}.ours.pcm", cfg.label), &our_out);
        let note = if fallbacks > 0 {
            format!("  [{fallbacks} CELT fallback]")
        } else {
            String::new()
        };

        match libopus_roundtrip(&input, cfg, &demo) {
            Some((lib_out, lib_bytes)) => {
                let (_, lib_snr) = aligned_snr(&pcm[..n], &lib_out);
                let lib_kbps = kbps(lib_bytes, n);
                println!(
                    "{:<14} {:>10.1} {:>8.2}dB {:>12.1} {:>7.2}dB   {:>+7.2}{note}",
                    cfg.label,
                    our_kbps,
                    our_snr,
                    lib_kbps,
                    lib_snr,
                    our_snr - lib_snr
                );
            },
            None => {
                println!(
                    "{:<14} {:>10.1} {:>8.2}dB {:>12} {:>9}   {:>8}{note}",
                    cfg.label, our_kbps, our_snr, "n/a", "n/a", "-"
                );
            },
        }
    }
}

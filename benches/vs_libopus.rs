//! Throughput comparison: pure-Rust `ruopus` vs libopus (1.6, via the `opus` FFI crate) - decode and encode,
//! in-process, on identical data.
//!
//!   cargo bench --bench vs_libopus --features std
//!
//! Reports nanoseconds per 20 ms frame, × realtime (one core), and the ratio
//! (how many times faster/slower we are than libopus). The encode comparison
//! runs libopus at complexity 0 (algorithmically closest to ours) and at its
//! default complexity 10, since that is what callers actually get.
#![allow(missing_docs)]

use std::hint::black_box;
use std::time::Instant;

use opus::{Application, Bandwidth, Bitrate, Channels};

const FS: usize = 48_000;
const FRAME: usize = 960; // 20 ms
const SECONDS: usize = 6;

/// A reproducible speech/music-like mono signal: voiced tones under a syllabic
/// envelope, plus light noise and a high partial.
fn signal() -> Vec<f32> {
    let mut seed = 0x1234_5678u32;
    (0..FRAME * 50 * SECONDS)
        .map(|i| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let n = (seed >> 9) as f32 / f32::from(u16::MAX) - 0.5;
            let t = i as f32 / FS as f32;
            let env = 0.5 + 0.45 * (2.0 * std::f32::consts::PI * 3.0 * t).sin().abs();
            env * (0.45 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                + 0.25 * (2.0 * std::f32::consts::PI * 1400.0 * t).sin()
                + 0.15 * (2.0 * std::f32::consts::PI * 6500.0 * t).sin())
                + 0.02 * n
        })
        .collect()
}

fn our_bw(bw: Bandwidth) -> ruopus::Bandwidth {
    match bw {
        Bandwidth::Narrowband => ruopus::Bandwidth::NarrowBand,
        Bandwidth::Mediumband => ruopus::Bandwidth::MediumBand,
        Bandwidth::Wideband => ruopus::Bandwidth::WideBand,
        Bandwidth::Superwideband => ruopus::Bandwidth::SuperWideBand,
        _ => ruopus::Bandwidth::FullBand,
    }
}

/// Times `f` over enough repeats to exceed ~0.4 s wall time; returns ns/frame.
fn time_per_frame(frames: usize, mut f: impl FnMut()) -> f64 {
    let mut reps = 1usize;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let el = t.elapsed().as_secs_f64();
        if el > 0.4 || reps > 1 << 20 {
            return el / (reps * frames) as f64 * 1e9;
        }
        reps = (reps * 2).max((reps as f64 * 0.45 / el.max(1e-9)) as usize + 1);
    }
}

fn xrt(ns_per_frame: f64) -> f64 {
    0.02 / (ns_per_frame / 1e9) // 20 ms of audio per frame
}

fn main() {
    let pcm = signal();
    let frames: Vec<&[f32]> = pcm.chunks_exact(FRAME).collect();
    let nframes = frames.len();
    println!(
        "ruopus vs libopus {} - {} frames ({} s audio), one core\n",
        opus::version(),
        nframes,
        SECONDS
    );

    let configs = [
        ("SILK  WB  16k", Bandwidth::Wideband, 16_000u32, Application::Voip),
        ("hybrid FB 32k", Bandwidth::Fullband, 32_000, Application::Voip),
        ("CELT  FB  64k", Bandwidth::Fullband, 64_000, Application::Audio),
    ];

    // ---- DECODE ----
    println!("DECODE                ruopus            libopus 1.6     speedup");
    for &(label, bw, br, app) in &configs {
        // Reference packets: encode the signal once with libopus.
        let mut refenc = opus::Encoder::new(FS as u32, Channels::Mono, app).unwrap();
        refenc.set_bitrate(Bitrate::Bits(br as i32)).unwrap();
        refenc.set_bandwidth(bw).unwrap();
        refenc.set_vbr(true).unwrap();
        let packets: Vec<Vec<u8>> = frames
            .iter()
            .map(|f| refenc.encode_vec_float(f, 1275).unwrap())
            .collect();

        let mut od = ruopus::OpusDecoder::new(1);
        for p in &packets {
            let _ = od.decode_packet(p);
        }
        let ours = time_per_frame(nframes, || {
            let mut od = ruopus::OpusDecoder::new(1);
            for p in &packets {
                black_box(od.decode_packet(black_box(p)).unwrap());
            }
        });

        let mut buf = vec![0.0f32; FRAME * 6];
        let theirs = time_per_frame(nframes, || {
            let mut d = opus::Decoder::new(FS as u32, Channels::Mono).unwrap();
            for p in &packets {
                let n = d.decode_float(black_box(p), &mut buf, false).unwrap();
                black_box(&buf[..n]);
            }
        });
        println!(
            "{label:<14} {:>8.0} ns {:>6.0}× {:>9.0} ns {:>6.0}× {:>6.2}×",
            ours,
            xrt(ours),
            theirs,
            xrt(theirs),
            theirs / ours
        );
    }

    // ---- ENCODE ----
    // Like-for-like: our encoder at the SAME complexity as libopus (0 and 10),
    // so each column is a fair comparison rather than our full-quality encoder
    // against libopus's stripped-down complexity-0 mode.
    println!("\nENCODE             ruopus c0 / libopus c0      ruopus c10 / libopus c10");
    for &(label, bw, br, app) in &configs {
        let bench_ours = |complexity: u8| -> f64 {
            let mut oe = ruopus::OpusEncoder::new(1);
            oe.set_bandwidth(our_bw(bw));
            oe.set_bitrate(Some(br));
            oe.set_complexity(complexity);
            for f in &frames {
                let _ = oe.encode_auto(f, 1275); // warm up
            }
            time_per_frame(nframes, || {
                let mut oe = ruopus::OpusEncoder::new(1);
                oe.set_bandwidth(our_bw(bw));
                oe.set_bitrate(Some(br));
                oe.set_complexity(complexity);
                for f in &frames {
                    black_box(oe.encode_auto(black_box(f), 1275).unwrap());
                }
            })
        };
        let bench_libopus = |complexity: i32| -> f64 {
            time_per_frame(nframes, || {
                let mut e = opus::Encoder::new(FS as u32, Channels::Mono, app).unwrap();
                e.set_bitrate(Bitrate::Bits(br as i32)).unwrap();
                e.set_bandwidth(bw).unwrap();
                e.set_vbr(true).unwrap();
                e.set_complexity(complexity).unwrap();
                for f in &frames {
                    black_box(e.encode_vec_float(black_box(f), 1275).unwrap());
                }
            })
        };
        let (o0, l0) = (bench_ours(0), bench_libopus(0));
        let (o10, l10) = (bench_ours(10), bench_libopus(10));
        println!(
            "{label:<14} {:>5.0}× / {:>5.0}× ({:.2}×)        {:>5.0}× / {:>5.0}× ({:.2}×)",
            xrt(o0),
            xrt(l0),
            l0 / o0,
            xrt(o10),
            xrt(l10),
            l10 / o10,
        );
    }
}

//! SILK-mode round-trip check: our OpusEncoder::encode_silk -> our decoder,
//! plus a .bit dump (with recorded final ranges) for opus_demo / libopus
//! interop verification (`opus_demo -d 16000 1 ours_silk.bit out.raw`).
use opus_native::packet::Bandwidth;
use opus_native::{OpusDecoder, OpusEncoder};

fn run(bw: Bandwidth, frame_ms: usize, bit_path: &str) -> usize {
    let spf = frame_ms * 48; // samples per frame at 48 kHz (mono)
    let mut enc = OpusEncoder::new(1);
    enc.set_bandwidth(bw);
    let mut dec = OpusDecoder::new(1);
    let mut bit = Vec::new();
    let mut mismatches = 0usize;
    for f in 0..100 {
        let pcm: Vec<f32> = (0..spf)
            .map(|i| {
                let t = (f * spf + i) as f32 / 48_000.0;
                0.4 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                    + 0.15 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
            })
            .collect();
        let packet = enc.encode_silk(&pcm, 1275).expect("encode_silk");
        let out = dec.decode_packet(&packet).expect("decode");
        assert_eq!(out.len(), spf);
        if dec.final_range() != enc.final_range() {
            mismatches += 1;
        }
        bit.extend_from_slice(&(packet.len() as u32).to_be_bytes());
        bit.extend_from_slice(&enc.final_range().to_be_bytes());
        bit.extend_from_slice(&packet);
    }
    std::fs::write(bit_path, &bit).unwrap();
    println!("{bw:?} {frame_ms}ms -> {bit_path}: {mismatches} range mismatches");
    mismatches
}

fn run_stereo(bw: Bandwidth, frame_ms: usize, bit_path: &str) -> usize {
    let spf = frame_ms * 48;
    let mut enc = OpusEncoder::new(2);
    enc.set_bandwidth(bw);
    let mut dec = OpusDecoder::new(2);
    let mut bit = Vec::new();
    let mut mismatches = 0usize;
    for f in 0..100 {
        let mut pcm = Vec::with_capacity(spf * 2);
        for i in 0..spf {
            let t = (f * spf + i) as f32 / 48_000.0;
            let l = 0.4 * (2.0 * std::f32::consts::PI * 200.0 * t).sin();
            let r = 0.2 * (2.0 * std::f32::consts::PI * 200.0 * t + 0.3).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 360.0 * t).sin();
            pcm.push(l);
            pcm.push(r);
        }
        let packet = enc.encode_silk(&pcm, 1275).expect("stereo encode_silk");
        let out = dec.decode_packet(&packet).expect("decode");
        assert_eq!(out.len(), spf * 2);
        if dec.final_range() != enc.final_range() {
            mismatches += 1;
        }
        bit.extend_from_slice(&(packet.len() as u32).to_be_bytes());
        bit.extend_from_slice(&enc.final_range().to_be_bytes());
        bit.extend_from_slice(&packet);
    }
    std::fs::write(bit_path, &bit).unwrap();
    println!("{bw:?} {frame_ms}ms stereo -> {bit_path}: {mismatches} range mismatches");
    mismatches
}

fn main() {
    let mut bad = 0;
    bad += run(Bandwidth::WideBand, 20, "/tmp/ours_silk_wb.bit");
    bad += run(Bandwidth::MediumBand, 20, "/tmp/ours_silk_mb.bit");
    bad += run(Bandwidth::NarrowBand, 20, "/tmp/ours_silk_nb.bit");
    bad += run(Bandwidth::WideBand, 40, "/tmp/ours_silk_wb40.bit");
    bad += run_stereo(Bandwidth::WideBand, 20, "/tmp/ours_silk_wb_st.bit");
    println!("total range mismatches: {bad}");
    assert_eq!(bad, 0, "self round-trip range mismatches");
}

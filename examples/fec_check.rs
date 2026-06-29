//! Differential FEC check against a libopus reference decode.
use opus_rs::{OpusDecoder, Packet};

fn main() {
    let bits = std::fs::read("/tmp/fec.bit").unwrap();
    let mut packets = Vec::new();
    let mut off = 0usize;
    while off + 8 <= bits.len() {
        let len = u32::from_be_bytes(bits[off..off + 4].try_into().unwrap()) as usize;
        off += 8;
        packets.push(&bits[off..off + len]);
        off += len;
    }
    let mut dec = OpusDecoder::new(2);
    let mut pcm: Vec<f32> = Vec::new();
    for i in 0..packets.len() {
        if i % 10 == 7 && i + 1 < packets.len() {
            let parsed = Packet::parse(packets[i]).unwrap();
            let dur = parsed.frames().len() * parsed.toc().frame_size().samples_per_channel_48k();
            pcm.extend(dec.decode_fec(packets[i + 1], dur).unwrap());
        } else {
            pcm.extend(dec.decode_packet(packets[i]).unwrap());
        }
    }
    let refbytes = std::fs::read("/tmp/fec_ref.f32").unwrap();
    let refpcm: Vec<f32> = refbytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    println!("ours {} ref {}", pcm.len(), refpcm.len());
    let n = pcm.len().min(refpcm.len());
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for j in 0..n {
        sig += f64::from(refpcm[j]) * f64::from(refpcm[j]);
        noise += f64::from(pcm[j] - refpcm[j]) * f64::from(pcm[j] - refpcm[j]);
    }
    println!(
        "SNR vs libopus with 10% FEC recovery: {:.1} dB",
        10.0 * (sig / noise.max(1e-30)).log10()
    );
    if let Some(first) = (0..n).find(|&j| (pcm[j] - refpcm[j]).abs() > 1e-4) {
        println!(
            "first divergence at interleaved sample {first} (packet ~{})",
            first / 1920
        );
    }
}

//! Differential decode-rate check against a libopus reference decode.
use ruopus::OpusDecoder;

fn main() {
    let vector = std::env::args().nth(1).expect("vector");
    let fs: u32 = std::env::args().nth(2).expect("rate").parse().unwrap();
    let data = std::fs::read(format!("tests/vectors/{vector}.bit")).unwrap();
    let mut packets = Vec::new();
    let mut off = 0usize;
    while off + 8 <= data.len() {
        let len = u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        off += 8;
        packets.push(&data[off..off + len]);
        off += len;
    }
    let mut dec = OpusDecoder::with_rate(fs, 2);
    let mut pcm: Vec<f32> = Vec::new();
    for pkt in &packets {
        pcm.extend(dec.decode_packet(pkt).unwrap());
    }
    let refbytes = std::fs::read("/tmp/rate_ref.f32").unwrap();
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
        "{vector} at {fs} Hz: SNR vs libopus {:.1} dB",
        10.0 * (sig / noise.max(1e-30)).log10()
    );
}

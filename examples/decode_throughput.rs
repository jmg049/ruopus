//! Measures Opus decode throughput over an `opus_demo` bitstream file:
//!
//! ```sh
//! cargo run --release --example decode_throughput tests/vectors/testvector05.bit
//! ```
//!
//! Reports decoded audio seconds per wall-clock second (× realtime).

use std::time::Instant;

use opus_native::OpusDecoder;

fn main() {
    let path = std::env::args().nth(1).expect("usage: decode_throughput <file.bit>");
    let data = std::fs::read(&path).expect("read bitstream file");

    // opus_demo framing: 4-byte BE length, 4-byte BE final range, payload.
    let mut packets = Vec::new();
    let mut off = 0usize;
    while off + 8 <= data.len() {
        let len = u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        off += 8;
        packets.push(&data[off..off + len]);
        off += len;
    }

    let mut decoder = OpusDecoder::new(2);
    let mut samples = 0u64;
    let start = Instant::now();
    for pkt in &packets {
        let pcm = decoder.decode_packet(pkt).expect("valid packet");
        samples += (pcm.len() / 2) as u64;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let audio_secs = samples as f64 / 48_000.0;
    println!(
        "{}: {} packets, {audio_secs:.1} s audio in {elapsed:.3} s - {:.0}× realtime",
        path,
        packets.len(),
        audio_secs / elapsed
    );
}

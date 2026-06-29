//! Decodes a surround Ogg Opus file, dumps the packets + layout for the
//! libopus differential harness, then compares if a reference exists.
use ruopus::ogg::{ChannelMapping, OggOpusReader};

fn main() {
    let path = std::env::args().nth(1).expect("usage: surround_check <file.opus>");
    let data = std::fs::read(&path).unwrap();

    // Dump packets + layout for the C harness.
    {
        let mut r = OggOpusReader::new(&data).unwrap();
        let head = r.head().clone();
        let ChannelMapping::Table {
            stream_count,
            coupled_count,
            mapping,
            ..
        } = &head.channel_mapping
        else {
            panic!("not a multistream file");
        };
        let mut f: Vec<u8> = Vec::new();
        f.push(head.channel_count);
        f.push(*stream_count);
        f.push(*coupled_count);
        f.extend_from_slice(mapping);
        while let Some(p) = r.next() {
            f.extend_from_slice(&(p.data.len() as u32).to_be_bytes());
            f.extend_from_slice(&p.data);
        }
        std::fs::write("/tmp/surround_pkts.bin", &f).unwrap();
        println!(
            "channels={} streams={stream_count} coupled={coupled_count} mapping={mapping:?} pre_skip={}",
            head.channel_count, head.pre_skip
        );
    }

    let (pcm, head) = ruopus::decode_ogg_opus(&data).unwrap();
    println!(
        "decoded {} samples x {} ch",
        pcm.len() / usize::from(head.channel_count),
        head.channel_count
    );

    if let Ok(refbytes) = std::fs::read("/tmp/surround_ref.f32") {
        let refpcm: Vec<f32> = refbytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // The reference decode includes pre-skip and no end trim; ours is
        // trimmed. Compare the overlap with pre-skip alignment.
        let ch = usize::from(head.channel_count);
        let skip = usize::from(head.pre_skip) * ch;
        let n = pcm.len().min(refpcm.len() - skip);
        let (mut sig, mut noise) = (0.0f64, 0.0f64);
        for j in 0..n {
            sig += f64::from(refpcm[skip + j]) * f64::from(refpcm[skip + j]);
            noise += f64::from(pcm[j] - refpcm[skip + j]) * f64::from(pcm[j] - refpcm[skip + j]);
        }
        println!(
            "SNR vs libopus multistream: {:.1} dB",
            10.0 * (sig / noise.max(1e-30)).log10()
        );
    }
}

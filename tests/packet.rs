//! Packet framing tests, including the worked examples of RFC 6716 §3.3 and
//! one test per well-formedness requirement [R1]-[R7].

use opus_rs::{Bandwidth, FrameSize, Mode, Packet, PacketError, Toc};

#[test]
fn toc_table_2_is_complete() {
    // Spot-check each row of RFC 6716 Table 2 plus the field packing
    // (bit 0 is the MSB: config in the top five bits).
    let cases: [(u8, Mode, Bandwidth, FrameSize); 12] = [
        (0, Mode::SilkOnly, Bandwidth::NarrowBand, FrameSize::Ms10),
        (3, Mode::SilkOnly, Bandwidth::NarrowBand, FrameSize::Ms60),
        (4, Mode::SilkOnly, Bandwidth::MediumBand, FrameSize::Ms10),
        (8, Mode::SilkOnly, Bandwidth::WideBand, FrameSize::Ms10),
        (11, Mode::SilkOnly, Bandwidth::WideBand, FrameSize::Ms60),
        (12, Mode::Hybrid, Bandwidth::SuperWideBand, FrameSize::Ms10),
        (13, Mode::Hybrid, Bandwidth::SuperWideBand, FrameSize::Ms20),
        (14, Mode::Hybrid, Bandwidth::FullBand, FrameSize::Ms10),
        (16, Mode::CeltOnly, Bandwidth::NarrowBand, FrameSize::Ms2_5),
        (20, Mode::CeltOnly, Bandwidth::WideBand, FrameSize::Ms2_5),
        (27, Mode::CeltOnly, Bandwidth::SuperWideBand, FrameSize::Ms20),
        (31, Mode::CeltOnly, Bandwidth::FullBand, FrameSize::Ms20),
    ];
    for (config, mode, bw, size) in cases {
        let toc = Toc::from_parts(config, false, 0);
        assert_eq!(toc.config(), config);
        assert_eq!(toc.mode(), mode, "config {config}");
        assert_eq!(toc.bandwidth(), bw, "config {config}");
        assert_eq!(toc.frame_size(), size, "config {config}");
        assert_eq!(Toc::new(toc.byte()), toc, "round-trip config {config}");
    }

    let stereo = Toc::from_parts(31, true, 3);
    assert_eq!(stereo.byte(), 0xFF);
    assert!(stereo.stereo());
    assert_eq!(stereo.channels(), 2);
    assert_eq!(stereo.frame_count_code(), 3);
}

#[test]
fn frame_size_carries_correct_durations() {
    assert_eq!(FrameSize::Ms2_5.samples_per_channel_48k(), 120);
    assert_eq!(FrameSize::Ms5.samples_per_channel_48k(), 240);
    assert_eq!(FrameSize::Ms10.samples_per_channel_48k(), 480);
    assert_eq!(FrameSize::Ms20.samples_per_channel_48k(), 960);
    assert_eq!(FrameSize::Ms40.samples_per_channel_48k(), 1920);
    assert_eq!(FrameSize::Ms60.samples_per_channel_48k(), 2880);
}

// ---- RFC 6716 §3.3 worked examples -----------------------------------------

#[test]
fn rfc_figure_8_code0_silk_nb_20ms() {
    // "Simplest case, one NB mono 20 ms SILK frame": config 1, s=0, c=0.
    let data = [0x08u8, 0xDE, 0xAD, 0xBE, 0xEF];
    let pkt = Packet::parse(&data).expect("valid");
    assert_eq!(pkt.toc().config(), 1);
    assert_eq!(pkt.toc().mode(), Mode::SilkOnly);
    assert_eq!(pkt.toc().bandwidth(), Bandwidth::NarrowBand);
    assert_eq!(pkt.toc().frame_size(), FrameSize::Ms20);
    assert!(!pkt.toc().stereo());
    assert_eq!(pkt.frames(), &[&data[1..]]);
}

#[test]
fn rfc_figure_9_code1_two_equal_celt_frames() {
    // "Two FB mono 5 ms CELT frames of the same compressed size":
    // config 29, c=1.
    let data = [0xE9u8, 1, 2, 3, 4, 5, 6];
    let pkt = Packet::parse(&data).expect("valid");
    assert_eq!(pkt.toc().config(), 29);
    assert_eq!(pkt.toc().mode(), Mode::CeltOnly);
    assert_eq!(pkt.toc().bandwidth(), Bandwidth::FullBand);
    assert_eq!(pkt.toc().frame_size(), FrameSize::Ms5);
    assert_eq!(pkt.frames(), &[&[1u8, 2, 3][..], &[4u8, 5, 6][..]]);
}

#[test]
fn rfc_figure_10_code3_vbr_two_hybrid_frames() {
    // "Two FB mono 20 ms Hybrid frames of different compressed size":
    // config 15, c=3, frame count byte v=1, p=0, M=2, then N1.
    let data = [0x7Bu8, 0x82, 3, 10, 11, 12, 20, 21];
    let pkt = Packet::parse(&data).expect("valid");
    assert_eq!(pkt.toc().config(), 15);
    assert_eq!(pkt.toc().mode(), Mode::Hybrid);
    assert_eq!(pkt.frames(), &[&[10u8, 11, 12][..], &[20u8, 21][..]]);
    assert_eq!(pkt.duration(), core::time::Duration::from_millis(40));
}

#[test]
fn rfc_figure_11_code3_cbr_four_celt_frames() {
    // "Four FB stereo 20 ms CELT frames of the same compressed size":
    // config 31, s=1, c=3, frame count byte v=0, p=0, M=4.
    let mut data = vec![0xFFu8, 0x04];
    data.extend((0..20).map(|i| i as u8));
    let pkt = Packet::parse(&data).expect("valid");
    assert_eq!(pkt.toc().config(), 31);
    assert!(pkt.toc().stereo());
    assert_eq!(pkt.frames().len(), 4);
    for (i, frame) in pkt.frames().iter().enumerate() {
        assert_eq!(frame.len(), 5, "frame {i}");
    }
    assert_eq!(pkt.duration(), core::time::Duration::from_millis(80));
}

// ---- Framing details --------------------------------------------------------

#[test]
fn code0_empty_frame_is_valid_dtx() {
    // A packet that is only a TOC byte: one zero-length frame.
    let pkt = Packet::parse(&[0x08]).expect("valid");
    assert_eq!(pkt.frames(), &[&[] as &[u8]]);
}

#[test]
fn code2_two_byte_length_coding() {
    // First length 252..=255 needs a second byte: len = b1*4 + b0.
    // 252 + 4*1 = 256 bytes for frame 1.
    let mut data = vec![0x0Au8, 252, 1];
    data.extend(core::iter::repeat_n(0xAA, 256));
    data.extend_from_slice(&[1, 2, 3]);
    let pkt = Packet::parse(&data).expect("valid");
    assert_eq!(pkt.frames()[0].len(), 256);
    assert_eq!(pkt.frames()[1], &[1, 2, 3]);
}

#[test]
fn code3_padding_chains_with_255() {
    // p=1; padding bytes 255, 1 -> 254 + 1 = 255 padding bytes after the
    // frames. M=1 VBR? Use CBR M=1: remaining = frame.
    let mut data = vec![0x03u8, 0b0100_0001, 255, 1];
    data.extend_from_slice(&[9, 9, 9, 9]); // one 4-byte frame
    data.extend(core::iter::repeat_n(0u8, 255)); // padding
    let pkt = Packet::parse(&data).expect("valid");
    assert_eq!(pkt.frames(), &[&[9u8, 9, 9, 9][..]]);
    assert_eq!(pkt.padding(), 255);
}

#[test]
fn code3_vbr_dtx_zero_length_frame() {
    // VBR with M=2: first frame has explicit length 0 (DTX), second takes
    // the remainder.
    let data = [0x0Bu8, 0x82, 0, 7, 7];
    let pkt = Packet::parse(&data).expect("valid");
    assert_eq!(pkt.frames(), &[&[] as &[u8], &[7u8, 7][..]]);
}

// ---- Malformed packets: one test per requirement ----------------------------

#[test]
fn r1_empty_packet_rejected() {
    assert_eq!(Packet::parse(&[]), Err(PacketError::Empty));
}

#[test]
fn r2_oversized_implicit_frame_rejected() {
    // Code 0 with a 1276-byte payload exceeds the 1275-byte frame limit.
    let data = vec![0x08u8; 1277];
    assert_eq!(Packet::parse(&data), Err(PacketError::FrameTooLarge));
}

#[test]
fn r3_code1_odd_payload_rejected() {
    let data = [0x09u8, 1, 2, 3];
    assert_eq!(Packet::parse(&data), Err(PacketError::Code1UnevenPayload));
}

#[test]
fn r4_code2_truncated_or_overrunning_length_rejected() {
    // A 1-byte code 2 packet is always invalid.
    assert_eq!(Packet::parse(&[0x0A]), Err(PacketError::InvalidFrameLength));
    // Two bytes with the second in 252..=255: the length field itself is cut.
    assert_eq!(Packet::parse(&[0x0A, 252]), Err(PacketError::InvalidFrameLength));
    // Length runs past the end of the packet.
    assert_eq!(Packet::parse(&[0x0A, 5, 1, 2]), Err(PacketError::InvalidFrameLength));
    // The RFC calls out that the only valid 2-byte code 2 packet has both
    // frames of length zero.
    let pkt = Packet::parse(&[0x0A, 0]).expect("valid");
    assert_eq!(pkt.frames(), &[&[] as &[u8], &[] as &[u8]]);
}

#[test]
fn r5_code3_frame_count_limits() {
    // M = 0 is invalid.
    assert_eq!(Packet::parse(&[0x0B, 0x00]), Err(PacketError::InvalidFrameCount));
    // 7 * 20 ms = 140 ms > 120 ms.
    let mut data = vec![0x0Bu8, 0x07];
    data.extend_from_slice(&[0; 14]);
    assert_eq!(Packet::parse(&data), Err(PacketError::InvalidFrameCount));
    // 48 * 2.5 ms = 120 ms is exactly legal (config 16, CELT 2.5 ms).
    let mut data = vec![0x83u8, 48];
    data.extend(core::iter::repeat_n(1u8, 48));
    let pkt = Packet::parse(&data).expect("48 x 2.5 ms is legal");
    assert_eq!(pkt.frames().len(), 48);
    // ... but 49 frames is not.
    let mut data = vec![0x83u8, 49];
    data.extend(core::iter::repeat_n(1u8, 49));
    assert_eq!(Packet::parse(&data), Err(PacketError::InvalidFrameCount));
}

#[test]
fn r6_cbr_violations_rejected() {
    // Code 3 packet missing its frame count byte.
    assert_eq!(Packet::parse(&[0x0B]), Err(PacketError::InvalidFrameLength));
    // Padding larger than the remaining packet.
    assert_eq!(Packet::parse(&[0x0B, 0x41, 200]), Err(PacketError::InvalidPadding));
    // Truncated padding length chain.
    assert_eq!(Packet::parse(&[0x0B, 0x41, 255]), Err(PacketError::InvalidPadding));
    // CBR payload (5 bytes) not divisible by M (2).
    assert_eq!(
        Packet::parse(&[0x0B, 0x02, 1, 2, 3, 4, 5]),
        Err(PacketError::CbrPayloadNotDivisible)
    );
}

#[test]
fn r7_vbr_violations_rejected() {
    // VBR M=2: explicit first-frame length missing entirely.
    assert_eq!(Packet::parse(&[0x0B, 0x82]), Err(PacketError::InvalidFrameLength));
    // VBR M=2: first-frame length exceeds the remaining bytes.
    assert_eq!(
        Packet::parse(&[0x0B, 0x82, 10, 1, 2]),
        Err(PacketError::InvalidFrameLength)
    );
}

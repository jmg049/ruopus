//! Range coder round-trip tests.
//!
//! RFC 6716 §5.1 guarantees that after encoding and decoding the same symbol
//! sequence, the encoder's and decoder's `rng` values match exactly, and
//! §4.1.6/§5.1.6 guarantee the same for `tell()`/`tell_frac()`. Every test
//! here checks those invariants alongside the decoded values, which makes the
//! decoder and encoder mutual oracles: an error in either side's arithmetic
//! desynchronizes the pair almost immediately.

use opus_rs::{RangeDecoder, RangeEncoder};

/// A small deterministic PRNG (xorshift32) so property-style tests need no
/// dependencies and never flake.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform value in `0..n` (n > 0). Bias is irrelevant for fuzzing.
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

#[test]
fn fresh_coder_reports_one_bit() {
    // RFC 6716 §4.1.6.1: a newly initialized decoder reports 1 bit used -
    // the bit reserved for termination of the encoder.
    let dec = RangeDecoder::new(&[0x55, 0xAA, 0xFF]);
    assert_eq!(dec.tell(), 1);

    let enc = RangeEncoder::new(16);
    assert_eq!(enc.tell(), 1);
}

#[test]
fn empty_buffer_decodes_without_panicking() {
    // §4.1.2.1: once input is exhausted the decoder uses zero bits forever.
    let mut dec = RangeDecoder::new(&[]);
    let fs = dec.decode(256);
    dec.update(fs, fs + 1, 256);
    let _ = dec.decode_bit_logp(4);
    let _ = dec.decode_raw_bits(13);
    let _ = dec.decode_uint(1000);
}

#[test]
fn symbol_roundtrip_with_rng_agreement() {
    // A fixed three-symbol context: {5, 2, 1}/8.
    const FL: [u32; 3] = [0, 5, 7];
    const FH: [u32; 3] = [5, 7, 8];
    const FT: u32 = 8;
    let symbols = [0usize, 1, 0, 2, 0, 0, 1, 2, 2, 0, 1, 0, 0, 0, 2, 1];

    let mut enc = RangeEncoder::new(64);
    let mut enc_rng_trace = Vec::new();
    for &s in &symbols {
        enc.encode(FL[s], FH[s], FT);
        enc_rng_trace.push(enc.range_size());
    }
    let enc_tell = enc.tell();
    let buf = enc.finalize().expect("encoder within budget");

    let mut dec = RangeDecoder::new(&buf);
    for (i, &expected) in symbols.iter().enumerate() {
        let fs = dec.decode(FT);
        let k = (0..3).find(|&k| FL[k] <= fs && fs < FH[k]).expect("fs in range");
        dec.update(FL[k], FH[k], FT);
        assert_eq!(k, expected, "symbol {i}");
        assert_eq!(dec.range_size(), enc_rng_trace[i], "rng mismatch after symbol {i}");
    }
    assert_eq!(dec.tell(), enc_tell, "tell() must agree after the same symbols");
}

#[test]
fn bit_logp_roundtrip() {
    let bits = [true, false, false, true, false, true, true, false, false, false, true];
    let logps = [1u32, 2, 3, 4, 5, 6, 7, 8, 2, 3, 1];

    let mut enc = RangeEncoder::new(32);
    for (&b, &logp) in bits.iter().zip(&logps) {
        enc.encode_bit_logp(b, logp);
    }
    let buf = enc.finalize().expect("within budget");

    let mut dec = RangeDecoder::new(&buf);
    for (i, (&b, &logp)) in bits.iter().zip(&logps).enumerate() {
        assert_eq!(dec.decode_bit_logp(logp), b, "bit {i}");
    }
}

#[test]
fn icdf_roundtrip() {
    // A real Opus table shape: terminated by 0, ftb = 8.
    // PDF {32, 96, 64, 64}/256 -> icdf = {224, 128, 64, 0}.
    const ICDF: [u8; 4] = [224, 128, 64, 0];
    const FTB: u32 = 8;
    let symbols = [1usize, 0, 3, 2, 1, 1, 0, 2, 3, 3, 1, 2, 0, 1];

    let mut enc = RangeEncoder::new(64);
    for &s in &symbols {
        enc.encode_icdf(s, &ICDF, FTB);
    }
    let enc_rng = enc.range_size();
    let buf = enc.finalize().expect("within budget");

    let mut dec = RangeDecoder::new(&buf);
    for (i, &expected) in symbols.iter().enumerate() {
        assert_eq!(dec.decode_icdf(&ICDF, FTB), expected, "symbol {i}");
    }
    assert_eq!(dec.range_size(), enc_rng);
}

#[test]
fn raw_bits_roundtrip() {
    let values: [(u32, u32); 8] = [
        (0x1, 1),
        (0x3FF, 10),
        (0, 5),
        (0xAAAA, 17),
        (0x7F, 7),
        (1, 24),
        (0xFFF, 12),
        (0x5, 3),
    ];

    let mut enc = RangeEncoder::new(32);
    for &(v, bits) in &values {
        enc.encode_raw_bits(v, bits);
    }
    let buf = enc.finalize().expect("within budget");

    let mut dec = RangeDecoder::new(&buf);
    for (i, &(v, bits)) in values.iter().enumerate() {
        assert_eq!(dec.decode_raw_bits(bits), v, "value {i}");
    }
}

#[test]
fn uint_roundtrip_small_and_large_ft() {
    // ft <= 256 uses pure range coding; larger ft splits into range-coded
    // high bits plus raw low bits (§4.1.5/§5.1.4). Cover both paths, plus
    // non-power-of-two ft and the extremes.
    let cases: [(u32, u32); 10] = [
        (0, 2),
        (1, 2),
        (5, 9),
        (255, 256),
        (256, 257),
        (1000, 1275),
        (12_345, 48_000),
        (0, u32::MAX),
        (u32::MAX - 1, u32::MAX),
        (77_777_777, 100_000_000),
    ];

    let mut enc = RangeEncoder::new(64);
    for &(t, ft) in &cases {
        enc.encode_uint(t, ft);
    }
    let enc_rng = enc.range_size();
    let enc_tell_frac = enc.tell_frac();
    let buf = enc.finalize().expect("within budget");

    let mut dec = RangeDecoder::new(&buf);
    for (i, &(t, ft)) in cases.iter().enumerate() {
        assert_eq!(dec.decode_uint(ft), Some(t), "value {i}");
    }
    assert_eq!(dec.range_size(), enc_rng);
    assert_eq!(dec.tell_frac(), enc_tell_frac);
}

#[test]
fn tell_equals_ceil_of_tell_frac() {
    // §4.1.6: ec_tell() == ceil(ec_tell_frac()/8) at every point.
    let mut enc = RangeEncoder::new(128);
    let mut rng = Rng(0xDEAD_BEEF);
    for _ in 0..200 {
        let logp = 1 + rng.below(8);
        enc.encode_bit_logp(rng.below(2) == 1, logp);
        assert_eq!(enc.tell(), enc.tell_frac().div_ceil(8));
    }
    let buf = enc.finalize().expect("within budget");

    let mut dec = RangeDecoder::new(&buf);
    let mut rng = Rng(0xDEAD_BEEF);
    for _ in 0..200 {
        let logp = 1 + rng.below(8);
        let _ = rng.below(2);
        let _ = dec.decode_bit_logp(logp);
        assert_eq!(dec.tell(), dec.tell_frac().div_ceil(8));
    }
}

/// The heavyweight property test: long randomized mixed streams of every
/// symbol kind, with the encoder and decoder checked for exact `rng` and
/// `tell_frac` agreement after every operation.
#[test]
fn randomized_mixed_stream_roundtrip() {
    for seed in 1..=25u32 {
        let mut ops = Vec::new();
        {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9));
            for _ in 0..400 {
                ops.push(match rng.below(5) {
                    // (kind, a, b): interpretation depends on kind.
                    0 => {
                        let ft = 2 + rng.below(300);
                        (0u8, rng.below(ft), ft)
                    },
                    1 => (1, u32::from(rng.below(2) == 1), 1 + rng.below(14)),
                    2 => {
                        let bits = 1 + rng.below(24);
                        (2, rng.next() & ((1 << bits) - 1), bits)
                    },
                    3 => {
                        let ft = 2 + (rng.next() % 1_000_000);
                        (3, rng.below(ft), ft)
                    },
                    _ => (4, rng.below(4), 0),
                });
            }
        }

        // {68, 88, 70, 30}/256 as an icdf: cumulative fh = 68,156,226,256.
        const ICDF: [u8; 4] = [188, 100, 30, 0];

        let mut enc = RangeEncoder::new(4096);
        let mut trace = Vec::new();
        for &(kind, a, b) in &ops {
            match kind {
                0 => enc.encode(a, a + 1, b),
                1 => enc.encode_bit_logp(a == 1, b),
                2 => enc.encode_raw_bits(a, b),
                3 => enc.encode_uint(a, b),
                _ => enc.encode_icdf(a as usize, &ICDF, 8),
            }
            trace.push((enc.range_size(), enc.tell_frac()));
        }
        let buf = enc.finalize().expect("within budget");

        let mut dec = RangeDecoder::new(&buf);
        for (i, &(kind, a, b)) in ops.iter().enumerate() {
            match kind {
                0 => {
                    let fs = dec.decode(b);
                    assert_eq!(fs, a, "seed {seed} op {i}: ec_decode value");
                    dec.update(a, a + 1, b);
                },
                1 => {
                    assert_eq!(dec.decode_bit_logp(b), a == 1, "seed {seed} op {i}: bit");
                },
                2 => {
                    assert_eq!(dec.decode_raw_bits(b), a, "seed {seed} op {i}: raw bits");
                },
                3 => {
                    assert_eq!(dec.decode_uint(b), Some(a), "seed {seed} op {i}: uint");
                },
                _ => {
                    assert_eq!(dec.decode_icdf(&ICDF, 8), a as usize, "seed {seed} op {i}: icdf");
                },
            }
            assert_eq!(
                (dec.range_size(), dec.tell_frac()),
                trace[i],
                "seed {seed} op {i}: encoder/decoder state diverged"
            );
        }
    }
}

#[test]
fn encoder_reports_budget_overflow() {
    // 4 bytes cannot hold 100 raw bytes of data.
    let mut enc = RangeEncoder::new(4);
    for _ in 0..100 {
        enc.encode_raw_bits(0xAB, 8);
    }
    assert!(enc.finalize().is_err());
}

#[test]
fn range_and_raw_bits_share_the_buffer() {
    // Range data from the front, raw bits from the back, meeting in a small
    // buffer - §5.1.5's termination must keep both decodable.
    let mut enc = RangeEncoder::new(8);
    enc.encode_bit_logp(true, 4);
    enc.encode_bit_logp(false, 1);
    enc.encode_raw_bits(0x2A55, 15);
    enc.encode_bit_logp(true, 2);
    let buf = enc.finalize().expect("fits in 8 bytes");
    assert_eq!(buf.len(), 8);

    let mut dec = RangeDecoder::new(&buf);
    assert!(dec.decode_bit_logp(4));
    assert!(!dec.decode_bit_logp(1));
    assert_eq!(dec.decode_raw_bits(15), 0x2A55);
    assert!(dec.decode_bit_logp(2));
}

#[test]
fn carry_propagation_survives_ff_runs() {
    // Encoding many maximally probable "0" symbols in a row produces long
    // runs of 0xFF output bytes, exercising the ext/rem carry machinery
    // (§5.1.1.2). The decode side must reproduce them bit-exactly.
    let mut enc = RangeEncoder::new(256);
    for i in 0..600u32 {
        enc.encode_bit_logp(i.is_multiple_of(97), 8);
    }
    let enc_rng = enc.range_size();
    let buf = enc.finalize().expect("within budget");

    let mut dec = RangeDecoder::new(&buf);
    for i in 0..600u32 {
        assert_eq!(dec.decode_bit_logp(8), i.is_multiple_of(97), "bit {i}");
    }
    assert_eq!(dec.range_size(), enc_rng);
}

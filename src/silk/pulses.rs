//! Excitation pulse decoding (RFC 6716 §4.2.7.8; normative
//! `decode_pulses.c`, `shell_coder.c`, `code_signs.c`).
//!
//! The excitation is coded per 16-sample *shell block*: a rate level picks
//! the iCDF for each block's total pulse count; each total is then split
//! recursively (16 → 8+8 → 4+4 → 2+2 → 1+1) with count-conditional iCDFs -
//! the shell decomposition; totals that saturate the count table add
//! least-significant-bit planes instead; finally each nonzero sample gets a
//! sign whose probability is conditioned on signal type, quantisation
//! offset type, and the block's pulse count.

#![allow(dead_code, reason = "consumed incrementally as the SILK decoder stages land")]

use alloc::vec;
use alloc::vec::Vec;

use crate::range::RangeDecoder;

use super::math::smulbb;
use super::tables::{
    LSB_ICDF, PULSES_PER_BLOCK_ICDF, RATE_LEVELS_ICDF, SHELL_CODE_TABLE_OFFSETS, SHELL_CODE_TABLE0, SHELL_CODE_TABLE1,
    SHELL_CODE_TABLE2, SHELL_CODE_TABLE3, SIGN_ICDF,
};

/// `SHELL_CODEC_FRAME_LENGTH`: samples per shell block.
pub(crate) const SHELL_CODEC_FRAME_LENGTH: usize = 16;
/// `LOG2_SHELL_CODEC_FRAME_LENGTH`.
const LOG2_SHELL_CODEC_FRAME_LENGTH: usize = 4;
/// `SILK_MAX_PULSES`: maximum pulses directly codable per block.
const SILK_MAX_PULSES: usize = 16;
/// `N_RATE_LEVELS`: number of rate levels (the last is the LSB-extension
/// table).
const N_RATE_LEVELS: usize = 10;

/// `decode_split`: one node of the shell decomposition.
#[inline]
fn decode_split(dec: &mut RangeDecoder, p: i32, shell_table: &[u8]) -> (i32, i32) {
    if p > 0 {
        let off = SHELL_CODE_TABLE_OFFSETS[p as usize] as usize;
        let child1 = dec.decode_icdf(&shell_table[off..], 8) as i32;
        (child1, p - child1)
    } else {
        (0, 0)
    }
}

/// `silk_shell_decoder`: decodes one 16-sample shell block of nonnegative
/// pulse amplitudes whose total is `pulses4`.
pub(crate) fn shell_decoder(pulses0: &mut [i16], dec: &mut RangeDecoder, pulses4: i32) {
    debug_assert!(pulses0.len() >= SHELL_CODEC_FRAME_LENGTH);
    let mut pulses3 = [0i32; 2];
    let mut pulses2 = [0i32; 4];
    let mut pulses1 = [0i32; 8];

    (pulses3[0], pulses3[1]) = decode_split(dec, pulses4, &SHELL_CODE_TABLE3);

    (pulses2[0], pulses2[1]) = decode_split(dec, pulses3[0], &SHELL_CODE_TABLE2);

    (pulses1[0], pulses1[1]) = decode_split(dec, pulses2[0], &SHELL_CODE_TABLE1);
    let mut leaf = |dec: &mut RangeDecoder, i: usize, p: i32| {
        let (a, b) = decode_split(dec, p, &SHELL_CODE_TABLE0);
        pulses0[i] = a as i16;
        pulses0[i + 1] = b as i16;
    };
    leaf(dec, 0, pulses1[0]);
    leaf(dec, 2, pulses1[1]);

    (pulses1[2], pulses1[3]) = decode_split(dec, pulses2[1], &SHELL_CODE_TABLE1);
    leaf(dec, 4, pulses1[2]);
    leaf(dec, 6, pulses1[3]);

    (pulses2[2], pulses2[3]) = decode_split(dec, pulses3[1], &SHELL_CODE_TABLE2);

    (pulses1[4], pulses1[5]) = decode_split(dec, pulses2[2], &SHELL_CODE_TABLE1);
    leaf(dec, 8, pulses1[4]);
    leaf(dec, 10, pulses1[5]);

    (pulses1[6], pulses1[7]) = decode_split(dec, pulses2[3], &SHELL_CODE_TABLE1);
    leaf(dec, 12, pulses1[6]);
    leaf(dec, 14, pulses1[7]);
}

/// `silk_decode_signs`: attaches signs to the nonzero decoded pulses.
///
/// `sum_pulses` per block conditions the sign probability (with LSB counts
/// folded in at bit 5+, exactly as the reference passes them).
pub(crate) fn decode_signs(
    dec: &mut RangeDecoder,
    pulses: &mut [i16],
    length: usize,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    let icdf_base = smulbb(7, quant_offset_type + (signal_type << 1)) as usize;
    let icdf_ptr = &SIGN_ICDF[icdf_base..];
    let n_blocks = (length + SHELL_CODEC_FRAME_LENGTH / 2) >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    let mut icdf = [0u8; 2];
    for (i, block) in pulses.chunks_mut(SHELL_CODEC_FRAME_LENGTH).take(n_blocks).enumerate() {
        let p = sum_pulses[i];
        if p > 0 {
            icdf[0] = icdf_ptr[((p & 0x1f) as usize).min(6)];
            for q in block.iter_mut() {
                if *q > 0 {
                    // dec_map: 0 → -1, 1 → +1.
                    *q *= (2 * dec.decode_icdf(&icdf, 8) as i16) - 1;
                }
            }
        }
    }
}

/// `silk_decode_pulses`: decodes the excitation for one SILK frame of
/// `frame_length` samples - rate level, per-block pulse counts, shell
/// decomposition, LSB planes, and signs.
///
/// Returns the signed excitation, padded to a whole number of shell blocks
/// (the caller uses the first `frame_length` samples).
pub(crate) fn decode_pulses(
    dec: &mut RangeDecoder,
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: usize,
) -> Vec<i16> {
    // Decode rate level.
    let rate_level_index = dec.decode_icdf(&RATE_LEVELS_ICDF[(signal_type >> 1) as usize], 8);

    // Number of shell blocks (rounded up only for 10 ms @ 12 kHz).
    let mut iter = frame_length >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    if iter * SHELL_CODEC_FRAME_LENGTH < frame_length {
        debug_assert_eq!(frame_length, 12 * 10);
        iter += 1;
    }

    // Sum-weighted pulse counts per block, with LSB extension.
    let mut sum_pulses = vec![0i32; iter];
    let mut n_lshifts = vec![0i32; iter];
    let cdf = &PULSES_PER_BLOCK_ICDF[rate_level_index];
    for i in 0..iter {
        sum_pulses[i] = dec.decode_icdf(cdf, 8) as i32;
        while sum_pulses[i] == (SILK_MAX_PULSES + 1) as i32 {
            n_lshifts[i] += 1;
            // With 10 LSB planes already, shift the table to exclude
            // another (SILK_MAX_PULSES + 1).
            let table = &PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1];
            let table = if n_lshifts[i] == 10 { &table[1..] } else { &table[..] };
            sum_pulses[i] = dec.decode_icdf(table, 8) as i32;
        }
    }

    // Shell decoding.
    let mut pulses = vec![0i16; iter * SHELL_CODEC_FRAME_LENGTH];
    for i in 0..iter {
        if sum_pulses[i] > 0 {
            shell_decoder(&mut pulses[i * SHELL_CODEC_FRAME_LENGTH..], dec, sum_pulses[i]);
        }
    }

    // LSB decoding.
    for i in 0..iter {
        if n_lshifts[i] > 0 {
            let n_ls = n_lshifts[i];
            let block = &mut pulses[i * SHELL_CODEC_FRAME_LENGTH..(i + 1) * SHELL_CODEC_FRAME_LENGTH];
            for q in block.iter_mut() {
                let mut abs_q = i32::from(*q);
                for _ in 0..n_ls {
                    abs_q = (abs_q << 1) + dec.decode_icdf(&LSB_ICDF, 8) as i32;
                }
                *q = abs_q as i16;
            }
            // Mark the pulse count nonzero for sign decoding.
            sum_pulses[i] |= n_ls << 5;
        }
    }

    // Signs.
    decode_signs(
        dec,
        &mut pulses,
        frame_length,
        signal_type,
        quant_offset_type,
        &sum_pulses,
    );
    pulses
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::range::{RangeDecoder, RangeEncoder};

    use super::super::tables::{
        SHELL_CODE_TABLE_OFFSETS, SHELL_CODE_TABLE0, SHELL_CODE_TABLE1, SHELL_CODE_TABLE2, SHELL_CODE_TABLE3, SIGN_ICDF,
    };
    use super::*;

    /// `silk_shell_encoder` (shell_coder.c), ported for round-trip testing.
    fn shell_encoder(enc: &mut RangeEncoder, pulses0: &[i32; 16]) {
        fn combine(input: &[i32]) -> Vec<i32> {
            input.chunks_exact(2).map(|p| p[0] + p[1]).collect()
        }
        fn encode_split(enc: &mut RangeEncoder, child1: i32, p: i32, table: &[u8]) {
            if p > 0 {
                let off = SHELL_CODE_TABLE_OFFSETS[p as usize] as usize;
                enc.encode_icdf(child1 as usize, &table[off..], 8);
            }
        }
        let pulses1 = combine(pulses0);
        let pulses2 = combine(&pulses1);
        let pulses3 = combine(&pulses2);
        let pulses4 = combine(&pulses3);

        encode_split(enc, pulses3[0], pulses4[0], &SHELL_CODE_TABLE3);

        encode_split(enc, pulses2[0], pulses3[0], &SHELL_CODE_TABLE2);
        encode_split(enc, pulses1[0], pulses2[0], &SHELL_CODE_TABLE1);
        encode_split(enc, pulses0[0], pulses1[0], &SHELL_CODE_TABLE0);
        encode_split(enc, pulses0[2], pulses1[1], &SHELL_CODE_TABLE0);
        encode_split(enc, pulses1[2], pulses2[1], &SHELL_CODE_TABLE1);
        encode_split(enc, pulses0[4], pulses1[2], &SHELL_CODE_TABLE0);
        encode_split(enc, pulses0[6], pulses1[3], &SHELL_CODE_TABLE0);

        encode_split(enc, pulses2[2], pulses3[1], &SHELL_CODE_TABLE2);
        encode_split(enc, pulses1[4], pulses2[2], &SHELL_CODE_TABLE1);
        encode_split(enc, pulses0[8], pulses1[4], &SHELL_CODE_TABLE0);
        encode_split(enc, pulses0[10], pulses1[5], &SHELL_CODE_TABLE0);
        encode_split(enc, pulses1[6], pulses2[3], &SHELL_CODE_TABLE1);
        encode_split(enc, pulses0[12], pulses1[6], &SHELL_CODE_TABLE0);
        encode_split(enc, pulses0[14], pulses1[7], &SHELL_CODE_TABLE0);
    }

    /// `silk_encode_signs` (code_signs.c), ported for round-trip testing.
    fn encode_signs(
        enc: &mut RangeEncoder,
        pulses: &[i32],
        length: usize,
        signal_type: i32,
        quant_offset_type: i32,
        sum_pulses: &[i32],
    ) {
        let icdf_base = (7 * (quant_offset_type + (signal_type << 1))) as usize;
        let icdf_ptr = &SIGN_ICDF[icdf_base..];
        let n_blocks = (length + SHELL_CODEC_FRAME_LENGTH / 2) >> LOG2_SHELL_CODEC_FRAME_LENGTH;
        let mut icdf = [0u8; 2];
        for (i, block) in pulses.chunks(SHELL_CODEC_FRAME_LENGTH).take(n_blocks).enumerate() {
            if sum_pulses[i] > 0 {
                icdf[0] = icdf_ptr[((sum_pulses[i] & 0x1f) as usize).min(6)];
                for &q in block {
                    if q != 0 {
                        enc.encode_icdf(((q >> 31) + 1) as usize, &icdf, 8);
                    }
                }
            }
        }
    }

    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *seed
    }

    /// Shell blocks of every total 0..=16, plus signed round trips through
    /// the sign coder, mirroring the reference call sequence.
    #[test]
    fn shell_and_sign_round_trip() {
        let mut seed = 0x5eed_u32;
        for total in 0..=16i32 {
            for st in 0..3i32 {
                for qot in 0..2i32 {
                    // Random nonnegative split of `total` over 16 samples.
                    let mut amp = [0i32; 16];
                    for _ in 0..total {
                        amp[(lcg(&mut seed) % 16) as usize] += 1;
                    }
                    // Random signs on nonzero amplitudes.
                    let signed: Vec<i32> = amp
                        .iter()
                        .map(|&a| if a > 0 && lcg(&mut seed) & 1 == 1 { -a } else { a })
                        .collect();

                    let mut enc = RangeEncoder::new(256);
                    shell_encoder(&mut enc, &amp);
                    encode_signs(&mut enc, &signed, 16, st, qot, &[total]);
                    let bytes = enc.finalize().expect("encode fits");

                    let mut dec = RangeDecoder::new(&bytes);
                    let mut got = [0i16; 16];
                    if total > 0 {
                        shell_decoder(&mut got, &mut dec, total);
                    }
                    let amps: Vec<i32> = got.iter().map(|&v| i32::from(v)).collect();
                    assert_eq!(amps, amp, "amplitudes (total={total})");
                    decode_signs(&mut dec, &mut got, 16, st, qot, &[total]);
                    let vals: Vec<i32> = got.iter().map(|&v| i32::from(v)).collect();
                    assert_eq!(vals, signed, "signs (total={total} st={st} qot={qot})");
                }
            }
        }
    }
}

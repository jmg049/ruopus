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

use crate::range::{RangeDecoder, RangeEncoder};

use super::math::smulbb;
use super::tables::{
    LSB_ICDF, MAX_PULSES_TABLE, PULSES_PER_BLOCK_BITS_Q5, PULSES_PER_BLOCK_ICDF, RATE_LEVELS_BITS_Q5, RATE_LEVELS_ICDF,
    SHELL_CODE_TABLE_OFFSETS, SHELL_CODE_TABLE0, SHELL_CODE_TABLE1, SHELL_CODE_TABLE2, SHELL_CODE_TABLE3, SIGN_ICDF,
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

/// `combine_and_check`: pairwise-sum `pulses_in` into `pulses_comb`,
/// returning `true` if any combined count exceeds `max_pulses`.
fn combine_and_check(pulses_comb: &mut [i32], pulses_in: &[i32], max_pulses: i32, len: usize) -> bool {
    let mut scale_down = false;
    for k in 0..len {
        let sum = pulses_in[2 * k] + pulses_in[2 * k + 1];
        if sum > max_pulses {
            scale_down = true;
        }
        pulses_comb[k] = sum;
    }
    scale_down
}

/// `silk_shell_encoder` (shell_coder.c): code one 16-sample block's pulse
/// magnitudes by recursive binary splits, top down.
fn shell_encoder(enc: &mut RangeEncoder, pulses0: &[i32]) {
    // Build the four levels of the pulse binary tree on the stack (mirrors
    // libopus's silk_shell_encoder fixed arrays). The previous `collect()` form
    // allocated four Vecs per 16-sample block - ~80 heap allocations per frame.
    fn combine(out: &mut [i32], input: &[i32]) {
        for (o, p) in out.iter_mut().zip(input.chunks_exact(2)) {
            *o = p[0] + p[1];
        }
    }
    fn encode_split(enc: &mut RangeEncoder, child1: i32, p: i32, table: &[u8]) {
        if p > 0 {
            let off = SHELL_CODE_TABLE_OFFSETS[p as usize] as usize;
            enc.encode_icdf(child1 as usize, &table[off..], 8);
        }
    }
    let mut pulses1 = [0i32; 8];
    let mut pulses2 = [0i32; 4];
    let mut pulses3 = [0i32; 2];
    let mut pulses4 = [0i32; 1];
    combine(&mut pulses1, pulses0);
    combine(&mut pulses2, &pulses1);
    combine(&mut pulses3, &pulses2);
    combine(&mut pulses4, &pulses3);

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

/// `silk_encode_signs` (code_signs.c): code the sign of each nonzero pulse,
/// with the probability conditioned on signal type, quantisation offset and
/// the block's pulse count.
fn encode_signs(
    enc: &mut RangeEncoder,
    pulses: &[i8],
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
                    enc.encode_icdf(((i32::from(q) >> 31) + 1) as usize, &icdf, 8);
                }
            }
        }
    }
}

/// `silk_encode_pulses`: code the excitation `pulses` (signed quantisation
/// indices, `frame_length` of them) - rate level, per-block pulse counts
/// (with LSB extension when a block saturates), the shell magnitudes, the
/// extra LSB planes, and the signs. The exact inverse of [`decode_pulses`].
pub(crate) fn encode_pulses(
    enc: &mut RangeEncoder,
    signal_type: i32,
    quant_offset_type: i32,
    pulses: &[i8],
    frame_length: usize,
) {
    let mut iter = frame_length >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    if iter * SHELL_CODEC_FRAME_LENGTH < frame_length {
        debug_assert_eq!(frame_length, 12 * 10);
        iter += 1;
    }
    let padded = iter * SHELL_CODEC_FRAME_LENGTH;

    // Zero-pad the pulses to a whole number of shell blocks (the reference
    // memsets the tail of the caller's buffer).
    let mut spulses = vec![0i8; padded];
    spulses[..frame_length].copy_from_slice(pulses);
    let pulses = &spulses[..];

    // Absolute values.
    let mut abs_pulses = vec![0i32; padded];
    for (a, &p) in abs_pulses.iter_mut().zip(pulses.iter()) {
        *a = i32::from(p.unsigned_abs());
    }

    // Per-block sum of pulses, downscaling (LSB planes) until it fits.
    let mut sum_pulses = vec![0i32; iter];
    let mut n_rshifts = vec![0i32; iter];
    for i in 0..iter {
        let block = &mut abs_pulses[i * SHELL_CODEC_FRAME_LENGTH..(i + 1) * SHELL_CODEC_FRAME_LENGTH];
        loop {
            let mut comb = [0i32; 8];
            let mut scale_down = combine_and_check(&mut comb, block, i32::from(MAX_PULSES_TABLE[0]), 8);
            let mut comb4 = [0i32; 4];
            scale_down |= combine_and_check(&mut comb4, &comb, i32::from(MAX_PULSES_TABLE[1]), 4);
            let mut comb2 = [0i32; 2];
            scale_down |= combine_and_check(&mut comb2, &comb4, i32::from(MAX_PULSES_TABLE[2]), 2);
            let mut comb1 = [0i32; 1];
            scale_down |= combine_and_check(&mut comb1, &comb2, i32::from(MAX_PULSES_TABLE[3]), 1);
            if scale_down {
                n_rshifts[i] += 1;
                for q in block.iter_mut() {
                    *q >>= 1;
                }
            } else {
                sum_pulses[i] = comb1[0];
                break;
            }
        }
    }

    // Rate level minimising the per-block count bits.
    let mut min_bits = i32::MAX;
    let mut rate_level = 0usize;
    for k in 0..N_RATE_LEVELS - 1 {
        let nbits = &PULSES_PER_BLOCK_BITS_Q5[k];
        let mut sum_bits = i32::from(RATE_LEVELS_BITS_Q5[(signal_type >> 1) as usize][k]);
        for i in 0..iter {
            sum_bits += i32::from(if n_rshifts[i] > 0 {
                nbits[SILK_MAX_PULSES + 1]
            } else {
                nbits[sum_pulses[i] as usize]
            });
        }
        if sum_bits < min_bits {
            min_bits = sum_bits;
            rate_level = k;
        }
    }
    enc.encode_icdf(rate_level, &RATE_LEVELS_ICDF[(signal_type >> 1) as usize], 8);

    // Per-block pulse counts (with the LSB-extension escapes).
    let cdf = &PULSES_PER_BLOCK_ICDF[rate_level];
    let last = &PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1];
    for i in 0..iter {
        if n_rshifts[i] == 0 {
            enc.encode_icdf(sum_pulses[i] as usize, cdf, 8);
        } else {
            enc.encode_icdf(SILK_MAX_PULSES + 1, cdf, 8);
            for _ in 0..n_rshifts[i] - 1 {
                enc.encode_icdf(SILK_MAX_PULSES + 1, last, 8);
            }
            enc.encode_icdf(sum_pulses[i] as usize, last, 8);
        }
    }

    // Shell magnitudes.
    for i in 0..iter {
        if sum_pulses[i] > 0 {
            shell_encoder(enc, &abs_pulses[i * SHELL_CODEC_FRAME_LENGTH..]);
        }
    }

    // Extra LSB planes (most significant first).
    for i in 0..iter {
        if n_rshifts[i] > 0 {
            let n_ls = n_rshifts[i] - 1;
            let block = &pulses[i * SHELL_CODEC_FRAME_LENGTH..];
            for &p in block.iter().take(SHELL_CODEC_FRAME_LENGTH) {
                let abs_q = i32::from(p.unsigned_abs());
                for j in (1..=n_ls).rev() {
                    enc.encode_icdf(((abs_q >> j) & 1) as usize, &LSB_ICDF, 8);
                }
                enc.encode_icdf((abs_q & 1) as usize, &LSB_ICDF, 8);
            }
            sum_pulses[i] |= n_rshifts[i] << 5;
        }
    }

    encode_signs(enc, pulses, frame_length, signal_type, quant_offset_type, &sum_pulses);
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::range::{RangeDecoder, RangeEncoder};

    use super::*;

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
                    // Random signs on nonzero amplitudes (as i8 pulses).
                    let signed: Vec<i8> = amp
                        .iter()
                        .map(|&a| {
                            if a > 0 && lcg(&mut seed) & 1 == 1 {
                                -a as i8
                            } else {
                                a as i8
                            }
                        })
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
                    let vals: Vec<i8> = got.iter().map(|&v| v as i8).collect();
                    assert_eq!(vals, signed, "signs (total={total} st={st} qot={qot})");
                }
            }
        }
    }

    /// Full excitation round trip: `encode_pulses` → `decode_pulses`
    /// reproduces the pulses, across frame lengths, signal/offset types, and
    /// magnitudes large enough to exercise the LSB-extension planes.
    #[test]
    fn encode_pulses_round_trips() {
        let mut seed = 0x1234_5678_u32;
        for &frame_length in &[80usize, 160, 320, 120] {
            for st in 0..3i32 {
                for qot in 0..2i32 {
                    for &max_amp in &[1i32, 3, 8, 40] {
                        let mut pulses = vec![0i8; frame_length];
                        for p in &mut pulses {
                            // Sparse excitation: mostly zero, occasional pulses.
                            if lcg(&mut seed) % 4 == 0 {
                                let a = (lcg(&mut seed) as i32 % (max_amp + 1)) as i8;
                                *p = if lcg(&mut seed) & 1 == 1 { -a } else { a };
                            }
                        }

                        let mut enc = RangeEncoder::new(512);
                        encode_pulses(&mut enc, st, qot, &pulses, frame_length);
                        let bytes = enc.finalize().expect("encode fits");

                        let mut dec = RangeDecoder::new(&bytes);
                        let got = decode_pulses(&mut dec, st, qot, frame_length);
                        let got_i8: Vec<i8> = got.iter().take(frame_length).map(|&v| v as i8).collect();
                        assert_eq!(
                            got_i8, pulses,
                            "pulses (len={frame_length} st={st} qot={qot} max={max_amp})"
                        );
                    }
                }
            }
        }
    }
}

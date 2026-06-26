//! Energy envelope decoding (RFC 6716 §4.3.2).
//!
//! CELT transmits each band's energy in base-2 log units ("DB" in the
//! reference) in up to three refinement stages:
//!
//! 1. **Coarse** (§4.3.2.1): 6 dB resolution, predicted in time (from the previous frame, unless the frame is *intra*)
//!    and in frequency (from the previous band), with the prediction error Laplace-coded.
//! 2. **Fine** (§4.3.2.2): per-band refinement bits assigned by the bit allocation, read as raw bits.
//! 3. **Finalise**: any bits left over at the very end of the frame add one extra half-resolution bit per band per
//!    channel, by priority.
//!
//! Energies persist across frames in [`EnergyState`] - the time-domain
//! predictor's memory. The float arithmetic here mirrors the reference float
//! build (the test vectors accept either build's output).

use super::laplace::ec_laplace_decode;
use super::modes::{BETA_COEF, BETA_INTRA, E_PROB_MODEL, MAX_FINE_BITS, NB_EBANDS, PRED_COEF};
use crate::range::RangeDecoder;

/// ICDF for the 2-bit fallback coarse-energy code (`small_energy_icdf`).
const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

/// Per-channel band energies carried across frames (`oldEBands`).
///
/// Indexed `[channel][band]`; units are base-2 log energy. Fresh state is
/// all zeros (matching the reference decoder's initialization).
#[derive(Debug, Clone, Default)]
pub struct EnergyState {
    /// Log energies from the previous frame, the time-domain predictor input.
    pub old_ebands: [[f32; NB_EBANDS]; 2],
}

/// Decodes the coarse energy for bands `start..end` (RFC 6716 §4.3.2.1,
/// `unquant_coarse_energy`).
///
/// `budget_bits` is the frame size in bits (`storage * 8`): when fewer than
/// 15 bits remain the coder falls back to cheaper codes, and below 1 bit the
/// prediction runs on a fixed -1 delta. `lm` is the frame-size exponent
/// (frame duration `120 << lm` samples), `intra` disables the time-domain
/// predictor.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the reference unquant_coarse_energy signature"
)]
pub fn decode_coarse_energy(
    dec: &mut RangeDecoder,
    state: &mut EnergyState,
    start: usize,
    end: usize,
    intra: bool,
    channels: usize,
    lm: usize,
    budget_bits: u32,
) {
    debug_assert!(end <= NB_EBANDS && start <= end);
    debug_assert!(channels == 1 || channels == 2);
    debug_assert!(lm < 4);

    let prob_model = &E_PROB_MODEL[lm][usize::from(intra)];
    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm], BETA_COEF[lm])
    };
    let budget = i64::from(budget_bits);

    let mut prev = [0.0f32; 2];
    for i in start..end {
        for (c, prev_c) in prev.iter_mut().enumerate().take(channels) {
            let tell = i64::from(dec.tell());
            let qi: i32 = if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                ec_laplace_decode(dec, u32::from(prob_model[pi]) << 7, u32::from(prob_model[pi + 1]) << 6)
            } else if budget - tell >= 2 {
                let q = dec.decode_icdf(&SMALL_ENERGY_ICDF, 2) as i32;
                // {0, 1, 2} -> {0, -1, +1}.
                (q >> 1) ^ -(q & 1)
            } else if budget - tell >= 1 {
                -i32::from(dec.decode_bit_logp(1))
            } else {
                -1
            };

            let q = qi as f32;
            let old = state.old_ebands[c][i].max(-9.0);
            state.old_ebands[c][i] = coef * old + *prev_c + q;
            *prev_c += q - beta * q;
        }
    }
}

/// Decodes the fine energy refinement (RFC 6716 §4.3.2.2,
/// `unquant_fine_energy`). `fine_quant[i]` is the bit count B_i from the
/// allocation; the refinement maps to `(f + 1/2) / 2^B_i - 1/2`.
pub fn decode_fine_energy(
    dec: &mut RangeDecoder,
    state: &mut EnergyState,
    start: usize,
    end: usize,
    fine_quant: &[i32],
    channels: usize,
) {
    for (i, &bits) in fine_quant.iter().enumerate().take(end).skip(start) {
        if bits <= 0 {
            continue;
        }
        for c in 0..channels {
            let q2 = dec.decode_raw_bits(bits as u32) as f32;
            let offset = (q2 + 0.5) / (1 << bits) as f32 - 0.5;
            state.old_ebands[c][i] += offset;
        }
    }
}

/// Spends the bits left at the end of the frame on extra half-resolution
/// energy refinements (RFC 6716 §4.3.2.2, `unquant_energy_finalise`).
///
/// Bands of priority 0 receive one extra bit per channel first (in band
/// order), then priority 1; whatever then remains is discarded.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the reference unquant_energy_finalise signature"
)]
pub fn decode_energy_finalise(
    dec: &mut RangeDecoder,
    state: &mut EnergyState,
    start: usize,
    end: usize,
    fine_quant: &[i32],
    fine_priority: &[bool],
    mut bits_left: i32,
    channels: usize,
) {
    for prio in [false, true] {
        let mut i = start;
        while i < end && bits_left >= channels as i32 {
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                i += 1;
                continue;
            }
            for c in 0..channels {
                let q2 = dec.decode_raw_bits(1) as f32;
                let offset = (q2 - 0.5) / (1 << (fine_quant[i] + 1)) as f32;
                state.old_ebands[c][i] += offset;
                bits_left -= 1;
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::celt::laplace::ec_laplace_encode;
    use crate::range::{RangeDecoder, RangeEncoder};

    /// Mirrors the decoder's prediction filter to compute expected energies
    /// from a known delta sequence.
    #[allow(clippy::needless_range_loop, reason = "indices mirror the decoder loop under test")]
    fn predict(deltas: &[i32], lm: usize, intra: bool, channels: usize) -> Vec<[f32; NB_EBANDS]> {
        let (coef, beta) = if intra {
            (0.0, BETA_INTRA)
        } else {
            (PRED_COEF[lm], BETA_COEF[lm])
        };
        let mut out = vec![[0.0f32; NB_EBANDS]; channels];
        let mut prev = [0.0f32; 2];
        let mut it = deltas.iter();
        for i in 0..NB_EBANDS {
            for (c, prev_c) in prev.iter_mut().enumerate().take(channels) {
                let q = *it.next().expect("enough deltas") as f32;
                let old = out[c][i].max(-9.0);
                out[c][i] = coef * old + *prev_c + q;
                *prev_c += q - beta * q;
            }
        }
        out
    }

    /// Encodes a delta sequence exactly as `quant_coarse_energy` does in its
    /// fully budgeted path, then checks the decoder reproduces the energies.
    #[test]
    #[allow(clippy::needless_range_loop, reason = "indices mirror the decoder loop under test")]
    fn coarse_energy_round_trip() {
        for lm in 0..4usize {
            for intra in [false, true] {
                for channels in [1usize, 2] {
                    let prob = &E_PROB_MODEL[lm][usize::from(intra)];
                    // A deterministic, varied delta pattern.
                    let deltas: Vec<i32> = (0..NB_EBANDS * channels).map(|i| ((i as i32 * 5) % 7) - 3).collect();

                    let mut enc = RangeEncoder::new(256);
                    let mut coded = Vec::new();
                    let mut it = deltas.iter();
                    for i in 0..NB_EBANDS {
                        for _ in 0..channels {
                            let pi = 2 * i.min(20);
                            coded.push(ec_laplace_encode(
                                &mut enc,
                                *it.next().expect("delta"),
                                u32::from(prob[pi]) << 7,
                                u32::from(prob[pi + 1]) << 6,
                            ));
                        }
                    }
                    assert_eq!(coded, deltas, "small deltas never saturate");
                    let enc_rng = enc.range_size();
                    let buf = enc.finalize().expect("within budget");

                    let mut dec = RangeDecoder::new(&buf);
                    let mut state = EnergyState::default();
                    decode_coarse_energy(
                        &mut dec,
                        &mut state,
                        0,
                        NB_EBANDS,
                        intra,
                        channels,
                        lm,
                        buf.len() as u32 * 8,
                    );
                    assert_eq!(dec.range_size(), enc_rng, "lm={lm} intra={intra} C={channels}");

                    let expected = predict(&deltas, lm, intra, channels);
                    for c in 0..channels {
                        for i in 0..NB_EBANDS {
                            assert!(
                                (state.old_ebands[c][i] - expected[c][i]).abs() < 1e-5,
                                "lm={lm} intra={intra} c={c} band {i}: {} vs {}",
                                state.old_ebands[c][i],
                                expected[c][i]
                            );
                        }
                    }
                }
            }
        }
    }

    /// With a starved budget the decoder must take the cheap fallback paths
    /// and never read past the budget by more than a symbol's bound.
    #[test]
    fn coarse_energy_respects_starved_budget() {
        // 2 bytes = 16 bits: after one or two Laplace symbols the budget
        // drops below 15 and the fallbacks engage; below 1 bit the delta is
        // pinned to -1 without reading at all.
        let buf = [0xA5u8, 0x3C];
        let mut dec = RangeDecoder::new(&buf);
        let mut state = EnergyState::default();
        decode_coarse_energy(&mut dec, &mut state, 0, NB_EBANDS, false, 2, 3, 16);
        // All 42 band/channel slots were filled deterministically.
        for c in 0..2 {
            for i in 0..NB_EBANDS {
                assert!(state.old_ebands[c][i].is_finite());
            }
        }
    }

    #[test]
    fn fine_energy_refines_toward_centre() {
        // One band, 3 fine bits: q2 = 5 -> offset (5.5/8 - 0.5) = 0.1875.
        let mut enc = RangeEncoder::new(16);
        enc.encode_raw_bits(5, 3);
        let buf = enc.finalize().expect("fits");

        let mut dec = RangeDecoder::new(&buf);
        let mut state = EnergyState::default();
        let mut fine_quant = vec![0i32; NB_EBANDS];
        fine_quant[0] = 3;
        decode_fine_energy(&mut dec, &mut state, 0, 1, &fine_quant, 1);
        assert!((state.old_ebands[0][0] - 0.1875).abs() < 1e-6);
    }

    #[test]
    fn finalise_spends_priority_zero_first() {
        // Bands 0 and 1 have priority 1 and 0 respectively; with exactly one
        // bit left, band 1 (priority 0) gets it.
        let mut enc = RangeEncoder::new(16);
        enc.encode_raw_bits(1, 1); // the single finalise bit: +offset
        let buf = enc.finalize().expect("fits");

        let mut dec = RangeDecoder::new(&buf);
        let mut state = EnergyState::default();
        let fine_quant = vec![0i32; NB_EBANDS];
        let mut fine_priority = vec![true; NB_EBANDS];
        fine_priority[1] = false;
        decode_energy_finalise(&mut dec, &mut state, 0, 2, &fine_quant, &fine_priority, 1, 1);

        assert_eq!(state.old_ebands[0][0], 0.0, "priority-1 band untouched");
        assert!((state.old_ebands[0][1] - 0.25).abs() < 1e-6, "q2=1 at B=0: (1-0.5)/2");
    }
}

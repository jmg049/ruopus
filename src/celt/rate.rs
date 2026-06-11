//! Bit allocation (RFC 6716 §4.3.3; normative `rate.c`).
//!
//! CELT never transmits the allocation: encoder and decoder each derive the
//! identical per-band budget from nothing but the frame size in bits, the
//! coded band range, and a handful of explicitly coded parameters (boosts,
//! trim, skips, stereo flags). The derivation, in order:
//!
//! 1. Interpolate between the static quality vectors
//!    ([`super::tables::BAND_ALLOCATION`]) to find the highest quality that
//!    fits `total`, then bisect 1/64th steps between it and the next.
//! 2. Working from the top band down, decide which bands are *skipped*
//!    (their bits redistributed downward) - one explicitly coded bit per
//!    skippable band.
//! 3. Code the intensity-stereo band boundary and dual-stereo flag (stereo
//!    frames only).
//! 4. Split each band's budget between fine energy
//!    ([`Allocation::fine_quant`]) and PVQ shape bits
//!    ([`Allocation::shape_bits`]), with capping/excess rebalancing.
//!
//! All quantities are in 1/8-bit units (`BITRES = 3`) unless noted.
//!
//! The pulse cache ([`super::tables::CACHE_BITS`]) maps between bit budgets
//! and PVQ pulse counts per band size: [`bits2pulses`]/[`pulses2bits`] with
//! the pseudo-pulse scale of [`get_pulses`].

use crate::range::RangeDecoder;

use super::modes::{EBANDS, LOG_N, MAX_FINE_BITS, NB_EBANDS};
use super::tables::{BAND_ALLOCATION, CACHE_BITS, CACHE_CAPS, CACHE_INDEX, LOG2_FRAC_TABLE, NB_ALLOC_VECTORS};

/// Fractional-bit resolution: all budgets are in 1/8 bit units.
pub const BITRES: u32 = 3;

/// Bisection steps between adjacent allocation quality vectors.
const ALLOC_STEPS: i32 = 6;

/// Fine-energy offset constant (`FINE_OFFSET`).
const FINE_OFFSET: i32 = 21;

/// Number of pseudo-pulse levels (`LOG_MAX_PSEUDO` bisection steps).
const LOG_MAX_PSEUDO: u32 = 6;

/// Converts a pseudo-pulse index to an actual pulse count (`get_pulses`):
/// exact up to 8, then 8 values per octave.
#[must_use]
pub const fn get_pulses(i: i32) -> i32 {
    if i < 8 { i } else { (8 + (i & 7)) << ((i >> 3) - 1) }
}

/// The pulse cache row for `(band, lm)`: entry 0 is the highest pseudo-pulse
/// level, entry `k` the cost of `get_pulses(k)` pulses in 1/8 bits minus one.
///
/// `lm` may be -1 (a fully time-split partition), selecting cache row 0.
fn cache_row(band: usize, lm: i32) -> &'static [u8] {
    let idx = CACHE_INDEX[(lm + 1) as usize * NB_EBANDS + band];
    debug_assert!(idx >= 0, "single-bin bands have no PVQ cache");
    &CACHE_BITS[idx as usize..]
}

/// The largest leaf budget the pulse cache can represent for `(band, lm)`,
/// in 1/8 bits (`cache[cache[0]]` in the reference's split condition).
#[must_use]
pub(crate) fn cache_max_bits(band: usize, lm: i32) -> i32 {
    let cache = cache_row(band, lm);
    i32::from(cache[usize::from(cache[0])])
}

/// Largest pseudo-pulse count whose cost fits in `bits` (1/8 bit units),
/// rounded to the nearest cost (`bits2pulses`).
#[must_use]
pub fn bits2pulses(band: usize, lm: i32, bits: i32) -> i32 {
    let cache = cache_row(band, lm);
    let mut lo = 0i32;
    let mut hi = i32::from(cache[0]);
    let bits = bits - 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid = (lo + hi + 1) >> 1;
        if i32::from(cache[mid as usize]) >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_cost = if lo == 0 { -1 } else { i32::from(cache[lo as usize]) };
    if bits - lo_cost <= i32::from(cache[hi as usize]) - bits {
        lo
    } else {
        hi
    }
}

/// Cost in 1/8 bits of `pulses` pseudo-pulses in `(band, lm)`
/// (`pulses2bits`).
#[must_use]
pub fn pulses2bits(band: usize, lm: i32, pulses: i32) -> i32 {
    if pulses == 0 {
        0
    } else {
        i32::from(cache_row(band, lm)[pulses as usize]) + 1
    }
}

/// Maximum allocation per band in 1/8 bits (`init_caps`, celt.c).
#[must_use]
pub fn init_caps(lm: usize, channels: usize) -> [i32; NB_EBANDS] {
    let mut cap = [0i32; NB_EBANDS];
    for (i, cap_i) in cap.iter_mut().enumerate() {
        let n = i32::from(EBANDS[i + 1] - EBANDS[i]) << lm;
        *cap_i = ((i32::from(CACHE_CAPS[NB_EBANDS * (2 * lm + channels - 1) + i]) + 64) * channels as i32 * n) >> 2;
    }
    cap
}

/// The result of the bit allocation for one frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allocation {
    /// Bands `start..coded_bands` receive shape bits; the rest are skipped.
    pub coded_bands: usize,
    /// PVQ shape budget per band, in 1/8 bits (`pulses` in the reference).
    pub shape_bits: [i32; NB_EBANDS],
    /// Fine energy bits per band per channel (`ebits`).
    pub fine_quant: [i32; NB_EBANDS],
    /// Whether each band is first (false) or second (true) priority for the
    /// final fine-energy bits.
    pub fine_priority: [bool; NB_EBANDS],
    /// Unspent shape bits carried into the band loop's rebalancing.
    pub balance: i32,
    /// First band coded in intensity stereo (`== coded_bands` disables it).
    pub intensity: usize,
    /// Whether mid/side coding is replaced by dual (independent) stereo.
    pub dual_stereo: bool,
}

/// Computes the allocation, reading the skip/intensity/dual-stereo decisions
/// from the bitstream (RFC 6716 §4.3.3, decoder side of
/// `compute_allocation`).
///
/// `offsets` are the per-band boosts in 1/8 bits decoded by the caller
/// (dynalloc), `alloc_trim` the decoded trim parameter (0..=10, default 5),
/// `total` the bits available for the remainder of the frame in 1/8 bits.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the reference compute_allocation signature"
)]
#[must_use]
pub fn compute_allocation(
    dec: &mut RangeDecoder,
    start: usize,
    end: usize,
    offsets: &[i32; NB_EBANDS],
    cap: &[i32; NB_EBANDS],
    alloc_trim: i32,
    total: i32,
    channels: usize,
    lm: usize,
) -> Allocation {
    let c = channels as i32;
    let len = NB_EBANDS;
    let mut total = total.max(0);
    let mut skip_start = start;

    // Reserve a bit to signal the end of manually skipped bands.
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;

    // Reserve bits for the intensity and dual-stereo parameters.
    let mut intensity_rsv = 0i32;
    let mut dual_stereo_rsv = 0i32;
    if channels == 2 {
        intensity_rsv = i32::from(LOG2_FRAC_TABLE[end - start]);
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }

    let mut thresh = [0i32; NB_EBANDS];
    let mut trim_offset = [0i32; NB_EBANDS];
    for j in start..end {
        let width = i32::from(EBANDS[j + 1] - EBANDS[j]);
        // Below this threshold no PVQ bits are allocated.
        thresh[j] = (c << BITRES).max((((3 * width) << lm) << BITRES) >> 4);
        // Tilt of the allocation curve.
        trim_offset[j] =
            (c * width * (alloc_trim - 5 - lm as i32) * (end as i32 - j as i32 - 1) * (1 << (lm as u32 + BITRES))) >> 6;
        // Single-coefficient bands get less resolution.
        if width << lm == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }

    // Find the highest static quality vector that fits.
    let mut lo = 1usize;
    let mut hi = NB_ALLOC_VECTORS - 1;
    loop {
        let mut done = false;
        let mut psum = 0i32;
        let mid = (lo + hi) >> 1;
        for j in (start..end).rev() {
            let n = i32::from(EBANDS[j + 1] - EBANDS[j]);
            let mut bitsj = ((c * n * i32::from(BAND_ALLOCATION[mid * len + j])) << lm) >> 2;
            if bitsj > 0 {
                bitsj = (bitsj + trim_offset[j]).max(0);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                psum += bitsj.min(cap[j]);
            } else if bitsj >= c << BITRES {
                psum += c << BITRES;
            }
        }
        if psum > total {
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
        if lo > hi {
            break;
        }
    }
    let q_hi = lo;
    let q_lo = lo - 1;

    // Per-band budgets at the two bracketing qualities.
    let mut bits1 = [0i32; NB_EBANDS];
    let mut bits2 = [0i32; NB_EBANDS];
    for j in start..end {
        let n = i32::from(EBANDS[j + 1] - EBANDS[j]);
        let mut bits1j = ((c * n * i32::from(BAND_ALLOCATION[q_lo * len + j])) << lm) >> 2;
        let mut bits2j = if q_hi >= NB_ALLOC_VECTORS {
            cap[j]
        } else {
            ((c * n * i32::from(BAND_ALLOCATION[q_hi * len + j])) << lm) >> 2
        };
        if bits1j > 0 {
            bits1j = (bits1j + trim_offset[j]).max(0);
        }
        if bits2j > 0 {
            bits2j = (bits2j + trim_offset[j]).max(0);
        }
        if q_lo > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2j = (bits2j - bits1j).max(0);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }

    interp_bits2pulses(
        dec,
        InterpParams {
            start,
            end,
            skip_start,
            bits1,
            bits2,
            thresh,
            cap: *cap,
            total,
            skip_rsv,
            intensity_rsv,
            dual_stereo_rsv,
            channels,
            lm,
        },
    )
}

/// Bundled inputs of [`interp_bits2pulses`], to keep the signature sane.
struct InterpParams {
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: [i32; NB_EBANDS],
    bits2: [i32; NB_EBANDS],
    thresh: [i32; NB_EBANDS],
    cap: [i32; NB_EBANDS],
    total: i32,
    skip_rsv: i32,
    intensity_rsv: i32,
    dual_stereo_rsv: i32,
    channels: usize,
    lm: usize,
}

/// The second half of the allocation (`interp_bits2pulses`): bisects between
/// the two bracketing quality vectors, decodes band skips and the stereo
/// parameters, then splits each band's budget into fine-energy and shape
/// bits.
fn interp_bits2pulses(dec: &mut RangeDecoder, p: InterpParams) -> Allocation {
    let InterpParams {
        start,
        end,
        skip_start,
        bits1,
        bits2,
        thresh,
        cap,
        mut total,
        skip_rsv,
        mut intensity_rsv,
        mut dual_stereo_rsv,
        channels,
        lm,
    } = p;
    let c = channels as i32;
    let stereo = u32::from(channels > 1);
    let alloc_floor = c << BITRES;
    let log_m = (lm as i32) << BITRES;

    let mut bits = [0i32; NB_EBANDS];
    let mut ebits = [0i32; NB_EBANDS];
    let mut fine_priority = [false; NB_EBANDS];

    // Bisect the interpolation fraction in 1/64 steps.
    let mut lo = 0i32;
    let mut hi = 1 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid * bits2[j]) >> ALLOC_STEPS);
            if tmp >= thresh[j] || done {
                done = true;
                psum += tmp.min(cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let mut psum = 0i32;
    let mut done = false;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo * bits2[j]) >> ALLOC_STEPS);
        if tmp < thresh[j] && !done {
            tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
        } else {
            done = true;
        }
        let tmp = tmp.min(cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    // Decide which bands to skip, working backwards from the end.
    let mut coded_bands = end;
    let coded_bands = loop {
        debug_assert!(coded_bands > start);
        let j = coded_bands - 1;
        // Never skip the first band nor a dynalloc-boosted band.
        if j <= skip_start {
            // Give the reserved skip bit back.
            total += skip_rsv;
            break coded_bands;
        }
        // Bits that redistribution would hand to this band.
        let left = total - psum;
        let width_all = i32::from(EBANDS[coded_bands] - EBANDS[start]);
        let percoeff = left / width_all;
        let left = left - width_all * percoeff;
        let rem = (left - i32::from(EBANDS[j] - EBANDS[start])).max(0);
        let band_width = i32::from(EBANDS[coded_bands] - EBANDS[j]);
        let mut band_bits = bits[j] + percoeff * band_width + rem;

        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            // The skip decision is explicitly coded.
            if dec.decode_bit_logp(1) {
                break coded_bands;
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        // Reclaim this band's bits; re-size the intensity parameter.
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = i32::from(LOG2_FRAC_TABLE[j - start]);
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            // Enough for one fine-energy bit per channel.
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    };

    // Intensity and dual-stereo parameters.
    let intensity = if intensity_rsv > 0 {
        start + dec.decode_uint((coded_bands + 1 - start) as u32).unwrap_or(0) as usize
    } else {
        0
    };
    if intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    let dual_stereo = if dual_stereo_rsv > 0 {
        dec.decode_bit_logp(1)
    } else {
        false
    };

    // Distribute the remaining bits over the coded bands.
    let left = total - psum;
    let width_all = i32::from(EBANDS[coded_bands] - EBANDS[start]);
    let percoeff = left / width_all;
    let mut left = left - width_all * percoeff;
    for j in start..coded_bands {
        bits[j] += percoeff * i32::from(EBANDS[j + 1] - EBANDS[j]);
    }
    for j in start..coded_bands {
        let tmp = left.min(i32::from(EBANDS[j + 1] - EBANDS[j]));
        bits[j] += tmp;
        left -= tmp;
    }

    // Split each band between fine energy and shape bits.
    let mut balance = 0i32;
    for j in start..coded_bands {
        debug_assert!(bits[j] >= 0);
        let n0 = i32::from(EBANDS[j + 1] - EBANDS[j]);
        let n = n0 << lm;
        let bit = bits[j] + balance;
        let excess;

        if n > 1 {
            excess = (bit - cap[j]).max(0);
            bits[j] = bit - excess;

            // Stereo mid/side adds one degree of freedom.
            let den = c * n + i32::from(channels == 2 && n > 2 && !dual_stereo && j < intensity);
            let nclog_n = den * (i32::from(LOG_N[j]) + log_m);

            // Offset fine bits by log2(N)/2 + FINE_OFFSET from their share.
            let mut offset = (nclog_n >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += (den << BITRES) >> 2;
            }
            // Bias the second and third fine bits.
            if bits[j] + offset < (den * 2) << BITRES {
                offset += nclog_n >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nclog_n >> 3;
            }

            // Divide with rounding.
            ebits[j] = ((bits[j] + offset + (den << (BITRES - 1))) / (den << BITRES)).max(0);
            // Never bust the band's own budget.
            if c * ebits[j] > bits[j] >> BITRES {
                ebits[j] = (bits[j] >> stereo) >> BITRES;
            }
            ebits[j] = ebits[j].min(MAX_FINE_BITS);
            // Rounded-down or capped bands get final-bit priority.
            fine_priority[j] = ebits[j] * (den << BITRES) >= bits[j] + offset;
            // The rest is the PVQ shape budget.
            bits[j] -= (c * ebits[j]) << BITRES;
        } else {
            // Single-bin bands: everything but one sign bit goes to fine.
            excess = (bit - (c << BITRES)).max(0);
            bits[j] = bit - excess;
            ebits[j] = 0;
            fine_priority[j] = true;
        }

        // Re-balance excess over the cap into extra fine bits here.
        if excess > 0 {
            let extra_fine = (excess >> (stereo + BITRES)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = (extra_fine * c) << BITRES;
            fine_priority[j] = extra_bits >= excess - balance;
            balance = excess - extra_bits;
        } else {
            balance = 0;
        }
        debug_assert!(bits[j] >= 0);
        debug_assert!(ebits[j] >= 0);
    }

    // Skipped bands spend all their bits on fine energy.
    for j in coded_bands..end {
        ebits[j] = (bits[j] >> stereo) >> BITRES;
        debug_assert!((c * ebits[j]) << BITRES == bits[j]);
        bits[j] = 0;
        fine_priority[j] = ebits[j] < 1;
    }

    Allocation {
        coded_bands,
        shape_bits: bits,
        fine_quant: ebits,
        fine_priority,
        balance,
        intensity,
        dual_stereo,
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use super::*;
    use crate::range::{RangeDecoder, RangeEncoder};

    #[test]
    fn get_pulses_matches_pseudo_scale() {
        for i in 0..8 {
            assert_eq!(get_pulses(i), i);
        }
        assert_eq!(get_pulses(8), 8);
        assert_eq!(get_pulses(15), 15);
        assert_eq!(get_pulses(16), 16);
        assert_eq!(get_pulses(17), 18);
        assert_eq!(get_pulses(24), 32);
        assert_eq!(get_pulses(32), 64);
        assert_eq!(get_pulses(39), 120);
        assert_eq!(get_pulses(40), 128, "MAX_PSEUDO maps to MAX_PULSES");
    }

    #[test]
    fn pulse_cache_round_trips() {
        // For every multi-bin band and LM: pulses2bits and bits2pulses must
        // agree on every representable pseudo-pulse level.
        for lm in 0..4i32 {
            for band in 0..NB_EBANDS {
                if CACHE_INDEX[(lm + 1) as usize * NB_EBANDS + band] < 0 {
                    continue;
                }
                let levels = i32::from(cache_row(band, lm)[0]);
                let mut prev_bits = 0;
                for pseudo in 1..=levels {
                    let bits = pulses2bits(band, lm, pseudo);
                    assert!(bits >= prev_bits, "cache must be non-decreasing");
                    prev_bits = bits;
                    // bits2pulses quantizes a budget to a level; with equal-
                    // cost plateaus in the cache it returns the lowest level
                    // of the plateau. The codec-relevant invariant is cost
                    // agreement, not exact level identity.
                    let back = bits2pulses(band, lm, bits);
                    assert!(back <= pseudo, "band {band} lm {lm} pseudo {pseudo}");
                    assert_eq!(
                        pulses2bits(band, lm, back),
                        bits,
                        "band {band} lm {lm} pseudo {pseudo}: same cost"
                    );
                }
            }
        }
    }

    #[test]
    fn caps_are_positive_and_scale_with_size() {
        for lm in 0..4 {
            for channels in [1, 2] {
                let caps = init_caps(lm, channels);
                for (i, &cap) in caps.iter().enumerate() {
                    assert!(cap > 0, "lm={lm} C={channels} band {i}");
                }
            }
        }
        // Wider bands and more channels can hold more.
        assert!(init_caps(3, 2)[20] > init_caps(0, 1)[20]);
    }

    /// Drives the decoder-side allocation with synthetic skip/stereo
    /// decisions and checks the structural invariants the band loop relies
    /// on: budgets non-negative, fine bits bounded, no band above its cap,
    /// and conservation against the granted total.
    #[test]
    fn allocation_invariants_across_rates_and_frames() {
        for lm in 0..4usize {
            for channels in [1usize, 2] {
                for &total_bits in &[80i32, 300, 1200, 4000, 12_000] {
                    // The bitstream the allocator reads: make every explicit
                    // decision a "don't skip"/zero so the path is deterministic.
                    let mut enc = RangeEncoder::new(64);
                    for _ in 0..NB_EBANDS {
                        enc.encode_bit_logp(false, 1);
                    }
                    let buf = enc.finalize().expect("fits");
                    let mut dec = RangeDecoder::new(&buf);

                    let cap = init_caps(lm, channels);
                    let offsets = [0i32; NB_EBANDS];
                    let total = total_bits << BITRES;
                    let alloc = compute_allocation(&mut dec, 0, NB_EBANDS, &offsets, &cap, 5, total, channels, lm);

                    assert!(alloc.coded_bands > 0 && alloc.coded_bands <= NB_EBANDS);
                    let mut spent = alloc.balance;
                    for (j, &cap_j) in cap.iter().enumerate() {
                        assert!(alloc.shape_bits[j] >= 0, "band {j}");
                        assert!(
                            (0..=MAX_FINE_BITS).contains(&alloc.fine_quant[j]),
                            "band {j} fine {}",
                            alloc.fine_quant[j]
                        );
                        assert!(
                            alloc.shape_bits[j] <= cap_j,
                            "band {j}: {} over cap {}",
                            alloc.shape_bits[j],
                            cap_j
                        );
                        spent += alloc.shape_bits[j] + ((channels as i32 * alloc.fine_quant[j]) << BITRES);
                    }
                    assert!(spent <= total, "lm={lm} C={channels} total={total}: spent {spent}");
                }
            }
        }
    }

    /// An explicitly coded skip must reduce the number of coded bands, and
    /// the intensity/dual-stereo parameters must round-trip.
    #[test]
    fn skip_and_stereo_parameters_decode() {
        let lm = 3usize;
        let channels = 2usize;
        let cap = init_caps(lm, channels);
        let offsets = [0i32; NB_EBANDS];
        let total = 2000i32 << BITRES;

        // First: skip immediately (top band), intensity = 7, dual = true.
        let mut enc = RangeEncoder::new(64);
        enc.encode_bit_logp(true, 1); // skip the top band, ending skipping
        // intensity in 0..=coded_bands-start, coded as uint.
        enc.encode_uint(7, (NB_EBANDS - 1 + 1) as u32);
        enc.encode_bit_logp(true, 1); // dual stereo
        let buf = enc.finalize().expect("fits");
        let mut dec = RangeDecoder::new(&buf);

        let alloc = compute_allocation(&mut dec, 0, NB_EBANDS, &offsets, &cap, 5, total, channels, lm);
        assert_eq!(alloc.coded_bands, NB_EBANDS - 1, "one band skipped");
        assert_eq!(alloc.intensity, 7);
        assert!(alloc.dual_stereo);
    }
}

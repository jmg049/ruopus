//! PVQ codeword enumeration - "coding with replacement and signs"
//! (RFC 6716 §4.3.4.2).
//!
//! CELT codes each band's normalized shape as an N-dimensional vector of K
//! signed unit pulses, transmitted as a single uniformly distributed integer
//! index in `0..V(N, K)`, where
//!
//! - `V(N, K)` is the number of N-dimensional pulse vectors with K pulses (signs included), and
//! - `U(N, K) = (V(N-1, K-1) + V(N, K-1)) / 2`, the enumeration's working function, symmetric in its arguments and
//!   obeying `U(N, K) = U(N-1, K) + U(N, K-1) + U(N-1, K-1)`.
//!
//! This uses a **table-driven** fast path: the `U` rows live in a precomputed flat table
//! (`CELT_PVQ_U_DATA`, 1272 `u32`, generated once from the recurrence on first
//! use), so decoding/encoding a band is O(N) table lookups instead of the O(NK)
//! on-the-fly recurrence - the dominant cost of CELT band decode. The table
//! exploits the symmetry `U(n,k) = U(k,n)`; the row index `min(n,k)` is ≤ 14
//! for every `(n,k)` CELT's bit allocation produces.

use crate::range::{RangeDecoder, RangeEncoder};

/// Offsets of each `U` row in [`pvq_u_data`] (`CELT_PVQ_U_ROW`, 1272-entry
/// non-custom build). Row `m` holds `U(m, c)`; `PVQ_U_ROW[15] = 1272` is the
/// total length.
const PVQ_U_ROW: [usize; 16] = [
    0, 176, 351, 525, 698, 870, 1041, 1131, 1178, 1207, 1226, 1240, 1248, 1254, 1257, 1272,
];

/// The flat table, built once from the recurrence
/// `U(a,b) = U(a-1,b) + U(a,b-1) + U(a-1,b-1)` (base `U(0,0)=1`). Row `m`'s
/// data starts at column `m`: `data[PVQ_U_ROW[m] + c] = U(m, c)` for `c ≥ m`
/// (so `PVQ_U_ROW[m]` is the raw start minus `m`). Entries past where `U` fits
/// 32 bits are never accessed (their wrap is harmless).
const PVQ_U_DATA_LEN: usize = PVQ_U_ROW[15];

/// The `U(n, k)` table, computed at compile time from the recurrence (so there
/// is no runtime initialisation and no `std`/allocator dependency).
const fn compute_pvq_u_data() -> [u32; PVQ_U_DATA_LEN] {
    // U(a, b) for a ≤ 14, b ≤ 176 (the widest standard band).
    let mut uu = [[0u32; 177]; 15];
    uu[0][0] = 1; // base: U(0,0)=1, U(0,b>0)=U(a>0,0)=0; recurrence only for a,b≥1.
    let mut a = 1;
    while a < 15 {
        let mut b = 1;
        while b < 177 {
            uu[a][b] = uu[a - 1][b].wrapping_add(uu[a][b - 1]).wrapping_add(uu[a - 1][b - 1]);
            b += 1;
        }
        a += 1;
    }
    let mut d = [0u32; PVQ_U_DATA_LEN];
    let mut m = 0;
    while m < 15 {
        let start = PVQ_U_ROW[m] + m; // raw start of row m
        let end = if m < 14 {
            PVQ_U_ROW[m + 1] + (m + 1)
        } else {
            PVQ_U_ROW[15]
        };
        let mut p = start;
        while p < end {
            d[p] = uu[m][p - PVQ_U_ROW[m]];
            p += 1;
        }
        m += 1;
    }
    d
}

static PVQ_U_DATA: [u32; PVQ_U_DATA_LEN] = compute_pvq_u_data();

fn pvq_u_data() -> &'static [u32] {
    &PVQ_U_DATA
}

/// `U(n, k)` (`CELT_PVQ_U`) via the symmetric table (`data = pvq_u_data()`,
/// fetched once by the caller to keep it out of the inner loop).
#[inline]
fn celt_pvq_u(data: &[u32], n: usize, k: usize) -> u32 {
    let (lo, hi) = if n < k { (n, k) } else { (k, n) };
    data[PVQ_U_ROW[lo] + hi]
}

/// `U(row, col)` directly (`CELT_PVQ_U_ROW[row][col]`); `row` is the symmetry
/// min (≤ 14) at every call site, matching the reference.
#[inline]
fn urow(data: &[u32], row: usize, col: usize) -> u32 {
    data[PVQ_U_ROW[row] + col]
}

/// Decodes codeword index `i` into the pulse vector `y` (length `n`, `k`
/// pulses), via the `U` table.
fn cwrsi(n0: usize, k0: usize, mut i: u32, y: &mut [i32]) {
    debug_assert!(n0 > 1 && k0 > 0);
    let data = pvq_u_data();
    let mut n = n0;
    let mut k = k0;
    let mut yi = 0usize;
    while n > 2 {
        if k >= n {
            // Lots of pulses: row index is `n` (≤ 14 here).
            let p = urow(data, n, k + 1);
            let s = -i32::from(i >= p);
            if s != 0 {
                i -= p;
            }
            let k0_dim = k;
            let q = urow(data, n, n);
            let p = if q > i {
                k = n;
                loop {
                    k -= 1;
                    let p = urow(data, k, n);
                    if p <= i {
                        break p;
                    }
                }
            } else {
                let mut p = urow(data, n, k);
                while p > i {
                    k -= 1;
                    p = urow(data, n, k);
                }
                p
            };
            i -= p;
            let val = (k0_dim - k) as i32;
            y[yi] = (val + s) ^ s;
            yi += 1;
        } else {
            // Lots of dimensions: row index is `k` (≤ 14 here).
            let p = urow(data, k, n);
            let q = urow(data, k + 1, n);
            if p <= i && i < q {
                i -= p;
                y[yi] = 0;
            } else {
                let s = -i32::from(i >= q);
                if s != 0 {
                    i -= q;
                }
                let k0_dim = k;
                let p = loop {
                    k -= 1;
                    let p = urow(data, k, n);
                    if p <= i {
                        break p;
                    }
                };
                i -= p;
                let val = (k0_dim - k) as i32;
                y[yi] = (val + s) ^ s;
            }
            yi += 1;
        }
        n -= 1;
    }
    // n == 2.
    let p = 2 * k as u32 + 1;
    let s = -i32::from(i >= p);
    if s != 0 {
        i -= p;
    }
    let k0_dim = k;
    k = ((i + 1) >> 1) as usize;
    if k != 0 {
        i -= 2 * k as u32 - 1;
    }
    let val = (k0_dim - k) as i32;
    y[yi] = (val + s) ^ s;
    yi += 1;
    // n == 1.
    let s = -(i as i32);
    y[yi] = (k as i32 + s) ^ s;
}

/// Computes the codeword index of pulse vector `y` (`icwrs()`, table-based).
fn icwrs(y: &[i32]) -> u32 {
    let n = y.len();
    debug_assert!(n >= 2);
    let data = pvq_u_data();
    let mut j = n - 1;
    let mut i = u32::from(y[j] < 0);
    let mut k = y[j].unsigned_abs() as usize;
    loop {
        j -= 1;
        i += celt_pvq_u(data, n - j, k);
        k += y[j].unsigned_abs() as usize;
        if y[j] < 0 {
            i += celt_pvq_u(data, n - j, k + 1);
        }
        if j == 0 {
            break;
        }
    }
    i
}

/// The size of the PVQ codebook: the number of N-dimensional vectors of K
/// signed pulses, `V(N, K) = U(N, K) + U(N, K + 1)`.
///
/// Requires `n >= 2` and `k >= 1`; the result must fit in 32 bits, which
/// holds for every (N, K) pair CELT's bit allocation can produce.
#[must_use]
pub fn pvq_codebook_size(n: usize, k: usize) -> u32 {
    let data = pvq_u_data();
    celt_pvq_u(data, n, k) + celt_pvq_u(data, n, k + 1)
}

/// Decodes K signed unit pulses into `y` (RFC 6716 §4.3.4.2,
/// `decode_pulses()`).
///
/// `y.len()` is the band size N (≥ 2); `k` ≥ 1. Returns `None` when the
/// uniformly coded index is out of range, which indicates frame corruption.
#[must_use]
pub fn decode_pulses(dec: &mut RangeDecoder, y: &mut [i32], k: usize) -> Option<()> {
    debug_assert!(y.len() >= 2 && k >= 1);
    let v = pvq_codebook_size(y.len(), k);
    let i = dec.decode_uint(v)?;
    cwrsi(y.len(), k, i, y);
    Some(())
}

/// Encodes the pulse vector `y` (sum of magnitudes K ≥ 1, length N ≥ 2);
/// mirror of [`decode_pulses`] (`encode_pulses()`).
pub fn encode_pulses(enc: &mut RangeEncoder, y: &[i32], k: usize) {
    debug_assert!(y.len() >= 2 && k >= 1);
    enc.encode_uint(icwrs(y), pvq_codebook_size(y.len(), k));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// V(N, K) for N, K < 10.
    const V_TABLE: [[u32; 10]; 10] = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [1, 4, 8, 12, 16, 20, 24, 28, 32, 36],
        [1, 6, 18, 38, 66, 102, 146, 198, 258, 326],
        [1, 8, 32, 88, 192, 360, 608, 952, 1408, 1992],
        [1, 10, 50, 170, 450, 1002, 1970, 3530, 5890, 9290],
        [1, 12, 72, 292, 912, 2364, 5336, 10836, 20256, 35436],
        [1, 14, 98, 462, 1666, 4942, 12642, 28814, 59906, 115598],
        [1, 16, 128, 688, 2816, 9424, 27008, 68464, 157184, 332688],
        [1, 18, 162, 978, 4482, 16722, 53154, 148626, 374274, 864146],
    ];

    #[test]
    fn codebook_sizes_match_reference_table() {
        for (n, row) in V_TABLE.iter().enumerate().skip(2) {
            for (k, &expected) in row.iter().enumerate().skip(1) {
                assert_eq!(pvq_codebook_size(n, k), expected, "V({n}, {k})");
            }
        }
    }

    /// The enumeration is a bijection: every index in `0..V(N, K)` decodes to
    /// a distinct vector with exactly K pulses, and re-encodes to itself.
    #[test]
    fn exhaustive_index_bijection_small_nk() {
        for n in 2..=6usize {
            for k in 1..=6usize {
                let v = pvq_codebook_size(n, k);
                for i in 0..v {
                    let mut y = alloc::vec![0i32; n];
                    cwrsi(n, k, i, &mut y);

                    let pulses: u32 = y.iter().map(|x| x.unsigned_abs()).sum();
                    assert_eq!(pulses, k as u32, "N={n} K={k} i={i}: pulse count");

                    assert_eq!(icwrs(&y), i, "N={n} K={k}: index round-trip");
                }
            }
        }
    }

    /// Pulse vectors survive an actual range coder round trip, and the
    /// encoder/decoder `rng` states agree afterwards.
    #[test]
    fn range_coder_round_trip() {
        // A deterministic spread of shapes, including larger N and K. All
        // (N, K) pairs keep V(N, K) within 32 bits, the invariant CELT's bit
        // allocation guarantees (V(24, 10) or V(96, 6) would overflow - the
        // allocation can never produce those).
        let cases: [(usize, usize); 6] = [(2, 1), (4, 3), (8, 8), (16, 4), (24, 5), (96, 3)];

        let mut enc = RangeEncoder::new(1024);
        let mut vectors = alloc::vec::Vec::new();
        for &(n, k) in &cases {
            // Deterministic pulse pattern: alternate signs, spread across dims.
            let mut y = vec![0i32; n];
            for p in 0..k {
                let at = (p * 7) % n;
                y[at] += if p % 2 == 0 { 1 } else { -1 };
            }
            // Fix up: ensure the sum of magnitudes is exactly k (collisions of
            // opposite sign would cancel; regenerate deterministically).
            let total: u32 = y.iter().map(|x| x.unsigned_abs()).sum();
            if total != k as u32 {
                y = vec![0i32; n];
                for p in 0..k {
                    y[p % n] += 1;
                }
            }
            encode_pulses(&mut enc, &y, k);
            vectors.push((n, k, y));
        }
        let enc_rng = enc.range_size();
        let buf = enc.finalize().expect("within budget");

        let mut dec = RangeDecoder::new(&buf);
        for (n, k, expected) in vectors {
            let mut y = vec![0i32; n];
            decode_pulses(&mut dec, &mut y, k).expect("in range");
            assert_eq!(y, expected, "N={n} K={k}");
        }
        assert_eq!(dec.range_size(), enc_rng);
    }

    extern crate alloc;
}

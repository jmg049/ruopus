//! Static data of the standard CELT mode (48 kHz, 960-sample MDCT with 2.5,
//! 5, 10, and 20 ms frames) - RFC 6716 Table 55 and the normative
//! `modes.c`/`static_modes_*.h`.
//!
//! Opus uses exactly one CELT mode; the "custom modes" of the reference
//! implementation are not part of RFC 6716 and are not represented here.

/// Number of energy bands.
pub const NB_EBANDS: usize = 21;

/// Band boundaries in MDCT bins at the shortest (2.5 ms, 120-sample) frame
/// size; band `i` spans bins `EBANDS[i] << LM .. EBANDS[i+1] << LM` at frame
/// size `120 << LM`. The top of band 20 (bin 100) is the highest coded bin;
/// the remaining spectrum is not transmitted.
pub const EBANDS: [i16; NB_EBANDS + 1] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// Mean log2 band energy removed before coarse coding and restored after
/// (`eMeans`, quantized in Q4 and converted back to float).
pub const E_MEANS: [f32; NB_EBANDS] = [
    6.437_5, 6.25, 5.75, 5.312_5, 5.062_5, 4.812_5, 4.5, 4.375, 4.875, 4.687_5, 4.562_5, 4.437_5, 4.875, 4.625,
    4.312_5, 4.5, 4.375, 4.625, 4.75, 4.437_5, 3.75,
];

/// `log2(EBANDS[i+1] - EBANDS[i])` in 1/8-bit (BITRES) units at the shortest
/// frame size (`logN400`); used by the bit allocation and spreading decisions.
pub const LOG_N: [i16; NB_EBANDS] = [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 21, 21, 24, 29, 34, 36];

/// Time-domain prediction coefficients for inter-frame coarse energy, one per
/// LM (frame size 120 << LM): 0.9, 0.8, 0.65, 0.5 in Q15.
pub const PRED_COEF: [f32; 4] = [
    29440.0 / 32768.0,
    26112.0 / 32768.0,
    21248.0 / 32768.0,
    16384.0 / 32768.0,
];

/// Frequency-domain prediction feedback for inter frames, one per LM (Q15).
pub const BETA_COEF: [f32; 4] = [
    30147.0 / 32768.0,
    22282.0 / 32768.0,
    12124.0 / 32768.0,
    6554.0 / 32768.0,
];

/// Frequency-domain prediction feedback for intra frames (Q15: 4915/32768).
pub const BETA_INTRA: f32 = 4915.0 / 32768.0;

/// Maximum number of fine energy bits per band (`MAX_FINE_BITS`, rate.h).
pub const MAX_FINE_BITS: i32 = 8;

/// Laplace probability model for coarse energy deltas
/// (`e_prob_model[LM][intra]`): per band, the probability of zero and the
/// decay rate, both in Q8.
pub const E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    // 120-sample frames (LM 0).
    [
        [
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128, 64, 128, 92, 78, 92, 79, 92, 78, 90, 79,
            116, 41, 115, 40, 114, 40, 132, 26, 132, 26, 145, 17, 161, 12, 176, 10, 177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132, 55, 132, 61, 114, 70, 96, 74, 88, 75, 88,
            87, 74, 89, 66, 91, 67, 100, 59, 108, 50, 120, 40, 122, 37, 97, 43, 78, 50,
        ],
    ],
    // 240-sample frames (LM 1).
    [
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74, 93, 74, 109, 40, 114, 36, 117, 34, 117, 34, 143,
            17, 145, 18, 146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190, 8, 177, 9,
        ],
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91, 73, 91, 78, 89, 86, 80, 92, 66, 93, 64, 102, 59,
            103, 60, 104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97, 38, 77, 45,
        ],
    ],
    // 480-sample frames (LM 2).
    [
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38, 112, 38, 124, 26, 132, 27, 136, 19, 140, 20,
            155, 14, 159, 16, 158, 18, 170, 13, 177, 10, 187, 8, 192, 6, 175, 9, 159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73, 87, 72, 92, 75, 98, 72, 105, 58, 107, 54, 115,
            52, 114, 55, 112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98, 35, 77, 42,
        ],
    ],
    // 960-sample frames (LM 3).
    [
        [
            42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36, 119, 33, 127, 33, 134, 34, 139, 21, 147, 23,
            152, 20, 158, 25, 154, 26, 166, 21, 173, 16, 184, 13, 184, 10, 150, 13, 139, 15,
        ],
        [
            22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72, 96, 67, 101, 73, 107, 72, 113, 55, 118, 52, 125,
            52, 118, 52, 117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97, 33, 77, 40,
        ],
    ],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_table_is_monotonic_and_complete() {
        for w in EBANDS.windows(2) {
            assert!(w[0] < w[1], "band boundaries strictly increase");
        }
        assert_eq!(EBANDS[NB_EBANDS], 100, "top coded bin at the shortest frame size");
    }

    #[test]
    fn log_n_matches_band_widths() {
        for i in 0..NB_EBANDS {
            let width = f64::from(EBANDS[i + 1] - EBANDS[i]);
            let expected = (width.log2() * 8.0).round() as i64;
            assert!(
                (i64::from(LOG_N[i]) - expected).abs() <= 1,
                "band {i}: logN {} vs computed {expected}",
                LOG_N[i]
            );
        }
    }
}

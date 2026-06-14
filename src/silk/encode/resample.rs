//! Decimating resamplers used by the pitch analysis (RFC 6716 §5.2;
//! normative `silk/resampler_down2.c`, `silk/resampler_down2_3.c`,
//! `silk/resampler_private_AR2.c`).
//!
//! Pitch estimation runs at 4 kHz (first stage) and 8 kHz (second stage),
//! so the internal-rate frame is decimated by 2 (16→8, 8→4 kHz) with the
//! half-band allpass [`down2`], or by 2/3 (12→8 kHz) with [`down2_3`]. Both
//! are fixed-point and pinned bit-exactly against the compiled reference.

extern crate alloc;
use alloc::vec;

use super::super::math::{rshift_round, smlawb, smulwb};

/// `silk_resampler_down2_0` / `_1` (Q15-ish allpass coefficients).
const DOWN2_0: i32 = 9872;
const DOWN2_1: i32 = 39809 - 65536; // -25727

/// `silk_Resampler_2_3_COEFS_LQ`: AR coefficients (`[0..2]`, Q14) and FIR
/// interpolation coefficients (`[2..6]`).
const COEFS_2_3: [i32; 6] = [-2797, -6507, 4697, 10739, 1567, 8276];
const ORDER_FIR: usize = 4;
/// `RESAMPLER_MAX_BATCH_SIZE_IN` (10 ms × 48 kHz).
const MAX_BATCH_SIZE_IN: usize = 480;

/// `silk_SAT16`.
fn sat16(a: i32) -> i16 {
    a.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// `silk_resampler_down2`: decimate `inp` by two through a second-order
/// allpass half-band filter. `out` receives `inp.len()/2` samples; `s` is
/// the 2-element state carried across calls.
pub(crate) fn down2(s: &mut [i32; 2], out: &mut [i16], inp: &[i16]) {
    let len2 = inp.len() / 2;
    for k in 0..len2 {
        let in32 = i32::from(inp[2 * k]) << 10;
        let y = in32.wrapping_sub(s[0]);
        let x = smlawb(y, y, DOWN2_1);
        let mut out32 = s[0].wrapping_add(x);
        s[0] = in32.wrapping_add(x);

        let in32 = i32::from(inp[2 * k + 1]) << 10;
        let y = in32.wrapping_sub(s[1]);
        let x = smulwb(y, DOWN2_0);
        out32 = out32.wrapping_add(s[1]).wrapping_add(x);
        s[1] = in32.wrapping_add(x);

        out[k] = sat16(rshift_round(out32, 11));
    }
}

/// `silk_resampler_private_AR2`: second-order AR filter, output in Q8.
fn ar2(s: &mut [i32], out_q8: &mut [i32], inp: &[i16], a_q14: &[i32]) {
    for (k, &xn) in inp.iter().enumerate() {
        let out32 = s[0].wrapping_add(i32::from(xn) << 8);
        out_q8[k] = out32;
        let out32 = out32 << 2;
        s[0] = smlawb(s[1], out32, a_q14[0]);
        s[1] = smulwb(out32, a_q14[1]);
    }
}

/// `silk_resampler_down2_3`: decimate `inp` by 2/3 (12→8 kHz). `out`
/// receives `floor(2*inp.len()/3)` samples; `s` is the 6-element state.
pub(crate) fn down2_3(s: &mut [i32; 6], out: &mut [i16], inp: &[i16]) {
    let mut buf = vec![0i32; MAX_BATCH_SIZE_IN + ORDER_FIR];
    buf[..ORDER_FIR].copy_from_slice(&s[..ORDER_FIR]);

    let mut in_off = 0usize;
    let mut in_len = inp.len();
    let mut out_off = 0usize;
    let mut n_samples;

    loop {
        n_samples = in_len.min(MAX_BATCH_SIZE_IN);
        // AR2 state lives in s[ORDER_FIR..ORDER_FIR+2]; the FIR history in
        // s[0..ORDER_FIR] is untouched here. Output into buf[ORDER_FIR..].
        ar2(
            &mut s[ORDER_FIR..ORDER_FIR + 2],
            &mut buf[ORDER_FIR..ORDER_FIR + n_samples],
            &inp[in_off..in_off + n_samples],
            &COEFS_2_3,
        );

        let mut bp = 0usize;
        let mut counter = n_samples as i32;
        while counter > 2 {
            let mut r = smulwb(buf[bp], COEFS_2_3[2]);
            r = smlawb(r, buf[bp + 1], COEFS_2_3[3]);
            r = smlawb(r, buf[bp + 2], COEFS_2_3[5]);
            r = smlawb(r, buf[bp + 3], COEFS_2_3[4]);
            out[out_off] = sat16(rshift_round(r, 6));
            out_off += 1;

            let mut r = smulwb(buf[bp + 1], COEFS_2_3[4]);
            r = smlawb(r, buf[bp + 2], COEFS_2_3[5]);
            r = smlawb(r, buf[bp + 3], COEFS_2_3[3]);
            r = smlawb(r, buf[bp + 4], COEFS_2_3[2]);
            out[out_off] = sat16(rshift_round(r, 6));
            out_off += 1;

            bp += 3;
            counter -= 3;
        }

        in_off += n_samples;
        in_len -= n_samples;
        if in_len > 0 {
            buf.copy_within(n_samples..n_samples + ORDER_FIR, 0);
        } else {
            break;
        }
    }
    s[..ORDER_FIR].copy_from_slice(&buf[n_samples..n_samples + ORDER_FIR]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn pin_input() -> [i16; 48] {
        core::array::from_fn(|i| ((i as i32 * 1237 + 11) % 9001 - 4500) as i16)
    }

    /// Bit-exact pin against the compiled reference `silk_resampler_down2`.
    #[test]
    fn down2_matches_reference_pin() {
        let inp = pin_input();
        let mut s = [0i32; 2];
        let mut out = [0i16; 24];
        down2(&mut s, &mut out, &inp);
        let expected: [i16; 24] = [
            -1608, -3676, 351, 2063, 1561, -3464, 1555, 2108, -1840, -1968, 2388, 262, -3774, 644, 2240, 1790, -3263,
            1773, 2315, -1627, -1758, 2600, 472, -3562,
        ];
        assert_eq!(out, expected, "down2 output disagrees with reference");
        assert_eq!(s, [1_850_487, 176_361], "down2 state disagrees with reference");
    }

    /// Bit-exact pin against the compiled reference `silk_resampler_down2_3`.
    #[test]
    fn down2_3_matches_reference_pin() {
        let inp = pin_input();
        let mut s = [0i32; 6];
        let mut out = [0i16; 32];
        down2_3(&mut s, &mut out, &inp);
        let expected: [i16; 32] = [
            0, -1287, -4180, -1508, 499, 1945, 3188, -2326, -2463, 1251, 2064, 1704, -4002, -1061, 1242, 2366, -605,
            -4390, -8, 980, 2852, 2250, -3529, -466, 1757, 2921, -62, -3847, 538, 1524, 2536, -2665,
        ];
        assert_eq!(out, expected, "down2_3 output disagrees with reference");
    }

    /// A low-frequency tone passes through `down2`; a near-Nyquist tone is
    /// attenuated (the half-band filter's purpose).
    #[test]
    fn down2_attenuates_high_frequencies() {
        let n = 480;
        let low: alloc::vec::Vec<i16> = (0..n).map(|i| ((i as f32 * 0.05).sin() * 10000.0) as i16).collect();
        let high: alloc::vec::Vec<i16> = (0..n)
            .map(|i| ((i as f32 * core::f32::consts::PI).sin() * 10000.0) as i16)
            .collect();
        let mut s = [0i32; 2];
        let mut out_low = vec![0i16; n / 2];
        down2(&mut s, &mut out_low, &low);
        let mut s = [0i32; 2];
        let mut out_high = vec![0i16; n / 2];
        down2(&mut s, &mut out_high, &high);
        let e_low: i64 = out_low.iter().map(|&v| i64::from(v) * i64::from(v)).sum();
        let e_high: i64 = out_high.iter().map(|&v| i64::from(v) * i64::from(v)).sum();
        assert!(
            e_low > 10 * e_high,
            "high freq not attenuated: low {e_low} high {e_high}"
        );
    }
}

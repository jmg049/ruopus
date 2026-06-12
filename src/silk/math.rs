//! SILK fixed-point arithmetic kernels (normative `silk/macros.h`,
//! `SigProc_FIX.h`, `Inlines.h`, `lin2log.c`, `log2lin.c`).
//!
//! The SILK decoder is integer-only: unlike CELT's float build, *every*
//! operation here is bitstream-affecting and must match the reference
//! bit-for-bit. The portable 64-bit forms of the reference macros are used
//! throughout; additions that the reference performs in (two's-complement)
//! machine arithmetic are `wrapping_*` here.

#![allow(dead_code, reason = "consumed incrementally as the SILK decoder stages land")]

/// `silk_SMULWB`: `(a * (i64)(i16)b) >> 16`.
#[inline]
pub(crate) const fn smulwb(a: i32, b: i32) -> i32 {
    ((a as i64 * (b as i16 as i64)) >> 16) as i32
}

/// `silk_SMLAWB`: `a + ((b * (i64)(i16)c) >> 16)`.
#[inline]
pub(crate) const fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((b as i64 * (c as i16 as i64)) >> 16) as i32)
}

/// `silk_SMULWT`: `(a * (i64)(b >> 16)) >> 16`.
#[inline]
pub(crate) const fn smulwt(a: i32, b: i32) -> i32 {
    ((a as i64 * (b >> 16) as i64) >> 16) as i32
}

/// `silk_SMLAWT`: `a + ((b * (i64)(c >> 16)) >> 16)`.
#[inline]
pub(crate) const fn smlawt(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((b as i64 * (c >> 16) as i64) >> 16) as i32)
}

/// `silk_SMULBB`: `(i32)(i16)a * (i32)(i16)b`.
#[inline]
pub(crate) const fn smulbb(a: i32, b: i32) -> i32 {
    (a as i16 as i32).wrapping_mul(b as i16 as i32)
}

/// `silk_SMLABB`: `a + (i32)(i16)b * (i32)(i16)c`.
#[inline]
pub(crate) const fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add((b as i16 as i32).wrapping_mul(c as i16 as i32))
}

/// `silk_SMULWW`: `((i64)a * b) >> 16`.
#[inline]
pub(crate) const fn smulww(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 16) as i32
}

/// `silk_SMLAWW`: `(i32)(a + (((i64)b * c) >> 16))`.
#[inline]
pub(crate) const fn smlaww(a: i32, b: i32, c: i32) -> i32 {
    (a as i64).wrapping_add((b as i64 * c as i64) >> 16) as i32
}

/// `silk_SMMUL`: `((i64)a * b) >> 32`.
#[inline]
pub(crate) const fn smmul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 32) as i32
}

/// `silk_ADD_SAT32`.
#[inline]
pub(crate) const fn add_sat32(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

/// `silk_SUB_SAT32`.
#[inline]
pub(crate) const fn sub_sat32(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}

/// `silk_LSHIFT_SAT32`: saturating left shift.
#[inline]
pub(crate) const fn lshift_sat32(a: i32, shift: i32) -> i32 {
    let lo = i32::MIN >> shift;
    let hi = i32::MAX >> shift;
    let a = if a < lo {
        lo
    } else if a > hi {
        hi
    } else {
        a
    };
    a << shift
}

/// `silk_RSHIFT_ROUND`: right shift with round-to-nearest (shift ≥ 1).
#[inline]
pub(crate) const fn rshift_round(a: i32, shift: i32) -> i32 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}

/// `silk_ROR32`: rotate right by `rot` (left for negative `rot`).
#[inline]
pub(crate) const fn ror32(a: i32, rot: i32) -> i32 {
    if rot == 0 {
        a
    } else if rot < 0 {
        (a as u32).rotate_left((-rot) as u32) as i32
    } else {
        (a as u32).rotate_right(rot as u32) as i32
    }
}

/// `silk_CLZ32`: leading zeros of the 32-bit pattern (32 for zero).
#[inline]
pub(crate) const fn clz32(a: i32) -> i32 {
    (a as u32).leading_zeros() as i32
}

/// `silk_CLZ_FRAC`: leading zeros and the 7 bits right after the leading
/// one.
#[inline]
pub(crate) const fn clz_frac(a: i32) -> (i32, i32) {
    let lz = clz32(a);
    (lz, ror32(a, 24 - lz) & 0x7f)
}

/// `silk_SQRT_APPROX`: approximate square root of a positive value.
pub(crate) const fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let (lz, frac_q7) = clz_frac(x);
    let mut y = if lz & 1 != 0 { 32768 } else { 46214 }; // 46214 = sqrt(2) * 32768
    // Get the scaling right, then refine from the fractional part.
    y >>= lz >> 1;
    smlawb(y, y, smulbb(213, frac_q7))
}

/// `silk_lin2log`: approximate `128 * log2(x)` (Q7).
pub(crate) const fn lin2log(x: i32) -> i32 {
    let (lz, frac_q7) = clz_frac(x);
    // Piece-wise parabolic approximation.
    smlawb(frac_q7, frac_q7.wrapping_mul(128 - frac_q7), 179).wrapping_add((31 - lz) << 7)
}

/// `silk_log2lin`: approximate `2^(x/128)` (input Q7).
pub(crate) const fn log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    } else if in_log_q7 >= 3967 {
        return i32::MAX;
    }
    let out = 1i32 << (in_log_q7 >> 7);
    let frac_q7 = in_log_q7 & 0x7f;
    // Piece-wise parabolic approximation.
    if in_log_q7 < 2048 {
        out + ((out.wrapping_mul(smlawb(frac_q7, smulbb(frac_q7, 128 - frac_q7), -174))) >> 7)
    } else {
        out.wrapping_add((out >> 7).wrapping_mul(smlawb(frac_q7, smulbb(frac_q7, 128 - frac_q7), -174)))
    }
}

/// `silk_INVERSE32_varQ`: approximate `(1 << q_res) / b` (`b != 0`,
/// `q_res > 0`).
pub(crate) const fn inverse32_var_q(b32: i32, q_res: i32) -> i32 {
    debug_assert!(b32 != 0);
    debug_assert!(q_res > 0);

    let b_headrm = clz32(b32.abs()) - 1;
    let b32_nrm = b32 << b_headrm; // Q: b_headrm
    // Inverse of b32 with 14 bits of precision (Q: 29 + 16 - b_headrm).
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    // First approximation (Q: 61 - b_headrm).
    let mut result = b32_inv << 16;
    // Residual of one minus denominator times approximation (Q32).
    let err_q32 = (1i32 << 29).wrapping_sub(smulwb(b32_nrm, b32_inv)) << 3;
    // Refinement.
    result = smlaww(result, err_q32, b32_inv);

    let lshift = 61 - b_headrm - q_res;
    if lshift <= 0 {
        lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// `silk_DIV32_varQ`: approximate `(a << q_res) / b` (`b != 0`,
/// `q_res >= 0`).
pub(crate) const fn div32_var_q(a32: i32, b32: i32, q_res: i32) -> i32 {
    debug_assert!(b32 != 0);
    debug_assert!(q_res >= 0);

    let a_headrm = clz32(a32.abs()) - 1;
    let a32_nrm = a32 << a_headrm; // Q: a_headrm
    let b_headrm = clz32(b32.abs()) - 1;
    let b32_nrm = b32 << b_headrm; // Q: b_headrm
    // Inverse of b32 with 14 bits of precision (Q: 29 + 16 - b_headrm).
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    // First approximation (Q: 29 + a_headrm - b_headrm).
    let mut result = smulwb(a32_nrm, b32_inv);
    // Residual; wrapping is fine, the final a32_nrm is always small.
    let a32_nrm = a32_nrm.wrapping_sub(smmul(b32_nrm, result) << 3); // Q: a_headrm
    // Refinement.
    result = smlawb(result, a32_nrm, b32_inv);

    let lshift = 29 + a_headrm - b_headrm - q_res;
    if lshift < 0 {
        lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// `silk_SMULL`: full 64-bit product.
#[inline]
pub(crate) const fn smull(a: i32, b: i32) -> i64 {
    a as i64 * b as i64
}

/// `silk_RSHIFT_ROUND64`: 64-bit right shift with round-to-nearest.
#[inline]
pub(crate) const fn rshift_round64(a: i64, shift: i32) -> i64 {
    ((a >> (shift - 1)) + 1) >> 1
}

/// `silk_MUL`: 32-bit multiply (the reference asserts no overflow).
#[inline]
pub(crate) const fn mul(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins generated by compiling the reference `SigProc_FIX.h`/`Inlines.h`
    /// and `lin2log.c`/`log2lin.c` directly and printing these exact calls.
    #[test]
    fn kernels_match_reference_pins() {
        for (x, want) in [
            (1, 1),
            (2, 1),
            (3, 1),
            (7, 2),
            (100, 9),
            (12345, 108),
            (987_654, 987),
            (2_000_000_000, 44483),
            (-5, 0),
            (i32::MAX, 46293),
        ] {
            assert_eq!(sqrt_approx(x), want, "sqrt_approx({x})");
        }
        for (x, want) in [
            (1, 0),
            (2, 128),
            (3, 203),
            (7, 360),
            (100, 851),
            (12345, 1739),
            (987_654, 2549),
            (2_000_000_000, 3955),
            (i32::MAX, 3967),
        ] {
            assert_eq!(lin2log(x), want, "lin2log({x})");
        }
        for (x, want) in [
            (0, 1),
            (100, 1),
            (1000, 225),
            (2047, 65024),
            (2048, 65536),
            (3000, 11_337_728),
            (3966, 2_122_317_824),
            (3967, i32::MAX),
            (-1, 0),
        ] {
            assert_eq!(log2lin(x), want, "log2lin({x})");
        }
        assert_eq!(inverse32_var_q(3, 16), 21845);
        assert_eq!(inverse32_var_q(-7, 20), -149_797);
        assert_eq!(inverse32_var_q(48000, 16), 1);
        assert_eq!(div32_var_q(1000, 3, 10), 341_333);
        assert_eq!(div32_var_q(-31337, 7, 2), -17907);
        assert_eq!(div32_var_q(2_000_000_000, -3, 0), -666_666_669);
        assert_eq!(smulwb(0x1234_5678, 0xffff_abcdu32 as i32), -100_453_581);
        assert_eq!(smlawt(7, -1_000_000, 0x7fff_0000), -499_978);
        assert_eq!(rshift_round(1001, 1), 501);
        assert_eq!(rshift_round(-1001, 3), -125);
        assert_eq!(ror32(0x8000_0001u32 as i32, -7), 192);
    }
}

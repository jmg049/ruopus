//! The Opus range coder (RFC 6716 §4.1 and §5.1).
//!
//! Opus entropy-codes nearly everything through a single range coder, which
//! doubles as the codec's bit-packer. Three kinds of data share one buffer:
//!
//! - **Range-coded symbols** with static probability models, packed MSB-first from the *front* of the frame.
//! - **Raw bits** (used by the CELT layer), packed LSB-first from the *end* of the frame, bypassing the range coder.
//! - **Uniform integers**, split between a range-coded high part and raw-bit low part by
//!   [`RangeDecoder::decode_uint`]/[`RangeEncoder::encode_uint`].
//!
//! The two directions may legally *overlap* in the middle of the buffer: the
//! range decoder reads several bytes ahead, and the encoder terminates the
//! stream (§5.1.5) such that decoding stays correct regardless of what the raw
//! bits put there.
//!
//! All arithmetic is bit-exact per the RFC. After encoding and decoding the
//! same symbol sequence, the encoder's and decoder's `rng` values are
//! guaranteed to be identical - the tests in this module rely on that property
//! (RFC 6716 §5.1) as a built-in correctness oracle.
//!
//! # Naming
//!
//! Methods keep a recognizable mapping to the reference implementation:
//!
//! | This crate | Reference |
//! |------------|------------------------------------|
//! | `decode`/`encode` | `ec_decode`/`ec_encode` |
//! | `decode_bin`/`encode_bin` | `ec_decode_bin`/`ec_encode_bin` |
//! | `update` | `ec_dec_update` |
//! | `decode_bit_logp`/`encode_bit_logp` | `ec_dec_bit_logp`/`ec_enc_bit_logp` |
//! | `decode_icdf`/`encode_icdf` | `ec_dec_icdf`/`ec_enc_icdf` |
//! | `decode_raw_bits`/`encode_raw_bits` | `ec_dec_bits`/`ec_enc_bits` |
//! | `decode_uint`/`encode_uint` | `ec_dec_uint`/`ec_enc_uint` |
//! | `tell`/`tell_frac` | `ec_tell`/`ec_tell_frac` |

mod decoder;
mod encoder;

pub use decoder::RangeDecoder;
pub use encoder::{RangeEncoder, RangeEncoderError};

/// Number of bits in a coder symbol (one byte). RFC 6716 §4.1.
pub(crate) const SYM_BITS: u32 = 8;

/// Maximum value of a coder symbol.
pub(crate) const SYM_MAX: u32 = (1 << SYM_BITS) - 1;

/// Total bits in the coder state values `val` and `rng`.
pub(crate) const CODE_BITS: u32 = 32;

/// The top of the coder range: 2³¹.
pub(crate) const CODE_TOP: u32 = 1 << (CODE_BITS - 1);

/// Renormalization threshold: 2²³. After renormalization, `rng > CODE_BOT`.
pub(crate) const CODE_BOT: u32 = CODE_TOP >> SYM_BITS;

/// Carry-out shift: the top 9 bits of `val` (8 data bits + carry) sit above
/// this bit position during encoder renormalization.
pub(crate) const CODE_SHIFT: u32 = CODE_BITS - SYM_BITS - 1;

/// Bits per raw-bits window flush; raw-bit reads/writes are limited to
/// `WINDOW_SIZE - SYM_BITS` = 24 bits per call, matching the reference
/// implementation.
pub(crate) const WINDOW_SIZE: u32 = u32::BITS;

/// Number of binary digits needed to represent `x`; `ilog(0) == 0`.
///
/// Equivalent to `EC_ILOG()` in the reference implementation: the position of
/// the highest set bit plus one.
#[inline]
#[must_use]
pub(crate) const fn ilog(x: u32) -> u32 {
    u32::BITS - x.leading_zeros()
}

#[cfg(test)]
mod ilog_tests {
    use super::ilog;

    #[test]
    fn ilog_matches_definition() {
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(2), 2);
        assert_eq!(ilog(3), 2);
        assert_eq!(ilog(4), 3);
        assert_eq!(ilog(u32::MAX), 32);
        assert_eq!(ilog(1 << 31), 32);
    }
}

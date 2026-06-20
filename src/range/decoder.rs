//! Range decoder (RFC 6716 §4.1).

use super::{CODE_BOT, SYM_BITS, SYM_MAX, WINDOW_SIZE, ilog};

/// The Opus range decoder.
///
/// Decodes the three kinds of entropy-coded data Opus packs into a frame:
/// range-coded symbols (from the front of the buffer), raw bits (from the
/// end), and uniformly distributed integers (a combination of the two). See
/// the [module documentation](super) for the layout.
///
/// # Robustness
///
/// Following RFC 6716 §4.1.2.1, the decoder never fails on truncated input:
/// once the buffer is exhausted it continues with zero bits. Corrupt streams
/// therefore decode to *some* symbol sequence; upper layers detect corruption
/// via range/`tell` checks where the spec requires them.
#[derive(Debug, Clone)]
pub struct RangeDecoder<'a> {
    buf: &'a [u8],
    /// Next front byte to read (range-coder direction).
    offs: usize,
    /// Number of bytes consumed from the end (raw-bits direction).
    end_offs: usize,
    /// Raw-bits window: bits already read from the end, LSB-first.
    end_window: u32,
    /// Number of valid bits in `end_window`.
    nend_bits: u32,
    /// The leftover (least significant) bit of the most recent front byte.
    leftover_bit: u32,
    /// Difference between the high end of the current range and the coded
    /// value, minus one (RFC 6716 §4.1).
    val: u32,
    /// Size of the current range. Invariant after renormalization:
    /// `rng > 2^23`.
    rng: u32,
    /// Conservative upper bound on whole bits consumed so far, including bits
    /// buffered in `rng` (RFC 6716 §4.1.6).
    nbits_total: u32,
}

impl<'a> RangeDecoder<'a> {
    /// Initializes a decoder over one Opus frame (RFC 6716 §4.1.1).
    ///
    /// `rng` starts at 128 and `val` at `127 - (b0>>1)` where `b0` is the
    /// first input byte (zero for an empty frame), followed immediately by
    /// renormalization. `nbits_total` starts at 9 so that a fresh decoder
    /// reports [`tell()`](Self::tell) = 1: the bit reserved for encoder
    /// termination.
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        let b0 = u32::from(buf.first().copied().unwrap_or(0));
        let mut dec = RangeDecoder {
            buf,
            offs: usize::from(!buf.is_empty()),
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            leftover_bit: b0 & 1,
            val: 127 - (b0 >> 1),
            rng: 128,
            nbits_total: 9,
        };
        dec.normalize();
        dec
    }

    /// Reads the next byte in the range-coder (front) direction, or zero once
    /// the buffer is exhausted (RFC 6716 §4.1.2.1).
    #[inline]
    fn read_byte(&mut self) -> u32 {
        match self.buf.get(self.offs) {
            Some(&b) => {
                self.offs += 1;
                u32::from(b)
            },
            None => 0,
        }
    }

    /// Reads the next byte in the raw-bits (back) direction, or zero once the
    /// buffer is exhausted (RFC 6716 §4.1.4).
    ///
    /// May legally overlap with bytes the range coder has read.
    #[inline]
    fn read_byte_from_end(&mut self) -> u32 {
        if self.end_offs < self.buf.len() {
            self.end_offs += 1;
            u32::from(self.buf[self.buf.len() - self.end_offs])
        } else {
            0
        }
    }

    /// Renormalization (RFC 6716 §4.1.2.1): restores the invariant
    /// `rng > 2^23`, consuming one byte per iteration.
    ///
    /// Each new 8-bit symbol `sym` is formed from the leftover bit of the
    /// previous byte (high bit) and the top 7 bits of the byte just read; the
    /// new byte's low bit becomes the next leftover.
    fn normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.nbits_total += SYM_BITS;
            self.rng <<= SYM_BITS;

            let byte = self.read_byte();
            let sym = (self.leftover_bit << 7) | (byte >> 1);
            self.leftover_bit = byte & 1;

            self.val = ((self.val << SYM_BITS) + (SYM_MAX - sym)) & 0x7FFF_FFFF;
        }
    }

    /// First step of decoding a symbol (RFC 6716 §4.1.2, `ec_decode`).
    ///
    /// Returns `fs`, a value lying within the range of some symbol in a
    /// context with cumulative frequency `ft`. The caller locates the symbol
    /// `k` with `fl[k] <= fs < fh[k]` and finishes with
    /// [`update`](Self::update).
    ///
    /// `ft` must be at most 2¹⁶ - 1 (all Opus contexts satisfy this).
    #[must_use]
    pub fn decode(&mut self, ft: u32) -> u32 {
        debug_assert!(ft > 0 && ft <= u32::from(u16::MAX));
        ft - (self.val / (self.rng / ft) + 1).min(ft)
    }

    /// Like [`decode`](Self::decode) with `ft = 1 << ftb`, avoiding a
    /// division (RFC 6716 §4.1.3.1, `ec_decode_bin`).
    #[must_use]
    pub fn decode_bin(&mut self, ftb: u32) -> u32 {
        debug_assert!(ftb <= 16);
        let ft = 1u32 << ftb;
        ft - (self.val / (self.rng >> ftb) + 1).min(ft)
    }

    /// Second step of decoding a symbol (RFC 6716 §4.1.2, `ec_dec_update`):
    /// applies the three-tuple `(fl, fh, ft)` of the symbol identified from
    /// the value returned by [`decode`](Self::decode), then renormalizes.
    pub fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        debug_assert!(fl < fh && fh <= ft);
        let s = self.rng / ft;
        self.val -= s * (ft - fh);
        self.rng = if fl > 0 {
            s * (fh - fl)
        } else {
            self.rng - s * (ft - fh)
        };
        self.normalize();
    }

    /// Decodes one binary symbol whose probability of being "1" is
    /// `1 / 2^logp` (RFC 6716 §4.1.3.2, `ec_dec_bit_logp`).
    #[must_use]
    pub fn decode_bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let bit = d < s;
        if !bit {
            self.val = d - s;
        }
        self.rng = if bit { s } else { r - s };
        self.normalize();
        bit
    }

    /// Decodes one symbol from a table-based context of up to 8 bits
    /// (RFC 6716 §4.1.3.3, `ec_dec_icdf`).
    ///
    /// `icdf[k]` holds `ft - fh[k]` (an "inverse" CDF) with `ft = 1 << ftb`;
    /// the table is terminated by a zero entry. This is the primary SILK-layer
    /// interface to the range decoder.
    ///
    /// # Panics
    ///
    /// Panics if `icdf` is not zero-terminated (malformed table - a programmer
    /// error in static tables, not a data error).
    #[must_use]
    pub fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        let d = self.val;
        let r = self.rng >> ftb;

        let mut k = 0usize;
        let mut t = self.rng;
        let mut s = r * u32::from(icdf[0]);
        while d < s {
            k += 1;
            t = s;
            s = r * u32::from(icdf[k]);
        }

        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        k
    }

    /// Reads `bits` raw bits from the end of the frame, LSB-first
    /// (RFC 6716 §4.1.4, `ec_dec_bits`). `bits` must be at most 24.
    #[must_use]
    pub fn decode_raw_bits(&mut self, bits: u32) -> u32 {
        debug_assert!(bits > 0 && bits <= WINDOW_SIZE - SYM_BITS);
        if self.nend_bits < bits {
            loop {
                self.end_window |= self.read_byte_from_end() << self.nend_bits;
                self.nend_bits += SYM_BITS;
                if self.nend_bits > WINDOW_SIZE - SYM_BITS {
                    break;
                }
            }
        }
        let ret = self.end_window & ((1u32 << bits) - 1);
        self.end_window >>= bits;
        self.nend_bits -= bits;
        self.nbits_total += bits;
        ret
    }

    /// Decodes one of `ft` equiprobable values in `0..ft`
    /// (RFC 6716 §4.1.5, `ec_dec_uint`). `ft` may be as large as 2³² - 1 and
    /// need not be a power of two; `ft` must be at least 2.
    ///
    /// Values requiring more than 8 bits are split between a range-coded high
    /// part and raw-bit low part. Returns `None` if the decoded value falls
    /// outside `0..ft`, which indicates a corrupt frame per the RFC ("the
    /// decoder should assume there has been an error").
    #[must_use]
    pub fn decode_uint(&mut self, ft: u32) -> Option<u32> {
        debug_assert!(ft > 1);
        let ftb = ilog(ft - 1);
        if ftb <= 8 {
            let t = self.decode(ft);
            self.update(t, t + 1, ft);
            Some(t)
        } else {
            let ft_hi = ((ft - 1) >> (ftb - 8)) + 1;
            let t = self.decode(ft_hi);
            self.update(t, t + 1, ft_hi);
            let t = (t << (ftb - 8)) | self.decode_raw_bits(ftb - 8);
            (t < ft).then_some(t)
        }
    }

    /// Conservative upper bound on the whole number of bits consumed so far,
    /// counting both range-coder and raw bits (RFC 6716 §4.1.6.1, `ec_tell`).
    ///
    /// A freshly initialized decoder reports 1: the bit reserved for encoder
    /// termination. Guaranteed to equal `ceil(tell_frac() / 8)`.
    #[inline]
    #[must_use]
    pub fn tell(&self) -> u32 {
        self.nbits_total - ilog(self.rng)
    }

    /// Like [`tell`](Self::tell) to fractional 1/8th-bit precision
    /// (RFC 6716 §4.1.6.2, `ec_tell_frac`).
    #[must_use]
    pub fn tell_frac(&self) -> u32 {
        tell_frac(self.nbits_total, self.rng)
    }

    /// Advances the bit-usage accounting to `bits` total, as if the
    /// intervening bits had been consumed, without reading them.
    ///
    /// Used by the CELT silence path, which "pretends to have read all the
    /// remaining bits" (RFC 6716 §4.3, reference `celt_decode_with_ec`) so
    /// downstream budget checks behave identically to the reference.
    #[cfg(feature = "std")]
    pub(crate) fn force_tell(&mut self, bits: u32) {
        let current = self.tell();
        // On a valid stream the silence flag is read early, so `current <= bits`
        // and we advance to the target. Corrupted input can leave the coder
        // already past it; advancing then is a no-op (never underflow).
        if bits > current {
            self.nbits_total += bits - current;
        }
    }

    /// Truncates the buffer to `new_len` bytes (`dec.storage -=
    /// redundancy_bytes` in `opus_decoder.c`): an embedded redundant frame
    /// occupies the tail, and subsequent raw bits must not read into it.
    ///
    /// # Panics
    ///
    /// Panics if raw bits were already consumed past `new_len`.
    pub fn shrink_storage(&mut self, new_len: usize) {
        assert!(new_len <= self.buf.len());
        assert!(self.end_offs == 0, "shrink before reading raw bits");
        self.buf = &self.buf[..new_len];
    }

    /// The current range size.
    ///
    /// After decoding a symbol sequence this must exactly equal the encoder's
    /// `rng` after encoding the same sequence (RFC 6716 §5.1) - a powerful
    /// cross-check used throughout this crate's tests.
    #[inline]
    #[must_use]
    pub fn range_size(&self) -> u32 {
        self.rng
    }
}

/// Shared `ec_tell_frac` computation (RFC 6716 §4.1.6.2); also used by the
/// encoder, whose value must match the decoder's exactly.
pub(crate) fn tell_frac(nbits_total: u32, rng: u32) -> u32 {
    let mut lg = ilog(rng);
    // r is a Q15 value representing the fractional part of rng:
    // 32768 <= r < 65536.
    let mut r = rng >> (lg - 16);
    // Each iteration adds one bit of precision to lg.
    for _ in 0..3 {
        r = (r * r) >> 15;
        let bit = r >> 16;
        lg = 2 * lg + bit;
        r >>= bit;
    }
    nbits_total * 8 - lg
}

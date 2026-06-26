//! `no_std` floating-point transcendentals, backed by [`libm`].
//!
//! On `std` builds the inherent `f32`/`f64` methods (`x.sin()`, `x.sqrt()`, ...)
//! are used and this module is not even compiled. Without `std` those methods
//! do not exist, so the codec brings [`FloatExt`] into scope and the same
//! `x.sin()` call sites resolve to these `libm`-backed implementations instead.
//! Keeping the call sites identical means the `std` build is byte-for-byte
//! unchanged (and stays conformance-exact); only `no_std` routes through `libm`.

/// The `std`-only `f32`/`f64` transcendental methods the codec uses, provided
/// for `no_std` via `libm`.
#[allow(dead_code)]
pub trait FloatExt {
    #[must_use]
    fn sin(self) -> Self;
    #[must_use]
    fn cos(self) -> Self;
    #[must_use]
    fn sqrt(self) -> Self;
    #[must_use]
    fn exp(self) -> Self;
    #[must_use]
    fn ln(self) -> Self;
    #[must_use]
    fn log2(self) -> Self;
    #[must_use]
    fn log10(self) -> Self;
    #[must_use]
    fn floor(self) -> Self;
    #[must_use]
    fn ceil(self) -> Self;
    #[must_use]
    fn round(self) -> Self;
    #[must_use]
    fn trunc(self) -> Self;
    #[must_use]
    fn mul_add(self, a: Self, b: Self) -> Self;
    #[must_use]
    fn powf(self, n: Self) -> Self;
    #[must_use]
    fn powi(self, n: i32) -> Self;
    #[must_use]
    fn sin_cos(self) -> (Self, Self)
    where
        Self: Sized;
    #[must_use]
    fn round_ties_even(self) -> Self;
}

impl FloatExt for f32 {
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    fn exp(self) -> Self {
        libm::expf(self)
    }
    fn ln(self) -> Self {
        libm::logf(self)
    }
    fn log2(self) -> Self {
        libm::log2f(self)
    }
    fn log10(self) -> Self {
        libm::log10f(self)
    }
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    fn ceil(self) -> Self {
        libm::ceilf(self)
    }
    fn round(self) -> Self {
        libm::roundf(self)
    }
    fn trunc(self) -> Self {
        libm::truncf(self)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fmaf(self, a, b)
    }
    fn powf(self, n: Self) -> Self {
        libm::powf(self, n)
    }
    fn powi(self, n: i32) -> Self {
        // Exponentiation by squaring, matching `f32::powi`'s integer semantics
        // (no fractional-power rounding) without a `powf` round-trip.
        let mut acc = 1.0f32;
        let mut base = if n < 0 { 1.0 / self } else { self };
        let mut e = n.unsigned_abs();
        while e > 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base *= base;
            e >>= 1;
        }
        acc
    }
    fn sin_cos(self) -> (Self, Self) {
        libm::sincosf(self)
    }
    fn round_ties_even(self) -> Self {
        libm::roundevenf(self)
    }
}

impl FloatExt for f64 {
    fn sin(self) -> Self {
        libm::sin(self)
    }
    fn cos(self) -> Self {
        libm::cos(self)
    }
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    fn exp(self) -> Self {
        libm::exp(self)
    }
    fn ln(self) -> Self {
        libm::log(self)
    }
    fn log2(self) -> Self {
        libm::log2(self)
    }
    fn log10(self) -> Self {
        libm::log10(self)
    }
    fn floor(self) -> Self {
        libm::floor(self)
    }
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
    fn round(self) -> Self {
        libm::round(self)
    }
    fn trunc(self) -> Self {
        libm::trunc(self)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fma(self, a, b)
    }
    fn powf(self, n: Self) -> Self {
        libm::pow(self, n)
    }
    fn powi(self, n: i32) -> Self {
        let mut acc = 1.0f64;
        let mut base = if n < 0 { 1.0 / self } else { self };
        let mut e = n.unsigned_abs();
        while e > 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base *= base;
            e >>= 1;
        }
        acc
    }
    fn sin_cos(self) -> (Self, Self) {
        libm::sincos(self)
    }
    fn round_ties_even(self) -> Self {
        libm::roundeven(self)
    }
}

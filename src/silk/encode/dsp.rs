//! Shared floating-point analysis kernels for the SILK encoder (RFC 6716
//! §5.2). These small building blocks are used by
//! more than one analysis stage (noise shaping, pitch analysis): the sine
//! window, autocorrelation, the Schur recursion, reflection→prediction
//! conversion, bandwidth expansion, an energy accumulator, and the LPC
//! analysis filter.

/// Upper bound on the order these helpers handle (`MAX_SHAPE_LPC_ORDER`).
const MAX_ORDER: usize = 24;

/// Window `px` with a sine (`win_type==1`) or
/// cosine (`win_type==2`) slope of `length` samples (a multiple of 4).
pub(crate) fn apply_sine_window(px_win: &mut [f32], px: &[f32], win_type: i32, length: usize) {
    debug_assert!(win_type == 1 || win_type == 2);
    debug_assert!(length & 3 == 0);
    let freq = core::f32::consts::PI / (length as f32 + 1.0);
    let c = 2.0 - freq * freq;
    let (mut s0, mut s1) = if win_type < 2 {
        (0.0f32, freq)
    } else {
        (1.0f32, 0.5 * c)
    };
    let mut k = 0;
    while k < length {
        px_win[k] = px[k] * 0.5 * (s0 + s1);
        px_win[k + 1] = px[k + 1] * s1;
        s0 = c * s1 - s0;
        px_win[k + 2] = px[k + 2] * 0.5 * (s1 + s0);
        px_win[k + 3] = px[k + 3] * s0;
        s1 = c * s0 - s1;
        k += 4;
    }
}

/// The first `count` autocorrelation taps.
pub(crate) fn autocorrelation(results: &mut [f32], input: &[f32], count: usize) {
    let n = input.len();
    let count = count.min(n);
    for (i, r) in results.iter_mut().enumerate().take(count) {
        *r = crate::simd::dot_f64(&input[..n - i], &input[i..]) as f32;
    }
}

/// Reflection coefficients from the autocorrelation,
/// returning the residual energy.
pub(crate) fn schur(refl_coef: &mut [f32], auto_corr: &[f32], order: usize) -> f32 {
    let mut c = [[0.0f64; 2]; MAX_ORDER + 1];
    for k in 0..=order {
        c[k][0] = f64::from(auto_corr[k]);
        c[k][1] = f64::from(auto_corr[k]);
    }
    for k in 0..order {
        let rc_tmp = -c[k + 1][0] / c[0][1].max(1e-9);
        refl_coef[k] = rc_tmp as f32;
        for n in 0..order - k {
            let ctmp1 = c[n + k + 1][0];
            let ctmp2 = c[n][1];
            c[n + k + 1][0] = ctmp1 + ctmp2 * rc_tmp;
            c[n][1] = ctmp2 + ctmp1 * rc_tmp;
        }
    }
    c[0][1] as f32
}

/// Reflection coefficients to prediction coefficients.
pub(crate) fn k2a(a: &mut [f32], rc: &[f32], order: usize) {
    for k in 0..order {
        let rck = rc[k];
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a[n];
            let tmp2 = a[k - n - 1];
            a[n] = tmp1 + tmp2 * rck;
            a[k - n - 1] = tmp2 + tmp1 * rck;
        }
        a[k] = -rck;
    }
}

/// Chirp the AR filter towards the unit circle.
pub(crate) fn bwexpander(ar: &mut [f32], order: usize, chirp: f32) {
    let mut cfac = chirp;
    for v in ar.iter_mut().take(order - 1) {
        *v *= cfac;
        cfac *= chirp;
    }
    ar[order - 1] *= cfac;
}

/// Sum of squares in double precision.
pub(crate) fn energy(data: &[f32]) -> f64 {
    crate::simd::dot_f64(data, data)
}

/// The LPC prediction residual of `s`
/// (`r[ix] = s[ix] - Σ_j s[ix-1-j]·a[j]`), with the first `order` outputs
/// set to zero (the filter starts from zero state).
pub(crate) fn lpc_analysis_filter_flp(r: &mut [f32], a: &[f32], s: &[f32], length: usize, order: usize) {
    // `pred = Σ_j s[ix-1-j]·a[j]`, a short (≤16-tap) dot evaluated once per
    // output sample. A vectorised `simd::dot` here loses: the per-call
    // horizontal fold does not amortise over so few taps and dwarfs the work
    // (it dominated `dot_avx2`). Dispatch on the (small, fixed) order to a
    // const-generic inner loop the compiler fully unrolls - using the
    // left-to-right accumulation order so it stays bit-identical to the
    // reference - and pipelines across the independent output samples.
    match order {
        6 => filter_n::<6>(r, a, s, length),
        8 => filter_n::<8>(r, a, s, length),
        10 => filter_n::<10>(r, a, s, length),
        12 => filter_n::<12>(r, a, s, length),
        16 => filter_n::<16>(r, a, s, length),
        _ => filter_dyn(r, a, s, length, order),
    }
}

/// Const-order LPC analysis filter inner loop (fully unrolled by the compiler).
#[inline]
fn filter_n<const N: usize>(r: &mut [f32], a: &[f32], s: &[f32], length: usize) {
    let a: &[f32; N] = a[..N].try_into().expect("a has at least N coefficients");
    for ix in N..length {
        let hist: &[f32; N] = s[ix - N..ix].try_into().expect("window is N wide");
        let mut pred = 0.0f32;
        for j in 0..N {
            pred += hist[N - 1 - j] * a[j];
        }
        r[ix] = s[ix] - pred;
    }
    for v in r.iter_mut().take(N) {
        *v = 0.0;
    }
}

/// Fallback for orders outside the SILK set (kept for completeness).
fn filter_dyn(r: &mut [f32], a: &[f32], s: &[f32], length: usize, order: usize) {
    let a = &a[..order];
    for ix in order..length {
        let hist = &s[ix - order..ix];
        let mut pred = 0.0f32;
        for j in 0..order {
            pred += hist[order - 1 - j] * a[j];
        }
        r[ix] = s[ix] - pred;
    }
    for v in r.iter_mut().take(order) {
        *v = 0.0;
    }
}

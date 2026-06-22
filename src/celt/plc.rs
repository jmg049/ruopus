//! Packet-loss-concealment kernels: pitch search over the synthesis history,
//! linear prediction, and the FIR/IIR filters the extrapolation runs in.
//!
//! PLC output is not normatively fixed by RFC 6716 - these mirror the
//! reference so concealment quality (and differential testing) match.

#![allow(dead_code, reason = "consumed by celt_decode_lost as the PLC lands")]

use alloc::vec;

/// Order of the CELT short-term LPC predictor.
pub(crate) const CELT_LPC_ORDER: usize = 24;
/// Length of the synthesis history kept for concealment.
pub(crate) const MAX_PERIOD: usize = 1024;
/// PLC pitch lag bounds: 480 Hz .. 66.6 Hz.
pub(crate) const PLC_PITCH_LAG_MAX: usize = 720;
pub(crate) const PLC_PITCH_LAG_MIN: usize = 100;

/// Cross-correlation `xcorr[i] = Σ_j x[j]·y[j+i]`.
fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    for (i, out) in xcorr.iter_mut().enumerate().take(max_pitch) {
        let mut sum = 0.0f32;
        for j in 0..len {
            sum += x[j] * y[j + i];
        }
        *out = sum;
    }
}

/// Windowed autocorrelation of `x` up to `lag` (float build).
pub(crate) fn celt_autocorr(x: &[f32], ac: &mut [f32], window: &[f32], overlap: usize, lag: usize) {
    let n = x.len();
    let fast_n = n - lag;
    let mut xx = vec![0.0f32; n];
    xx.copy_from_slice(x);
    for i in 0..overlap {
        xx[i] = x[i] * window[i];
        xx[n - i - 1] = x[n - i - 1] * window[i];
    }
    pitch_xcorr(&xx, &xx, ac, fast_n, lag + 1);
    for k in 0..=lag {
        let mut d = 0.0f32;
        for i in k + fast_n..n {
            d += xx[i] * xx[i - k];
        }
        ac[k] += d;
    }
}

/// Levinson-Durbin recursion (float build), bailing at 30 dB gain.
pub(crate) fn celt_lpc(lpc: &mut [f32], ac: &[f32]) {
    let p = lpc.len();
    lpc.fill(0.0);
    let mut error = ac[0];
    if ac[0] > 1e-10 {
        for i in 0..p {
            // This iteration's reflection coefficient.
            let mut rr = 0.0f32;
            for j in 0..i {
                rr += lpc[j] * ac[i - j];
            }
            rr += ac[i + 1];
            let r = -rr / error;
            lpc[i] = r;
            for j in 0..(i + 1) >> 1 {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + r * tmp2;
                lpc[i - 1 - j] = tmp2 + r * tmp1;
            }
            error -= r * r * error;
            if error <= 0.001 * ac[0] {
                break;
            }
        }
    }
}

/// Analysis filter. `input` carries `num.len()` history samples
/// followed by `out.len()` signal samples.
pub(crate) fn celt_fir(input: &[f32], num: &[f32], out: &mut [f32]) {
    let ord = num.len();
    debug_assert_eq!(input.len(), out.len() + ord);
    for (i, o) in out.iter_mut().enumerate() {
        let mut sum = input[ord + i];
        for (j, &c) in num.iter().enumerate() {
            sum += c * input[ord + i - j - 1];
        }
        *o = sum;
    }
}

/// Synthesis filter in place, with rolling memory `mem`
/// (`mem[0]` is the most recent output).
pub(crate) fn celt_iir(x: &mut [f32], den: &[f32], mem: &mut [f32]) {
    let ord = den.len();
    for v in x.iter_mut() {
        let mut sum = *v;
        for j in 0..ord {
            sum -= den[j] * mem[j];
        }
        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        mem[0] = sum;
        *v = sum;
    }
}

/// Fixed 5-tap analysis filter used by the pitch downsampler.
fn celt_fir5(x: &mut [f32], num: &[f32; 5]) {
    let mut mem = [0.0f32; 5];
    for v in x.iter_mut() {
        let sum = *v + num[0] * mem[0] + num[1] * mem[1] + num[2] * mem[2] + num[3] * mem[3] + num[4] * mem[4];
        mem[4] = mem[3];
        mem[3] = mem[2];
        mem[2] = mem[1];
        mem[1] = mem[0];
        mem[0] = *v;
        *v = sum;
    }
}

/// 2× downsample with channel mixdown, then whiten with a 4th-order LPC
/// (plus a zero).
pub(crate) fn pitch_downsample(x: &[&[f32]], x_lp: &mut [f32], len: usize) {
    for i in 1..len >> 1 {
        x_lp[i] = 0.25 * x[0][2 * i - 1] + 0.25 * x[0][2 * i + 1] + 0.5 * x[0][2 * i];
    }
    x_lp[0] = 0.25 * x[0][1] + 0.5 * x[0][0];
    if x.len() == 2 {
        for i in 1..len >> 1 {
            x_lp[i] += 0.25 * x[1][2 * i - 1] + 0.25 * x[1][2 * i + 1] + 0.5 * x[1][2 * i];
        }
        x_lp[0] += 0.25 * x[1][1] + 0.5 * x[1][0];
    }

    let mut ac = [0.0f32; 5];
    celt_autocorr(&x_lp[..len >> 1], &mut ac, &[], 0, 4);

    // Noise floor -40 dB, then lag windowing.
    ac[0] *= 1.0001;
    for (i, a) in ac.iter_mut().enumerate().skip(1) {
        *a -= *a * (0.008 * i as f32) * (0.008 * i as f32);
    }

    let mut lpc = [0.0f32; 4];
    celt_lpc(&mut lpc, &ac);
    let mut tmp = 1.0f32;
    for c in &mut lpc {
        tmp *= 0.9;
        *c *= tmp;
    }
    // Add a zero.
    let c1 = 0.8f32;
    let lpc2 = [
        lpc[0] + 0.8,
        lpc[1] + c1 * lpc[0],
        lpc[2] + c1 * lpc[1],
        lpc[3] + c1 * lpc[2],
        c1 * lpc[3],
    ];
    celt_fir5(&mut x_lp[..len >> 1], &lpc2);
}

/// The two best normalised-correlation candidates.
fn find_best_pitch(xcorr: &[f32], y: &[f32], len: usize, max_pitch: usize) -> [usize; 2] {
    let mut best_num = [-1.0f32; 2];
    let mut best_den = [0.0f32; 2];
    let mut best_pitch = [0usize, 1];
    let mut syy = 1.0f32;
    for &v in &y[..len] {
        syy += v * v;
    }
    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            // Scaled to avoid both underflow and overflow when squaring.
            let xcorr16 = xcorr[i] * 1e-12;
            let num = xcorr16 * xcorr16;
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
    }
    best_pitch
}

/// Coarse 4× search, refined 2× search, then pseudo-interpolation.
/// `y` is `x_lp` extended `max_pitch` samples into
/// the past (i.e. `x_lp = &y[max_pitch>>1..]` in the PLC caller).
pub(crate) fn pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
    let lag = len + max_pitch;

    let x_lp4: alloc::vec::Vec<f32> = (0..len >> 2).map(|j| x_lp[2 * j]).collect();
    let y_lp4: alloc::vec::Vec<f32> = (0..lag >> 2).map(|j| y[2 * j]).collect();
    let mut xcorr = vec![0.0f32; max_pitch >> 1];

    // Coarse search with 4x decimation.
    pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr, len >> 2, max_pitch >> 2);
    let best = find_best_pitch(&xcorr, &y_lp4, len >> 2, max_pitch >> 2);

    // Finer search with 2x decimation around the two candidates.
    for i in 0..max_pitch >> 1 {
        xcorr[i] = 0.0;
        if (i as i32 - 2 * best[0] as i32).abs() > 2 && (i as i32 - 2 * best[1] as i32).abs() > 2 {
            continue;
        }
        let mut sum = 0.0f32;
        for j in 0..len >> 1 {
            sum += x_lp[j] * y[i + j];
        }
        xcorr[i] = sum.max(-1.0);
    }
    let best = find_best_pitch(&xcorr, y, len >> 1, max_pitch >> 1);

    // Pseudo-interpolation.
    let offset: i32 = if best[0] > 0 && best[0] < (max_pitch >> 1) - 1 {
        let a = xcorr[best[0] - 1];
        let b = xcorr[best[0]];
        let c = xcorr[best[0] + 1];
        if c - a > 0.7 * (b - a) {
            1
        } else if a - c > 0.7 * (b - c) {
            -1
        } else {
            0
        }
    } else {
        0
    };
    (2 * best[0] as i32 - offset) as usize
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    /// Pins generated by running the reference float kernels over this exact
    /// synthetic signal.
    #[test]
    fn kernels_match_reference_pins() {
        let buf: Vec<f32> = (0..2048)
            .map(|i| {
                let i = i as f32;
                1000.0 * (2.0 * core::f32::consts::PI * i / 171.0).sin()
                    + 200.0 * (2.0 * core::f32::consts::PI * i / 53.0).sin()
            })
            .collect();
        let mut lp = [0.0f32; 1024];
        pitch_downsample(&[&buf], &mut lp, 2048);
        assert!((lp[0] - 15.097_536).abs() < 1e-3, "lp[0]={}", lp[0]);
        assert!((lp[100] - 60.605_133).abs() < 1e-3, "lp[100]={}", lp[100]);
        assert!((lp[500] - 8.821_564).abs() < 1e-3, "lp[500]={}", lp[500]);
        assert!((lp[1023] - 4.188_85).abs() < 1e-3, "lp[1023]={}", lp[1023]);

        let pitch = pitch_search(
            &lp[PLC_PITCH_LAG_MAX >> 1..],
            &lp,
            2048 - PLC_PITCH_LAG_MAX,
            PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN,
        );
        assert_eq!(PLC_PITCH_LAG_MAX - pitch, 688);

        let mut ac = [0.0f32; 25];
        celt_autocorr(&lp, &mut ac, &[], 0, 24);
        assert!((ac[0] - 1_905_144.0).abs() / ac[0] < 1e-5, "ac0={}", ac[0]);
        assert!((ac[1] - 1_881_565.4).abs() / ac[1] < 1e-5, "ac1={}", ac[1]);
        let mut lpc = [0.0f32; 24];
        celt_lpc(&mut lpc, &ac);
        assert!((lpc[0] + 1.982_927).abs() < 1e-4, "lpc0={}", lpc[0]);
        assert!((lpc[5] - 0.969_61).abs() < 1e-4, "lpc5={}", lpc[5]);
        assert!((lpc[23] - 0.143_381).abs() < 1e-4, "lpc23={}", lpc[23]);
    }
}

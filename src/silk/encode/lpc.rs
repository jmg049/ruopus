//! Linear-prediction analysis for the SILK encoder (RFC 6716 §5.2.3;
//! normative `silk/float/burg_modified_FLP.c`).
//!
//! [`burg_modified`] is Burg's method, modified as in the reference: it
//! computes short-term prediction coefficients from the signal (summed over
//! the subframes of an LPC analysis block) while enforcing a minimum inverse
//! prediction gain, so the resulting synthesis filter is guaranteed stable.
//! The accumulators are `f64` to match the reference's `double` precision.

extern crate alloc;
use alloc::vec;

/// Conditioning factor added to the zero-lag autocorrelation
/// (`FIND_LPC_COND_FAC`), regularising the solve.
const FIND_LPC_COND_FAC: f64 = 1e-5;

/// Sum of squares (`silk_energy_FLP`), accumulated in `f64`.
fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
}

/// Dot product `Σ a[i]·b[i]` (`silk_inner_product_FLP`), accumulated in `f64`.
fn inner_product(a: &[f32], b: &[f32], n: usize) -> f64 {
    (0..n).map(|i| f64::from(a[i]) * f64::from(b[i])).sum()
}

/// Burg's method, modified (`silk_burg_modified_FLP`).
///
/// Fills `a` (length `d`) with the short-term prediction coefficients and
/// returns the residual energy. `x` holds `nb_subfr` subframes of
/// `subfr_length` samples each (each subframe beginning with the `d` history
/// samples the analysis predicts from). `min_inv_gain` caps the prediction
/// gain (`1/gain`), bounding the synthesis filter away from instability.
///
/// # Panics
///
/// Panics if `a.len() < d`, `d == 0`, `d > 24`, or `x` is shorter than
/// `nb_subfr * subfr_length`.
pub(crate) fn burg_modified(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> f32 {
    assert!(d > 0 && d <= 24 && a.len() >= d);
    assert!(x.len() >= nb_subfr * subfr_length && subfr_length > d);
    let min_inv_gain = f64::from(min_inv_gain);

    // Autocorrelations summed over the subframes.
    let c0 = energy(&x[..nb_subfr * subfr_length]);
    let mut c_first_row = vec![0.0f64; d];
    for s in 0..nb_subfr {
        let xs = &x[s * subfr_length..];
        for n in 1..=d {
            c_first_row[n - 1] += inner_product(xs, &xs[n..], subfr_length - n);
        }
    }
    let mut c_last_row = c_first_row.clone();

    let mut caf = vec![0.0f64; d + 1];
    let mut cab = vec![0.0f64; d + 1];
    caf[0] = c0 + FIND_LPC_COND_FAC * c0 + 1e-9;
    cab[0] = caf[0];

    let mut af = vec![0.0f64; d];
    let mut inv_gain = 1.0f64;
    let mut reached_max_gain = false;

    for n in 0..d {
        // Update the correlation rows and C·Af / C·flipud(Af).
        for s in 0..nb_subfr {
            let xs = &x[s * subfr_length..];
            let mut tmp1 = f64::from(xs[n]);
            let mut tmp2 = f64::from(xs[subfr_length - n - 1]);
            for k in 0..n {
                c_first_row[k] -= f64::from(xs[n]) * f64::from(xs[n - k - 1]);
                c_last_row[k] -= f64::from(xs[subfr_length - n - 1]) * f64::from(xs[subfr_length - n + k]);
                let atmp = af[k];
                tmp1 += f64::from(xs[n - k - 1]) * atmp;
                tmp2 += f64::from(xs[subfr_length - n + k]) * atmp;
            }
            for k in 0..=n {
                caf[k] -= tmp1 * f64::from(xs[n - k]);
                cab[k] -= tmp2 * f64::from(xs[subfr_length - n + k - 1]);
            }
        }
        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        for k in 0..n {
            let atmp = af[k];
            tmp1 += c_last_row[n - k - 1] * atmp;
            tmp2 += c_first_row[n - k - 1] * atmp;
        }
        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;

        // Reflection (PARCOR) coefficient for this order.
        let mut num = cab[n + 1];
        let mut nrg_b = cab[0];
        let mut nrg_f = caf[0];
        for k in 0..n {
            let atmp = af[k];
            num += cab[n - k] * atmp;
            nrg_b += cab[k + 1] * atmp;
            nrg_f += caf[k + 1] * atmp;
        }
        let mut rc = -2.0 * num / (nrg_f + nrg_b);

        // Enforce the minimum inverse prediction gain.
        let tmp = inv_gain * (1.0 - rc * rc);
        if tmp <= min_inv_gain {
            rc = (1.0 - min_inv_gain / inv_gain).sqrt();
            if num > 0.0 {
                rc = -rc;
            }
            inv_gain = min_inv_gain;
            reached_max_gain = true;
        } else {
            inv_gain = tmp;
        }

        // Update the AR coefficients.
        for k in 0..(n + 1) >> 1 {
            let t1 = af[k];
            let t2 = af[n - k - 1];
            af[k] = t1 + rc * t2;
            af[n - k - 1] = t2 + rc * t1;
        }
        af[n] = rc;

        if reached_max_gain {
            for af_k in af.iter_mut().take(d).skip(n + 1) {
                *af_k = 0.0;
            }
            break;
        }

        // Update C·Af and C·Ab.
        for k in 0..=n + 1 {
            let t1 = caf[k];
            caf[k] += rc * cab[n - k + 1];
            cab[n - k + 1] += rc * t1;
        }
    }

    if reached_max_gain {
        for k in 0..d {
            a[k] = -af[k] as f32;
        }
        // Subtract the energy of the preceding samples from C0.
        let mut c0 = c0;
        for s in 0..nb_subfr {
            c0 -= energy(&x[s * subfr_length..s * subfr_length + d]);
        }
        (c0 * inv_gain) as f32
    } else {
        let mut nrg_f = caf[0];
        let mut tmp1 = 1.0f64;
        for k in 0..d {
            let atmp = af[k];
            nrg_f += caf[k + 1] * atmp;
            tmp1 += atmp * atmp;
            a[k] = -atmp as f32;
        }
        nrg_f -= FIND_LPC_COND_FAC * c0 * tmp1;
        nrg_f as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Burg should recover the coefficients of a synthetic AR(2) process and
    /// reduce the residual energy far below the input energy.
    #[test]
    fn recovers_ar2_coefficients() {
        // x[n] = 1.3·x[n-1] - 0.6·x[n-2] + e[n]; a stable AR(2).
        let (a1, a2) = (1.3f32, -0.6f32);
        let n = 480usize;
        let mut x = vec![0.0f32; n];
        // Deterministic pseudo-noise excitation.
        let mut seed = 12345u32;
        let mut prev = (0.0f32, 0.0f32);
        for v in &mut x {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let e = ((seed >> 16) as f32 / 32_768.0) - 1.0;
            let s = a1 * prev.0 + a2 * prev.1 + 0.1 * e;
            *v = s;
            prev = (s, prev.0);
        }

        let mut a = [0.0f32; 16];
        let order = 2usize;
        let resid = burg_modified(&mut a, &x, 1.0 / 1e4, n, 1, order);

        // The predictor coefficients should be close to (a1, a2).
        assert!((a[0] - a1).abs() < 0.1, "a0={} expected {a1}", a[0]);
        assert!((a[1] - a2).abs() < 0.1, "a1={} expected {a2}", a[1]);
        // The predictor whitens the signal: the residual collapses to the
        // excitation energy, far below the AR-amplified input energy.
        let input_energy = energy(&x) as f32;
        assert!(
            resid > 0.0 && resid < 0.3 * input_energy,
            "resid {resid} vs energy {input_energy}"
        );
    }

    /// The minimum-inverse-gain cap must keep the filter stable: the returned
    /// residual energy stays positive and finite for a near-singular input.
    #[test]
    fn enforces_stability_on_a_pure_tone() {
        let n = 320usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut a = [0.0f32; 16];
        let resid = burg_modified(&mut a, &x, 1.0 / 1e4, n, 1, 10);
        assert!(resid.is_finite() && resid >= 0.0, "resid {resid}");
        assert!(a.iter().all(|v| v.is_finite() && v.abs() < 8.0), "coefs {a:?}");
    }
}

//! Low-level LPC bindings: autocorrelation, Levinson-Durbin, LPC residual /
//! synthesis, and long-term prediction (LTP) helpers.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// LPC prediction coefficients returned by :func:`lpc_analysis` or
/// :func:`levinson_durbin`.
#[pyclass(module = "ruopus.lowlevel", name = "LpcCoefficients")]
#[derive(Clone)]
pub struct LpcCoefficients {
    order: usize,
    /// High-precision coefficients for internal Rust use.
    coeffs_f64: Vec<f64>,
}

#[pymethods]
impl LpcCoefficients {
    /// Number of prediction coefficients.
    #[getter]
    fn order(&self) -> usize {
        self.order
    }

    /// Prediction coefficients ``a[0..order]`` such that
    /// ``e[n] = x[n] - a[0]*x[n-1] - ... - a[order-1]*x[n-order]``.
    #[getter]
    fn coeffs(&self) -> Vec<f32> {
        self.coeffs_f64.iter().map(|&x| x as f32).collect()
    }

    fn __repr__(&self) -> String {
        format!("LpcCoefficients(order={})", self.order)
    }
}

// ── internal helpers ────────────────────────────────────────────────────────

fn autocorrelation_impl(sig: &[f32], order: usize) -> Vec<f64> {
    let n = sig.len();
    let mut ac = vec![0.0f64; order + 1];
    for lag in 0..=order {
        let mut sum = 0.0f64;
        for i in lag..n {
            sum += f64::from(sig[i]) * f64::from(sig[i - lag]);
        }
        ac[lag] = sum;
    }
    ac
}

fn levinson_solve(r: &[f64], order: usize) -> Option<Vec<f64>> {
    if r.len() < order + 1 || r[0] == 0.0 {
        return None;
    }
    let mut a = vec![0.0f64; order];
    let mut err = r[0];
    for m in 0..order {
        let mut lambda = r[m + 1];
        for k in 0..m {
            lambda += a[k] * r[m - k];
        }
        let km = -lambda / err;
        let mut new_a = a.clone();
        for k in 0..m {
            new_a[k] += km * a[m - 1 - k];
        }
        new_a[m] = km;
        a = new_a;
        err *= 1.0 - km * km;
        if err <= 0.0 {
            return None;
        }
    }
    Some(a)
}

fn residual_impl(sig: &[f32], coeffs: &[f64], history: &[f32]) -> Vec<f32> {
    let order = coeffs.len();
    let h: Vec<f64> = history.iter().map(|&x| f64::from(x)).collect();
    let mut out = vec![0.0f32; sig.len()];
    for n in 0..sig.len() {
        let mut pred = 0.0f64;
        for k in 0..order {
            let sample = if n > k {
                f64::from(sig[n - 1 - k])
            } else {
                let hist_idx = h.len().saturating_sub(1 + k - n);
                *h.get(hist_idx).unwrap_or(&0.0)
            };
            pred += coeffs[k] * sample;
        }
        out[n] = (f64::from(sig[n]) - pred) as f32;
    }
    out
}

fn synthesis_impl(res: &[f32], coeffs: &[f64], history: &[f32]) -> Vec<f32> {
    let order = coeffs.len();
    let h: Vec<f64> = history.iter().map(|&x| f64::from(x)).collect();
    let mut out_f64 = vec![0.0f64; res.len()];
    for n in 0..res.len() {
        let mut pred = 0.0f64;
        for k in 0..order {
            let sample = if n > k {
                out_f64[n - 1 - k]
            } else {
                let hist_idx = h.len().saturating_sub(1 + k - n);
                *h.get(hist_idx).unwrap_or(&0.0)
            };
            pred += coeffs[k] * sample;
        }
        out_f64[n] = f64::from(res[n]) + pred;
    }
    out_f64.iter().map(|&x| x as f32).collect()
}

// ── Python functions ────────────────────────────────────────────────────────

/// Compute the biased autocorrelation sequence ``R[0..=order]`` of `sig`.
///
/// Parameters
/// ----------
/// sig : numpy.ndarray
///     1-D ``float32`` input signal.
/// order : int
///     Maximum lag (output length is ``order + 1``).
///
/// Returns
/// -------
/// numpy.ndarray
///     1-D ``float64`` array of shape ``(order + 1,)``.
#[pyfunction]
pub fn compute_autocorrelation<'py>(
    py: Python<'py>,
    sig: PyReadonlyArray1<'_, f32>,
    order: usize,
) -> Bound<'py, PyArray1<f64>> {
    let ac = autocorrelation_impl(sig.as_slice().unwrap(), order);
    PyArray1::from_vec(py, ac)
}

/// Solve the Yule-Walker equations via the Levinson-Durbin recursion.
///
/// Parameters
/// ----------
/// ac : numpy.ndarray
///     1-D ``float64`` autocorrelation sequence of length ``>= order + 1``
///     (as returned by :func:`compute_autocorrelation`).
/// order : int
///     AR model order.
///
/// Returns
/// -------
/// LpcCoefficients or None
///     ``None`` when the autocorrelation matrix is singular or the filter
///     becomes unstable during the recursion.
#[pyfunction]
pub fn levinson_durbin(_py: Python<'_>, ac: PyReadonlyArray1<'_, f64>, order: usize) -> Option<LpcCoefficients> {
    let r = ac.as_slice().unwrap();
    levinson_solve(r, order).map(|coeffs_f64| LpcCoefficients { order, coeffs_f64 })
}

/// Estimate LPC coefficients of order `order` from signal `sig`.
///
/// Uses the autocorrelation method (Yule-Walker + Levinson-Durbin) with a
/// small regularisation factor (``1e-4 * R[0]``) to ensure positive-definiteness
/// in ill-conditioned cases (e.g. pure tones at high order).
///
/// Parameters
/// ----------
/// sig : numpy.ndarray
///     1-D ``float32`` input signal.
/// order : int
///     AR model order.
///
/// Returns
/// -------
/// LpcCoefficients
///
/// Raises
/// ------
/// ValueError
///     If the autocorrelation matrix is singular even after regularisation.
#[pyfunction]
pub fn lpc_analysis(_py: Python<'_>, sig: PyReadonlyArray1<'_, f32>, order: usize) -> PyResult<LpcCoefficients> {
    let mut ac = autocorrelation_impl(sig.as_slice().unwrap(), order);
    // Regularise R[0] to keep the Toeplitz matrix positive-definite even for
    // near-singular cases such as pure sinusoids at high LPC order.
    ac[0] *= 1.0 + 1e-4;
    levinson_solve(&ac, order)
        .map(|coeffs_f64| LpcCoefficients { order, coeffs_f64 })
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("singular autocorrelation matrix"))
}

/// Apply the LPC analysis filter to `sig`, returning the prediction residual.
///
/// ``e[n] = sig[n] - coeffs[0]*sig[n-1] - ... - coeffs[p-1]*sig[n-p]``
///
/// Samples before the start of `sig` are treated as zero.
#[pyfunction]
pub fn lpc_residual<'py>(
    py: Python<'py>,
    sig: PyReadonlyArray1<'_, f32>,
    coeffs: &LpcCoefficients,
) -> Bound<'py, PyArray1<f32>> {
    let out = residual_impl(sig.as_slice().unwrap(), &coeffs.coeffs_f64, &[]);
    PyArray1::from_vec(py, out)
}

/// Apply the LPC synthesis filter to `res`, reconstructing the signal.
///
/// ``x[n] = res[n] + coeffs[0]*x[n-1] + ... + coeffs[p-1]*x[n-p]``
///
/// Samples before the start of `res` are treated as zero.
#[pyfunction]
pub fn lpc_synthesis<'py>(
    py: Python<'py>,
    res: PyReadonlyArray1<'_, f32>,
    coeffs: &LpcCoefficients,
) -> Bound<'py, PyArray1<f32>> {
    let out = synthesis_impl(res.as_slice().unwrap(), &coeffs.coeffs_f64, &[]);
    PyArray1::from_vec(py, out)
}

/// Stateful LPC analysis filter: returns ``(residual, state)``.
///
/// `state` is a Python list of the last ``order`` input samples, suitable for
/// passing to a subsequent :func:`lpc_synthesis_stateful` call.
///
/// Parameters
/// ----------
/// sig : numpy.ndarray
///     1-D ``float32`` input signal.
/// coeffs : LpcCoefficients
///     LPC coefficients.
///
/// Returns
/// -------
/// tuple[numpy.ndarray, list[float]]
///     Residual array (``float32``) and filter state (list of ``float``).
#[pyfunction]
pub fn lpc_residual_stateful<'py>(
    py: Python<'py>,
    sig: PyReadonlyArray1<'_, f32>,
    coeffs: &LpcCoefficients,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<f32>)> {
    let s = sig.as_slice().unwrap();
    let out = residual_impl(s, &coeffs.coeffs_f64, &[]);
    let order = coeffs.order;
    let state: Vec<f32> = if s.len() >= order {
        s[s.len() - order..].to_vec()
    } else {
        let mut st = vec![0.0f32; order - s.len()];
        st.extend_from_slice(s);
        st
    };
    Ok((PyArray1::from_vec(py, out), state))
}

/// Stateful LPC synthesis filter: returns ``(signal, state)``.
///
/// Parameters
/// ----------
/// res : numpy.ndarray
///     1-D ``float32`` residual.
/// coeffs : LpcCoefficients
///     LPC coefficients.
/// state : list[float], optional
///     Filter state from a previous call (length must equal ``coeffs.order``).
///
/// Returns
/// -------
/// tuple[numpy.ndarray, list[float]]
///     Reconstructed signal (``float32``) and updated filter state.
#[pyfunction]
#[pyo3(signature = (res, coeffs, state = None))]
pub fn lpc_synthesis_stateful<'py>(
    py: Python<'py>,
    res: PyReadonlyArray1<'_, f32>,
    coeffs: &LpcCoefficients,
    state: Option<Vec<f32>>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<f32>)> {
    let r = res.as_slice().unwrap();
    let history = state.unwrap_or_else(|| vec![0.0f32; coeffs.order]);
    let out = synthesis_impl(r, &coeffs.coeffs_f64, &history);
    let order = coeffs.order;
    let new_state: Vec<f32> = if out.len() >= order {
        out[out.len() - order..].to_vec()
    } else {
        let mut st = history[out.len()..].to_vec();
        st.extend_from_slice(&out);
        st
    };
    Ok((PyArray1::from_vec(py, out), new_state))
}

/// Estimate the fundamental pitch lag of `sig` by peak-picking the
/// normalized autocorrelation in ``[min_lag, max_lag]``.
///
/// Parameters
/// ----------
/// sig : numpy.ndarray
///     1-D ``float32`` input signal.
/// min_lag : int, optional
///     Minimum lag to consider (default 20).
/// max_lag : int, optional
///     Maximum lag to consider (default 200).
///
/// Returns
/// -------
/// int
///     Estimated pitch lag in samples, or ``min_lag`` if the signal is silent.
#[pyfunction]
#[pyo3(signature = (sig, min_lag = 20, max_lag = 200))]
pub fn estimate_pitch(_py: Python<'_>, sig: PyReadonlyArray1<'_, f32>, min_lag: usize, max_lag: usize) -> usize {
    let s = sig.as_slice().unwrap();
    let energy: f64 = s.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    if energy == 0.0 {
        return min_lag;
    }
    let max_lag = max_lag.min(s.len().saturating_sub(1));
    let mut best_lag = min_lag;
    let mut best_corr = f64::NEG_INFINITY;
    for lag in min_lag..=max_lag {
        let mut corr = 0.0f64;
        let mut norm = 0.0f64;
        for i in lag..s.len() {
            corr += f64::from(s[i]) * f64::from(s[i - lag]);
            norm += f64::from(s[i - lag]) * f64::from(s[i - lag]);
        }
        let ncorr = if norm > 0.0 { corr / norm.sqrt() } else { 0.0 };
        if ncorr > best_corr {
            best_corr = ncorr;
            best_lag = lag;
        }
    }
    best_lag
}

/// Long-term prediction residual: subtract a single-tap LTP predictor.
///
/// ``e[n] = sig[n] - gain * sig[n - lag]``
///
/// Samples before the start of `sig` (index ``< lag``) are treated as zero.
///
/// Parameters
/// ----------
/// sig : numpy.ndarray
///     1-D ``float32`` input signal.
/// lag : int
///     Pitch lag in samples.
/// gain : float
///     LTP gain coefficient.
///
/// Returns
/// -------
/// numpy.ndarray
///     1-D ``float32`` LTP residual, same shape as `sig`.
#[pyfunction]
pub fn ltp_residual<'py>(
    py: Python<'py>,
    sig: PyReadonlyArray1<'_, f32>,
    lag: usize,
    gain: f32,
) -> Bound<'py, PyArray1<f32>> {
    let s = sig.as_slice().unwrap();
    let out: Vec<f32> = s
        .iter()
        .enumerate()
        .map(|(n, &x)| {
            let pred = if n >= lag { gain * s[n - lag] } else { 0.0 };
            x - pred
        })
        .collect();
    PyArray1::from_vec(py, out)
}

/// Long-term prediction synthesis: reconstruct a signal from an LTP residual.
///
/// ``x[n] = res[n] + gain * x[n - lag]``
///
/// Parameters
/// ----------
/// res : numpy.ndarray
///     1-D ``float32`` LTP residual.
/// lag : int
///     Pitch lag in samples.
/// gain : float
///     LTP gain coefficient.
///
/// Returns
/// -------
/// numpy.ndarray
///     1-D ``float32`` reconstructed signal, same shape as `res`.
#[pyfunction]
pub fn ltp_synthesis<'py>(
    py: Python<'py>,
    res: PyReadonlyArray1<'_, f32>,
    lag: usize,
    gain: f32,
) -> Bound<'py, PyArray1<f32>> {
    let r = res.as_slice().unwrap();
    let mut out = vec![0.0f32; r.len()];
    for n in 0..r.len() {
        let pred = if n >= lag { gain * out[n - lag] } else { 0.0 };
        out[n] = r[n] + pred;
    }
    PyArray1::from_vec(py, out)
}

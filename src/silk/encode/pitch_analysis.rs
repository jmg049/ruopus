//! Pitch (lag) estimation for the SILK encoder (RFC 6716 §5.2; normative
//! `silk/float/pitch_analysis_core_FLP.c`).
//!
//! [`pitch_analysis_core`] decides voiced/unvoiced and, when voiced, finds
//! the per-subframe pitch lags. It works in three stages of increasing
//! resolution: a coarse normalised-correlation search at 4 kHz that lists
//! candidate lags, a refinement at 8 kHz over a small contour codebook, and
//! (above 8 kHz) a final search at the internal rate. The lag/contour
//! indices it emits are read back by the decoder's `decode_pitch`.

extern crate alloc;
use alloc::vec;

use super::super::indices::MAX_NB_SUBFR;
use super::super::tables::{CB_LAGS_STAGE2, CB_LAGS_STAGE2_10_MS, CB_LAGS_STAGE3, CB_LAGS_STAGE3_10_MS};
use super::resample::down2;

const PE_LTP_MEM_LENGTH_MS: usize = 20;
const PE_SUBFR_LENGTH_MS: usize = 5;
const PE_MIN_LAG_MS: i32 = 2;
const PE_MAX_LAG_MS: i32 = 18;
const PE_MAX_LAG: usize = (PE_MAX_LAG_MS * 16) as usize;
const PE_D_SRCH_LENGTH: usize = 24;
const PE_NB_STAGE3_LAGS: usize = 5;
const PE_NB_CBKS_STAGE2: usize = 3;
const PE_NB_CBKS_STAGE2_EXT: usize = 11;
const PE_NB_CBKS_STAGE2_10MS: usize = 3;
const PE_NB_CBKS_STAGE3_MAX: usize = 34;
const PE_NB_CBKS_STAGE3_10MS: usize = 12;
const PE_SHORTLAG_BIAS: f32 = 0.2;
const PE_PREVLAG_BIAS: f32 = 0.2;
const PE_FLATCONTOUR_BIAS: f32 = 0.05;
const C_STRIDE: usize = (PE_MAX_LAG >> 1) + 5;

/// `silk_Lag_range_stage3` - `[complexity][subframe][lo, hi]`.
const LAG_RANGE_STAGE3: [[[i8; 2]; 4]; 3] = [
    [[-5, 8], [-1, 6], [-1, 6], [-4, 10]],
    [[-6, 10], [-2, 6], [-1, 6], [-5, 10]],
    [[-9, 12], [-3, 7], [-2, 7], [-7, 13]],
];
/// `silk_Lag_range_stage3_10_ms`.
const LAG_RANGE_STAGE3_10MS: [[i8; 2]; 2] = [[-3, 7], [-2, 7]];
/// `silk_nb_cbk_searchs_stage3` (`PE_NB_CBKS_STAGE3_{MIN,MID,MAX}`).
const NB_CBK_SEARCHS_STAGE3: [usize; 3] = [16, 24, 34];

fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
}

fn inner(a: &[f32], b: &[f32], n: usize) -> f64 {
    (0..n).map(|i| f64::from(a[i]) * f64::from(b[i])).sum()
}

/// `celt_pitch_xcorr`: `out[i] = <x, y[i..]>` for `i` in `0..max`.
fn xcorr(x: &[f32], y: &[f32], out: &mut [f32], len: usize, max: usize) {
    for (i, o) in out.iter_mut().enumerate().take(max) {
        *o = inner(x, &y[i..], len) as f32;
    }
}

/// `silk_float2short_array`: round-to-nearest with saturation to i16.
fn float2short(out: &mut [i16], x: &[f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v.round().clamp(-32768.0, 32767.0) as i16;
    }
}

/// `silk_insertion_sort_decreasing_FLP`: sort the largest `k` of `a`
/// descending, recording their original indices in `idx`.
fn insertion_sort_decreasing(a: &mut [f32], idx: &mut [usize], l: usize, k: usize) {
    for (i, id) in idx.iter_mut().enumerate().take(k) {
        *id = i;
    }
    for i in 1..k {
        let value = a[i];
        let mut j = i as isize - 1;
        while j >= 0 && value > a[j as usize] {
            a[(j + 1) as usize] = a[j as usize];
            idx[(j + 1) as usize] = idx[j as usize];
            j -= 1;
        }
        a[(j + 1) as usize] = value;
        idx[(j + 1) as usize] = i;
    }
    for i in k..l {
        let value = a[i];
        if value > a[k - 1] {
            let mut j = k as isize - 2;
            while j >= 0 && value > a[j as usize] {
                a[(j + 1) as usize] = a[j as usize];
                idx[(j + 1) as usize] = idx[j as usize];
                j -= 1;
            }
            a[(j + 1) as usize] = value;
            idx[(j + 1) as usize] = i;
        }
    }
}

/// `silk_pitch_analysis_core_FLP`. `frame` is the whitened residual of
/// length `(20 + nb_subfr*5) * fs_khz`. Returns
/// `(voicing, pitch_out, lag_index, contour_index)` where `voicing` is 0 for
/// voiced and 1 for unvoiced; `ltp_corr` is updated with the normalised
/// correlation (and read as the previous frame's value on entry).
#[allow(clippy::too_many_arguments, reason = "mirrors the reference signature")]
#[allow(clippy::needless_range_loop, reason = "computed index ranges mirror the reference")]
pub(crate) fn pitch_analysis_core(
    frame: &[f32],
    fs_khz: i32,
    complexity: usize,
    nb_subfr: usize,
    mut prev_lag: i32,
    search_thres1: f32,
    search_thres2: f32,
    ltp_corr: &mut f32,
) -> (i32, [i32; MAX_NB_SUBFR], i16, i8) {
    let n_units = PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS;
    let frame_length = n_units * fs_khz as usize;
    let frame_length_8khz = n_units * 8;
    let frame_length_4khz = n_units * 4;
    let sf_length = PE_SUBFR_LENGTH_MS * fs_khz as usize;
    let sf_length_8khz = PE_SUBFR_LENGTH_MS * 8;
    let sf_length_4khz = PE_SUBFR_LENGTH_MS * 4;
    let min_lag = PE_MIN_LAG_MS * fs_khz;
    let min_lag_4khz = PE_MIN_LAG_MS * 4;
    let min_lag_8khz = PE_MIN_LAG_MS * 8;
    let max_lag = PE_MAX_LAG_MS * fs_khz - 1;
    let max_lag_4khz = PE_MAX_LAG_MS * 4;
    let max_lag_8khz = PE_MAX_LAG_MS * 8 - 1;

    let mut pitch_out = [0i32; MAX_NB_SUBFR];

    // --- Resample to 8 kHz then 4 kHz ---
    let mut frame_8khz = vec![0.0f32; frame_length_8khz];
    let mut frame_8_fix = vec![0i16; frame_length_8khz];
    if fs_khz == 16 {
        let mut frame_16_fix = vec![0i16; frame_length];
        float2short(&mut frame_16_fix, &frame[..frame_length]);
        let mut s = [0i32; 2];
        down2(&mut s, &mut frame_8_fix, &frame_16_fix);
        for (o, &v) in frame_8khz.iter_mut().zip(frame_8_fix.iter()) {
            *o = f32::from(v);
        }
    } else if fs_khz == 12 {
        let mut frame_12_fix = vec![0i16; frame_length];
        float2short(&mut frame_12_fix, &frame[..frame_length]);
        let mut s = [0i32; 6];
        super::resample::down2_3(&mut s, &mut frame_8_fix, &frame_12_fix);
        for (o, &v) in frame_8khz.iter_mut().zip(frame_8_fix.iter()) {
            *o = f32::from(v);
        }
    } else {
        // 8 kHz: the input is already at the second-stage rate.
        float2short(&mut frame_8_fix, &frame[..frame_length_8khz]);
    }

    let mut frame_4_fix = vec![0i16; frame_length_4khz];
    let mut s = [0i32; 2];
    down2(&mut s, &mut frame_4_fix, &frame_8_fix);
    let mut frame_4khz: alloc::vec::Vec<f32> = frame_4_fix.iter().map(|&v| f32::from(v)).collect();

    // Low-pass (running sum) on the 4 kHz signal.
    for i in (1..frame_length_4khz).rev() {
        frame_4khz[i] = (frame_4khz[i] + frame_4khz[i - 1]).clamp(-32768.0, 32767.0);
    }

    // --- First stage: coarse normalised correlation at 4 kHz ---
    let mut c = vec![0.0f32; MAX_NB_SUBFR * C_STRIDE];
    let mut xcorr_buf = vec![0.0f32; (max_lag_4khz - min_lag_4khz + 1) as usize];
    let mut tgt = sf_length_4khz << 2; // middle of frame
    for _k in 0..nb_subfr >> 1 {
        let basis0 = (tgt as i32 - min_lag_4khz) as usize;
        xcorr(
            &frame_4khz[tgt..],
            &frame_4khz[(tgt as i32 - max_lag_4khz) as usize..],
            &mut xcorr_buf,
            sf_length_8khz,
            (max_lag_4khz - min_lag_4khz + 1) as usize,
        );
        let mut cross = f64::from(xcorr_buf[(max_lag_4khz - min_lag_4khz) as usize]);
        let mut normalizer = energy(&frame_4khz[tgt..tgt + sf_length_8khz])
            + energy(&frame_4khz[basis0..basis0 + sf_length_8khz])
            + sf_length_8khz as f64 * 4000.0;
        c[min_lag_4khz as usize] += (2.0 * cross / normalizer) as f32;
        for d in min_lag_4khz + 1..=max_lag_4khz {
            let basis = (tgt as i32 - d) as usize;
            cross = f64::from(xcorr_buf[(max_lag_4khz - d) as usize]);
            normalizer += f64::from(frame_4khz[basis]) * f64::from(frame_4khz[basis])
                - f64::from(frame_4khz[basis + sf_length_8khz]) * f64::from(frame_4khz[basis + sf_length_8khz]);
            c[d as usize] += (2.0 * cross / normalizer) as f32;
        }
        tgt += sf_length_8khz;
    }

    // Short-lag bias.
    for i in (min_lag_4khz..=max_lag_4khz).rev() {
        c[i as usize] -= c[i as usize] * i as f32 / 4096.0;
    }

    // Sort the strongest candidates.
    let mut length_d_srch = 4 + 2 * complexity;
    let mut d_srch = [0i32; PE_D_SRCH_LENGTH];
    {
        let mut idx = [0usize; (PE_MAX_LAG >> 1) + 5];
        let span = (max_lag_4khz - min_lag_4khz + 1) as usize;
        insertion_sort_decreasing(
            &mut c[min_lag_4khz as usize..min_lag_4khz as usize + span],
            &mut idx,
            span,
            length_d_srch,
        );
        for i in 0..length_d_srch {
            d_srch[i] = idx[i] as i32;
        }
    }

    let cmax = c[min_lag_4khz as usize];
    if cmax < 0.2 {
        *ltp_corr = 0.0;
        return (1, [0; MAX_NB_SUBFR], 0, 0);
    }

    let threshold = search_thres1 * cmax;
    for i in 0..length_d_srch {
        if c[min_lag_4khz as usize + i] > threshold {
            d_srch[i] = (d_srch[i] + min_lag_4khz) << 1;
        } else {
            length_d_srch = i;
            break;
        }
    }

    // Build the lag mask at 8 kHz and convolve to widen it.
    let mut d_comp = [0i16; (PE_MAX_LAG >> 1) + 5];
    for i in 0..length_d_srch {
        d_comp[d_srch[i] as usize] = 1;
    }
    for i in (min_lag_8khz..=max_lag_8khz + 3).rev() {
        d_comp[i as usize] += d_comp[(i - 1) as usize] + d_comp[(i - 2) as usize];
    }
    length_d_srch = 0;
    for i in min_lag_8khz..max_lag_8khz + 1 {
        if d_comp[(i + 1) as usize] > 0 {
            d_srch[length_d_srch] = i;
            length_d_srch += 1;
        }
    }
    for i in (min_lag_8khz..=max_lag_8khz + 3).rev() {
        d_comp[i as usize] += d_comp[(i - 1) as usize] + d_comp[(i - 2) as usize] + d_comp[(i - 3) as usize];
    }
    let mut length_d_comp = 0usize;
    for i in min_lag_8khz..max_lag_8khz + 4 {
        if d_comp[i as usize] > 0 {
            d_comp[length_d_comp] = (i - 2) as i16;
            length_d_comp += 1;
        }
    }

    // --- Second stage: refine at 8 kHz over the contour codebook ---
    for v in c.iter_mut() {
        *v = 0.0;
    }
    let stage2: &[f32] = if fs_khz == 8 { frame } else { &frame_8khz };
    let mut tgt = PE_LTP_MEM_LENGTH_MS * 8;
    for k in 0..nb_subfr {
        let energy_tmp = energy(&stage2[tgt..tgt + sf_length_8khz]) + 1.0;
        for j in 0..length_d_comp {
            let d = d_comp[j] as usize;
            let basis = tgt - d;
            let cross = inner(&stage2[basis..], &stage2[tgt..], sf_length_8khz);
            if cross > 0.0 {
                let e = energy(&stage2[basis..basis + sf_length_8khz]);
                c[k * C_STRIDE + d] = (2.0 * cross / (e + energy_tmp)) as f32;
            }
        }
        tgt += sf_length_8khz;
    }

    let mut ccmax = 0.0f32;
    let mut ccmax_b = -1000.0f32;
    let mut cb_imax = 0usize;
    let mut lag = -1i32;
    let prev_lag_log2 = if prev_lag > 0 {
        if fs_khz == 12 {
            prev_lag = (prev_lag << 1) / 3;
        } else if fs_khz == 16 {
            prev_lag >>= 1;
        }
        (prev_lag as f32).log2()
    } else {
        0.0
    };

    let (cbk_size, lag_cb, nb_cbk_search): (usize, &[[i8; 11]], usize) = if nb_subfr == MAX_NB_SUBFR {
        let n = if fs_khz == 8 && complexity > 0 {
            PE_NB_CBKS_STAGE2_EXT
        } else {
            PE_NB_CBKS_STAGE2
        };
        (PE_NB_CBKS_STAGE2_EXT, &CB_LAGS_STAGE2, n)
    } else {
        (PE_NB_CBKS_STAGE2_10MS, &[], PE_NB_CBKS_STAGE2_10MS)
    };

    let mut cc = [0.0f32; PE_NB_CBKS_STAGE2_EXT];
    for k in 0..length_d_srch {
        let d = d_srch[k];
        for j in 0..nb_cbk_search {
            cc[j] = 0.0;
            for i in 0..nb_subfr {
                let off = if nb_subfr == MAX_NB_SUBFR {
                    i32::from(lag_cb[i][j])
                } else {
                    i32::from(CB_LAGS_STAGE2_10_MS[i][j])
                };
                cc[j] += c[i * C_STRIDE + (d + off) as usize];
            }
        }
        let mut ccmax_new = -1000.0f32;
        let mut cb_imax_new = 0usize;
        for i in 0..nb_cbk_search {
            if cc[i] > ccmax_new {
                ccmax_new = cc[i];
                cb_imax_new = i;
            }
        }
        let lag_log2 = (d as f32).log2();
        let mut ccmax_new_b = ccmax_new - PE_SHORTLAG_BIAS * nb_subfr as f32 * lag_log2;
        if prev_lag > 0 {
            let mut delta = lag_log2 - prev_lag_log2;
            delta *= delta;
            ccmax_new_b -= PE_PREVLAG_BIAS * nb_subfr as f32 * *ltp_corr * delta / (delta + 0.5);
        }
        if ccmax_new_b > ccmax_b && ccmax_new > nb_subfr as f32 * search_thres2 {
            ccmax_b = ccmax_new_b;
            ccmax = ccmax_new;
            lag = d;
            cb_imax = cb_imax_new;
        }
    }

    if lag == -1 {
        *ltp_corr = 0.0;
        return (1, [0; MAX_NB_SUBFR], 0, 0);
    }
    *ltp_corr = ccmax / nb_subfr as f32;

    if fs_khz > 8 {
        // --- Third stage: refine at the internal rate ---
        lag = if fs_khz == 12 {
            ((lag * 3 + 1) >> 1).clamp(min_lag, max_lag)
        } else {
            (lag << 1).clamp(min_lag, max_lag)
        };
        let start_lag = (lag - 2).max(min_lag);
        let end_lag = (lag + 2).min(max_lag);
        let mut lag_new = lag;
        cb_imax = 0;
        let contour_bias = PE_FLATCONTOUR_BIAS / lag as f32;

        let (nb_cbk_search3, cbk_size3): (usize, usize) = if nb_subfr == MAX_NB_SUBFR {
            (NB_CBK_SEARCHS_STAGE3[complexity], PE_NB_CBKS_STAGE3_MAX)
        } else {
            (PE_NB_CBKS_STAGE3_10MS, PE_NB_CBKS_STAGE3_10MS)
        };

        let cross_st3 = calc_corr_st3(frame, start_lag, sf_length, nb_subfr, complexity);
        let energies_st3 = calc_energy_st3(frame, start_lag, sf_length, nb_subfr, complexity);

        let tgt0 = PE_LTP_MEM_LENGTH_MS * fs_khz as usize;
        let energy_tmp = energy(&frame[tgt0..tgt0 + nb_subfr * sf_length]) + 1.0;
        let mut ccmax3 = -1000.0f32;
        for (lag_counter, d) in (start_lag..=end_lag).enumerate() {
            for j in 0..nb_cbk_search3 {
                let mut cross = 0.0f64;
                let mut e = energy_tmp;
                for k in 0..nb_subfr {
                    cross += f64::from(cross_st3[k][j][lag_counter]);
                    e += f64::from(energies_st3[k][j][lag_counter]);
                }
                let mut ccmax_new = if cross > 0.0 { (2.0 * cross / e) as f32 } else { 0.0 };
                ccmax_new *= 1.0 - contour_bias * j as f32;
                if ccmax_new > ccmax3 && d + i32::from(CB_LAGS_STAGE3[0][j]) <= max_lag {
                    ccmax3 = ccmax_new;
                    lag_new = d;
                    cb_imax = j;
                }
            }
        }

        for k in 0..nb_subfr {
            let off = if nb_subfr == MAX_NB_SUBFR {
                i32::from(CB_LAGS_STAGE3[k][cb_imax])
            } else {
                i32::from(CB_LAGS_STAGE3_10_MS[k][cb_imax])
            };
            pitch_out[k] = (lag_new + off).clamp(min_lag, PE_MAX_LAG_MS * fs_khz);
        }
        let _ = cbk_size3;
        ((0), pitch_out, (lag_new - min_lag) as i16, cb_imax as i8)
    } else {
        for k in 0..nb_subfr {
            let off = if nb_subfr == MAX_NB_SUBFR {
                i32::from(lag_cb[k][cb_imax])
            } else {
                i32::from(CB_LAGS_STAGE2_10_MS[k][cb_imax])
            };
            pitch_out[k] = (lag + off).clamp(min_lag_8khz, PE_MAX_LAG_MS * 8);
        }
        let _ = cbk_size;
        (0, pitch_out, (lag - min_lag_8khz) as i16, cb_imax as i8)
    }
}

type St3 = alloc::vec::Vec<[[f32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]>;

/// `silk_P_Ana_calc_corr_st3`: stage-3 cross-correlations per subframe,
/// codebook vector and lag offset.
#[allow(clippy::needless_range_loop, reason = "computed index ranges mirror the reference")]
fn calc_corr_st3(frame: &[f32], start_lag: i32, sf_length: usize, nb_subfr: usize, complexity: usize) -> St3 {
    let mut out: St3 = vec![[[0.0; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; nb_subfr];
    let (nb_cbk_search, cbk_max): (usize, bool) = if nb_subfr == MAX_NB_SUBFR {
        (NB_CBK_SEARCHS_STAGE3[complexity], true)
    } else {
        (PE_NB_CBKS_STAGE3_10MS, false)
    };
    let mut tgt = sf_length << 2;
    for k in 0..nb_subfr {
        let (lag_low, lag_high) = lag_range(k, complexity, cbk_max);
        let span = (lag_high - lag_low + 1) as usize;
        let mut scratch = [0.0f32; 22];
        let base = (tgt as i32 - start_lag - lag_high) as usize;
        let mut xb = [0.0f32; 22];
        xcorr(&frame[tgt..], &frame[base..], &mut xb, sf_length, span);
        for (lc, j) in (lag_low..=lag_high).enumerate() {
            scratch[lc] = xb[(lag_high - j) as usize];
        }
        let delta = lag_low;
        for i in 0..nb_cbk_search {
            let cb = if cbk_max {
                i32::from(CB_LAGS_STAGE3[k][i])
            } else {
                i32::from(CB_LAGS_STAGE3_10_MS[k][i])
            };
            let idx = (cb - delta) as usize;
            out[k][i].copy_from_slice(&scratch[idx..idx + PE_NB_STAGE3_LAGS]);
        }
        tgt += sf_length;
    }
    out
}

/// `silk_P_Ana_calc_energy_st3`: stage-3 energies (recursive sliding window).
#[allow(clippy::needless_range_loop, reason = "computed index ranges mirror the reference")]
fn calc_energy_st3(frame: &[f32], start_lag: i32, sf_length: usize, nb_subfr: usize, complexity: usize) -> St3 {
    let mut out: St3 = vec![[[0.0; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; nb_subfr];
    let (nb_cbk_search, cbk_max): (usize, bool) = if nb_subfr == MAX_NB_SUBFR {
        (NB_CBK_SEARCHS_STAGE3[complexity], true)
    } else {
        (PE_NB_CBKS_STAGE3_10MS, false)
    };
    let mut tgt = sf_length << 2;
    for k in 0..nb_subfr {
        let (lag_low, lag_high) = lag_range(k, complexity, cbk_max);
        let mut scratch = [0.0f32; 22];
        let basis0 = (tgt as i32 - (start_lag + lag_low)) as usize;
        let mut e = energy(&frame[basis0..basis0 + sf_length]) + 1e-3;
        scratch[0] = e as f32;
        let lag_diff = (lag_high - lag_low + 1) as usize;
        for i in 1..lag_diff {
            // basis_ptr = tgt - (start_lag + lag_low); window slides by -i.
            e -= f64::from(frame[basis0 + sf_length - i]) * f64::from(frame[basis0 + sf_length - i]);
            e += f64::from(frame[basis0 - i]) * f64::from(frame[basis0 - i]);
            scratch[i] = e as f32;
        }
        let delta = lag_low;
        for i in 0..nb_cbk_search {
            let cb = if cbk_max {
                i32::from(CB_LAGS_STAGE3[k][i])
            } else {
                i32::from(CB_LAGS_STAGE3_10_MS[k][i])
            };
            let idx = (cb - delta) as usize;
            out[k][i].copy_from_slice(&scratch[idx..idx + PE_NB_STAGE3_LAGS]);
        }
        tgt += sf_length;
    }
    out
}

/// Stage-3 lag search range for subframe `k` (`silk_Lag_range_stage3`).
fn lag_range(k: usize, complexity: usize, cbk_max: bool) -> (i32, i32) {
    if cbk_max {
        let r = LAG_RANGE_STAGE3[complexity][k];
        (i32::from(r[0]), i32::from(r[1]))
    } else {
        let r = LAG_RANGE_STAGE3_10MS[k];
        (i32::from(r[0]), i32::from(r[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-exact pin against the compiled reference
    /// `silk_pitch_analysis_core_FLP` for a periodic 16 kHz frame.
    #[test]
    fn pitch_core_matches_reference_pin() {
        let (fs_khz, nb) = (16i32, 4usize);
        let frame_length = (20 + nb * 5) * fs_khz as usize;
        let frame: alloc::vec::Vec<f32> = (0..frame_length)
            .map(|i| {
                let mut s = 3000.0 * (core::f32::consts::TAU * i as f32 / 120.0).sin();
                s += 1200.0 * (core::f32::consts::TAU * i as f32 / 60.0).sin();
                s += ((i as i32 * 2719 + 7) % 211 - 105) as f32 * 1.5;
                s
            })
            .collect();

        let mut ltp_corr = 0.0f32;
        let (voicing, pitch_out, lag_index, contour_index) =
            pitch_analysis_core(&frame, fs_khz, 2, nb, 0, 0.6, 0.4, &mut ltp_corr);

        assert_eq!(voicing, 0, "should be voiced");
        assert_eq!(pitch_out, [120, 120, 120, 120], "pitch lags disagree with reference");
        assert_eq!(lag_index, 88, "lag index disagrees with reference");
        assert_eq!(contour_index, 0, "contour index disagrees with reference");
        assert!((ltp_corr - 0.998095).abs() < 1e-3, "LTPCorr {ltp_corr} off");
    }
}

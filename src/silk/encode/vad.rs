//! Voice activity detection for the SILK encoder (RFC 6716 §5.2).
//!
//! [`VadState::get_sa_q8`] produces the speech-activity probability
//! (`speech_activity_Q8`, 0-255), the spectral-tilt measure
//! (`input_tilt_Q15`), and the per-band input-quality measures
//! (`input_quality_bands_Q15`) that the noise-shaping and gain stages use to
//! tune the coding. It splits the frame into four bands with a tree of
//! half-band analysis filters, tracks a per-band noise-floor estimate, and
//! maps the per-band signal-to-noise ratios through sigmoids.
//!
//! Fixed-point throughout; carries the cross-frame noise-estimation state.

extern crate alloc;

use super::super::math::{lin2log, smlabb, smlawb, smulwb, smulww, sqrt_approx};

const VAD_N_BANDS: usize = 4;
const VAD_INTERNAL_SUBFRAMES_LOG2: usize = 2;
const VAD_INTERNAL_SUBFRAMES: usize = 1 << VAD_INTERNAL_SUBFRAMES_LOG2;
const VAD_NOISE_LEVELS_BIAS: i32 = 50;
const VAD_NEGATIVE_OFFSET_Q5: i32 = 128;
const VAD_SNR_FACTOR_Q16: i32 = 45000;
const VAD_SNR_SMOOTH_COEF_Q18: i32 = 4096;
const VAD_NOISE_LEVEL_SMOOTH_COEF_Q16: i32 = 1024;

/// Half-band analysis filter allpass coefficients (`A_fb1_20` / `A_fb1_21`).
const A_FB1_20: i32 = 5394 << 1;
const A_FB1_21: i32 = -24290;

/// Piecewise-linear sigmoid, input Q5, output Q15.
fn sigm_q15(in_q5: i32) -> i32 {
    const SLOPE_Q10: [i32; 6] = [237, 153, 73, 30, 12, 7];
    const POS_Q15: [i32; 6] = [16384, 23955, 28861, 31213, 32178, 32548];
    const NEG_Q15: [i32; 6] = [16384, 8812, 3906, 1554, 589, 219];
    if in_q5 < 0 {
        let in_q5 = -in_q5;
        if in_q5 >= 6 * 32 {
            0
        } else {
            let ind = (in_q5 >> 5) as usize;
            NEG_Q15[ind] - smlabb(0, SLOPE_Q10[ind], in_q5 & 0x1F)
        }
    } else if in_q5 >= 6 * 32 {
        32767
    } else {
        let ind = (in_q5 >> 5) as usize;
        POS_Q15[ind] + smlabb(0, SLOPE_Q10[ind], in_q5 & 0x1F)
    }
}

/// Saturating add of two non-negative values.
fn add_pos_sat32(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

/// Split `input` (length `n`) into a low band and a
/// high band (each `n/2`) via a first-order allpass pair. `s` is the
/// 2-element filter state, carried across frames.
fn ana_filt_bank_1(input: &[i16], s: &mut [i32; 2], n: usize) -> (alloc::vec::Vec<i16>, alloc::vec::Vec<i16>) {
    let n2 = n >> 1;
    let mut out_l = alloc::vec![0i16; n2];
    let mut out_h = alloc::vec![0i16; n2];
    for k in 0..n2 {
        let in32 = i32::from(input[2 * k]) << 10;
        let y = in32.wrapping_sub(s[0]);
        let x = smlawb(y, y, A_FB1_21);
        let out_1 = s[0].wrapping_add(x);
        s[0] = in32.wrapping_add(x);

        let in32 = i32::from(input[2 * k + 1]) << 10;
        let y = in32.wrapping_sub(s[1]);
        let x = smulwb(y, A_FB1_20);
        let out_2 = s[1].wrapping_add(x);
        s[1] = in32.wrapping_add(x);

        out_l[k] = sat16(rshift_round(out_2.wrapping_add(out_1), 11));
        out_h[k] = sat16(rshift_round(out_2.wrapping_sub(out_1), 11));
    }
    (out_l, out_h)
}

/// Saturate to i16.
fn sat16(a: i32) -> i16 {
    a.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// Arithmetic right shift with rounding.
fn rshift_round(a: i32, shift: u32) -> i32 {
    (a + (1 << (shift - 1))) >> shift
}

/// Cross-frame VAD state (`silk_VAD_state`).
#[derive(Clone)]
pub(crate) struct VadState {
    ana_state: [i32; 2],
    ana_state1: [i32; 2],
    ana_state2: [i32; 2],
    xnrg_subfr: [i32; VAD_N_BANDS],
    nl: [i32; VAD_N_BANDS],
    inv_nl: [i32; VAD_N_BANDS],
    nrg_ratio_smth_q8: [i32; VAD_N_BANDS],
    hp_state: i16,
    counter: i32,
    noise_level_bias: [i32; VAD_N_BANDS],
}

/// The per-frame VAD outputs.
pub(crate) struct VadResult {
    /// Speech-activity probability, 0-255.
    pub speech_activity_q8: i32,
    /// Spectral-tilt measure, Q15.
    pub input_tilt_q15: i32,
    /// Per-band input quality, Q15.
    pub input_quality_bands_q15: [i32; VAD_N_BANDS],
}

impl VadState {
    /// A reset VAD state (`silk_VAD_Init`).
    #[must_use]
    pub(crate) fn new() -> Self {
        let mut noise_level_bias = [0i32; VAD_N_BANDS];
        for (b, v) in noise_level_bias.iter_mut().enumerate() {
            *v = (VAD_NOISE_LEVELS_BIAS / (b as i32 + 1)).max(1);
        }
        let mut nl = [0i32; VAD_N_BANDS];
        let mut inv_nl = [0i32; VAD_N_BANDS];
        for b in 0..VAD_N_BANDS {
            nl[b] = 100 * noise_level_bias[b];
            inv_nl[b] = i32::MAX / nl[b];
        }
        VadState {
            ana_state: [0; 2],
            ana_state1: [0; 2],
            ana_state2: [0; 2],
            xnrg_subfr: [0; VAD_N_BANDS],
            nl,
            inv_nl,
            nrg_ratio_smth_q8: [100 * 256; VAD_N_BANDS],
            hp_state: 0,
            counter: 15,
            noise_level_bias,
        }
    }

    /// Update the per-band noise-floor estimate
    /// from the current subband energies.
    #[allow(
        clippy::needless_range_loop,
        reason = "parallel per-band state arrays indexed together"
    )]
    fn get_noise_levels(&mut self, px: &[i32; VAD_N_BANDS]) {
        let min_coef = if self.counter < 1000 {
            let m = i32::from(i16::MAX) / ((self.counter >> 4) + 1);
            self.counter += 1;
            m
        } else {
            0
        };
        for k in 0..VAD_N_BANDS {
            let nl = self.nl[k];
            let nrg = add_pos_sat32(px[k], self.noise_level_bias[k]);
            let inv_nrg = i32::MAX / nrg;
            let mut coef = if nrg > (nl << 3) {
                VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 >> 3
            } else if nrg < nl {
                VAD_NOISE_LEVEL_SMOOTH_COEF_Q16
            } else {
                smulwb(smulww(inv_nrg, nl), VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 << 1)
            };
            coef = coef.max(min_coef);
            self.inv_nl[k] = smlawb(self.inv_nl[k], inv_nrg - self.inv_nl[k], coef);
            let nl = (i32::MAX / self.inv_nl[k]).min(0x00FF_FFFF);
            self.nl[k] = nl;
        }
    }

    /// Compute the speech activity, tilt and per-band
    /// quality for one frame of `pin` (`frame_length` samples at `fs_khz`).
    #[allow(clippy::needless_range_loop, reason = "computed index ranges mirror the reference")]
    pub(crate) fn get_sa_q8(&mut self, pin: &[i16], frame_length: usize, fs_khz: i32) -> VadResult {
        // Filter and decimate into four bands.
        let (l0, h0) = ana_filt_bank_1(pin, &mut self.ana_state, frame_length);
        let (l1, h1) = ana_filt_bank_1(&l0, &mut self.ana_state1, l0.len());
        let (l2, h2) = ana_filt_bank_1(&l1, &mut self.ana_state2, l1.len());
        // band 0 = 0-1 kHz, 1 = 1-2 kHz, 2 = 2-4 kHz, 3 = 4-8 kHz.
        let mut bands = [l2, h2, h1, h0];

        // HP filter (differentiator) on the lowest band.
        let dl = bands[0].len();
        bands[0][dl - 1] >>= 1;
        let hp_state_tmp = bands[0][dl - 1];
        for i in (1..dl).rev() {
            bands[0][i - 1] >>= 1;
            bands[0][i] -= bands[0][i - 1];
        }
        bands[0][0] -= self.hp_state;
        self.hp_state = hp_state_tmp;

        // Energy per band, accumulated over internal subframes.
        let mut xnrg = [0i32; VAD_N_BANDS];
        for b in 0..VAD_N_BANDS {
            let band = &bands[b];
            let dec_subframe_length = band.len() >> VAD_INTERNAL_SUBFRAMES_LOG2;
            let mut off = 0usize;
            xnrg[b] = self.xnrg_subfr[b];
            let mut sum_squared = 0i32;
            for s in 0..VAD_INTERNAL_SUBFRAMES {
                sum_squared = 0;
                for i in 0..dec_subframe_length {
                    let x_tmp = i32::from(band[off + i]) >> 3;
                    sum_squared = smlabb(sum_squared, x_tmp, x_tmp);
                }
                if s < VAD_INTERNAL_SUBFRAMES - 1 {
                    xnrg[b] = add_pos_sat32(xnrg[b], sum_squared);
                } else {
                    xnrg[b] = add_pos_sat32(xnrg[b], sum_squared >> 1);
                }
                off += dec_subframe_length;
            }
            self.xnrg_subfr[b] = sum_squared;
        }

        self.get_noise_levels(&xnrg);

        // Signal-plus-noise to noise ratio per band → tilt, sum of squares.
        const TILT_WEIGHTS: [i32; VAD_N_BANDS] = [30000, 6000, -12000, -12000];
        let mut sum_squared = 0i32;
        let mut input_tilt = 0i32;
        let mut nrg_to_noise_ratio_q8 = [0i32; VAD_N_BANDS];
        for b in 0..VAD_N_BANDS {
            let speech_nrg = xnrg[b] - self.nl[b];
            if speech_nrg > 0 {
                nrg_to_noise_ratio_q8[b] = if xnrg[b] & 0xFF80_0000u32 as i32 == 0 {
                    (xnrg[b] << 8) / (self.nl[b] + 1)
                } else {
                    xnrg[b] / ((self.nl[b] >> 8) + 1)
                };
                let mut snr_q7 = lin2log(nrg_to_noise_ratio_q8[b]) - 8 * 128;
                sum_squared = smlabb(sum_squared, snr_q7, snr_q7);
                if speech_nrg < (1 << 20) {
                    snr_q7 = smulwb(sqrt_approx(speech_nrg) << 6, snr_q7);
                }
                input_tilt = smlawb(input_tilt, TILT_WEIGHTS[b], snr_q7);
            } else {
                nrg_to_noise_ratio_q8[b] = 256;
            }
        }
        sum_squared /= VAD_N_BANDS as i32;
        let p_snr_db_q7 = 3 * sqrt_approx(sum_squared);

        let mut sa_q15 = sigm_q15(smulwb(VAD_SNR_FACTOR_Q16, p_snr_db_q7) - VAD_NEGATIVE_OFFSET_Q5);
        let input_tilt_q15 = (sigm_q15(input_tilt) - 16384) << 1;

        // Scale the activity by the (noise-removed) power level.
        let mut speech_nrg = 0i32;
        for b in 0..VAD_N_BANDS {
            speech_nrg += (b as i32 + 1) * ((xnrg[b] - self.nl[b]) >> 4);
        }
        if frame_length == 20 * fs_khz as usize {
            speech_nrg >>= 1;
        }
        if speech_nrg <= 0 {
            sa_q15 >>= 1;
        } else if speech_nrg < 16384 {
            speech_nrg <<= 16;
            speech_nrg = sqrt_approx(speech_nrg);
            sa_q15 = smulwb(32768 + speech_nrg, sa_q15);
        }
        let speech_activity_q8 = (sa_q15 >> 7).min(i32::from(u8::MAX));

        // Smoothed energy-to-noise ratio per band → quality.
        let mut smooth_coef_q16 = smulwb(VAD_SNR_SMOOTH_COEF_Q18, smulwb(sa_q15, sa_q15));
        if frame_length == 10 * fs_khz as usize {
            smooth_coef_q16 >>= 1;
        }
        let mut input_quality_bands_q15 = [0i32; VAD_N_BANDS];
        for b in 0..VAD_N_BANDS {
            self.nrg_ratio_smth_q8[b] = smlawb(
                self.nrg_ratio_smth_q8[b],
                nrg_to_noise_ratio_q8[b] - self.nrg_ratio_smth_q8[b],
                smooth_coef_q16,
            );
            let snr_q7 = 3 * (lin2log(self.nrg_ratio_smth_q8[b]) - 8 * 128);
            input_quality_bands_q15[b] = sigm_q15((snr_q7 - 16 * 128) >> 4);
        }

        VadResult {
            speech_activity_q8,
            input_tilt_q15,
            input_quality_bands_q15,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-exact pin against the reference over a
    /// five-frame sequence (so the noise-estimation state evolves).
    #[test]
    fn vad_matches_reference_pin() {
        let (fs_khz, frame_length) = (16i32, 320usize);
        let mut st = VadState::new();
        // (SA_Q8, tilt_Q15, [quality×4]) from the reference, per frame.
        let expected = [
            (255, 32766, [23731, 23494, 23494, 23494]),
            (255, 32766, [23731, 23494, 23494, 23494]),
            (255, 32766, [23494, 23257, 23257, 23257]),
            (255, 32766, [23494, 23257, 23257, 23257]),
            (255, 32766, [23257, 23020, 23020, 23020]),
        ];
        for (f, (exp_sa, exp_tilt, exp_q)) in expected.into_iter().enumerate() {
            let input: alloc::vec::Vec<i16> = (0..frame_length)
                .map(|i| {
                    let n = (f * frame_length + i) as f64;
                    let mut s = 3000.0 * (core::f64::consts::TAU * n / 80.0).sin();
                    s += 1500.0 * (core::f64::consts::TAU * n / 27.0).sin();
                    s += ((n as i64 * 1237 + 11).rem_euclid(401) - 200) as f64;
                    s as i16
                })
                .collect();
            let r = st.get_sa_q8(&input, frame_length, fs_khz);
            assert_eq!(r.speech_activity_q8, exp_sa, "frame {f} SA");
            assert_eq!(r.input_tilt_q15, exp_tilt, "frame {f} tilt");
            assert_eq!(r.input_quality_bands_q15, exp_q, "frame {f} quality");
        }
    }
}

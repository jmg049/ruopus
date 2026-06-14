//! The SILK resampler, decoder paths (normative `resampler.c`,
//! `resampler_private_up2_HQ.c`, `resampler_private_IIR_FIR.c`,
//! tables from `resampler_rom.h`).
//!
//! The decoder upsamples the internal 8/12/16 kHz signal to the API rate:
//! a 2× allpass upsampler (three second-order allpass sections per phase
//! plus a notch), optionally followed by fractional FIR interpolation for
//! non-power-of-two ratios. Equal rates copy through. The downsampling paths
//! (`down_FIR` over an `AR2` prefilter) serve both the decoder's higher-rate
//! outputs and the encoder front-end ([`Resampler::new_enc`], API rate →
//! internal rate, including the 1:3/1:4/1:6 ratios).

#![allow(dead_code, reason = "consumed incrementally as the SILK decoder stages land")]

use alloc::vec;

use super::math::{rshift_round, smlabb, smlawb, smulbb, smulwb, smulww};

/// `RESAMPLER_ORDER_FIR_12`.
const ORDER_FIR_12: usize = 8;
/// `RESAMPLER_MAX_BATCH_SIZE_MS`.
const MAX_BATCH_SIZE_MS: usize = 10;

/// `silk_resampler_up2_hq_0` / `_1` (allpass coefficients per phase).
const UP2_HQ_0: [i32; 3] = [1746, 14986, 39083 - 65536];
const UP2_HQ_1: [i32; 3] = [6854, 25769, 55542 - 65536];

/// `silk_resampler_frac_FIR_12`: interpolation fractions 1/24 .. 23/24.
const FRAC_FIR_12: [[i16; 4]; 12] = [
    [189, -600, 617, 30567],
    [117, -159, -1070, 29704],
    [52, 221, -2392, 28276],
    [-4, 529, -3350, 26341],
    [-48, 758, -3956, 23973],
    [-80, 905, -4235, 21254],
    [-99, 972, -4222, 18278],
    [-107, 967, -3957, 15143],
    [-103, 896, -3487, 11950],
    [-91, 773, -2865, 8798],
    [-71, 611, -2143, 5784],
    [-46, 425, -1375, 2996],
];

/// `silk_Resampler_3_4_COEFS`: 2 AR2 taps then the FIR half-tables.
const COEFS_3_4: [i16; 29] = [
    -20694, -13867, -49, 64, 17, -157, 353, -496, 163, 11047, 22205, -39, 6, 91, -170, 186, 23, -896, 6336, 19928, -19,
    -36, 102, -89, -24, 328, -951, 2568, 15909,
];

/// `silk_Resampler_2_3_COEFS`: 2 AR2 taps then the FIR half-tables.
const COEFS_2_3: [i16; 20] = [
    -14457, -14019, 64, 128, -122, 36, 310, -768, 584, 9267, 17733, 12, 128, 18, -142, 288, -117, -865, 4123, 14459,
];

/// `silk_Resampler_1_2_COEFS`: 2 AR2 taps then the FIR half-tables.
const COEFS_1_2: [i16; 14] = [
    616, -14323, -10, 39, 58, -46, -84, 120, 184, -315, -541, 1284, 5380, 9024,
];

/// `silk_Resampler_1_3_COEFS`: 2 AR2 taps then the 18 symmetric FIR taps
/// (`RESAMPLER_DOWN_ORDER_FIR2` = 36). Encoder front-end (48→16, 24→8).
const COEFS_1_3: [i16; 20] = [
    16102, -15162, -13, 0, 20, 26, 5, -31, -43, -4, 65, 90, 7, -157, -248, -44, 593, 1583, 2612, 3271,
];
/// `silk_Resampler_1_4_COEFS` (48→12).
const COEFS_1_4: [i16; 20] = [
    22500, -15099, 3, -14, -20, -15, 2, 25, 37, 25, -16, -71, -107, -79, 50, 292, 623, 982, 1288, 1464,
];
/// `silk_Resampler_1_6_COEFS` (48→8).
const COEFS_1_6: [i16; 20] = [
    27540, -15257, 17, 12, 8, 1, -10, -22, -30, -32, -22, 3, 44, 100, 168, 243, 317, 381, 429, 455,
];

/// `delay_matrix_dec[in][out]` over rates [8, 12, 16] × [8, 12, 16, 24, 48].
const DELAY_MATRIX_DEC: [[i8; 5]; 3] = [[4, 0, 2, 0, 0], [0, 9, 4, 7, 4], [0, 3, 12, 7, 7]];
/// `delay_matrix_enc[in][out]` over rates [8, 12, 16, 24, 48] × [8, 12, 16].
const DELAY_MATRIX_ENC: [[i8; 3]; 5] = [[6, 0, 3], [0, 7, 3], [0, 1, 10], [0, 2, 6], [18, 10, 12]];

/// `rateID`: [8000, 12000, 16000, 24000, 48000] → 0..=4.
const fn rate_id(r: i32) -> usize {
    ((((r >> 12) - (r > 16000) as i32) >> (r > 24000) as i32) - 1) as usize
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    /// Equal rates.
    Copy,
    /// Exactly 2×: the allpass upsampler alone.
    Up2Hq,
    /// Other upsampling ratios: 2× allpass + fractional FIR.
    IirFir,
    /// Downsampling: AR2 filter + FIR interpolation.
    DownFir,
}

/// Decoder-side resampler state (`silk_resampler_state_struct`).
#[derive(Debug, Clone)]
pub(crate) struct Resampler {
    s_iir: [i32; 6],
    s_fir: [i16; ORDER_FIR_12],
    /// Down-FIR history (i32 domain).
    s_fir32: [i32; 36],
    fir_order: usize,
    fir_fracs: i32,
    coefs: &'static [i16],
    delay_buf: [i16; 48],
    input_delay: usize,
    fs_in_khz: usize,
    fs_out_khz: usize,
    batch_size: usize,
    inv_ratio_q16: i32,
    method: Method,
}

impl Resampler {
    /// `silk_resampler_init` (decoder direction): `fs_hz_in` must be 8, 12
    /// or 16 kHz; `fs_hz_out` 8-48 kHz, not below the input.
    pub fn new(fs_hz_in: i32, fs_hz_out: i32) -> Self {
        assert!(
            fs_hz_in == 8000 || fs_hz_in == 12000 || fs_hz_in == 16000,
            "decoder input rate"
        );
        assert!(
            matches!(fs_hz_out, 8000 | 12000 | 16000 | 24000 | 48000),
            "unsupported decoder output rate {fs_hz_out}"
        );

        // (FIR order, fractional phases, coefficients) for the down paths.
        let mut fir_order = 0usize;
        let mut fir_fracs = 0i32;
        let mut coefs: &'static [i16] = &[];
        let method = if fs_hz_out == fs_hz_in {
            Method::Copy
        } else if fs_hz_out == 2 * fs_hz_in {
            Method::Up2Hq
        } else if fs_hz_out > fs_hz_in {
            Method::IirFir
        } else if fs_hz_out * 4 == fs_hz_in * 3 {
            fir_fracs = 3;
            fir_order = 18; // RESAMPLER_DOWN_ORDER_FIR0
            coefs = &COEFS_3_4;
            Method::DownFir
        } else if fs_hz_out * 3 == fs_hz_in * 2 {
            fir_fracs = 2;
            fir_order = 18;
            coefs = &COEFS_2_3;
            Method::DownFir
        } else if fs_hz_out * 2 == fs_hz_in {
            fir_fracs = 1;
            fir_order = 24; // RESAMPLER_DOWN_ORDER_FIR1
            coefs = &COEFS_1_2;
            Method::DownFir
        } else {
            panic!("unsupported decoder rate pair {fs_hz_in}->{fs_hz_out}");
        };
        let up2x = i32::from(method == Method::IirFir);

        // Input/output ratio in Q16, rounded up.
        let mut inv_ratio_q16 = ((fs_hz_in << (14 + up2x)) / fs_hz_out) << 2;
        while smulww(inv_ratio_q16, fs_hz_out) < (fs_hz_in << up2x) {
            inv_ratio_q16 += 1;
        }

        let fs_in_khz = (fs_hz_in / 1000) as usize;
        Resampler {
            s_iir: [0; 6],
            s_fir: [0; ORDER_FIR_12],
            s_fir32: [0; 36],
            fir_order,
            fir_fracs,
            coefs,
            delay_buf: [0; 48],
            input_delay: DELAY_MATRIX_DEC[rate_id(fs_hz_in)][rate_id(fs_hz_out)] as usize,
            fs_in_khz,
            fs_out_khz: (fs_hz_out / 1000) as usize,
            batch_size: fs_in_khz * MAX_BATCH_SIZE_MS,
            inv_ratio_q16,
            method,
        }
    }

    /// `silk_resampler_init(forEnc=1)`: the encoder front-end resampler,
    /// converting the API rate `fs_hz_in` ∈ {8,12,16,24,48} kHz to the
    /// internal rate `fs_hz_out` ∈ {8,12,16} kHz. Adds the 1:3/1:4/1:6 down
    /// ratios the decoder direction never needs and uses the encoder delay
    /// table.
    #[must_use]
    pub fn new_enc(fs_hz_in: i32, fs_hz_out: i32) -> Self {
        assert!(
            matches!(fs_hz_in, 8000 | 12000 | 16000 | 24000 | 48000),
            "unsupported encoder input rate {fs_hz_in}"
        );
        assert!(
            fs_hz_out == 8000 || fs_hz_out == 12000 || fs_hz_out == 16000,
            "encoder output rate"
        );

        let mut fir_order = 0usize;
        let mut fir_fracs = 0i32;
        let mut coefs: &'static [i16] = &[];
        let method = if fs_hz_out == fs_hz_in {
            Method::Copy
        } else if fs_hz_out == 2 * fs_hz_in {
            Method::Up2Hq
        } else if fs_hz_out > fs_hz_in {
            Method::IirFir
        } else if fs_hz_out * 4 == fs_hz_in * 3 {
            fir_fracs = 3;
            fir_order = 18;
            coefs = &COEFS_3_4;
            Method::DownFir
        } else if fs_hz_out * 3 == fs_hz_in * 2 {
            fir_fracs = 2;
            fir_order = 18;
            coefs = &COEFS_2_3;
            Method::DownFir
        } else if fs_hz_out * 2 == fs_hz_in {
            fir_fracs = 1;
            fir_order = 24;
            coefs = &COEFS_1_2;
            Method::DownFir
        } else if fs_hz_out * 3 == fs_hz_in {
            fir_fracs = 1;
            fir_order = 36; // RESAMPLER_DOWN_ORDER_FIR2
            coefs = &COEFS_1_3;
            Method::DownFir
        } else if fs_hz_out * 4 == fs_hz_in {
            fir_fracs = 1;
            fir_order = 36;
            coefs = &COEFS_1_4;
            Method::DownFir
        } else if fs_hz_out * 6 == fs_hz_in {
            fir_fracs = 1;
            fir_order = 36;
            coefs = &COEFS_1_6;
            Method::DownFir
        } else {
            panic!("unsupported encoder rate pair {fs_hz_in}->{fs_hz_out}");
        };
        let up2x = i32::from(method == Method::IirFir);

        let mut inv_ratio_q16 = ((fs_hz_in << (14 + up2x)) / fs_hz_out) << 2;
        while smulww(inv_ratio_q16, fs_hz_out) < (fs_hz_in << up2x) {
            inv_ratio_q16 += 1;
        }

        let fs_in_khz = (fs_hz_in / 1000) as usize;
        Resampler {
            s_iir: [0; 6],
            s_fir: [0; ORDER_FIR_12],
            s_fir32: [0; 36],
            fir_order,
            fir_fracs,
            coefs,
            delay_buf: [0; 48],
            input_delay: DELAY_MATRIX_ENC[rate_id(fs_hz_in)][rate_id(fs_hz_out)] as usize,
            fs_in_khz,
            fs_out_khz: (fs_hz_out / 1000) as usize,
            batch_size: fs_in_khz * MAX_BATCH_SIZE_MS,
            inv_ratio_q16,
            method,
        }
    }

    /// The configured output rate in Hz.
    pub fn output_rate_hz(&self) -> i32 {
        (self.fs_out_khz * 1000) as i32
    }

    /// `silk_resampler`: converts `input` (≥ 1 ms) to the output rate;
    /// `out` receives `input.len() * fs_out / fs_in` samples.
    pub fn process(&mut self, out: &mut [i16], input: &[i16]) {
        let in_len = input.len();
        debug_assert!(in_len >= self.fs_in_khz);
        debug_assert!(self.input_delay <= self.fs_in_khz);
        debug_assert_eq!(out.len(), in_len * self.fs_out_khz / self.fs_in_khz);

        let n_samples = self.fs_in_khz - self.input_delay;

        // Head: delay buffer plus the start of the input (1 ms total).
        let mut head = [0i16; 48];
        head[..self.input_delay].copy_from_slice(&self.delay_buf[..self.input_delay]);
        head[self.input_delay..self.fs_in_khz].copy_from_slice(&input[..n_samples]);

        // The tail stops input_delay samples short of the end; that
        // remainder is only buffered for the next call.
        let tail = &input[n_samples..in_len - self.input_delay];
        let (out_head, out_tail) = out.split_at_mut(self.fs_out_khz);
        match self.method {
            Method::Up2Hq => {
                up2_hq_into(&mut self.s_iir, out_head, &head[..self.fs_in_khz]);
                up2_hq_into(&mut self.s_iir, out_tail, tail);
            },
            Method::IirFir => {
                self.iir_fir(out_head, &head[..self.fs_in_khz]);
                self.iir_fir(out_tail, tail);
            },
            Method::DownFir => {
                self.down_fir(out_head, &head[..self.fs_in_khz]);
                self.down_fir(out_tail, tail);
            },
            Method::Copy => {
                out_head.copy_from_slice(&head[..self.fs_in_khz]);
                out_tail.copy_from_slice(tail);
            },
        }

        // Save the input tail for the next call.
        self.delay_buf[..self.input_delay].copy_from_slice(&input[in_len - self.input_delay..]);
    }

    /// `silk_resampler_private_IIR_FIR`: 2× upsample then fractional FIR
    /// interpolation at `inv_ratio_q16`.
    fn iir_fir(&mut self, out: &mut [i16], input: &[i16]) {
        let mut buf = vec![0i16; 2 * self.batch_size + ORDER_FIR_12];
        buf[..ORDER_FIR_12].copy_from_slice(&self.s_fir);

        let index_increment_q16 = self.inv_ratio_q16;
        let mut in_off = 0usize;
        let mut out_off = 0usize;
        let mut n_samples_in;
        loop {
            n_samples_in = (input.len() - in_off).min(self.batch_size);

            // 2x upsample into the work buffer (after the FIR history).
            let work = &mut buf[ORDER_FIR_12..];
            up2_hq_into(
                &mut self.s_iir,
                &mut work[..2 * n_samples_in],
                &input[in_off..in_off + n_samples_in],
            );

            // Fractional interpolation over the upsampled signal.
            let max_index_q16 = (n_samples_in as i32) << 17;
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let table_index = smulwb(index_q16 & 0xffff, 12) as usize;
                let base = (index_q16 >> 16) as usize;
                let f = &FRAC_FIR_12[table_index];
                let g = &FRAC_FIR_12[11 - table_index];
                let mut res_q15 = smulbb(i32::from(buf[base]), i32::from(f[0]));
                res_q15 = smlabb(res_q15, i32::from(buf[base + 1]), i32::from(f[1]));
                res_q15 = smlabb(res_q15, i32::from(buf[base + 2]), i32::from(f[2]));
                res_q15 = smlabb(res_q15, i32::from(buf[base + 3]), i32::from(f[3]));
                res_q15 = smlabb(res_q15, i32::from(buf[base + 4]), i32::from(g[3]));
                res_q15 = smlabb(res_q15, i32::from(buf[base + 5]), i32::from(g[2]));
                res_q15 = smlabb(res_q15, i32::from(buf[base + 6]), i32::from(g[1]));
                res_q15 = smlabb(res_q15, i32::from(buf[base + 7]), i32::from(g[0]));
                out[out_off] = rshift_round(res_q15, 15).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
                out_off += 1;
                index_q16 += index_increment_q16;
            }

            in_off += n_samples_in;
            if in_off < input.len() {
                // Carry the tail of the upsampled signal as FIR history.
                buf.copy_within(2 * n_samples_in..2 * n_samples_in + ORDER_FIR_12, 0);
            } else {
                break;
            }
        }
        let start = 2 * n_samples_in;
        self.s_fir.copy_from_slice(&buf[start..start + ORDER_FIR_12]);
    }
}

impl Resampler {
    /// `silk_resampler_private_down_FIR`: a second-order AR filter (Q8)
    /// followed by polyphase FIR interpolation.
    fn down_fir(&mut self, out: &mut [i16], input: &[i16]) {
        let ord = self.fir_order;
        let mut buf = vec![0i32; self.batch_size + ord];
        buf[..ord].copy_from_slice(&self.s_fir32[..ord]);

        let a_q14 = &self.coefs[..2];
        let fir_coefs = &self.coefs[2..];
        let index_increment_q16 = self.inv_ratio_q16;
        let mut in_off = 0usize;
        let mut out_off = 0usize;
        let mut n_samples_in;
        loop {
            n_samples_in = (input.len() - in_off).min(self.batch_size);

            // Second-order AR filter (output in Q8).
            {
                let s = &mut self.s_iir;
                for (k, o) in buf[ord..ord + n_samples_in].iter_mut().enumerate() {
                    let out32 = s[0].wrapping_add(i32::from(input[in_off + k]) << 8);
                    *o = out32;
                    let out32 = out32 << 2;
                    s[0] = smlawb(s[1], out32, i32::from(a_q14[0]));
                    s[1] = smulwb(out32, i32::from(a_q14[1]));
                }
            }

            let max_index_q16 = (n_samples_in as i32) << 16;
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let base = (index_q16 >> 16) as usize;
                let res_q6 = match ord {
                    18 => {
                        // Fractional phase selects the half-table pair.
                        let ind = smulwb(index_q16 & 0xffff, self.fir_fracs) as usize;
                        let p1 = &fir_coefs[9 * ind..];
                        let p2 = &fir_coefs[9 * (self.fir_fracs as usize - 1 - ind)..];
                        let mut r = smulwb(buf[base], i32::from(p1[0]));
                        for t in 1..9 {
                            r = smlawb(r, buf[base + t], i32::from(p1[t]));
                        }
                        for t in 0..9 {
                            r = smlawb(r, buf[base + 17 - t], i32::from(p2[t]));
                        }
                        r
                    },
                    // Symmetric half-table (FIR1 = 24, FIR2 = 36).
                    _ => {
                        let half = ord / 2;
                        let mut r = smulwb(buf[base].wrapping_add(buf[base + ord - 1]), i32::from(fir_coefs[0]));
                        for t in 1..half {
                            r = smlawb(
                                r,
                                buf[base + t].wrapping_add(buf[base + ord - 1 - t]),
                                i32::from(fir_coefs[t]),
                            );
                        }
                        r
                    },
                };
                out[out_off] = rshift_round(res_q6, 6).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
                out_off += 1;
                index_q16 += index_increment_q16;
            }

            in_off += n_samples_in;
            if in_off < input.len() {
                buf.copy_within(n_samples_in..n_samples_in + ord, 0);
            } else {
                break;
            }
        }
        self.s_fir32[..ord].copy_from_slice(&buf[n_samples_in..n_samples_in + ord]);
    }
}

/// `silk_resampler_private_up2_HQ`: 2× allpass upsampler (Q10 state) -
/// three allpass sections per phase, the last a notch just above Nyquist.
fn up2_hq_into(s: &mut [i32; 6], out: &mut [i16], input: &[i16]) {
    for (k, &x) in input.iter().enumerate() {
        let in32 = i32::from(x) << 10;

        let y = in32 - s[0];
        let x0 = smulwb(y, UP2_HQ_0[0]);
        let out32_1 = s[0] + x0;
        s[0] = in32 + x0;

        let y = out32_1 - s[1];
        let x1 = smulwb(y, UP2_HQ_0[1]);
        let out32_2 = s[1] + x1;
        s[1] = out32_1 + x1;

        let y = out32_2 - s[2];
        let x2 = smlawb(y, y, UP2_HQ_0[2]);
        let out32_1 = s[2] + x2;
        s[2] = out32_2 + x2;

        out[2 * k] = rshift_round(out32_1, 10).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;

        let y = in32 - s[3];
        let x0 = smulwb(y, UP2_HQ_1[0]);
        let out32_1 = s[3] + x0;
        s[3] = in32 + x0;

        let y = out32_1 - s[4];
        let x1 = smulwb(y, UP2_HQ_1[1]);
        let out32_2 = s[4] + x1;
        s[4] = out32_1 + x1;

        let y = out32_2 - s[5];
        let x2 = smlawb(y, y, UP2_HQ_1[2]);
        let out32_1 = s[5] + x2;
        s[5] = out32_2 + x2;

        out[2 * k + 1] = rshift_round(out32_1, 10).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
}

#[cfg(test)]
mod down_tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    /// Pins from the compiled reference down-FIR paths (two consecutive
    /// frames so the AR2/FIR/delay state carries): per frame, the first 8
    /// outputs then the last 2.
    #[test]
    fn down_paths_match_reference_pins() {
        #[allow(clippy::type_complexity, reason = "pin fixture")]
        let cases: [(i32, i32, [(&[i16], [i16; 2]); 2]); 3] = [
            (
                16000,
                12000,
                [
                    (&[0, 0, 0, 2, -2, 3, -4, 11], [-1310, -1773]),
                    (&[-1121, -1372, -868, -1006, -584, -659, -295, -318], [392, 583]),
                ],
            ),
            (
                12000,
                8000,
                [
                    (&[0, -8, -3, -9, 9, -71, -1792, -1912], [1138, 1318]),
                    (&[1531, 1604, 2097, -297, -2363, -1222, -1705, -1010], [1755, 1726]),
                ],
            ),
            (
                16000,
                8000,
                [
                    (&[0, -4, -2, -7, 14, -94, -1745, -1823], [-2316, -1239]),
                    (&[-1511, -911, -946, -501, -432, -52, 70, 405], [214, 597]),
                ],
            ),
        ];
        for (fs_in, fs_out, frames) in cases {
            let mut r = Resampler::new(fs_in, fs_out);
            let in_len = (fs_in / 1000 * 20) as usize;
            let out_len = (fs_out / 1000 * 20) as usize;
            for (frame, (want_head, want_tail)) in frames.iter().enumerate() {
                let input: Vec<i16> = (0..in_len)
                    .map(|i| (((i + frame * in_len) as i32 * 119) % 4001 - 2000) as i16)
                    .collect();
                let mut out = vec![0i16; out_len];
                r.process(&mut out, &input);
                assert_eq!(&out[..8], *want_head, "{fs_in}->{fs_out} frame {frame} head");
                assert_eq!(&out[out_len - 2..], *want_tail, "{fs_in}->{fs_out} frame {frame} tail");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    /// Pins generated by compiling the reference resampler with this exact
    /// input over two consecutive 20 ms frames (so the delay buffer and
    /// filter states carry across calls). For each frame: the first 12
    /// output samples, then the last 3.
    #[test]
    fn matches_reference_pins() {
        #[allow(clippy::type_complexity, reason = "pin fixture")]
        let cases: [(i32, i32, [(&[i16], [i16; 3]); 2]); 5] = [
            (
                8000,
                48000,
                [
                    (&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [466, 500, 530]),
                    (
                        &[552, 567, 575, 581, 588, 600, 618, 642, 671, 703, 734, 761],
                        [-489, -415, -360],
                    ),
                ],
            ),
            (
                12000,
                48000,
                [
                    (&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1481, 1505, 1530]),
                    (
                        &[1559, 1591, 1625, 1659, 1690, 1717, 1743, 1770, 1798, 1826, 1845, 1838],
                        [-671, -1526, -2286],
                    ),
                ],
            ),
            (
                16000,
                48000,
                [
                    (&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [-1045, -1365, -1635]),
                    (
                        &[
                            -1593, -1274, -951, -876, -1055, -1266, -1277, -1064, -808, -712, -812, -958,
                        ],
                        [653, 706, 769],
                    ),
                ],
            ),
            (
                8000,
                16000,
                [
                    (
                        &[0, 0, 0, 0, -7, -70, -317, -898, -1736, -2383, -2358, -1780],
                        [368, 461, 550],
                    ),
                    (
                        &[580, 615, 699, 780, 818, 860, 937, 1012, 1056, 1103, 1175, 1246],
                        [-691, -499, -335],
                    ),
                ],
            ),
            (
                16000,
                16000,
                [
                    (&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [-1714, -1595, -1476]),
                    (
                        &[
                            -1357, -1238, -1119, -1000, -881, -762, -643, -524, -405, -286, -167, -48,
                        ],
                        [357, 476, 595],
                    ),
                ],
            ),
        ];

        for (fs_in, fs_out, frames) in cases {
            let mut r = Resampler::new(fs_in, fs_out);
            let in_len = (fs_in / 1000 * 20) as usize;
            let out_len = (fs_out / 1000 * 20) as usize;
            for (frame, (want_head, want_tail)) in frames.iter().enumerate() {
                let input: Vec<i16> = (0..in_len)
                    .map(|i| (((i + frame * in_len) as i32 * 119) % 4001 - 2000) as i16)
                    .collect();
                let mut out = vec![0i16; out_len];
                r.process(&mut out, &input);
                assert_eq!(&out[..12], *want_head, "{fs_in}->{fs_out} frame {frame} head");
                assert_eq!(&out[out_len - 3..], *want_tail, "{fs_in}->{fs_out} frame {frame} tail");
            }
        }
    }

    /// Encoder front-end resampler (`silk_resampler_init` forEnc=1), pinned
    /// against the compiled reference over three consecutive 20 ms frames.
    /// For each frame: the first 8 output samples, then the last 4.
    #[test]
    fn encoder_matches_reference_pins() {
        #[allow(clippy::type_complexity, reason = "pin fixture")]
        let cases: [(i32, i32, [([i16; 8], [i16; 4]); 3]); 3] = [
            (
                48000,
                16000,
                [
                    ([0, 0, 0, 0, 0, -1, 1, 0], [-7861, -8220, -8024, -7462]),
                    (
                        [-7189, -7143, -6513, -5574, -4923, -4546, -3620, -2398],
                        [-8264, -7881, -7697, -7777],
                    ),
                    (
                        [-7615, -6782, -6206, -5872, -5363, -4175, -3317, -2707],
                        [-7770, -7807, -8087, -7763],
                    ),
                ],
            ),
            (
                24000,
                16000,
                [
                    ([0, 0, 0, 0, 0, -1, 5, 2], [-7872, -8435, -8112, -7843]),
                    (
                        [-7398, -6869, -6268, -5492, -4953, -4759, -3712, -2794],
                        [-7934, -7777, -7572, -7709],
                    ),
                    (
                        [-7784, -7019, -6570, -5708, -5050, -4034, -3438, -3072],
                        [-8162, -8025, -7834, -7520],
                    ),
                ],
            ),
            (
                16000,
                16000,
                [
                    ([0, 0, 0, 0, 0, 0, 0, 0], [-7595, -8347, -8187, -7917]),
                    (
                        [-7541, -7061, -6486, -5820, -5073, -4254, -3373, -2442],
                        [-8291, -8241, -8081, -7811],
                    ),
                    (
                        [-7435, -6955, -6380, -5714, -4967, -4148, -3267, -3138],
                        [-8185, -8135, -7975, -7705],
                    ),
                ],
            ),
        ];

        for (fs_in, fs_out, frames) in cases {
            let mut r = Resampler::new_enc(fs_in, fs_out);
            let in_len = (fs_in / 1000 * 20) as usize;
            let out_len = (fs_out / 1000 * 20) as usize;
            for (frame, (want_head, want_tail)) in frames.iter().enumerate() {
                let input: Vec<i16> = (0..in_len)
                    .map(|i| {
                        let n = (frame * in_len + i) as f64;
                        let s = 8000.0 * (core::f64::consts::TAU * n / (f64::from(fs_in) / 300.0)).sin();
                        (s + ((n as i64 * 1237 + 11).rem_euclid(401) - 200) as f64 * 2.0) as i16
                    })
                    .collect();
                let mut out = vec![0i16; out_len];
                r.process(&mut out, &input);
                assert_eq!(&out[..8], want_head, "{fs_in}->{fs_out} frame {frame} head");
                assert_eq!(&out[out_len - 4..], want_tail, "{fs_in}->{fs_out} frame {frame} tail");
            }
        }
    }
}

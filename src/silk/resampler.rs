//! The SILK resampler, decoder paths (normative `resampler.c`,
//! `resampler_private_up2_HQ.c`, `resampler_private_IIR_FIR.c`,
//! tables from `resampler_rom.h`).
//!
//! The decoder upsamples the internal 8/12/16 kHz signal to the API rate:
//! a 2× allpass upsampler (three second-order allpass sections per phase
//! plus a notch), optionally followed by fractional FIR interpolation for
//! non-power-of-two ratios. Equal rates copy through. Downsampling paths
//! (`down_FIR`, `AR2`) are encoder-side and not ported.

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

/// `delay_matrix_dec[in][out]` over rates [8, 12, 16] × [8, 12, 16, 24, 48].
const DELAY_MATRIX_DEC: [[i8; 5]; 3] = [[4, 0, 2, 0, 0], [0, 9, 4, 7, 4], [0, 3, 12, 7, 7]];

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
}

/// Decoder-side resampler state (`silk_resampler_state_struct`).
#[derive(Debug, Clone)]
pub(crate) struct Resampler {
    s_iir: [i32; 6],
    s_fir: [i16; ORDER_FIR_12],
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
            matches!(fs_hz_out, 8000 | 12000 | 16000 | 24000 | 48000) && fs_hz_out >= fs_hz_in,
            "unsupported decoder output rate {fs_hz_out}"
        );

        let method = if fs_hz_out == fs_hz_in {
            Method::Copy
        } else if fs_hz_out == 2 * fs_hz_in {
            Method::Up2Hq
        } else {
            Method::IirFir
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
            delay_buf: [0; 48],
            input_delay: DELAY_MATRIX_DEC[rate_id(fs_hz_in)][rate_id(fs_hz_out)] as usize,
            fs_in_khz,
            fs_out_khz: (fs_hz_out / 1000) as usize,
            batch_size: fs_in_khz * MAX_BATCH_SIZE_MS,
            inv_ratio_q16,
            method,
        }
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
}

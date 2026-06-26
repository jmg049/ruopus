//! Signal analysis driving the encoder's mode/bandwidth decision.
//!
//! Only the decoder is conformance-normative, so any mode/bandwidth choice that
//! produces a valid, round-tripping Opus packet is correct. This module derives
//! the choice from a light-weight, FFT-free analysis of the 48 kHz time-domain
//! frame (keeping the core crate dependency-free):
//!
//! * a 6-band energy split (band edges ~4 / 6 / 8 / 12 kHz) from cheap one-pole filters;
//! * a spectral tilt (low-band vs high-band energy ratio): speech rolls off steeply above ~4 kHz, music keeps energy up
//!   to 20 kHz;
//! * a tonality estimate from the normalised one-lag autocorrelation: a high lag-1 correlation means a smooth, low-pass
//!   (voiced/tonal) signal, a low one means broadband/noisy content;
//! * a zero-crossing rate, a cheap voiced/unvoiced and low/high-frequency proxy.
//!
//! These combine into [`FrameAnalysis::music_probability`] (0 = clearly speech,
//! 1 = clearly music) and a recommended [`Bandwidth`] from where the signal's
//! energy dies out. The encoder smooths the probability across frames
//! (hysteresis) so the mode does not flip on a single borderline frame.

use crate::packet::Bandwidth;

/// Per-frame signal features and the decisions derived from them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameAnalysis {
    /// Likelihood the frame is music rather than speech, in `[0, 1]`.
    /// 0 = clearly speech, 1 = clearly music.
    pub music_probability: f32,
    /// The highest audio bandwidth the signal's spectrum actually occupies,
    /// i.e. the bandwidth above which there is negligible energy. Caps the
    /// automatic bandwidth selection.
    pub detected_bandwidth: Bandwidth,
    /// Frame mean-square energy (per sample), a silence/level gate.
    pub energy: f32,
}

/// Analyse one frame of interleaved 48 kHz f32 PCM (down-mixed to mono here).
///
/// The estimates are intentionally scale-tolerant: every discriminator is an
/// *energy ratio* or a *correlation*, so absolute level only matters for the
/// silence gate.
pub(crate) fn analyze_frame(pcm: &[f32], channels: usize) -> FrameAnalysis {
    let ch = channels.max(1);
    let n = pcm.len() / ch;
    if n == 0 {
        return FrameAnalysis {
            music_probability: 0.5,
            detected_bandwidth: Bandwidth::FullBand,
            energy: 0.0,
        };
    }

    // Everything below is gathered in ONE pass over the frame, down-mixing to
    // mono on the fly (no scratch buffer): total energy, four band-edge
    // low-pass energies, the lag-1 autocorrelation, and the zero-crossing
    // count. The per-sample arithmetic and its accumulation order are identical
    // to computing each in its own pass, so the analysis result - and every
    // mode/bandwidth decision it drives - is byte-for-byte unchanged; it is
    // just one sweep instead of seven.
    //
    // The band energies come from one-pole low-pass filters at the bandwidth
    // edges (Hz): 4000 (NB), 6000 (MB), 8000 (WB), 12000 (SWB); above 12 kHz is
    // FB. The slice energy is the increase between successive low-pass energies.
    // The lowpass coefficient for cutoff f is a = dt/(rc+dt), rc = 1/(2*pi*f),
    // dt = 1/48000.
    const FS: f32 = 48_000.0;
    let lp_coef = |f_hz: f32| -> f32 {
        let rc = 1.0 / (core::f32::consts::TAU * f_hz);
        let dt = 1.0 / FS;
        dt / (rc + dt)
    };
    let (a0, a1, a2, a3) = (lp_coef(4_000.0), lp_coef(6_000.0), lp_coef(8_000.0), lp_coef(12_000.0));

    let mut energy_acc = 0.0f32;
    let (mut y0, mut y1, mut y2, mut y3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut e0, mut e1, mut e2, mut e3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut r0, mut r1) = (0.0f32, 0.0f32);
    let mut zc = 0usize;
    let mut prev = 0.0f32;
    for i in 0..n {
        // Down-mix this sample to mono (the mode decision is a per-stream
        // property; the downmix is analysed as one signal).
        let x = if ch == 1 {
            pcm[i]
        } else {
            let mut acc = 0.0f32;
            for c in 0..ch {
                acc += pcm[i * ch + c];
            }
            acc / ch as f32
        };
        energy_acc += x * x;
        y0 += a0 * (x - y0);
        e0 += y0 * y0;
        y1 += a1 * (x - y1);
        e1 += y1 * y1;
        y2 += a2 * (x - y2);
        e2 += y2 * y2;
        y3 += a3 * (x - y3);
        e3 += y3 * y3;
        if i >= 1 {
            // Lag-1 autocorrelation (spectral tilt) and zero-crossing rate.
            r0 += x * x;
            r1 += x * prev;
            if (x >= 0.0) != (prev >= 0.0) {
                zc += 1;
            }
        }
        prev = x;
    }
    let energy = energy_acc / n as f32;
    let lp_energy = [e0 / n as f32, e1 / n as f32, e2 / n as f32, e3 / n as f32];

    // Band energies (energy in each frequency slice):
    //   b0: 0..4k, b1: 4..6k, b2: 6..8k, b3: 8..12k, b4: 12k..24k
    let e_total = energy.max(1e-12);
    let b0 = lp_energy[0];
    let b1 = (lp_energy[1] - lp_energy[0]).max(0.0);
    let b2 = (lp_energy[2] - lp_energy[1]).max(0.0);
    let b3 = (lp_energy[3] - lp_energy[2]).max(0.0);
    let b4 = (e_total - lp_energy[3]).max(0.0);

    // Fraction of energy above 8 kHz (music spreads here, speech does not),
    // in the top octave (>12 kHz, decisive for fullband music), and in the
    // <4 kHz speech formant region.
    let high_frac = (b3 + b4) / e_total;
    let top_frac = b4 / e_total;
    let low_frac = b0 / e_total;

    // Lag-1 autocorrelation: near +1 is a smooth low-pass signal (voiced
    // speech, bass-heavy tone), near 0 is broadband/noise-like.
    let lag1_corr = if r0 > 1e-12 { (r1 / r0).clamp(-1.0, 1.0) } else { 0.0 };
    // Zero-crossings per sample: low for voiced/bass, high for bright/noisy.
    let zcr = zc as f32 / n as f32;

    // --- Recommended bandwidth ---------------------------------------------
    // Pick the narrowest bandwidth that still contains essentially all the
    // signal energy, so we never spend bits coding empty high bands. Thresholds
    // are fractions of total energy above each edge.
    let detected_bandwidth = if top_frac > 0.02 {
        Bandwidth::FullBand
    } else if b3 / e_total > 0.02 {
        Bandwidth::SuperWideBand
    } else if b2 / e_total > 0.02 {
        Bandwidth::WideBand
    } else if b1 / e_total > 0.02 {
        Bandwidth::MediumBand
    } else {
        Bandwidth::NarrowBand
    };

    // --- Music vs speech ---------------------------------------------------
    // Combine the cues into a single probability. Each term pushes toward music
    // (1.0) or speech (0.0); we start neutral and accumulate evidence.
    //
    //  * Energy above 8 kHz is the strongest music cue - speech has almost none (it is band-limited to ~4 kHz, wideband
    //    to ~7 kHz).
    //  * Energy in the top octave (>12 kHz) is decisive for fullband music.
    //  * A dominant <4 kHz low band is a speech cue.
    //  * Very high lag-1 correlation with low ZCR is a strongly voiced/tonal cue (could be either, so weighted lightly
    //    toward speech).
    let mut p = 0.5f32;
    p += 1.6 * high_frac; // up to +~1.6 toward music
    p += 2.0 * top_frac; // top octave is decisive
    p -= 0.5 * low_frac; // strong low band → speech
    // Voiced/tonal low-pass signal: nudge toward speech.
    if lag1_corr > 0.9 && zcr < 0.1 && high_frac < 0.05 {
        p -= 0.2;
    }
    // Noise-like broadband content with lots of high frequency: music/noise.
    if zcr > 0.25 && high_frac > 0.1 {
        p += 0.2;
    }
    let music_probability = p.clamp(0.0, 1.0);

    FrameAnalysis {
        music_probability,
        detected_bandwidth,
        energy,
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    fn tone(freqs: &[(f32, f32)], n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                freqs
                    .iter()
                    .map(|&(f, a)| a * (core::f32::consts::TAU * f * t).sin())
                    .sum()
            })
            .collect()
    }

    #[test]
    fn speech_like_low_band_reads_as_speech() {
        // A 300 Hz voiced tone with a couple of low formants, nothing above
        // 3.5 kHz - unmistakably speech-band.
        let pcm = tone(&[(180.0, 0.5), (900.0, 0.25), (2400.0, 0.1)], 960);
        let a = analyze_frame(&pcm, 1);
        assert!(
            a.music_probability < 0.45,
            "expected speech, got p={}",
            a.music_probability
        );
        assert!(
            matches!(
                a.detected_bandwidth,
                Bandwidth::NarrowBand | Bandwidth::MediumBand | Bandwidth::WideBand
            ),
            "speech bandwidth {:?}",
            a.detected_bandwidth
        );
    }

    #[test]
    fn bright_broadband_reads_as_music() {
        // Strong content up to 15 kHz - fullband music.
        let pcm = tone(&[(220.0, 0.3), (3000.0, 0.3), (9000.0, 0.35), (15000.0, 0.3)], 960);
        let a = analyze_frame(&pcm, 1);
        assert!(
            a.music_probability > 0.55,
            "expected music, got p={}",
            a.music_probability
        );
        assert_eq!(a.detected_bandwidth, Bandwidth::FullBand);
    }

    #[test]
    fn silence_is_handled() {
        let a = analyze_frame(&[0.0; 960], 1);
        assert!(a.energy < 1e-9);
    }
}

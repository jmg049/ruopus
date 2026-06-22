//! Signal analysis driving the encoder's mode/bandwidth decision.
//!
//! libopus's real `run_analysis` (analysis.c) runs an MLP tonality model over a
//! sliding MDCT to produce a music-probability and a recommended bandwidth. We
//! are **not** required to match its decisions bit-for-bit - only the *decoder*
//! is conformance-normative, so any mode/bandwidth choice that produces a valid,
//! round-tripping Opus packet is correct. This module therefore implements a
//! lighter-weight, fully documented analysis from the 48 kHz time-domain frame:
//!
//! * a 6-band energy split (the same band edges libopus's bandwidth detector
//!   keys off: ~4 / 6 / 8 / 12 kHz) computed with cheap one-pole/one-zero
//!   filters, no FFT, keeping the core crate dependency-free;
//! * a **spectral tilt** (low-band vs high-band energy ratio) - speech rolls off
//!   steeply above ~4 kHz, music keeps energy up to 20 kHz;
//! * a **tonality** estimate from the normalised one-lag autocorrelation - a
//!   high lag-1 correlation means a smooth, low-pass (voiced/tonal) signal, a
//!   low one means broadband/noisy content;
//! * a **zero-crossing rate**, a classic cheap voiced/unvoiced and
//!   low/high-frequency proxy.
//!
//! These combine into [`FrameAnalysis::music_probability`] (0 = clearly speech,
//! 1 = clearly music) and a recommended [`Bandwidth`] from where the signal's
//! energy actually dies out. The encoder smooths the probability across frames
//! (hysteresis) so the mode does not flip every frame.

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

    // Down-mix to mono (the mode decision is a per-stream property; libopus
    // analyses the downmix too).
    let mut mono = alloc::vec![0.0f32; n];
    if ch == 1 {
        mono.copy_from_slice(&pcm[..n]);
    } else {
        for i in 0..n {
            let mut acc = 0.0f32;
            for c in 0..ch {
                acc += pcm[i * ch + c];
            }
            mono[i] = acc / ch as f32;
        }
    }

    // --- Total energy (level / silence gate) -------------------------------
    let energy = mono.iter().map(|&v| v * v).sum::<f32>() / n as f32;

    // --- Band energies via cheap RC filters --------------------------------
    // We measure energy in six cumulative high-pass-residual buckets by running
    // a chain of one-pole low-pass filters at the bandwidth edges and taking
    // the energy *removed* between adjacent cutoffs as the band energy. Cutoffs
    // (Hz): 4000 (NB edge), 6000 (MB), 8000 (WB), 12000 (SWB); everything above
    // 12 kHz is the FB band. The lowpass coefficient for cutoff f is
    // a = dt/(rc+dt), rc = 1/(2*pi*f), dt = 1/48000.
    const FS: f32 = 48_000.0;
    let lp_coef = |f_hz: f32| -> f32 {
        let rc = 1.0 / (core::f32::consts::TAU * f_hz);
        let dt = 1.0 / FS;
        dt / (rc + dt)
    };
    let cutoffs = [4_000.0f32, 6_000.0, 8_000.0, 12_000.0];
    // Energy of the signal low-passed at each cutoff.
    let mut lp_energy = [0.0f32; 4];
    for (k, &fc) in cutoffs.iter().enumerate() {
        let a = lp_coef(fc);
        let mut y = 0.0f32;
        let mut e = 0.0f32;
        for &x in &mono {
            y += a * (x - y);
            e += y * y;
        }
        lp_energy[k] = e / n as f32;
    }
    // Band energies (energy in each frequency slice):
    //   b0: 0..4k, b1: 4..6k, b2: 6..8k, b3: 8..12k, b4: 12k..24k
    // A wider low-pass passes everything a narrower one does plus more, so the
    // slice energy is the increase between successive low-pass energies; the
    // top slice is what the widest low-pass dropped vs the full signal.
    let e_total = energy.max(1e-12);
    let b0 = lp_energy[0];
    let b1 = (lp_energy[1] - lp_energy[0]).max(0.0);
    let b2 = (lp_energy[2] - lp_energy[1]).max(0.0);
    let b3 = (lp_energy[3] - lp_energy[2]).max(0.0);
    let b4 = (e_total - lp_energy[3]).max(0.0);

    // Fraction of energy above 8 kHz (the wideband edge): music spreads here,
    // narrowband/wideband speech does not.
    let high_frac = (b3 + b4) / e_total;
    // Fraction of energy in the very top octave (>12 kHz): only fullband music
    // / cymbals / sibilance reach here.
    let top_frac = b4 / e_total;
    // Low-band (speech formant region, <4 kHz) fraction.
    let low_frac = b0 / e_total;

    // --- Spectral tilt -----------------------------------------------------
    // The normalised one-lag autocorrelation. A value near +1 means a smooth,
    // strongly low-pass signal (voiced speech, bass-heavy tone); near 0 means a
    // broadband / noise-like signal. libopus's analysis uses a tonality measure
    // from the MDCT; this lag-1 correlation is the time-domain analogue.
    let mut r0 = 0.0f32;
    let mut r1 = 0.0f32;
    for i in 1..n {
        r0 += mono[i] * mono[i];
        r1 += mono[i] * mono[i - 1];
    }
    let lag1_corr = if r0 > 1e-12 { (r1 / r0).clamp(-1.0, 1.0) } else { 0.0 };

    // --- Zero-crossing rate ------------------------------------------------
    // Crossings per sample: low for voiced speech / bass, high for noisy or
    // bright (high-frequency) content. A cheap brightness proxy.
    let mut zc = 0usize;
    for i in 1..n {
        if (mono[i] >= 0.0) != (mono[i - 1] >= 0.0) {
            zc += 1;
        }
    }
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
    //  * Energy above 8 kHz is the strongest music cue - speech has almost
    //    none (it is band-limited to ~4 kHz, wideband to ~7 kHz).
    //  * Energy in the top octave (>12 kHz) is decisive for fullband music.
    //  * A dominant <4 kHz low band is a speech cue.
    //  * Very high lag-1 correlation with low ZCR is a strongly voiced/tonal
    //    cue (could be either, so weighted lightly toward speech).
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
    use super::*;
    use alloc::vec::Vec;

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
            matches!(a.detected_bandwidth, Bandwidth::NarrowBand | Bandwidth::MediumBand | Bandwidth::WideBand),
            "speech bandwidth {:?}",
            a.detected_bandwidth
        );
    }

    #[test]
    fn bright_broadband_reads_as_music() {
        // Strong content up to 15 kHz - fullband music.
        let pcm = tone(
            &[(220.0, 0.3), (3000.0, 0.3), (9000.0, 0.35), (15000.0, 0.3)],
            960,
        );
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

//! The SILK decoder (RFC 6716 §4.2; normative `silk/` reference sources).
//!
//! SILK is the linear-prediction half of Opus: an LP layer carrying
//! 10 or 20 ms frames at an internal rate of 8, 12 or 16 kHz
//! (narrowband/mediumband/wideband), used alone or under CELT in hybrid
//! mode. Decoding a frame (`silk_decode_frame`) runs, in bitstream order:
//! header flags (VAD/LBRR), stereo prediction weights and mid-only flag for
//! stereo, frame type, quantisation gains, normalised LSF indices (two-stage
//! VQ), pitch lags and LTP filter coefficients for voiced frames, the LTP
//! scaling factor, the noise seed, and the shell-coded excitation; synthesis
//! then runs LTP and short-term LPC filters over the excitation, followed by
//! stereo unmixing and resampling to the output rate.
//!
//! Unlike CELT's float build, the normative SILK decoder is entirely
//! fixed-point - every operation here is integer arithmetic and must be
//! bit-exact, not only the entropy decoding.
//!
//! Build order mirrors the CELT port: static tables first (mechanically
//! extracted), then the entropy layer bottom-up, then synthesis, validated
//! per stage and finally against the official test vectors' final-range and
//! PCM oracles.

pub(crate) mod indices;
pub(crate) mod math;
pub(crate) mod pulses;
pub mod tables;

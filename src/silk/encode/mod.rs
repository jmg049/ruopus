//! The SILK encoder (RFC 6716 §5.2; normative `silk/` and `silk/float/`).
//!
//! SILK is the speech/low-bitrate half of Opus. The full encode pipeline is
//! assembled here - short-term (Burg LPC → NLSF VQ) and long-term (pitch +
//! LTP) prediction, noise-shaping analysis, gain quantisation, the noise-
//! shaping quantiser, voice-activity detection, rate control, mid/side stereo,
//! and the index/pulse bitstream - driven by [`api::SilkEncoder`] (mono) and
//! [`api::SilkStereoEncoder`].
//!
//! The analysis uses the reference float build; the quantisation indices it
//! emits are read back by the existing bit-exact fixed-point decoder, so the
//! round trip is the conformance oracle.

pub mod api;
pub(crate) mod control;
pub(crate) mod dsp;
pub(crate) mod frame;
pub(crate) mod gains;
pub(crate) mod lpc;
pub(crate) mod ltp;
pub(crate) mod nlsf;
pub(crate) mod noise_shape;
pub(crate) mod nsq;
pub(crate) mod pitch_analysis;
pub(crate) mod resample;
pub(crate) mod resample_in;
pub(crate) mod stereo;
pub(crate) mod vad;

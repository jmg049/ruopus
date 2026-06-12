//! The CELT layer (RFC 6716 §4.3) - under construction.
//!
//! CELT is the MDCT half of Opus: band energies coded with a Laplace model
//! plus fine refinement, and band shapes coded as PVQ pulse vectors. The
//! decode pipeline (§4.3) is
//!
//! ```text
//! coarse energy → fine energy → bit allocation → PVQ shapes →
//! anti-collapse → denormalization → inverse MDCT → post-filter
//! ```
//!
//! Implemented so far, bottom-up - each kernel fully tested in isolation
//! before the pipeline is assembled:
//!
//! | Module | Reference | Contents |
//! |--------|-----------|----------|
//! | [`laplace`] | `laplace.c`, §4.3.2.1 | the Laplace coder for coarse energy deltas |
//! | [`cwrs`] | `cwrs.c`, §4.3.4.2 | PVQ codeword enumeration (pulse vectors ↔ indices) |
//! | [`modes`] | `modes.c`, Table 55 | static data of the standard 48 kHz mode |
//! | [`energy`] | `quant_bands.c`, §4.3.2 | coarse/fine/finalise energy envelope decoding |
//! | [`rate`] | `rate.c`, §4.3.3 | the bit allocation: quality interpolation, band skipping, fine/shape split |
//! | [`tables`] | `static_modes_*.h` | mechanically extracted allocation and pulse-cache tables |
//! | [`vq`] | `vq.c`, §4.3.4.3 | spreading rotation, PVQ shape decoding, renormalisation |
//! | [`bands`] | `bands.c`, §4.3.4 | the band loop: theta splits, stereo, folding, collapse masks |
//! | [`mdct`] | `mdct.c`, §4.3.7 | the low-overlap MDCT (forward + backward) with the FFT backend seam |
//! | [`decoder`] | `celt_decoder.c`, §4.3 | the frame driver: flags, post-filter, synthesis, de-emphasis |

pub mod bands;
pub mod cwrs;
pub mod decoder;
pub mod energy;
pub mod laplace;
pub mod mdct;
pub mod modes;
pub mod rate;
pub mod tables;
pub mod vq;

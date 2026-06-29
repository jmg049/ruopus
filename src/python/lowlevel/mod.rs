//! The `ruopus.lowlevel` submodule: direct access to the SILK, LPC, and CELT
//! layers beneath the Opus packet codec. These are advanced building blocks;
//! ordinary use should prefer the top-level `OpusEncoder`/`OpusDecoder`.

pub mod celt;
pub mod lpc;
pub mod silk;

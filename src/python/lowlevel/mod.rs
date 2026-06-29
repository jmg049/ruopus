//! The `opus_native.lowlevel` submodule: direct access to the SILK and CELT
//! layers beneath the Opus packet codec. These are advanced building blocks;
//! ordinary use should prefer the top-level `OpusEncoder`/`OpusDecoder`.

pub mod celt;
pub mod silk;

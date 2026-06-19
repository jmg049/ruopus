# Encoder - known limitations and future work

The SILK/CELT/hybrid encoder is functional and competitive with libopus on
real speech (within ~1 dB SNR at matched bitrate; see
`examples/encoder_quality.rs`). The items below are deliberate
simplifications or deferred refinements, with enough context to pick each up.

## SILK per-frame rate control (`silk/encode/frame.rs`)

`encode_frame` enforces a hard `max_bits` cap (hybrid path) with libopus's
gain-multiplier loop: scale the unquantised gains coarser until the coded
frame fits, geometric until the over/under budget is bracketed, then
bisection-interpolated toward the cap, restoring the best fitting attempt.
Differences from the reference (`silk_encode_frame_FLP`):

- **No zero-pulse "damage control."** When a frame still busts at the 4×
  (`gainMult_Q8 == 1024`) ceiling, the reference zeroes all pulses and codes
  mid gains as a last resort. We instead leave the frame over budget and let
  `OpusEncoder::encode_auto` fall back to CELT-only for it. Reason: the
  reference's damage control advances the NSQ state inconsistently with the
  coded (zero) excitation, which would desync **our** decoder (our oracle is
  per-packet final-range equality). Doing it properly needs a decoder-style
  synthesis pass to resync the encoder NSQ state after zeroing the pulses.
  Impact: on a 9.4 s speech clip exactly one loud-transient frame falls back
  (super-wideband 24 kb/s); the CELT fallback actually codes that frame
  *better* than extreme-gain SILK would, so this is low priority.
- **No per-subframe gain locking** (`gain_lock`/`best_gain_mult`) and **no
  lambda adjustment** (`Lambda *= 1.5`, `quantOffsetType = 0` when stuck).
  Both are quality refinements for capped frames.

## Hybrid SILK/CELT split (`encoder.rs`)

- `compute_silk_rate_for_hybrid` ports libopus's rate table but only the
  no-FEC column and the +300 bps super-wideband nudge; the FEC columns and
  the stereo -1000 bps tweak are unused (no FEC / stereo-hybrid cap yet).
- `celt_floor` (the reserved CELT high-band byte share) is a heuristic
  (`(bands)*3 + 3`), tuned by hand on one speech clip. A rate-aware split
  would be better.

## SILK LPC analysis (`find_pred_coefs`)

Not yet matched to the reference - `encode_frame` runs Burg on the raw frame,
not on the gain-normalised, LTP-pre-whitened `LPC_in_pre`. A faithful port
regressed on synthetic tones; it must be gated on the real-speech quality
harness (`examples/encoder_quality.rs`), which is the only trustworthy oracle
since the encoder is not bit-exact-defined. See the memory note
`opus-find-pred-coefs` for the full analysis.

## Other

- **DTX / silence frames** are not implemented (every frame is coded active).
- The CELT-only path fills `max_bytes` (CBR) or shrinks to the VBR target;
  there is no constrained-VBR mode.

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
  (Per-subframe gain locking and the lambda/quant-offset adjustment for stuck
  frames are now implemented; only the damage control remains.)

## Hybrid SILK/CELT split (`encoder.rs`)

- `compute_silk_rate_for_hybrid` ports libopus's rate table but only the
  no-FEC column and the +300 bps super-wideband nudge; the FEC columns and
  the stereo -1000 bps tweak are unused (no FEC / stereo-hybrid cap yet).
- `celt_floor` (the reserved CELT high-band byte share) is a heuristic
  (`(bands)*3 + 3`), tuned by hand on one speech clip. A rate-aware split
  would be better.

## `encode_ogg_opus` pre-skip vs mode

`pre_skip` is fixed at 120 (the CELT reconstruction delay), but hybrid/SILK
have a smaller delay (~69 in libopus). So a hybrid/SILK file decodes ~51
samples (~1 ms) misaligned in a *third-party* decoder (ffmpeg still reports
0.97 correlation; our own decoder round-trips consistently because it uses the
same pre_skip). To make cross-decoder alignment exact, set `pre_skip` from the
mode `encode_auto` will pick for the chosen bitrate (≤ 40 kb/s fullband →
hybrid → ~69; else CELT → 120), or flush the encoder delay explicitly. Note
our decoder's measured hybrid delay (120) differs from libopus's (~69) - worth
confirming our decoder's hybrid delay is conformant.

## DTX (`OpusEncoder::set_dtx`)

Implemented: after 200 ms of inactivity `encode_auto` emits a 1-byte TOC-only
packet (the decoder conceals it), with the libopus run bounds (≤ 400 ms, then
a refresh frame). Caveat: activity is decided by a **simple energy threshold**
(`frame_is_active`, ≈ -60 dBFS) rather than libopus's VAD/analysis activity
probability - it catches genuine silence and very quiet gaps but not noisy
speech pauses. Using the SILK VAD's `speech_activity_q8` (currently buried in
`encode_frame`) would make DTX trigger on real low-activity content too.

## Other

- The CELT-only path fills `max_bytes` (CBR) or shrinks to the VBR target;
  there is no constrained-VBR mode.

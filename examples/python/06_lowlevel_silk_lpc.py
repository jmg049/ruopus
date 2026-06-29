"""Low-level building blocks: the SILK codec and LPC analysis.

The `opus_rs.lowlevel` submodule exposes the layers beneath the Opus packet
codec. These are advanced tools - ordinary use should prefer OpusEncoder /
OpusDecoder - but they let you drive SILK directly or run linear-prediction
analysis on arbitrary signals.

Run: python examples/python/06_lowlevel_silk_lpc.py
"""

import numpy as np

from opus_rs import lowlevel as ll


def silk_roundtrip() -> None:
    # SILK works on int16 PCM at its internal rate (8/12/16 kHz). One 20 ms frame
    # at 16 kHz is nb_subfr * 5 * fs_khz = 4 * 5 * 16 = 320 samples.
    t = np.arange(320) / 16000.0
    pcm = (8000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)

    enc = ll.SilkEncoder(fs_khz=16, nb_subfr=4, bitrate=20000, complexity=8)
    payload = enc.encode(pcm)

    ctl = ll.DecControl(
        channels_internal=1,
        channels_api=1,
        internal_sample_rate=16000,
        api_sample_rate=16000,
        payload_size_ms=20,
    )
    dec = ll.SilkDecoder()
    out = dec.decode(payload, ctl)
    print(f"SILK: {len(pcm)} samples -> {len(payload)} bytes -> {out.shape} {out.dtype}")
    print(f"  encoder final range: 0x{enc.final_range:08x}")


def lpc_analysis() -> None:
    # Linear-prediction analysis of a synthetic signal.
    sig = (0.5 * np.sin(2 * np.pi * 5 * np.arange(400) / 400)).astype(np.float32)

    coeffs = ll.lpc_analysis(sig, order=16)
    print(f"\nLPC: {coeffs!r}")

    # residual -> synthesis reconstructs the signal almost exactly.
    residual = ll.lpc_residual(sig, coeffs)
    recon = ll.lpc_synthesis(residual, coeffs)
    err = float(np.abs(recon[: len(sig)] - sig).max())
    print(f"  residual {residual.shape} -> synthesis, peak reconstruction error {err:.2e}")

    pitch = ll.estimate_pitch(sig, sample_rate=16000)
    print(f"  estimated pitch (period_samples, confidence): {pitch}")


def main() -> None:
    silk_roundtrip()
    lpc_analysis()


if __name__ == "__main__":
    main()

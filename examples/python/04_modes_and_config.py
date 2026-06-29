"""Encoder modes and configuration.

Opus has three coding modes - SILK (speech), CELT (music/low-delay), and hybrid
(both) - plus tunable bitrate, complexity, bandwidth and DTX. Configuration is
exposed as plain properties on the encoder.

Run: python examples/python/04_modes_and_config.py
"""

import numpy as np

from ruopus import Bandwidth, OpusDecoder, OpusEncoder

SR = 48000
FRAME = 960  # 20 ms


def speech_like() -> np.ndarray:
    t = np.arange(FRAME) / SR
    sig = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.sin(2 * np.pi * 900 * t)
    return sig.astype(np.float32).reshape(FRAME, 1)


def main() -> None:
    frame = speech_like()

    enc = OpusEncoder(1, bitrate=24000)

    # Configuration is just properties.
    enc.complexity = 6
    enc.dtx = False
    print(f"configured: {enc!r}")
    print(f"  complexity={enc.complexity}, bitrate={enc.bitrate}")

    # The coding modes for the same 20 ms frame. SILK and CELT honour the
    # configured bandwidth; hybrid requires super-wideband or fullband.
    enc.bandwidth = Bandwidth.WideBand
    auto = enc.encode_auto(frame)  # picks a mode for you
    silk = enc.encode_silk(frame)  # force SILK (speech), wideband
    celt = enc.encode(frame)  # force CELT (music), wideband
    enc.bandwidth = Bandwidth.FullBand
    hybrid = enc.encode_hybrid(frame)  # force hybrid, fullband

    for name, packet in [("auto", auto), ("silk", silk), ("hybrid", hybrid), ("celt", celt)]:
        dec = OpusDecoder(1)
        pcm = dec.decode_packet(packet)
        print(f"  {name:6s}: {len(packet):4d}-byte packet -> {pcm.shape}")

    # DTX: after a run of silence, encode_auto emits 1-byte packets the decoder
    # conceals, dropping the silence bitrate to ~0.4 kb/s.
    enc.dtx = True
    silence = np.zeros((FRAME, 1), dtype=np.float32)
    sizes = [len(enc.encode_auto(silence)) for _ in range(20)]
    print(f"  DTX over 20 silent frames -> packet sizes settle to {min(sizes)} byte(s)")


if __name__ == "__main__":
    main()

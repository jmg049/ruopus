"""Basic encode -> decode round-trip.

Run: python examples/python/01_encode_decode.py
"""

import numpy as np

from opus_rs import OpusDecoder, OpusEncoder

SR = 48000
FRAME = 960  # 20 ms at 48 kHz


def main() -> None:
    # A 20 ms stereo frame: a 440 Hz tone, slightly quieter on the right.
    t = np.arange(FRAME) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    frame = np.stack([tone, tone * 0.9], axis=1)  # shape (960, 2), float32
    assert frame.shape == (FRAME, 2) and frame.dtype == np.float32

    enc = OpusEncoder(2, bitrate=64000)
    dec = OpusDecoder(2)

    packet = enc.encode(frame)  # -> bytes
    pcm = dec.decode_packet(packet)  # -> (960, 2) float32

    print(f"encoded {frame.nbytes} bytes of PCM into a {len(packet)}-byte packet")
    print(f"decoded back to {pcm.shape} {pcm.dtype}")

    # The range coder is the bit-exact conformance oracle: a correct encoder and
    # decoder finish the packet in the same internal state.
    assert enc.final_range == dec.final_range
    print(f"range-coder oracle matches: 0x{enc.final_range:08x}")

    # Reconstruction is lossy (Opus is a perceptual codec) but close.
    err = np.abs(pcm - frame).max()
    print(f"peak reconstruction error: {err:.4f}")


if __name__ == "__main__":
    main()

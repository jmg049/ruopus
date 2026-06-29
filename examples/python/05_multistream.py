"""Multistream (surround) decoding.

`MultistreamDecoder` decodes N elementary Opus streams per packet - the first
`coupled` as stereo, the rest mono - and routes their channels to the output
layout via `mapping`. A single ordinary Opus packet is a valid one-stream
multistream packet, which makes the routing easy to demonstrate.

Run: python examples/python/05_multistream.py
"""

import numpy as np

from ruopus import MultistreamDecoder, OpusEncoder

SR = 48000
FRAME = 960  # 20 ms


def main() -> None:
    t = np.arange(FRAME) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # One mono stream routed to one output channel.
    mono_packet = OpusEncoder(1).encode(tone.reshape(FRAME, 1))
    ms_mono = MultistreamDecoder(streams=1, coupled=0, mapping=[0])
    out = ms_mono.decode_packet(mono_packet)
    print(f"1 stream, 0 coupled, mapping=[0] -> {out.shape} ({ms_mono.channels}ch)")

    # One coupled (stereo) stream routed to two output channels.
    stereo_packet = OpusEncoder(2).encode(np.stack([tone, tone * 0.8], axis=1))
    ms_stereo = MultistreamDecoder(streams=1, coupled=1, mapping=[0, 1])
    out = ms_stereo.decode_packet(stereo_packet)
    print(f"1 stream, 1 coupled, mapping=[0,1] -> {out.shape} ({ms_stereo.channels}ch)")

    # `mapping=255` makes a silent output channel; here we duplicate the stereo
    # pair and add a silent third channel.
    ms_custom = MultistreamDecoder(streams=1, coupled=1, mapping=[0, 1, 255])
    out = ms_custom.decode_packet(stereo_packet)
    print(f"mapping=[0,1,255] -> {out.shape}; channel 2 is silent: {np.all(out[:, 2] == 0)}")

    print(
        "\n(True N-stream surround needs self-delimited multistream packets from a "
        "multistream encoder, which the high-level API does not yet produce.)"
    )


if __name__ == "__main__":
    main()

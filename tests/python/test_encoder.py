"""Phase 2 binding tests: encoder, properties, ogg round-trips."""

import numpy as np
import pytest

from ruopus import (
    Bandwidth,
    EncodeError,
    OpusDecoder,
    OpusEncoder,
    decode_ogg_opus,
    encode_ogg_opus,
)


def _tone(frames: int, channels: int, freq: float = 440.0) -> np.ndarray:
    t = np.arange(frames) / 48000.0
    mono = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 1:
        return mono.reshape(frames, 1)
    return np.stack([mono] * channels, axis=1)


@pytest.mark.parametrize("channels", [1, 2])
def test_encode_decode_roundtrip_and_range_oracle(channels):
    frame = _tone(960, channels)
    enc = OpusEncoder(channels, bitrate=64000)
    packet = enc.encode_auto(frame)
    assert isinstance(packet, bytes) and len(packet) >= 3

    dec = OpusDecoder(channels)
    out = dec.decode_packet(packet)
    assert out.shape == (960, channels) and out.dtype == np.float32
    # The range coder is the bit-exact conformance oracle: encoder and decoder
    # must finish the packet in the same state.
    assert enc.final_range == dec.final_range != 0


def test_encode_accepts_1d_and_2d():
    enc = OpusEncoder(2, bitrate=64000)
    frame2d = _tone(960, 2)
    assert isinstance(enc.encode(frame2d), bytes)
    assert isinstance(enc.encode(frame2d.reshape(-1)), bytes)  # interleaved 1-D


def test_encode_modes():
    enc = OpusEncoder(1, bitrate=32000)
    assert isinstance(enc.encode(_tone(480, 1)), bytes)         # CELT
    assert isinstance(enc.encode_silk(_tone(960, 1)), bytes)    # SILK
    assert isinstance(enc.encode_hybrid(_tone(960, 1)), bytes)  # hybrid


def test_bad_frame_size_raises_encode_error():
    enc = OpusEncoder(1)
    with pytest.raises(EncodeError):
        enc.encode(_tone(123, 1))  # not a valid frame size


def test_wrong_channel_columns_raises():
    enc = OpusEncoder(2)
    with pytest.raises(ValueError):
        enc.encode(_tone(960, 1))  # 1 column but 2-channel encoder


def test_properties_roundtrip_and_clamp():
    enc = OpusEncoder(2)
    assert enc.channels == 2
    enc.complexity = 3
    assert enc.complexity == 3
    enc.complexity = 99
    assert enc.complexity == 10  # clamped to 0..=10
    enc.bitrate = None
    assert enc.bitrate is None
    enc.bitrate = 24000
    assert enc.bitrate == 24000
    enc.dtx = True
    assert enc.dtx is True
    enc.bandwidth = Bandwidth.WideBand
    assert enc.bandwidth == Bandwidth.WideBand


def test_ogg_roundtrip_and_head():
    mono = _tone(48000, 1, freq=220.0).reshape(-1)
    ogg = encode_ogg_opus(mono, 1, 48000)
    assert isinstance(ogg, bytes) and len(ogg) > 0

    pcm, head = decode_ogg_opus(ogg)
    assert pcm.dtype == np.float32 and pcm.shape[1] == 1
    assert head.channel_count == 1
    assert head.version >= 1
    assert head.mapping_family == 0
    assert head.stream_count is None and head.channel_mapping is None
    assert isinstance(head.to_bytes(), bytes)

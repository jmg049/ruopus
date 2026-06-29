"""Phase 3 binding tests: Toc, Packet, MultistreamDecoder."""

import numpy as np
import pytest

from opus_rs import (
    Bandwidth,
    FrameSize,
    Mode,
    MultistreamDecoder,
    OpusEncoder,
    Packet,
    PacketError,
    Toc,
)


def _stereo_packet() -> bytes:
    t = np.arange(960) / 48000.0
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return OpusEncoder(2, bitrate=64000).encode(np.stack([tone, tone], axis=1))


def test_packet_is_a_sequence_of_frames():
    p = Packet(_stereo_packet())
    assert len(p) >= 1
    assert isinstance(p[0], bytes)
    assert isinstance(p[-1], bytes)  # negative indexing
    assert list(p) == p.frames  # __iter__ via __getitem__ matches the property
    assert all(isinstance(f, bytes) for f in p.frames)
    assert p.duration.total_seconds() > 0
    assert isinstance(p.padding, int)
    with pytest.raises(IndexError):
        _ = p[10_000]


def test_packet_parse_errors():
    with pytest.raises(PacketError):
        Packet(b"\xff\xff\xff")
    with pytest.raises(PacketError):
        Packet.parse_self_delimited(b"")


def test_toc_from_packet_has_enum_typed_fields():
    tc = Packet(_stereo_packet()).toc
    assert tc.channels == 2 and tc.stereo is True
    assert isinstance(tc.mode, Mode)
    assert isinstance(tc.bandwidth, Bandwidth)
    assert isinstance(tc.frame_size, FrameSize)
    assert int(tc) == tc.byte


def test_toc_value_semantics_and_from_parts():
    a = Toc.from_parts(28, True, 0)  # fullband CELT, stereo, one frame
    assert a.config == 28 and a.channels == 2 and a.mode == Mode.CeltOnly
    assert Toc(a.byte) == a
    assert hash(Toc(a.byte)) == hash(a)
    with pytest.raises(ValueError):
        Toc.from_parts(32, False, 0)
    with pytest.raises(ValueError):
        Toc.from_parts(0, False, 4)


def test_multistream_single_stream_mono():
    t = np.arange(960) / 48000.0
    mono = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32).reshape(960, 1)
    pkt = OpusEncoder(1, bitrate=48000).encode(mono)
    ms = MultistreamDecoder(streams=1, coupled=0, mapping=[0])
    out = ms.decode_packet(pkt)
    assert out.shape == (960, 1) and out.dtype == np.float32
    assert ms.channels == 1 and ms.sample_rate == 48000


def test_multistream_coupled_stereo():
    ms = MultistreamDecoder(streams=1, coupled=1, mapping=[0, 1])
    out = ms.decode_packet(_stereo_packet())
    assert out.shape == (960, 2) and ms.channels == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(streams=0, coupled=0, mapping=[0]),
        dict(streams=1, coupled=2, mapping=[0]),
        dict(streams=1, coupled=0, mapping=[9]),
        dict(streams=1, coupled=0, mapping=[]),
        dict(streams=1, coupled=0, mapping=[0], sample_rate=44100),
    ],
)
def test_multistream_validation(kwargs):
    with pytest.raises(ValueError):
        MultistreamDecoder(**kwargs)

"""Phase 1 binding tests: decoder, enums, exception hierarchy.

Full encode->decode round-trips arrive with the encoder bindings (Phase 2);
here we exercise the decoder's loss/DTX paths, the NumPy zero-copy output
contract, the packet enums, and the exception hierarchy.
"""

import numpy as np
import pytest

import ruopus as op
from ruopus import (
    Bandwidth,
    EncodeError,
    FrameSize,
    Mode,
    OggError,
    OpusDecoder,
    OpusError,
    PacketError,
)


def test_version():
    v = op.version()
    assert isinstance(v, str)
    assert v.count(".") >= 2  # semver-ish, e.g. "0.1.0"


def test_decode_lost_mono_shape_dtype_contiguous():
    dec = OpusDecoder(1)
    pcm = dec.decode_lost(960)
    assert pcm.shape == (960, 1)
    assert pcm.dtype == np.float32
    assert pcm.flags["C_CONTIGUOUS"]  # interleaved (frames, channels)


def test_decode_lost_stereo_and_rate():
    dec = OpusDecoder(2, sample_rate=24000)
    assert dec.channels == 2 and dec.sample_rate == 24000
    pcm = dec.decode_lost(480)
    assert pcm.shape == (480, 2)
    assert "channels=2" in repr(dec)


def test_decode_packet_dtx_and_i16():
    dec = OpusDecoder(1)
    assert dec.decode_packet(b"\x3c").shape == (120, 1)  # 1-byte TOC -> conceal
    out = dec.decode_packet_i16(b"")
    assert out.dtype == np.int16 and out.shape == (120, 1)


@pytest.mark.parametrize("channels", [0, 3])
def test_bad_channels_raise_value_error(channels):
    with pytest.raises(ValueError):
        OpusDecoder(channels)


def test_bad_sample_rate_raises():
    with pytest.raises(ValueError):
        OpusDecoder(1, sample_rate=44100)


def test_malformed_packet_raises_packet_error():
    dec = OpusDecoder(1)
    with pytest.raises(PacketError) as ei:
        dec.decode_packet(b"\xff\xff\xff")
    assert str(ei.value)  # non-empty message preserved from Rust Display
    assert isinstance(ei.value, OpusError)


def test_bandwidth_enum():
    assert int(Bandwidth.FullBand) == 4
    assert Bandwidth.FullBand.sample_rate_hz == 48000
    assert Bandwidth.FullBand.audio_bandwidth_hz == 20000
    assert Bandwidth.NarrowBand < Bandwidth.FullBand
    assert len({Bandwidth.FullBand, Bandwidth.FullBand}) == 1  # hashable


def test_mode_and_framesize_enums():
    assert repr(Mode.Hybrid) == "Mode.Hybrid"
    assert FrameSize.Ms20.tenth_ms == 200
    assert FrameSize.Ms20.samples_per_channel_48k == 960
    assert FrameSize.Ms20.duration.total_seconds() == pytest.approx(0.020)


def test_exception_hierarchy():
    for exc in (PacketError, EncodeError, OggError):
        assert issubclass(exc, OpusError)
    assert issubclass(OpusError, Exception)

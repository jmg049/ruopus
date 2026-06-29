"""Phase 4 binding tests: the opus_rs.lowlevel submodule (SILK/LPC/CELT)."""

import numpy as np
import pytest

from opus_rs import lowlevel as ll


def test_submodule_surface():
    expected = {
        "SilkEncoder", "SilkStereoEncoder", "SilkDecoder", "DecControl",
        "CeltEncoder", "CeltDecoder", "LpcCoefficients",
        "compute_autocorrelation", "levinson_durbin", "lpc_analysis",
        "lpc_residual", "lpc_synthesis", "lpc_residual_stateful",
        "lpc_synthesis_stateful", "estimate_pitch", "ltp_residual", "ltp_synthesis",
    }
    assert expected <= set(dir(ll))


def test_silk_encode_decode_roundtrip():
    t = np.arange(320) / 16000.0  # 20 ms at 16 kHz
    pcm = (8000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    enc = ll.SilkEncoder(16, 4, bitrate=20000, complexity=8)
    payload = enc.encode(pcm)
    assert isinstance(payload, bytes) and len(payload) > 0
    assert enc.final_range != 0

    ctl = ll.DecControl(1, 1, 16000, 16000, 20)
    dec = ll.SilkDecoder()
    out = dec.decode(payload, ctl)
    assert out.dtype == np.int16 and out.shape == (320, 1)
    assert dec.decode_lost(ctl).dtype == np.int16


def test_silk_encoder_validation_and_props():
    enc = ll.SilkEncoder(16, 4)
    enc.bitrate = 16000
    assert enc.bitrate == 16000
    enc.complexity = 99
    assert enc.complexity == 10
    with pytest.raises(ValueError):
        ll.SilkEncoder(11, 4)  # bad fs_khz
    with pytest.raises(ValueError):
        ll.SilkEncoder(16, 3)  # bad nb_subfr
    with pytest.raises(ValueError):
        enc.encode(np.zeros(7, dtype=np.int16))  # not a whole frame


def test_silk_stereo_encode():
    pcm = (8000 * np.sin(2 * np.pi * 300 * np.arange(320) / 16000)).astype(np.int16)
    payload = ll.SilkStereoEncoder(16, 4, bitrate=30000).encode(pcm, pcm)
    assert isinstance(payload, bytes) and len(payload) > 0


def test_deccontrol_fields():
    ctl = ll.DecControl(2, 2, 16000, 48000, 20)
    assert ctl.channels_internal == 2 and ctl.api_sample_rate == 48000
    ctl.payload_size_ms = 40
    assert ctl.payload_size_ms == 40


def test_lpc_roundtrip():
    sig = (0.5 * np.sin(2 * np.pi * 5 * np.arange(400) / 400)).astype(np.float32)
    coeffs = ll.lpc_analysis(sig, 16)
    assert coeffs.order == 16 and len(coeffs.coeffs) == 16
    res = ll.lpc_residual(sig, coeffs)
    rec = ll.lpc_synthesis(res, coeffs)
    assert res.dtype == np.float32
    assert np.abs(rec[: len(sig)] - sig).max() < 1e-4  # near-perfect reconstruction


def test_lpc_helpers():
    sig = (0.5 * np.sin(2 * np.pi * 5 * np.arange(400) / 400)).astype(np.float32)
    ac = ll.compute_autocorrelation(sig, 16)
    assert ac.dtype == np.float64 and ac.shape == (17,)
    coeffs = ll.levinson_durbin(ac, 16)
    assert coeffs is None or coeffs.order == 16
    out, state = ll.lpc_residual_stateful(sig, ll.lpc_analysis(sig, 16))
    assert out.dtype == np.float32 and isinstance(state, list)
    assert ll.ltp_synthesis(ll.ltp_residual(sig, 80, 0.5), 80, 0.5).dtype == np.float32


def test_celt_encode_and_conceal():
    cf = (0.3 * np.sin(2 * np.pi * 440 * np.arange(960) / 48000)).astype(np.float32)
    enc = ll.CeltEncoder(1, bitrate=64000)
    assert isinstance(enc.encode_frame(cf, 200), bytes)
    assert isinstance(enc.encode_frame_bw(cf, 200, 21), bytes)
    assert enc.final_range != 0

    dec = ll.CeltDecoder(1)
    lost = dec.decode_lost(960, 0, 21)
    assert lost.shape == (960, 1) and lost.dtype == np.float32

"""Packet-loss concealment (PLC) and forward error correction (FEC).

When a packet is lost, the decoder can conceal it from prior state with
`decode_lost`, or - if the encoder enabled in-band FEC (`inband_fec=True`) so
packets carry a redundant LBRR copy of the previous frame - genuinely recover
it from the *next* packet with `decode_fec`. FEC currently covers the mono
SILK-mode path (`encode_silk`).

Run: python examples/python/03_packet_loss.py
"""

import numpy as np

from opus_native import Bandwidth, OpusDecoder, OpusEncoder

SR = 48000
FRAME = 960  # 20 ms


def make_frame(i: int) -> np.ndarray:
    t = (np.arange(FRAME) + i * FRAME) / SR
    return (0.3 * np.sin(2 * np.pi * (300 + 40 * i) * t)).astype(np.float32).reshape(FRAME, 1)


def _maxcorr(a: np.ndarray, b: np.ndarray) -> float:
    """Delay-robust correlation (the codec adds algorithmic delay, so a naive
    sample-aligned compare understates the match)."""
    a, b = a.reshape(-1), b.reshape(-1)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    best = 0.0
    for lag in range(-200, 201, 4):
        x, y = a[max(0, lag) : n + min(0, lag)], b[max(0, -lag) : n + min(0, -lag)]
        if len(x) > 100 and x.std() > 1e-6 and y.std() > 1e-6:
            best = max(best, abs(float(np.corrcoef(x, y)[0, 1])))
    return best


def main() -> None:
    # Enable in-band FEC: each SILK packet carries an LBRR copy of the previous
    # packet's frame.
    enc = OpusEncoder(1, bitrate=24000, inband_fec=True, packet_loss_perc=30)
    enc.bandwidth = Bandwidth.WideBand
    packets = [enc.encode_silk(make_frame(i)) for i in range(5)]

    # Reference: a clean decode of frame 2 (carries the same codec delay).
    reference_frame2 = None
    d = OpusDecoder(1)
    for i, p in enumerate(packets):
        out = d.decode_packet(p)
        if i == 2:
            reference_frame2 = out

    # --- Loss recovered by FEC ---
    fec_dec = OpusDecoder(1)
    fec_dec.decode_packet(packets[0])
    fec_dec.decode_packet(packets[1])
    # Packet 2 is lost; recover it from packet 3's LBRR data.
    recovered = fec_dec.decode_fec(packets[3], FRAME)

    # --- Same loss with plain concealment, for comparison ---
    con_dec = OpusDecoder(1)
    con_dec.decode_packet(packets[0])
    con_dec.decode_packet(packets[1])
    concealed = con_dec.decode_lost(FRAME)

    fec_corr = _maxcorr(recovered, reference_frame2)
    con_corr = _maxcorr(concealed, reference_frame2)
    print(f"FEC recovery   -> {recovered.shape}, correlation with the lost frame: {fec_corr:.3f}")
    print(f"PLC concealment -> {concealed.shape}, correlation with the lost frame: {con_corr:.3f}")
    print(f"FEC reconstructs the lost frame ({fec_corr:.2f}); concealment can only guess ({con_corr:.2f}).")


if __name__ == "__main__":
    main()

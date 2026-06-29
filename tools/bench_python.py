#!/usr/bin/env python3
"""Benchmark the ruopus Python bindings.

Reports real numbers for the binding's performance claims:

* decode / encode throughput (× realtime, and per-call cost),
* the zero-copy FFI crossing: how much a naive "copy the PCM into NumPy"
  binding would cost on top of our ``PyArray1::from_vec`` move-out,
* GIL-release scaling: decode throughput across Python threads.

Run after installing the wheel (``python tools/build_python.py`` then
``pip install`` the wheel, or ``maturin develop``):

    python tools/bench_python.py
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import ruopus
from ruopus import OpusDecoder, OpusEncoder

SR = 48000
FRAME = 960  # 20 ms at 48 kHz
CHANNELS = 2
N_FRAMES = 2000
FRAME_SECONDS = FRAME / SR


def make_corpus() -> list[bytes]:
    """Encode N_FRAMES of varied stereo audio into Opus packets."""
    rng = np.random.default_rng(0)
    enc = OpusEncoder(CHANNELS, bitrate=96000)
    packets = []
    for i in range(N_FRAMES):
        t = (np.arange(FRAME) + i * FRAME) / SR
        freq = 220.0 + 40.0 * (i % 8)
        tone = 0.25 * np.sin(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(FRAME)
        frame = np.stack([tone, tone * 0.9], axis=1).astype(np.float32)
        packets.append(enc.encode_auto(frame))
    return packets


def timed(fn, *, iters: int, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_decode(packets: list[bytes]) -> None:
    def run():
        dec = OpusDecoder(CHANNELS)
        for p in packets:
            dec.decode_packet(p)

    secs = timed(run, iters=5)
    audio_secs = N_FRAMES * FRAME_SECONDS
    per_packet_us = secs / N_FRAMES * 1e6
    print(f"  decode   : {audio_secs / secs:8.1f}x realtime  |  {per_packet_us:7.2f} µs/packet")


def bench_encode() -> None:
    rng = np.random.default_rng(1)
    frames = [
        np.stack(
            [
                (0.25 * np.sin(2 * np.pi * 300 * np.arange(FRAME) / SR)
                 + 0.02 * rng.standard_normal(FRAME))
            ] * 2,
            axis=1,
        ).astype(np.float32)
        for _ in range(200)
    ]

    def run():
        enc = OpusEncoder(CHANNELS, bitrate=96000)
        for f in frames:
            enc.encode_auto(f)

    secs = timed(run, iters=5)
    audio_secs = len(frames) * FRAME_SECONDS
    per_frame_us = secs / len(frames) * 1e6
    print(f"  encode   : {audio_secs / secs:8.1f}x realtime  |  {per_frame_us:7.2f} µs/frame")


def bench_zero_copy() -> None:
    """Quantify the copy a naive binding would add, on a large decoded buffer.

    Our decode returns a NumPy array that OWNS the codec's output ``Vec``
    (``PyArray1::from_vec`` moves it - no second allocation). A binding that
    copied the PCM into a fresh array instead would pay one full copy of the
    output per decode. For a single 20 ms frame that copy is ~µs and lost in
    decode noise, so we measure it where it matters: a whole-stream decode whose
    output is large. ``arr.copy()`` is exactly the per-decode cost the move-out
    avoids.
    """
    from ruopus import decode_ogg_opus, encode_ogg_opus

    seconds = 10
    n = SR * seconds
    mono = (0.25 * np.sin(2 * np.pi * 330 * np.arange(n) / SR)).astype(np.float32)
    ogg = encode_ogg_opus(mono, 1, 96000)
    pcm, _ = decode_ogg_opus(ogg)  # large array, materialised by move-out

    t_copy = timed(lambda: pcm.copy(), iters=50)
    gbps = pcm.nbytes / t_copy / 1e9
    print(
        f"  zero-copy: a {seconds}s decode yields {pcm.nbytes / 1e6:.1f} MB; the move-out "
        f"avoids a {t_copy * 1e6:.0f} µs copy per decode ({gbps:.1f} GB/s)"
    )


def bench_threads(packets: list[bytes]) -> None:
    """Decode the corpus split across N Python threads; GIL is released in the
    decode kernel, so throughput should scale."""
    def decode_chunk(chunk):
        d = OpusDecoder(CHANNELS)
        for p in chunk:
            d.decode_packet(p)

    def run(n_threads):
        chunks = [packets[i::n_threads] for i in range(n_threads)]
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(decode_chunk, chunks))

    audio_secs = N_FRAMES * FRAME_SECONDS
    for n in (1, 2, 4):
        secs = timed(lambda n=n: run(n), iters=3)
        print(f"  threads={n}: {audio_secs / secs:8.1f}x realtime")


def main() -> None:
    print(f"ruopus {ruopus.version()}  |  {CHANNELS}ch {SR} Hz, "
          f"{N_FRAMES} x {FRAME / SR * 1000:.0f} ms frames\n")
    packets = make_corpus()
    avg_bytes = statistics.mean(len(p) for p in packets)
    print(f"corpus: {len(packets)} packets, {avg_bytes:.0f} bytes avg\n")
    bench_decode(packets)
    bench_encode()
    bench_zero_copy()
    print()
    bench_threads(packets)


if __name__ == "__main__":
    main()

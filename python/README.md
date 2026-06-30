<div align="center">

# ruopus

[![PyPI][pypi-img]][pypi] [![Docs][docs-img]][docs] [![License: MIT][license-img]][license]

## A fast Opus audio codec for Python, powered by Rust

</div>

`ruopus` encodes and decodes [Opus](https://opus-codec.org/) audio
([RFC 6716](https://www.rfc-editor.org/rfc/rfc6716)) with first-class NumPy
interop. The codec itself is a from-scratch Rust implementation with no C and
no FFI, but day to day you just call Python methods that take and return
NumPy arrays - the Rust is an implementation detail.

## Features

- **Encode and decode**: automatic mode selection (SILK / hybrid / CELT) with full control over bitrate, complexity, DTX, in-band FEC, bandwidth, and application profile
- **NumPy native**: PCM crosses the boundary as `(frames, channels)` float32 (or int16) arrays, moved out of Rust with no extra copy
- **No GIL stalls**: every encode/decode call releases the GIL
- **Ogg Opus files**: single-call `encode_ogg_opus` / `decode_ogg_opus` for complete `.opus` files
- **Packet loss handling**: concealment (`decode_lost`) and in-band FEC recovery (`decode_fec`)
- **Packet introspection**: parse the TOC byte and frame structure of raw Opus packets without decoding audio
- **Surround**: multistream decoding for 5.1/7.1 and other RFC 7845 layouts
- **Low-level access**: the `ruopus.lowlevel` submodule exposes the SILK and CELT codecs and the LPC analysis pipeline directly, for research and custom codec work
- **Fully typed**: ships with `.pyi` stubs and a `py.typed` marker

## Installation

```bash
pip install ruopus
```

Requires Python 3.9+ and NumPy 1.22+. Pre-built wheels are available for
Linux, macOS, and Windows (CPython 3.9-3.13) - no Rust toolchain needed.

## Quick Start

```python
import numpy as np
import ruopus

# Stereo encoder at 64 kbps, matching decoder
enc = ruopus.OpusEncoder(2, bitrate=64_000)
dec = ruopus.OpusDecoder(2)

# 20 ms stereo frame at 48 kHz -> 960 samples per channel
frame = np.zeros((960, 2), dtype=np.float32)

packet = enc.encode_auto(frame)        # bytes - encoded Opus packet
pcm    = dec.decode_packet(packet)     # (960, 2) float32 in [-1, 1]
```

Encoder input can be a 1-D interleaved array or a 2-D `(frames, channels)`
array; both are accepted without extra copies.

## Ogg Opus Files

Write and read complete `.opus` files in one call each:

```python
import numpy as np
import ruopus

sr = 48_000
t = np.linspace(0, 3, sr * 3, dtype=np.float32)
pcm = np.column_stack([
    np.sin(2 * np.pi * 440 * t),   # left:  A4
    np.sin(2 * np.pi * 880 * t),   # right: A5
])

ogg = ruopus.encode_ogg_opus(pcm, channels=2, bitrate=128_000)
with open("output.opus", "wb") as f:
    f.write(ogg)

with open("output.opus", "rb") as f:
    decoded_pcm, head = ruopus.decode_ogg_opus(f.read())

print(f"{head.channel_count}-ch, {decoded_pcm.shape[0] / sr:.2f}s")
```

## Packet Loss and FEC

```python
import ruopus

dec = ruopus.OpusDecoder(2)
FRAME = 960  # 20 ms at 48 kHz

for pkt in stream:
    if pkt is None:
        pcm = dec.decode_lost(frame_size=FRAME)        # concealment
    else:
        pcm = dec.decode_packet(pkt)

# Recover a lost packet from in-band FEC carried by its successor
recovered = dec.decode_fec(next_pkt, frame_size=FRAME)
```

## Choosing a Bitrate and Application

```python
# Phone-quality voice
enc_voice = ruopus.OpusEncoder(
    1,
    bitrate=8_000,
    application=ruopus.Application.Voip,
    signal=ruopus.Signal.Voice,
)

# High-quality music
enc_music = ruopus.OpusEncoder(
    2,
    bitrate=128_000,
    application=ruopus.Application.Audio,
    signal=ruopus.Signal.Music,
)

# Low-latency game voice (CELT only)
enc_ld = ruopus.OpusEncoder(
    1,
    bitrate=32_000,
    application=ruopus.Application.RestrictedLowDelay,
)
```

## Surround Decoding

```python
import ruopus

# 5.1 surround: 4 elementary streams, 2 stereo-coupled
dec = ruopus.MultistreamDecoder(
    streams=4,
    coupled=2,
    mapping=[0, 4, 1, 2, 3, 5],   # L R C LFE Ls Rs
)
pcm = dec.decode_packet(raw_packet)   # (frames, 6) float32
```

## Packet Inspection

```python
import ruopus

pkt = ruopus.Packet(raw_bytes)
print(f"mode={pkt.toc.mode}, bandwidth={pkt.toc.bandwidth}, frames={len(pkt)}")
```

## Low-Level API

For research or custom codec work, `ruopus.lowlevel` exposes the SILK and
CELT layers and the LPC analysis pipeline directly, below the Opus packet
format:

```python
import numpy as np
import ruopus.lowlevel as ll

enc = ll.CeltEncoder(channels=1, complexity=10, bitrate=64_000)
frame = np.zeros(960, dtype=np.float32)
body = enc.encode(frame)   # raw CELT frame body, no Opus framing
```

## Performance

Measured against libopus 1.6.1 (SIMD-enabled C), one core, pinned to a single
performance core. Figures are x realtime.

| Mode | ruopus | libopus |
|------|-------------|---------|
| Decode, SILK wideband 16 kb/s | 2095x | 1171x |
| Decode, CELT fullband 64 kb/s | 1389x | 1566x |
| Encode, matched complexity (all modes) | parity with libopus | |

See the [full benchmark breakdown](https://github.com/jmg049/ruopus#performance)
for the complete table and methodology.

## Documentation

The full guide and API reference are at
[jmg049.github.io/ruopus](https://jmg049.github.io/ruopus/), covering codec
modes, frame sizes, FEC/loss handling, multistream layouts, and the
low-level API in depth.

## License

MIT License.

## Links

- **GitHub**: <https://github.com/jmg049/ruopus>
- **Documentation**: <https://jmg049.github.io/ruopus/>
- **PyPI**: <https://pypi.org/project/ruopus/>

[pypi]: https://pypi.org/project/ruopus/
[pypi-img]: https://img.shields.io/pypi/v/ruopus?style=for-the-badge&color=009E73&label=PyPI

[docs]: https://jmg049.github.io/ruopus/
[docs-img]: https://img.shields.io/pypi/v/ruopus?style=for-the-badge&color=009E73&label=Docs

[license-img]: https://img.shields.io/pypi/l/ruopus?style=for-the-badge&label=license&labelColor=gray
[license]: https://github.com/jmg049/ruopus/blob/main/LICENSE

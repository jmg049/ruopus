# Python examples

Runnable examples for the `ruopus` Python bindings. Each is self-contained
(it synthesises its own audio - no external files needed) and runs against the
installed wheel:

```sh
python tools/build_python.py --release      # build the wheel
pip install target/wheels/ruopus-*.whl  # install it
python examples/python/01_encode_decode.py
```

| Example | Shows |
|---|---|
| `01_encode_decode.py` | Basic frame encode → decode round-trip, the range-coder oracle |
| `02_ogg_file.py` | Writing and reading a real `.opus` file (`encode_ogg_opus` / `decode_ogg_opus`) |
| `03_packet_loss.py` | Packet-loss concealment (`decode_lost`) and FEC (`decode_fec`) |
| `04_modes_and_config.py` | SILK / hybrid / CELT modes, bitrate / complexity / bandwidth / DTX |
| `05_multistream.py` | Surround decoding with `MultistreamDecoder` |
| `06_lowlevel_silk_lpc.py` | The `ruopus.lowlevel` SILK codec and LPC analysis |

PCM is always NumPy: `float32` for the Opus paths, `int16` for low-level SILK,
shaped `(frames, channels)`. Packets are `bytes`.

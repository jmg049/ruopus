ruopus Documentation
====================

**ruopus** is a pure-Rust implementation of the `Opus audio codec (RFC 6716)
<https://www.rfc-editor.org/rfc/rfc6716>`_ with first-class Python/NumPy interop.
PCM crosses the Rust/Python boundary as NumPy arrays shaped ``(frames, channels)``,
moved out of Rust without an extra copy. The GIL is released for all encode/decode
operations.

Features
--------

**Encoding:**

- **Three encode paths**: automatic mode selection (SILK/hybrid/CELT), forced CELT, forced SILK, or hybrid
- **Full encoder control**: complexity, bitrate (CBR/VBR), DTX, in-band FEC, bandwidth, signal hint, application profile
- **Ogg Opus output**: single-call convenience for writing complete Ogg Opus files

**Decoding:**

- **float32 and int16 output**: ``decode_packet`` / ``decode_packet_i16``
- **Packet loss concealment**: ``decode_lost`` extrapolates the last pitch period
- **In-band FEC recovery**: ``decode_fec`` reconstructs a lost frame from its successor
- **Ogg Opus input**: demuxing, pre-skip removal, end trimming, and output gain

**Packet Introspection:**

- **TOC parsing**: mode, bandwidth, frame size, channel count from the single header byte
- **Packet parsing**: frame extraction for standard and self-delimited framing (RFC 6716 §3 and Appendix B)

**Advanced:**

- **Multistream decoding**: surround layouts via ``MultistreamDecoder`` (RFC 7845)
- **Low-level layers**: direct access to SILK, LPC arithmetic, and CELT codecs below the packet layer

Quick Examples
--------------

Encode and Decode
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import ruopus

   enc = ruopus.OpusEncoder(2, bitrate=64_000)          # stereo, 64 kbps
   dec = ruopus.OpusDecoder(2)

   frame = np.zeros((960, 2), dtype=np.float32)         # 20 ms stereo at 48 kHz
   packet = enc.encode_auto(frame)                      # -> bytes
   pcm    = dec.decode_packet(packet)                   # -> (960, 2) float32

Ogg Opus Round-Trip
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import ruopus

   samples = np.random.randn(48_000 * 5).astype(np.float32)   # 5 s mono
   ogg_bytes = ruopus.encode_ogg_opus(samples, channels=1, bitrate=96_000)

   pcm, head = ruopus.decode_ogg_opus(ogg_bytes)
   print(f"Decoded {pcm.shape[0] / 48_000:.2f} s, {head.channel_count}-ch")

Packet Inspection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import ruopus

   pkt = ruopus.Packet(raw_bytes)
   print(f"mode={pkt.toc.mode}, bw={pkt.toc.bandwidth}, frames={len(pkt)}")

Installation
------------

.. code-block:: bash

   pip install ruopus

Build from source (requires a Rust toolchain and ``maturin``):

.. code-block:: bash

   git clone https://github.com/jmg049/ruopus
   cd ruopus
   pip install maturin
   maturin develop --release

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/installation
   guide/quickstart
   guide/codec_modes
   guide/frame_sizes
   guide/ogg
   guide/fec_and_loss
   guide/packet_inspection
   guide/multistream
   guide/lowlevel

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

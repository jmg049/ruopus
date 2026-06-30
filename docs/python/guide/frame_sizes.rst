Frame Sizes
===========

Opus operates on fixed-duration frames. The frame duration determines latency,
overhead, and which modes are available. This guide explains the trade-offs and
shows which frame sizes are valid for each encode method.

Valid Frame Sizes
-----------------

Opus supports six frame durations:

.. list-table::
   :header-rows: 1
   :widths: 15 25 20 40

   * - Duration
     - Samples at 48 kHz
     - ``FrameSize`` enum
     - Available modes
   * - 2.5 ms
     - 120
     - ``FrameSize.Ms2_5``
     - CELT-only
   * - 5 ms
     - 240
     - ``FrameSize.Ms5``
     - CELT-only
   * - 10 ms
     - 480
     - ``FrameSize.Ms10``
     - SILK, hybrid, CELT
   * - 20 ms
     - 960
     - ``FrameSize.Ms20``
     - SILK, hybrid, CELT
   * - 40 ms
     - 1920
     - ``FrameSize.Ms40``
     - SILK-only
   * - 60 ms
     - 2880
     - ``FrameSize.Ms60``
     - SILK-only

The sample counts above are *per channel*. A stereo 20 ms frame has
960 samples per channel, so the 2-D input array has shape ``(960, 2)``.

Choosing a Frame Size
---------------------

**20 ms** is the standard choice for most applications. It gives a good
balance of latency, packet overhead, and codec efficiency, and it is the only
frame size that works with all three modes (SILK, hybrid, CELT).

**10 ms** is the minimum latency frame that still supports SILK and hybrid. Use
it when 20 ms end-to-end latency is too much (e.g. a real-time call with tight
jitter buffer requirements).

**2.5 ms / 5 ms** are for CELT-only applications (e.g.
``Application.RestrictedLowDelay``) that need the absolute minimum algorithmic
delay.

**40 ms / 60 ms** are SILK-only and reduce packetisation overhead at the cost of
latency. Useful for store-and-forward voice (e.g. voicemail) or very
bandwidth-constrained links.

Per-Method Valid Frame Sizes
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Valid durations
   * - :meth:`~ruopus.OpusEncoder.encode_auto`
     - 2.5, 5, 10, 20, 40, 60 ms
   * - :meth:`~ruopus.OpusEncoder.encode`
     - 2.5, 5, 10, 20 ms (CELT-only)
   * - :meth:`~ruopus.OpusEncoder.encode_silk`
     - 10, 20, 40, 60 ms (SILK-only)
   * - :meth:`~ruopus.OpusEncoder.encode_hybrid`
     - 10, 20 ms (hybrid only)

Passing an invalid frame size raises :exc:`~ruopus.EncodeError`.

Samples vs Duration
-------------------

Since Opus always works at 48 kHz internally, compute sample counts from
durations like this:

.. code-block:: python

   import ruopus

   for fs in ruopus.FrameSize:
       print(
           f"{fs.name}: {fs.tenth_ms / 10:.1f} ms "
           f"= {fs.samples_per_channel_48k} samples/ch"
       )

Output::

   Ms2_5: 2.5 ms = 120 samples/ch
   Ms5:   5.0 ms = 240 samples/ch
   Ms10: 10.0 ms = 480 samples/ch
   Ms20: 20.0 ms = 960 samples/ch
   Ms40: 40.0 ms = 1920 samples/ch
   Ms60: 60.0 ms = 2880 samples/ch

Constructing frames of the right size:

.. code-block:: python

   import numpy as np
   import ruopus

   CHANNELS = 2
   FRAME_SIZE = ruopus.FrameSize.Ms20   # 960 samples/ch

   n = FRAME_SIZE.samples_per_channel_48k
   frame = np.zeros((n, CHANNELS), dtype=np.float32)

Encoder Lookahead
-----------------

The encoder adds a small amount of algorithmic delay (``pre_skip``) so the
MDCT windows can overlap properly. Read it from the encoder:

.. code-block:: python

   enc = ruopus.OpusEncoder(2)
   print(f"Lookahead: {enc.lookahead} samples at 48 kHz")
   # Typically 120 for fullband CELT (the MDCT overlap)

When the decoded output is aligned with the input for round-trip testing,
skip the first ``enc.lookahead`` output samples.

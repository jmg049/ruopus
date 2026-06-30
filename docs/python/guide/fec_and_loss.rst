Packet Loss Handling
====================

Opus has two mechanisms for coping with lost packets in real-time streams:
**packet loss concealment (PLC)** via :meth:`~ruopus.OpusDecoder.decode_lost`,
and **in-band forward error correction (FEC)** via
:meth:`~ruopus.OpusDecoder.decode_fec`. Both keep the decoder state consistent
so the stream can recover cleanly after loss.

Packet Loss Concealment
-----------------------

When a packet is lost, call :meth:`~ruopus.OpusDecoder.decode_lost` instead of
:meth:`~ruopus.OpusDecoder.decode_packet`. You must tell the decoder how many
samples to conceal; this should match the expected frame size of the lost
packet.

.. code-block:: python

   import ruopus

   dec = ruopus.OpusDecoder(2)

   for pkt in stream:
       if pkt is None:
           # Conceal one lost 20 ms frame
           concealed = dec.decode_lost(frame_size=960)
       else:
           pcm = dec.decode_packet(pkt)

CELT-mode concealment extrapolates the last pitch period, producing a short
fade-out. Frames following a SILK or hybrid packet fade to silence (full SILK
PLC is not yet ported). The ``final_range`` of a concealed decode is 0.

In-Band FEC (LBRR)
------------------

When in-band FEC is enabled on the encoder, SILK-mode packets carry a
low-bitrate redundant (LBRR) copy of the *previous* frame's audio alongside
the current frame. If a packet is lost, its content can be recovered from the
*next* received packet using :meth:`~ruopus.OpusDecoder.decode_fec`:

Enable FEC on the Encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import ruopus

   enc = ruopus.OpusEncoder(
       1,
       bitrate=24_000,
       application=ruopus.Application.Voip,
       signal=ruopus.Signal.Voice,
       inband_fec=True,
       packet_loss_perc=10,    # hint to the encoder: expect 10% loss
   )

``packet_loss_perc`` biases the encoder toward loss-robust coding (more
redundancy, lower raw bitrate efficiency). Values are clamped to 0-100.

Recover a Lost Packet Using FEC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dec = ruopus.OpusDecoder(1)
   FRAME = 960   # 20 ms at 48 kHz

   prev_pkt = None

   for pkt in stream:
       if pkt is None:
           if prev_pkt is not None:
               # next_pkt not yet available; use concealment as fallback
               recovered = dec.decode_lost(FRAME)
           # ... wait for next packet
       else:
           # If the previous slot was empty and we have a successor,
           # try FEC recovery first:
           if prev_pkt is None and next_pkt is not None:
               recovered = dec.decode_fec(next_pkt, frame_size=FRAME)
           pcm = dec.decode_packet(pkt)
       prev_pkt = pkt

The decoder automatically falls back to plain concealment when the packet
carries no usable FEC data (CELT-only modes, or when the requested frame size
is longer than the LBRR frame).

Realistic Loss Handling Loop
------------------------------

A production loss handler typically buffers packets and decides on decode vs
conceal vs FEC at playback time:

.. code-block:: python

   import ruopus
   import numpy as np

   enc = ruopus.OpusEncoder(
       1,
       bitrate=16_000,
       application=ruopus.Application.Voip,
       signal=ruopus.Signal.Voice,
       inband_fec=True,
       packet_loss_perc=5,
   )
   dec = ruopus.OpusDecoder(1)

   FRAME = 960   # 20 ms
   LOSS_RATE = 0.05   # simulate 5% loss

   # Simulate transmission
   rng = np.random.default_rng(42)
   pcm_in = np.random.randn(FRAME * 10, 1).astype(np.float32)
   packets = []
   for i in range(0, len(pcm_in), FRAME):
       pkt = enc.encode_auto(pcm_in[i:i+FRAME])
       packets.append(pkt if rng.random() > LOSS_RATE else None)

   # Decode with FEC / PLC
   output_blocks = []
   for i, pkt in enumerate(packets):
       if pkt is not None:
           output_blocks.append(dec.decode_packet(pkt))
       elif i + 1 < len(packets) and packets[i + 1] is not None:
           # FEC: use the next received packet to recover this lost one
           output_blocks.append(dec.decode_fec(packets[i + 1], FRAME))
       else:
           # No next packet yet: conceal
           output_blocks.append(dec.decode_lost(FRAME))

   recovered = np.concatenate(output_blocks, axis=0)
   print(f"Recovered {len(recovered) / 48_000:.2f} s of audio")

Tuning for Loss-Prone Networks
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Recommendation
   * - ``inband_fec=True``
     - Enable on the encoder when SILK modes will be used (Voip / Voice).
   * - ``packet_loss_perc``
     - Set to the *expected* loss rate. Higher values increase redundancy overhead but improve recovery quality.
   * - ``application``
     - ``Application.Voip`` or ``Application.Audio``; FEC data is only included in SILK/hybrid packets, and ``RestrictedLowDelay`` never uses SILK so FEC has no effect.
   * - Frame size
     - 20 ms is the most common SILK frame size and gives FEC the most redundancy budget.

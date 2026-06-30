Packet Inspection
=================

ruopus exposes two types for reading the structure of raw Opus packet bytes
without decoding audio: :class:`~ruopus.Toc` (the single header byte) and
:class:`~ruopus.Packet` (the full parsed packet including its frames).

The TOC Byte
------------

Every Opus packet starts with a table-of-contents (TOC) byte that encodes the
mode, bandwidth, frame size, channel count, and framing code in 8 bits
(RFC 6716 §3.1).

:class:`~ruopus.Toc` wraps that byte:

.. code-block:: python

   import ruopus

   # From a raw packet
   raw = b"\x78\x01\x02\x03\x04"
   toc = ruopus.Toc(raw[0])
   print(toc)

   # Or from the parsed packet:
   pkt = ruopus.Packet(raw)
   toc = pkt.toc

Reading TOC Fields
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   toc = ruopus.Toc(raw[0])

   print(f"Mode:       {toc.mode}")       # Mode.CeltOnly / Mode.Hybrid / Mode.SilkOnly
   print(f"Bandwidth:  {toc.bandwidth}")  # Bandwidth.FullBand, etc.
   print(f"Frame size: {toc.frame_size}") # FrameSize.Ms20, etc.
   print(f"Channels:   {toc.channels}")   # 1 or 2
   print(f"Stereo:     {toc.stereo}")     # bool
   print(f"Config:     {toc.config}")     # raw 5-bit config number (0-31)
   print(f"FCC:        {toc.frame_count_code}")  # frame framing code (0-3)
   print(f"Raw byte:   {toc.byte:#04x}")

Building a TOC from Parts
~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~ruopus.Toc.from_parts` constructs a TOC from its three bitfields, useful
for building test vectors or inspecting the config space:

.. code-block:: python

   toc = ruopus.Toc.from_parts(config=16, stereo=True, frame_count_code=0)
   print(toc.mode, toc.bandwidth, toc.frame_size)

Hashing and Equality
~~~~~~~~~~~~~~~~~~~~

:class:`~ruopus.Toc` is hashable and supports equality, so you can count unique
configurations in a stream:

.. code-block:: python

   from collections import Counter

   counts = Counter(ruopus.Toc(pkt[0]).config for pkt in raw_packets)
   print(counts.most_common(5))

Parsing a Full Packet
---------------------

:class:`~ruopus.Packet` parses the complete RFC 6716 §3 structure: the TOC byte
plus the compressed audio frames. It validates all R1-R7 bitstream constraints.

.. code-block:: python

   pkt = ruopus.Packet(raw_bytes)

   print(f"Frames:   {len(pkt)}")          # number of encoded frames
   print(f"Duration: {pkt.duration}")      # datetime.timedelta
   print(f"Padding:  {pkt.padding} bytes") # code-3 padding only

   for i, frame in enumerate(pkt.frames):
       print(f"  frame[{i}]: {len(frame)} bytes")

   # Sequence protocol: pkt[0] == pkt.frames[0]
   first_frame = pkt[0]     # bytes
   last_frame  = pkt[-1]    # negative indexing supported

Empty frames (zero bytes) indicate a DTX (discontinuous transmission) packet;
the decoder treats them as a short silence.

Self-Delimited Parsing
~~~~~~~~~~~~~~~~~~~~~~

Multistream payloads use *self-delimited* framing (RFC 6716 Appendix B) so
that each elementary stream's extent is unambiguous. Parse with
:meth:`~ruopus.Packet.parse_self_delimited`:

.. code-block:: python

   payload = b"..."   # concatenated self-delimited streams
   offset = 0
   while offset < len(payload):
       pkt, consumed = ruopus.Packet.parse_self_delimited(payload[offset:])
       print(f"Stream at {offset}: {len(pkt)} frames, {consumed} bytes")
       offset += consumed

Inspecting a Live Stream
------------------------

A typical monitoring loop that logs packet stats without touching audio:

.. code-block:: python

   import ruopus
   from collections import defaultdict

   mode_counts  = defaultdict(int)
   bw_counts    = defaultdict(int)
   total_bytes  = 0

   for raw_pkt in incoming_packets():
       pkt = ruopus.Packet(raw_pkt)
       mode_counts[pkt.toc.mode]      += 1
       bw_counts[pkt.toc.bandwidth]   += 1
       total_bytes                    += len(raw_pkt)

   print("Mode distribution:",  dict(mode_counts))
   print("Bandwidth distribution:", dict(bw_counts))
   print(f"Total compressed data: {total_bytes / 1024:.1f} KB")

Bit-Exactness Check
-------------------

After a round-trip encode → decode, compare the encoder's
:attr:`~ruopus.OpusEncoder.final_range` with the decoder's
:attr:`~ruopus.OpusDecoder.final_range`. A conformant implementation produces
identical values:

.. code-block:: python

   import numpy as np
   import ruopus

   enc = ruopus.OpusEncoder(1)
   dec = ruopus.OpusDecoder(1)

   frame  = np.zeros(960, dtype=np.float32)
   packet = enc.encode_auto(frame)
   _pcm   = dec.decode_packet(packet)

   assert enc.final_range == dec.final_range, "Bit-exactness violation!"
   print(f"Range coder state: {enc.final_range:#010x}")

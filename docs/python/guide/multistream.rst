Multistream Decoding
====================

Surround-sound Opus files use *multistream* encoding (RFC 7845 §5.1.1): several
independent elementary Opus streams are packed into each Ogg page, with a
channel-mapping table that routes decoded channels to the output layout.
:class:`~ruopus.MultistreamDecoder` handles this demultiplexing.

Understanding Multistream Layout
---------------------------------

A multistream packet contains ``streams`` elementary streams. The first
``coupled`` streams are decoded as stereo; the remaining ``streams - coupled``
are decoded as mono. The ``mapping`` list (one entry per output channel) routes
each decoded channel to the output array by index; the sentinel value ``255``
inserts a silent channel.

For example, a 5.1 surround layout has 4 streams (3 stereo-coupled, 1 mono):

.. code-block:: text

   streams = 4, coupled = 2
   mapping = [0, 1, 2, 3, 4, 5]
              L  R  C LFE  Ls  Rs

Creating a MultistreamDecoder
------------------------------

The ``mapping`` list must satisfy:

- Every entry is either ``255`` (silent) or less than ``streams + coupled``.
- At least one output channel.

.. code-block:: python

   import ruopus

   # 5.1 surround: 4 streams, 2 stereo-coupled
   dec = ruopus.MultistreamDecoder(
       streams=4,
       coupled=2,
       mapping=[0, 1, 2, 3, 4, 5],
       sample_rate=48_000,
   )

   print(f"Output channels: {dec.channels}")    # 6
   print(f"Sample rate:     {dec.sample_rate}") # 48000

Decoding Multistream Packets
-----------------------------

Each packet contains all elementary streams in self-delimited framing
(the last stream is standard-framed). Pass the raw bytes directly:

.. code-block:: python

   for raw_pkt in ogg_pages:
       pcm = dec.decode_packet(raw_pkt)   # (frames, 6) float32

Output is ``(frames, channels)`` float32, with channels in the output order
defined by ``mapping``.

Stereo-Only (Family 0)
-----------------------

For mono and stereo Ogg Opus files (channel mapping family 0), the simpler
:class:`~ruopus.OpusDecoder` is sufficient, since the Ogg container does not use
multistream framing for these layouts.

.. code-block:: python

   # mono or stereo: plain OpusDecoder, or decode_ogg_opus()
   pcm, head = ruopus.decode_ogg_opus(data)   # handles family-0 automatically

Surround Layouts from RFC 7845
-------------------------------

RFC 7845 Appendix A defines the mapping tables for the standard Vorbis-
compatible surround layouts. Common configurations:

.. code-block:: python

   # Stereo (family 0, use OpusDecoder instead)
   #   streams=1, coupled=1, mapping=[0, 1]

   # 3.0 (L C R)
   dec_30 = ruopus.MultistreamDecoder(
       streams=2, coupled=1, mapping=[0, 2, 1]
   )

   # 5.1 (L R C LFE Ls Rs, Vorbis channel order)
   dec_51 = ruopus.MultistreamDecoder(
       streams=4, coupled=2, mapping=[0, 4, 1, 2, 3, 5]
   )

   # 7.1 (L R C LFE Ls Rs Lss Rss)
   dec_71 = ruopus.MultistreamDecoder(
       streams=5, coupled=3, mapping=[0, 6, 1, 2, 3, 4, 5, 7]
   )

Silent Channels
---------------

A mapping entry of ``255`` produces a zero-filled output channel, useful for
layouts where one output slot has no source stream:

.. code-block:: python

   # 4.0 surround with a silent LFE slot
   dec = ruopus.MultistreamDecoder(
       streams=2,
       coupled=2,
       mapping=[0, 1, 255, 2, 3],   # channel 2 = silence
   )

Output Sample Rates
-------------------

Like :class:`~ruopus.OpusDecoder`, the output rate can be reduced for
low-rate playback pipelines:

.. code-block:: python

   dec = ruopus.MultistreamDecoder(
       streams=2, coupled=1, mapping=[0, 2, 1],
       sample_rate=16_000,   # 16 kHz output
   )

Valid rates: 48000, 24000, 16000, 12000, 8000.

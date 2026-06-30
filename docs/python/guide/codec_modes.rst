Codec Modes
===========

Opus contains two independent audio codecs that it combines to cover every use
case from 6 kbps narrowband phone calls to 512 kbps transparent music archival.
Understanding the three modes (SILK-only, CELT-only, and hybrid) helps you
pick the right encoder settings.

The Three Modes
---------------

SILK-Only
~~~~~~~~~

SILK is a wideband speech codec (LP-based, similar in heritage to iSAC).
It operates at internal rates of 8, 12, or 16 kHz and is optimised for speech
at low-to-medium bitrates.

- **Best for**: speech, voice calls, VoIP at 6-32 kbps
- **Frame sizes**: 10, 20, 40, or 60 ms
- **Bandwidth**: narrowband (NB), medium-band (MB), or wideband (WB)

.. code-block:: python

   enc = ruopus.OpusEncoder(
       1,
       bitrate=16_000,
       application=ruopus.Application.Voip,
       signal=ruopus.Signal.Voice,
   )
   packet = enc.encode_silk(frame)    # force SILK-only for this frame

CELT-Only
~~~~~~~~~

CELT is an MDCT-based codec (similar in heritage to Vorbis). It operates at
the full 48 kHz sample rate and excels at music, wideband audio, and
low-latency applications.

- **Best for**: music, game audio, low-delay voice at 32+ kbps
- **Frame sizes**: 2.5, 5, 10, or 20 ms
- **Bandwidth**: narrowband through fullband

.. code-block:: python

   enc = ruopus.OpusEncoder(
       2,
       bitrate=96_000,
       application=ruopus.Application.Audio,
       signal=ruopus.Signal.Music,
   )
   packet = enc.encode(frame)    # encode() always uses CELT

Hybrid
~~~~~~

Hybrid coding uses SILK for the low band (below 8 kHz) and CELT for the high
band above it. This gives SILK's speech quality at low bitrates while
preserving fullband presence.

- **Best for**: super-wideband or fullband speech at 16-32 kbps
- **Frame sizes**: 10 or 20 ms
- **Bandwidth**: super-wideband (SWB) or fullband (FB)

.. code-block:: python

   packet = enc.encode_hybrid(frame)    # force hybrid for this frame

Automatic Mode Selection
------------------------

:meth:`~ruopus.OpusEncoder.encode_auto` is the right choice for most
applications. The encoder runs a signal classifier on every frame and selects
SILK, hybrid, or CELT based on:

- The :attr:`~ruopus.OpusEncoder.signal` hint (``Auto``, ``Voice``, ``Music``)
- The :attr:`~ruopus.OpusEncoder.application` profile
- The target bitrate and forced/maximum bandwidth
- Cross-frame hysteresis (to avoid rapid mode flipping)

.. code-block:: python

   enc = ruopus.OpusEncoder(2, bitrate=32_000)
   # Auto mode: classifier picks SILK/hybrid/CELT per frame
   packet = enc.encode_auto(frame)

Checking the Mode of a Decoded Packet
--------------------------------------

After encoding, inspect the TOC byte to confirm which mode was used:

.. code-block:: python

   pkt = ruopus.Packet(packet)
   print(pkt.toc.mode)        # Mode.SilkOnly / Mode.Hybrid / Mode.CeltOnly
   print(pkt.toc.bandwidth)   # Bandwidth.FullBand, etc.
   print(pkt.toc.frame_size)  # FrameSize.Ms20, etc.

See :doc:`packet_inspection` for the full packet introspection API.

Application Profiles
--------------------

The :class:`~ruopus.Application` enum nudges the automatic classifier:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Effect
   * - ``Application.Audio``
     - Balanced default. Suitable for music and general audio.
   * - ``Application.Voip``
     - Biases toward SILK / hybrid; adds noise suppression tuned for speech.
   * - ``Application.RestrictedLowDelay``
     - Forces CELT-only on every frame. Lowest latency (encoder lookahead = 2.5 ms). Never uses SILK.

Signal Hints
------------

The :class:`~ruopus.Signal` enum biases the speech/music classifier:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Value
     - Effect
   * - ``Signal.Auto``
     - Run the built-in classifier (default).
   * - ``Signal.Voice``
     - Treat the source as speech; push toward SILK / hybrid modes.
   * - ``Signal.Music``
     - Treat the source as music; push toward CELT.

Bandwidth Control
-----------------

:class:`~ruopus.Bandwidth` controls the coded audio bandwidth:

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Enum value
     - Audio BW
     - Sample rate
     - Notes
   * - ``Bandwidth.NarrowBand``
     - 4 kHz
     - 8 kHz
     - SILK-only
   * - ``Bandwidth.MediumBand``
     - 6 kHz
     - 12 kHz
     - SILK-only
   * - ``Bandwidth.WideBand``
     - 8 kHz
     - 16 kHz
     - SILK or hybrid
   * - ``Bandwidth.SuperWideBand``
     - 12 kHz
     - 24 kHz
     - Hybrid or CELT
   * - ``Bandwidth.FullBand``
     - 20 kHz
     - 48 kHz
     - CELT or hybrid

Set :attr:`~ruopus.OpusEncoder.max_bandwidth` to cap automatic selection, or
:attr:`~ruopus.OpusEncoder.bandwidth` to pin it. Call
:meth:`~ruopus.OpusEncoder.set_auto_bandwidth` to restore automatic selection.

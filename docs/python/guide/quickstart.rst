Quickstart
==========

This guide covers the most common workflows: encode, decode, Ogg Opus files,
and a basic pipeline for streaming audio.

Basic Encode → Decode
---------------------

The simplest path: create an encoder and a decoder for the same channel count
and bounce a frame through both.

.. code-block:: python

   import numpy as np
   import ruopus

   # Stereo encoder at 64 kbps (default: fullband, complexity 10, auto mode)
   enc = ruopus.OpusEncoder(2, bitrate=64_000)
   dec = ruopus.OpusDecoder(2)

   # 20 ms stereo frame at 48 kHz  →  960 samples per channel
   frame = np.zeros((960, 2), dtype=np.float32)

   packet = enc.encode_auto(frame)        # bytes, encoded Opus packet
   pcm    = dec.decode_packet(packet)     # (960, 2) float32 in [-1, 1]

   print(f"Packet size: {len(packet)} bytes")
   print(f"Decoded shape: {pcm.shape}, dtype: {pcm.dtype}")

Encoder input can be a 1-D interleaved array or a 2-D ``(frames, channels)``
array. Both are accepted without extra copies.

Choosing a Bitrate
------------------

Bitrate controls the quality/size tradeoff:

.. code-block:: python

   # Phone-quality voice (8 kbps SILK narrowband)
   enc_voice = ruopus.OpusEncoder(
       1,
       bitrate=8_000,
       application=ruopus.Application.Voip,
       signal=ruopus.Signal.Voice,
   )

   # High-quality music (128 kbps CELT fullband)
   enc_music = ruopus.OpusEncoder(
       2,
       bitrate=128_000,
       application=ruopus.Application.Audio,
       signal=ruopus.Signal.Music,
   )

   # Low-latency game voice (32 kbps, restricted low delay = CELT only)
   enc_ld = ruopus.OpusEncoder(
       1,
       bitrate=32_000,
       application=ruopus.Application.RestrictedLowDelay,
   )

See :doc:`codec_modes` for an explanation of what these settings select under
the hood.

Mono Audio
----------

For single-channel audio pass ``channels=1``. The PCM array is
``(frames, 1)`` after decoding.

.. code-block:: python

   enc = ruopus.OpusEncoder(1, bitrate=32_000)
   dec = ruopus.OpusDecoder(1)

   frame = np.zeros((960, 1), dtype=np.float32)   # or shape (960,)
   packet = enc.encode_auto(frame)
   pcm    = dec.decode_packet(packet)              # (960, 1) float32

Decoding to int16
-----------------

:meth:`~ruopus.OpusDecoder.decode_packet_i16` returns ``int16`` PCM scaled to
``[-32768, 32767]``, identical to the ``opus_demo`` reference output.

.. code-block:: python

   dec = ruopus.OpusDecoder(2)
   pcm_i16 = dec.decode_packet_i16(packet)    # (frames, 2) int16

Streaming Pipeline
------------------

The encoder and decoder are both stateful: maintain them across frames so
inter-frame state (mode hysteresis, overlap buffers, concealment history) is
continuous:

.. code-block:: python

   import numpy as np
   import ruopus

   ENC = ruopus.OpusEncoder(2, bitrate=64_000)
   DEC = ruopus.OpusDecoder(2)
   FRAME = 960   # 20 ms at 48 kHz

   def process_stream(pcm_44k: np.ndarray) -> np.ndarray:
       """Resample, encode, and decode a stereo 44.1 kHz recording."""
       # ruopus always works at 48 kHz; resample externally if needed
       pcm = pcm_44k.astype(np.float32)
       packets, decoded = [], []
       for start in range(0, len(pcm) - FRAME, FRAME):
           chunk = pcm[start : start + FRAME]
           pkt   = ENC.encode_auto(chunk)
           out   = DEC.decode_packet(pkt)
           packets.append(pkt)
           decoded.append(out)
       return np.concatenate(decoded, axis=0)

Adjusting Encoder Settings at Runtime
--------------------------------------

All encoder properties are writable; changes take effect on the next frame:

.. code-block:: python

   enc = ruopus.OpusEncoder(2, bitrate=64_000)

   # Reduce bitrate mid-stream (e.g. bandwidth dropped)
   enc.bitrate = 24_000

   # Enable DTX to suppress silent frames
   enc.dtx = True

   # Bump complexity for a batch-encode job
   enc.complexity = 10

Next Steps
----------

- Understand :doc:`codec_modes` and when to use each encode method
- Learn about valid :doc:`frame_sizes`
- Encode and decode complete files with :doc:`ogg`
- Handle packet loss with :doc:`fec_and_loss`
- Inspect packet structure with :doc:`packet_inspection`

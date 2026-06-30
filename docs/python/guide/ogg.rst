Ogg Opus Files
==============

Ogg Opus (`RFC 7845 <https://www.rfc-editor.org/rfc/rfc7845>`_) is the
standard container for Opus audio. ruopus provides two convenience functions,
:func:`~ruopus.encode_ogg_opus` and :func:`~ruopus.decode_ogg_opus`, that
handle the full container round-trip in a single call, plus the
:class:`~ruopus.OpusHead` identification-header object returned by the decoder.

Encoding to Ogg Opus
--------------------

:func:`~ruopus.encode_ogg_opus` takes raw 48 kHz float32 PCM and returns a
complete Ogg Opus file as ``bytes``:

.. code-block:: python

   import numpy as np
   import ruopus

   # Generate 3 seconds of stereo audio
   sr = 48_000
   t  = np.linspace(0, 3, sr * 3, dtype=np.float32)
   pcm = np.column_stack([
       np.sin(2 * np.pi * 440 * t),   # left:  A4
       np.sin(2 * np.pi * 880 * t),   # right: A5
   ])   # shape (144000, 2)

   ogg = ruopus.encode_ogg_opus(pcm, channels=2, bitrate=128_000)

   with open("output.opus", "wb") as f:
       f.write(ogg)

   print(f"Ogg Opus file: {len(ogg) / 1024:.1f} KB")

Input can also be a 1-D interleaved array (samples interleaved L, R, L, R, …):

.. code-block:: python

   flat_pcm = pcm.ravel()    # shape (288000,)
   ogg = ruopus.encode_ogg_opus(flat_pcm, channels=2, bitrate=128_000)

Decoding from Ogg Opus
-----------------------

:func:`~ruopus.decode_ogg_opus` handles the complete demuxing pipeline:

- Reads the ``OpusHead`` identification header
- Applies the :attr:`~ruopus.OpusHead.pre_skip` (discards leading silence)
- Trims trailing granule padding
- Applies the :attr:`~ruopus.OpusHead.output_gain_q8` to the decoded PCM

.. code-block:: python

   with open("input.opus", "rb") as f:
       data = f.read()

   pcm, head = ruopus.decode_ogg_opus(data)

   print(f"Channels:    {head.channel_count}")
   print(f"Input rate:  {head.input_sample_rate} Hz")
   print(f"Pre-skip:    {head.pre_skip} samples")
   print(f"Output gain: {head.output_gain_q8} (Q7.8 dB)")
   print(f"Decoded PCM: {pcm.shape}, dtype={pcm.dtype}")

The output is always ``float32`` at 48 kHz, shaped ``(frames, channels)``.

Reading the OpusHead
--------------------

The :class:`~ruopus.OpusHead` object exposes every field of the RFC 7845
identification header:

.. code-block:: python

   pcm, head = ruopus.decode_ogg_opus(data)

   print(repr(head))
   # OpusHead(version=1, channel_count=2, pre_skip=312,
   #          input_sample_rate=44100, mapping_family=0)

   # Round-trip the header back to bytes (e.g. for remuxing):
   header_bytes = head.to_bytes()

Channel-mapping family 0 (mono and stereo) is the only family currently
supported by :func:`~ruopus.decode_ogg_opus`. For surround layouts, see
:doc:`multistream`.

Round-Trip Fidelity
-------------------

A full encode → decode round-trip introduces the encoder lookahead (typically
120 samples) as latency and the codec's lossy compression artifacts:

.. code-block:: python

   import numpy as np
   import ruopus

   original = np.random.randn(48_000).astype(np.float32)
   ogg      = ruopus.encode_ogg_opus(original, channels=1, bitrate=256_000)
   decoded, head = ruopus.decode_ogg_opus(ogg)

   # The decoded length may differ slightly due to pre_skip and frame alignment
   min_len = min(len(original), len(decoded))
   rms_err = np.sqrt(np.mean((original[:min_len] - decoded[:min_len, 0]) ** 2))
   print(f"RMS error vs original: {rms_err:.6f}")

Saving Decoded Audio
--------------------

The decoded float32 PCM can be saved with `audio_samples
<https://pypi.org/project/audio-samples/>`_:

.. code-block:: python

   import numpy as np
   import audio_samples as aus

   pcm, head = ruopus.decode_ogg_opus(data)
   samples = aus.AudioSamples.new_mono(pcm, 48_000)
   aus.io.save("decoded.wav", samples, as_type=np.float32)

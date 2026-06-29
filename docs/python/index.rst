ruopus
===========

A pure-Rust implementation of the Opus audio codec (`RFC 6716
<https://www.rfc-editor.org/rfc/rfc6716>`_) with first-class NumPy interop.
PCM crosses the Rust/Python boundary as NumPy ``float32`` / ``int16`` arrays
shaped ``(frames, channels)``, moved (not copied) out of Rust; packets are
``bytes``.

.. code-block:: python

   import numpy as np
   import ruopus

   enc = ruopus.OpusEncoder(2, bitrate=64000)
   dec = ruopus.OpusDecoder(2)

   frame = np.zeros((960, 2), dtype=np.float32)   # 20 ms stereo at 48 kHz
   packet = enc.encode(frame)                      # -> bytes
   pcm = dec.decode_packet(packet)                 # -> (960, 2) float32

.. toctree::
   :maxdepth: 2

   api

Indices
=======

* :ref:`genindex`

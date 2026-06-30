Low-Level API
=============

The ``ruopus.lowlevel`` submodule exposes the SILK, LPC, and CELT codec layers
directly, below the Opus packet layer. These are advanced building blocks for
research, educational, or custom codec work. **For ordinary encode/decode,
prefer** :class:`~ruopus.OpusEncoder` **and** :class:`~ruopus.OpusDecoder`.

.. code-block:: python

   import ruopus.lowlevel as ll

   # Inspect what's available
   print(dir(ll))

CELT Encoder and Decoder
------------------------

:class:`~ruopus.lowlevel.CeltEncoder` and :class:`~ruopus.lowlevel.CeltDecoder`
operate on raw CELT frame bodies, with no Opus TOC byte and no packet framing. They
are useful for research into the MDCT layer independently.

.. code-block:: python

   import numpy as np
   import ruopus.lowlevel as ll

   enc = ll.CeltEncoder(channels=1, complexity=10, bitrate=64_000)
   dec = ll.CeltDecoder(channels=1)

   frame = np.zeros(960, dtype=np.float32)
   body  = enc.encode(frame)           # bytes, raw CELT frame body
   print(f"Encoded: {len(body)} bytes")

   # CELT decoder also provides packet-loss concealment:
   concealed = dec.decode_lost(frame_size=960)   # (960, 1) float32

.. warning::

   The ``CeltDecoder`` does not expose a full ``decode`` method because CELT
   decoding requires the range-coder state from the encoder, which is not
   exported. Decoding real CELT packets should go through
   :class:`~ruopus.OpusDecoder`.

SILK Encoder and Decoder
-------------------------

:class:`~ruopus.lowlevel.SilkEncoder` operates at SILK's *internal* sample
rates (8, 12, or 16 kHz), **not** the 48 kHz Opus rate. Input PCM is
``int16``. :class:`~ruopus.lowlevel.SilkDecoder` decodes SILK bitstream bytes
back to ``int16`` PCM.

.. code-block:: python

   import numpy as np
   import ruopus.lowlevel as ll

   # 16 kHz, 20 ms frames (4 subframes)
   enc = ll.SilkEncoder(fs_khz=16, nb_subfr=4, bitrate=25_000, complexity=10)

   # One 20 ms frame at 16 kHz = 16000 * 0.02 = 320 samples
   frame_i16 = np.zeros(320, dtype=np.int16)
   payload = enc.encode(frame_i16)       # bytes, SILK bitstream

   print(f"SILK payload: {len(payload)} bytes")

SILK Stereo Encoding
~~~~~~~~~~~~~~~~~~~~

:class:`~ruopus.lowlevel.SilkStereoEncoder` extends the mono encoder with
mid-side (M/S) stereo coding:

.. code-block:: python

   enc_ms = ll.SilkStereoEncoder(fs_khz=16, nb_subfr=4, bitrate=40_000)
   stereo_frame = np.zeros(320 * 2, dtype=np.int16)   # interleaved L/R
   payload = enc_ms.encode(stereo_frame)

LPC Arithmetic
--------------

The ``ruopus.lowlevel`` module exposes the complete LPC analysis pipeline used
inside SILK. These functions are useful for DSP research, feature extraction,
and understanding speech codec internals.

.. code-block:: python

   import numpy as np
   import ruopus.lowlevel as ll

   signal = np.random.randn(320).astype(np.float32)

Autocorrelation
~~~~~~~~~~~~~~~

.. code-block:: python

   # Biased autocorrelation up to lag `order`
   ac = ll.compute_autocorrelation(signal, order=16)
   print(f"r[0] = {ac[0]:.4f}")   # signal energy

Levinson-Durbin
~~~~~~~~~~~~~~~

Solve the Yule-Walker equations to get LPC coefficients:

.. code-block:: python

   # From a pre-computed autocorrelation vector
   coeffs = ll.levinson_durbin(ac)
   print(coeffs)   # LpcCoefficients(order=16)
   print(coeffs.coeffs)   # list of float32

Full LPC Analysis
~~~~~~~~~~~~~~~~~

:func:`~ruopus.lowlevel.lpc_analysis` combines autocorrelation and
Levinson-Durbin in one call:

.. code-block:: python

   coeffs = ll.lpc_analysis(signal, order=16)
   print(f"LPC order: {coeffs.order}")

Residual and Synthesis
~~~~~~~~~~~~~~~~~~~~~~

Compute the LPC prediction residual (analysis filter) and reconstruct PCM
(synthesis filter):

.. code-block:: python

   # Stateless: history is passed explicitly
   history = np.zeros(coeffs.order, dtype=np.float32)
   residual = ll.lpc_residual(signal, coeffs, history)

   reconstructed = ll.lpc_synthesis(residual, coeffs, history)

   # Stateful wrappers that maintain a rolling history buffer:
   residual2 = ll.lpc_residual_stateful(signal, coeffs)
   recon2    = ll.lpc_synthesis_stateful(residual2, coeffs)

Long-Term Prediction (LTP)
~~~~~~~~~~~~~~~~~~~~~~~~~~

LTP models the pitch periodicity on top of the LPC short-term model:

.. code-block:: python

   pitch_lag = ll.estimate_pitch(signal, fs_khz=16)
   print(f"Estimated pitch lag: {pitch_lag} samples")

   ltp_res = ll.ltp_residual(signal, pitch_lag)
   ltp_syn = ll.ltp_synthesis(ltp_res, pitch_lag)

DecControl
----------

:class:`~ruopus.lowlevel.DecControl` carries the SILK decoder's control
parameters (sample rate, frame length, and similar). It is passed to / returned
from :class:`~ruopus.lowlevel.SilkDecoder`:

.. code-block:: python

   ctrl = ll.DecControl(fs_khz=16, nb_subfr=4)
   dec  = ll.SilkDecoder()
   pcm  = dec.decode(silk_payload, ctrl)   # (frame_samples,) int16

Installation
============

Requirements
------------

- Python 3.9 or higher
- NumPy 1.24 or higher

Install from PyPI
-----------------

.. code-block:: bash

   pip install ruopus

Build from Source
-----------------

Building from source requires a stable Rust toolchain (``rustup`` recommended)
and ``maturin``.

.. code-block:: bash

   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   # Clone and build
   git clone https://github.com/jmg049/ruopus
   cd ruopus
   pip install maturin
   maturin develop --release    # editable install, release-optimised

For a wheel you can distribute:

.. code-block:: bash

   maturin build --release

The ``--release`` flag is important: a debug build is 10-50× slower than a
release build.

Verifying the Install
---------------------

.. code-block:: python

   import ruopus
   print(ruopus.version())   # e.g. "0.1.0"

   enc = ruopus.OpusEncoder(1)
   dec = ruopus.OpusDecoder(1)
   import numpy as np
   frame = np.zeros(960, dtype=np.float32)
   packet = enc.encode_auto(frame)
   pcm = dec.decode_packet(packet)
   print(f"Encoded {len(packet)} bytes, decoded {pcm.shape}")

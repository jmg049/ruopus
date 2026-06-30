"""ruopus - a pure-Rust implementation of the Opus audio codec (RFC 6716).

PCM crosses the Rust/Python boundary as NumPy arrays shaped
``(frames, channels)``, moved out of Rust without an extra copy. The GIL is
released for every encode/decode call.

Example:
    >>> import numpy as np
    >>> import ruopus
    >>>
    >>> enc = ruopus.OpusEncoder(2, bitrate=64_000)  # stereo, 64 kbps
    >>> dec = ruopus.OpusDecoder(2)
    >>>
    >>> frame = np.zeros((960, 2), dtype=np.float32)  # 20 ms stereo at 48 kHz
    >>> packet = enc.encode_auto(frame)
    >>> pcm = dec.decode_packet(packet)
"""

from __future__ import annotations

import sys as _sys

from ._ruopus import *  # noqa: F401,F403
from ._ruopus import lowlevel

# Register the compiled `lowlevel` submodule under the public package's name
# so `import ruopus.lowlevel` works, not just attribute access on `ruopus`.
_sys.modules[f"{__name__}.lowlevel"] = lowlevel
del _sys

__all__ = [
    "Application",
    "Bandwidth",
    "EncodeError",
    "FrameSize",
    "Mode",
    "MultistreamDecoder",
    "OggError",
    "OpusDecoder",
    "OpusEncoder",
    "OpusError",
    "OpusHead",
    "Packet",
    "PacketError",
    "Signal",
    "Toc",
    "decode_ogg_opus",
    "encode_ogg_opus",
    "lowlevel",
    "version",
]

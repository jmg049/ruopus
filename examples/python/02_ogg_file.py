"""Write and read a real Ogg Opus (.opus) file.

`encode_ogg_opus` produces a complete standard Ogg Opus stream (RFC 7845) that
plays in ffmpeg / VLC; `decode_ogg_opus` reads one back, applying pre-skip,
end-trimming and the header's output gain, and returns the PCM plus the parsed
`OpusHead`.

Run: python examples/python/02_ogg_file.py
"""

import tempfile
from pathlib import Path

import numpy as np

from opus_rs import decode_ogg_opus, encode_ogg_opus

SR = 48000


def main() -> None:
    # 2 seconds of mono 220 Hz tone.
    n = SR * 2
    pcm = (0.25 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)).astype(np.float32)

    ogg_bytes = encode_ogg_opus(pcm, channels=1, bitrate=96000)

    path = Path(tempfile.gettempdir()) / "opus_rs_example.opus"
    path.write_bytes(ogg_bytes)
    print(f"wrote {path} ({len(ogg_bytes)} bytes)")

    decoded, head = decode_ogg_opus(path.read_bytes())
    print(f"decoded {decoded.shape} {decoded.dtype}")
    print(
        "OpusHead: "
        f"version={head.version}, channels={head.channel_count}, "
        f"pre_skip={head.pre_skip}, input_sample_rate={head.input_sample_rate}, "
        f"mapping_family={head.mapping_family}"
    )

    # Duration is preserved to within a frame after pre-skip / end trimming.
    print(f"input {n / SR:.2f}s -> decoded {decoded.shape[0] / SR:.2f}s")
    path.unlink()


if __name__ == "__main__":
    main()

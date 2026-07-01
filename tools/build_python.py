#!/usr/bin/env python3
"""Build the `ruopus` wheel with corrected type stubs.

PyO3's `experimental-inspect` pass emits complete `.pyi` stubs (types *and*
docstrings) straight from the Rust source, but two mechanical fixes are needed
that the introspector cannot do itself:

1. NumPy return/argument annotations are declared via
   ``#[pyo3(signature = ... -> "numpy.typing.NDArray[...]")]`` (rust-numpy maps
   `PyArray` to `_typeshed.Incomplete`), but the introspector adds no
   ``import numpy`` for those custom type strings.
2. The exception hierarchy is built with `create_exception!`, which the
   introspector cannot model: it omits the classes and tags the top module
   incomplete with a catch-all ``def __getattr__``. We strip that and inject the
   real, typed hierarchy.

This script runs ``maturin build --generate-stubs`` (passing through any extra
args, e.g. ``--release``) and patches every ``.pyi`` inside the resulting wheel,
updating the wheel ``RECORD`` so it stays valid. Because the stubs are
regenerated from the binary on every build, they cannot drift from the compiled
module.

    python tools/build_python.py                # debug wheel
    python tools/build_python.py --release       # release wheel
"""

from __future__ import annotations

import base64
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_IMPORTS = ("import numpy", "import numpy.typing")
GETATTR_TAINT = "def __getattr__(name: str) -> Incomplete: ..."
EXCEPTION_STUB = '''
class OpusError(Exception):
    """Base class for every error raised by ``ruopus``.

    Catch this to handle any codec failure regardless of its specific kind.
    """

class PacketError(OpusError):
    """Raised when an Opus packet is malformed (RFC 6716 §3.4 rules R1-R7).

    Corresponds to the Rust ``ruopus::PacketError``.
    """

class EncodeError(OpusError):
    """Raised when encoding fails: an unsupported frame size, or an output
    budget outside ``3..=1275`` bytes that the packet could not be made to fit.

    Corresponds to the Rust ``ruopus::EncodeError``.
    """

class OggError(OpusError):
    """Raised when decoding an Ogg Opus stream fails (bad container, bad packet,
    or an unsupported channel-mapping family).

    Corresponds to the Rust ``ruopus::OggDecodeError``.
    """
'''


def patch_pyi(name: str, text: str) -> str:
    lines = text.splitlines()
    if any("numpy." in ln for ln in lines):
        first_import = next(
            (i for i, ln in enumerate(lines) if ln.startswith(("import ", "from "))),
            len(lines),
        )
        missing = [imp for imp in REQUIRED_IMPORTS if imp not in lines]
        lines[first_import:first_import] = missing
    if name.endswith("/__init__.pyi") or name.endswith("__init__.pyi"):
        lines = [ln for ln in lines if ln.strip() != GETATTR_TAINT]
        return "\n".join(lines).rstrip() + "\n" + EXCEPTION_STUB
    return "\n".join(lines).rstrip() + "\n"


def record_line(arcname: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{arcname},sha256={digest},{len(data)}"


def patch_wheel(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        members = {info.filename: zf.read(info.filename) for info in zf.infolist()}

    patched = []
    for name in list(members):
        if name.endswith(".pyi"):
            new = patch_pyi(name, members[name].decode()).encode()
            if new != members[name]:
                members[name] = new
                patched.append(name)

    # Rebuild RECORD with fresh hashes for the files we changed.
    record_name = next(n for n in members if n.endswith(".dist-info/RECORD"))
    new_record = []
    for line in members[record_name].decode().splitlines():
        arcname = line.split(",", 1)[0]
        if arcname in patched:
            new_record.append(record_line(arcname, members[arcname]))
        else:
            new_record.append(line)
    members[record_name] = ("\n".join(new_record) + "\n").encode()

    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return patched


def resolve_maturin() -> str:
    # pip installs the `maturin` console script next to the interpreter that
    # ran pip (or into its Scripts/ dir on Windows). That directory isn't
    # necessarily on PATH -- e.g. CI invokes manylinux's per-version
    # interpreters by full path without exporting their bin dir -- so look
    # there first instead of trusting a bare "maturin" lookup.
    bin_dir = Path(sys.executable).parent
    for candidate in (bin_dir / "maturin", bin_dir / "maturin.exe", bin_dir / "Scripts" / "maturin.exe"):
        if candidate.exists():
            return str(candidate)
    found = shutil.which("maturin")
    if found:
        return found
    raise SystemExit("maturin executable not found near the running interpreter or on PATH")


def newest_wheel() -> Path:
    wheels = sorted((ROOT / "target" / "wheels").glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SystemExit("no wheel produced")
    return wheels[-1]


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--patch-only":
        # Patch already-built wheels in place (e.g. after maturin-action in CI).
        for path in args[1:]:
            patched = patch_wheel(Path(path))
            print(f"patched stubs in {Path(path).name}: {', '.join(patched) or '(none)'}")
        return 0

    # Build with maturin (passing through extra args, e.g. --release -i pythonX.Y)
    # then patch the produced wheel's stubs.
    subprocess.run([resolve_maturin(), "build", "--generate-stubs", *args], cwd=ROOT, check=True)
    wheel = newest_wheel()
    patched = patch_wheel(wheel)
    print(f"patched stubs in {wheel.name}: {', '.join(patched) or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

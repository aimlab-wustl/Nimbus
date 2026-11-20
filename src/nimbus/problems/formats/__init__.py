"""Readers/writers for the external and project input formats.

Public external formats (their own conventions are preserved; nothing here invents a standard):

- :mod:`.alist`        MacKay alist parity-check matrices (incl. the AFF3CT zero-padded variant).
- :mod:`.opb`          a documented subset of the PB24 OPB format and a recognizer for the scoped
                       constraint families (chains, disjoint covers, unit packing, cardinality).

Project schemas (versioned, explicitly *not* external standards):

- :mod:`.spinpoly`     ``nimbus.spinpoly/1``: spin-product objective + constraint blocks.
- :mod:`.matching_tensor` sparse third-order matching tensors (``.npz``).

Every reader raises :class:`nimbus.core.errors.FormatError` whose message names the file and,
where meaningful, the 1-based line (``<file>:<line>: <problem>``). Out-of-family inputs that are
syntactically valid raise :class:`nimbus.core.errors.UnsupportedProblem`.

Compressed files (``.gz``, ``.xz``) are decompressed transparently; the recorded ``sha256`` is
always the checksum of the bytes *as stored on disk* (i.e. of the compressed file), which is what
the benchmark manifest pins.
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import lzma
import os
from dataclasses import dataclass
from pathlib import Path

from nimbus.core.errors import FormatError

__all__ = ["SourceText", "read_source", "source_from_text", "format_error", "sha256_bytes", "sha256_file"]

#: refuse to read larger inputs unless the caller raises the limit explicitly (1 GiB uncompressed)
DEFAULT_MAX_BYTES = 1 << 30


@dataclass(frozen=True)
class SourceText:
    """Decoded text of one input plus its provenance."""

    name: str  # display name used in error messages
    text: str
    sha256: str | None  # checksum of the raw bytes as stored (None for in-memory strings)
    path: Path | None
    compression: str | None = None  # "gzip", "xz", "bz2" or None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def format_error(name: str, line: int | None, message: str) -> FormatError:
    """Build a FormatError ``<name>:<line>: <message>`` (line omitted when None)."""
    where = f"{name}:{line}" if line is not None else f"{name}"
    return FormatError(f"{where}: {message}")


def _decompress(raw: bytes, name: str, suffix: str, max_bytes: int) -> tuple[bytes, str | None]:
    openers = {".gz": ("gzip", gzip.decompress), ".xz": ("xz", lzma.decompress), ".bz2": ("bz2", bz2.decompress)}
    if suffix not in openers:
        return raw, None
    kind, fn = openers[suffix]
    try:
        data = fn(raw)
    except (OSError, EOFError, lzma.LZMAError, ValueError) as exc:
        raise format_error(name, None, f"cannot decompress {kind} data: {exc}") from exc
    if len(data) > max_bytes:
        raise format_error(name, None, f"decompressed size {len(data)} exceeds the limit {max_bytes} bytes")
    return data, kind


def read_source(path: str | os.PathLike, *, max_bytes: int = DEFAULT_MAX_BYTES, encoding: str = "utf-8") -> SourceText:
    """Read a (possibly compressed) text file; FormatError on missing file, size or decoding problems."""
    p = Path(path)
    name = str(p)
    try:
        size = p.stat().st_size
    except FileNotFoundError as exc:
        raise format_error(name, None, "file not found") from exc
    if size > max_bytes:
        raise format_error(name, None, f"file size {size} exceeds the limit {max_bytes} bytes")
    raw = p.read_bytes()
    data, kind = _decompress(raw, name, p.suffix.lower(), max_bytes)
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise format_error(name, None, f"not valid {encoding} text: {exc}") from exc
    return SourceText(name=name, text=text, sha256=sha256_bytes(raw), path=p, compression=kind)


def source_from_text(text: str, name: str = "<string>") -> SourceText:
    """Wrap an in-memory string (no checksum, no path)."""
    return SourceText(name=name, text=text, sha256=None, path=None, compression=None)


# Convenience re-exports, resolved lazily (PEP 562) so importing one format does not import all.
_LAZY = {
    "ParityCheckMatrix": "alist", "read_alist": "alist", "write_alist": "alist",
    "OPBModel": "opb", "read_opb": "opb", "recognize_constraints": "opb",
    "SpinPolyDocument": "spinpoly", "read_spinpoly": "spinpoly", "write_spinpoly": "spinpoly",
}
__all__ += list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f"{__name__}.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

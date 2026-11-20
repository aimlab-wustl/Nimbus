"""Explicit, cached, checksum-verified download of public benchmark files.

The committed manifest ``benchmarks/manifest.yaml`` (schema ``nimbus.benchmark_manifest/1``) lists
every file with its URL (GitHub raw URLs must embed the pinned 40-hex commit of their source),
destination path under the data root, size, sha256, format, expected parser metadata, target and
provenance/license notes. Nothing is fetched implicitly: tests and runs read the data root and
skip/fail when files are missing.

``fetch(set_name)``:

1. selects the entries of a set (e.g. ``matching``) or explicit ids;
2. reuses a cached file when its sha256 matches (manifest value, else the lock file);
3. otherwise downloads to a temporary file with a per-request timeout, an overall time limit and
   a size limit (the manifest ``bytes`` when known, else ``max_bytes``), verifies size and sha256,
   and renames it into place atomically;
4. when neither the manifest nor the lock has a checksum yet, records the observed sha256/size in
   ``<data_root>/benchmarks.lock.json`` (trust on first use) and verifies against it afterwards;
   the maintainer then copies the value into the manifest;
5. parses each file with its format reader and compares the manifest's ``expected`` metadata
   (``check=True``), and regenerates ``<data_root>/SOURCES.md`` (provenance + checksums).

Command line: ``python -m nimbus.experiments.fetch --set matching [--data-root DIR] [--force]``;
``--list`` shows entries, ``--verify`` checks the local copy offline.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from nimbus.core.errors import ConfigError, NimbusError
from nimbus.experiments import paths

__all__ = [
    "MANIFEST_SCHEMA",
    "LOCK_NAME",
    "FetchError",
    "ChecksumMismatch",
    "ManifestEntry",
    "Manifest",
    "FetchResult",
    "load_manifest",
    "fetch",
    "verify",
    "check_expected",
    "entry_path",
    "write_sources_md",
    "main",
]

MANIFEST_SCHEMA = "nimbus.benchmark_manifest/1"
LOCK_SCHEMA = "nimbus.benchmark_lock/1"
LOCK_NAME = "benchmarks.lock.json"
SOURCES_NAME = "SOURCES.md"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
USER_AGENT = "nimbus-fetch/1 (research benchmark fetcher)"

_ENTRY_KEYS = {"id", "sets", "source", "url", "path", "format", "sha256", "bytes", "expected", "target",
               "reference", "license", "notes"}
_REQUIRED_ENTRY_KEYS = {"id", "sets", "source", "url", "path", "format"}
_SOURCE_KEYS = {"name", "homepage", "default_branch", "revision", "revision_date", "license", "citation", "notes",
                "documentation"}
_FORMATS = {"alist", "text"}
_GITHUB_RAW = re.compile(r"^https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class FetchError(NimbusError):
    """Download, size, time or metadata failure while fetching benchmark data."""


class ChecksumMismatch(FetchError):
    """Downloaded or cached bytes do not match the pinned checksum."""


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    sets: tuple
    source: str
    url: str
    path: str  # POSIX path relative to the data root
    format: str
    sha256: str | None = None
    bytes: int | None = None
    expected: dict = field(default_factory=dict)
    target: dict | None = None
    reference: str | None = None
    license: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Manifest:
    schema: str
    sources: dict
    entries: tuple
    path: Path | None = None

    @property
    def sets(self) -> list[str]:
        return sorted({s for e in self.entries for s in e.sets})

    def entry(self, entry_id: str) -> ManifestEntry:
        for e in self.entries:
            if e.id == entry_id:
                return e
        raise ConfigError(f"no manifest entry {entry_id!r}")

    def select(self, set_name: str | None = None, ids: Iterable[str] | None = None) -> list[ManifestEntry]:
        if set_name is None and ids is None:
            raise ConfigError("select a set name or explicit ids")
        out = []
        if set_name is not None:
            if set_name not in self.sets:
                raise ConfigError(f"unknown benchmark set {set_name!r}; known sets: {self.sets}")
            out = [e for e in self.entries if set_name in e.sets]
        if ids is not None:
            for i in ids:
                e = self.entry(i)
                if e not in out:
                    out.append(e)
        return out


@dataclass(frozen=True)
class FetchResult:
    id: str
    path: Path
    status: str  # "cached" | "downloaded" | "replaced" | "recorded" (first checksum recorded in the lock)
    sha256: str
    bytes: int
    metadata_mismatches: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Manifest loading and validation
# ---------------------------------------------------------------------------------------------
def _err(path, msg) -> ConfigError:
    return ConfigError(f"{path}: {msg}")


def load_manifest(path: str | os.PathLike | None = None) -> Manifest:
    """Load and validate the benchmark manifest (default: ``<repo>/benchmarks/manifest.yaml``)."""
    import yaml

    p = Path(path) if path is not None else paths.manifest_path()
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise _err(p, "manifest not found") from None
    except yaml.YAMLError as exc:
        raise _err(p, f"invalid YAML: {exc}") from None
    if not isinstance(doc, dict):
        raise _err(p, "expected a mapping")
    unknown = set(doc) - {"schema", "sources", "entries", "description"}
    if unknown:
        raise _err(p, f"unknown top-level field(s) {sorted(unknown)}")
    if doc.get("schema") != MANIFEST_SCHEMA:
        raise _err(p, f"schema must be {MANIFEST_SCHEMA!r}")
    sources = doc.get("sources") or {}
    if not isinstance(sources, dict):
        raise _err(p, "sources must be a mapping")
    for key, src in sources.items():
        if not isinstance(src, dict):
            raise _err(p, f"sources.{key} must be a mapping")
        bad = set(src) - _SOURCE_KEYS
        if bad:
            raise _err(p, f"sources.{key}: unknown field(s) {sorted(bad)}")
        if "revision" in src and src["revision"] is not None and not _HEX40.match(str(src["revision"])):
            raise _err(p, f"sources.{key}.revision must be a full 40-hex commit SHA")
    entries = []
    ids, dests = set(), set()
    for k, raw in enumerate(doc.get("entries") or []):
        where = f"entries[{k}]"
        if not isinstance(raw, dict):
            raise _err(p, f"{where} must be a mapping")
        bad = set(raw) - _ENTRY_KEYS
        if bad:
            raise _err(p, f"{where} ({raw.get('id')}): unknown field(s) {sorted(bad)}")
        missing = _REQUIRED_ENTRY_KEYS - set(raw)
        if missing:
            raise _err(p, f"{where} ({raw.get('id')}): missing field(s) {sorted(missing)}")
        eid = str(raw["id"])
        if eid in ids:
            raise _err(p, f"duplicate entry id {eid!r}")
        ids.add(eid)
        sets = raw["sets"]
        if isinstance(sets, str) or not isinstance(sets, list) or not sets or not all(isinstance(s, str) for s in sets):
            raise _err(p, f"{eid}: sets must be a non-empty list of names")
        if raw["source"] not in sources:
            raise _err(p, f"{eid}: unknown source {raw['source']!r}")
        rel = PurePosixPath(str(raw["path"]))
        if rel.is_absolute() or ".." in rel.parts or "\\" in str(raw["path"]) or not rel.parts:
            raise _err(p, f"{eid}: path must be a relative POSIX path inside the data root")
        if str(rel) in dests or str(rel) in (LOCK_NAME, SOURCES_NAME):
            raise _err(p, f"{eid}: duplicate or reserved destination path {rel}")
        dests.add(str(rel))
        if raw["format"] not in _FORMATS:
            raise _err(p, f"{eid}: unknown format {raw['format']!r} (known: {sorted(_FORMATS)})")
        url = str(raw["url"])
        if not re.match(r"^(https?|file)://", url):
            raise _err(p, f"{eid}: url must be http(s):// or file://")
        m = _GITHUB_RAW.match(url)
        if m:
            ref = m.group(3)
            rev = sources[raw["source"]].get("revision")
            if not _HEX40.match(ref):
                raise _err(p, f"{eid}: GitHub raw URL must pin a 40-hex commit, found ref {ref!r}")
            if rev and ref != rev:
                raise _err(p, f"{eid}: URL commit {ref} differs from sources.{raw['source']}.revision {rev}")
        sha = raw.get("sha256")
        if sha is not None and not _HEX64.match(str(sha)):
            raise _err(p, f"{eid}: sha256 must be 64 lowercase hex characters or null")
        nbytes = raw.get("bytes")
        if nbytes is not None and (isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0):
            raise _err(p, f"{eid}: bytes must be a non-negative integer or null")
        expected = raw.get("expected") or {}
        if not isinstance(expected, dict):
            raise _err(p, f"{eid}: expected must be a mapping")
        target = raw.get("target")
        if target is not None and not isinstance(target, dict):
            raise _err(p, f"{eid}: target must be a mapping or null")
        entries.append(ManifestEntry(id=eid, sets=tuple(sets), source=str(raw["source"]), url=url, path=str(rel),
                                     format=raw["format"], sha256=sha, bytes=nbytes, expected=dict(expected),
                                     target=target, reference=raw.get("reference"), license=raw.get("license"),
                                     notes=raw.get("notes")))
    for e in entries:
        if e.reference is not None and e.reference not in ids:
            raise _err(p, f"{e.id}: reference {e.reference!r} is not an entry id")
        inst = e.expected.get("instance")
        if inst is not None and inst not in ids:
            raise _err(p, f"{e.id}: expected.instance {inst!r} is not an entry id")
    return Manifest(schema=MANIFEST_SCHEMA, sources=dict(sources), entries=tuple(entries), path=p)


# ---------------------------------------------------------------------------------------------
# Lock file
# ---------------------------------------------------------------------------------------------
def _lock_path(root: Path) -> Path:
    return root / LOCK_NAME


def _read_lock(root: Path) -> dict:
    p = _lock_path(root)
    if not p.exists():
        return {"schema": LOCK_SCHEMA, "entries": {}}
    try:
        lock = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FetchError(f"{p}: corrupt lock file: {exc}") from None
    if lock.get("schema") != LOCK_SCHEMA or not isinstance(lock.get("entries"), dict):
        raise FetchError(f"{p}: not a {LOCK_SCHEMA} file")
    return lock


def _write_json_atomic(p: Path, data: dict) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def _sha256_file(p: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


# ---------------------------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------------------------
def _download(url: str, dest_tmp: Path, *, limit: int, timeout_s: float, deadline: float) -> tuple[str, int]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    h = hashlib.sha256()
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(dest_tmp, "wb") as out:
            status = getattr(resp, "status", 200)
            if status is not None and status >= 400:
                raise FetchError(f"{url}: HTTP {status}")
            length = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
            if length is not None and length.isdigit() and int(length) > limit:
                raise FetchError(f"{url}: announced size {length} exceeds the size limit of {limit} bytes")
            while True:
                if time.monotonic() > deadline:
                    raise FetchError(f"{url}: time limit exceeded")
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                n += len(chunk)
                if n > limit:
                    raise FetchError(f"{url}: download exceeds the size limit of {limit} bytes")
                h.update(chunk)
                out.write(chunk)
    except FetchError:
        dest_tmp.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, OSError, ValueError) as exc:
        dest_tmp.unlink(missing_ok=True)
        raise FetchError(f"{url}: download failed: {exc}") from exc
    return h.hexdigest(), n


def entry_path(entry_id: str, data_root: str | os.PathLike | None = None, manifest: Manifest | None = None) -> Path:
    """Local path of a manifest entry under the data root (whether or not it was fetched)."""
    man = manifest or load_manifest()
    root = Path(data_root) if data_root is not None else paths.data_root()
    return root / man.entry(entry_id).path


def fetch(set_name: str | None = None, data_root: str | os.PathLike | None = None, force: bool = False, *,
          ids: Sequence[str] | None = None, manifest: Manifest | str | os.PathLike | None = None,
          timeout_s: float = 60.0, time_limit_s: float = 900.0, max_bytes: int = DEFAULT_MAX_BYTES,
          check: bool = True, write_sources: bool = True, log=None) -> list[FetchResult]:
    """Fetch a benchmark set (or explicit ids) into the data root; see the module docstring."""
    man = manifest if isinstance(manifest, Manifest) else load_manifest(manifest)
    root = Path(data_root) if data_root is not None else paths.data_root()
    root.mkdir(parents=True, exist_ok=True)
    entries = man.select(set_name, ids)
    lock = _read_lock(root)
    deadline = time.monotonic() + time_limit_s
    say = log or (lambda msg: None)
    results = []
    lock_dirty = False
    for e in entries:
        dest = root / e.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        locked = lock["entries"].get(e.id)
        if locked is not None and locked.get("url") != e.url:
            locked = None  # URL changed: the old lock record does not apply
        expected_sha = e.sha256 or (locked or {}).get("sha256")
        expected_bytes = e.bytes if e.bytes is not None else (locked or {}).get("bytes")
        status = None
        if dest.exists() and not force:
            sha, n = _sha256_file(dest)
            if expected_sha is None and expected_bytes is not None and n != expected_bytes:
                say(f"{e.id}: cached file has {n} bytes, expected {expected_bytes}; re-downloading")
            elif expected_sha is None:
                status = "recorded"
            elif sha == expected_sha:
                status = "cached"
            else:
                say(f"{e.id}: cached file has sha256 {sha[:12]}..., expected {expected_sha[:12]}...; re-downloading")
        if status is None:
            if expected_bytes is not None:
                limit = int(expected_bytes)
            else:
                limit = max_bytes
            tmp = dest.with_name(dest.name + ".part")
            sha, n = _download(e.url, tmp, limit=limit, timeout_s=timeout_s, deadline=deadline)
            if expected_bytes is not None and n != expected_bytes:
                tmp.unlink(missing_ok=True)
                raise FetchError(f"{e.id}: downloaded {n} bytes, manifest/lock says {expected_bytes}")
            if expected_sha is not None and sha != expected_sha:
                tmp.unlink(missing_ok=True)
                raise ChecksumMismatch(f"{e.id}: sha256 {sha} does not match the pinned {expected_sha} ({e.url})")
            existed = dest.exists()
            os.replace(tmp, dest)
            status = "recorded" if expected_sha is None else ("replaced" if existed else "downloaded")
            say(f"{e.id}: {status} ({n} bytes)")
        if expected_sha is None or locked is None or locked.get("sha256") != sha:
            lock["entries"][e.id] = {"sha256": sha, "bytes": n, "url": e.url, "path": e.path,
                                     "recorded_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
            lock_dirty = True
        mism = {}
        if check:
            mism = check_expected(e, dest, manifest=man, data_root=root)
            if mism:
                raise FetchError(f"{e.id}: parsed metadata differs from the manifest: {mism}")
        results.append(FetchResult(id=e.id, path=dest, status=status, sha256=sha, bytes=n, metadata_mismatches=mism))
    if lock_dirty:
        _write_json_atomic(_lock_path(root), lock)
    if write_sources:
        write_sources_md(root, man)
    return results


def verify(set_name: str | None = None, data_root: str | os.PathLike | None = None, *, ids: Sequence[str] | None = None,
           manifest: Manifest | str | os.PathLike | None = None, check: bool = True) -> dict:
    """Offline check of a set: ``{id: "ok" | "missing" | "no checksum" | "<problem>"}``."""
    man = manifest if isinstance(manifest, Manifest) else load_manifest(manifest)
    root = Path(data_root) if data_root is not None else paths.data_root()
    lock = _read_lock(root) if root.exists() else {"entries": {}}
    report = {}
    for e in man.select(set_name, ids):
        dest = root / e.path
        if not dest.exists():
            report[e.id] = "missing"
            continue
        sha, n = _sha256_file(dest)
        expected = e.sha256 or lock["entries"].get(e.id, {}).get("sha256")
        if expected is None:
            report[e.id] = "no checksum"
        elif sha != expected:
            report[e.id] = f"sha256 mismatch ({sha})"
        elif e.bytes is not None and n != e.bytes:
            report[e.id] = f"size {n} != {e.bytes}"
        else:
            mism = check_expected(e, dest, manifest=man, data_root=root) if check else {}
            report[e.id] = "ok" if not mism else f"metadata mismatch {mism}"
    return report


# ---------------------------------------------------------------------------------------------
# Expected-metadata checks through the real parsers
# ---------------------------------------------------------------------------------------------
def _cmp(expected: dict, actual: dict) -> dict:
    out = {}
    for k, v in expected.items():
        if k == "instance":
            continue
        if k not in actual:
            out[k] = (v, "<not computed>")
        elif actual[k] != v:
            out[k] = (v, actual[k])
    return out


def check_expected(entry: ManifestEntry, path: Path, *, manifest: Manifest | None = None,
                   data_root: Path | None = None) -> dict:
    """Parse ``path`` with the entry's format reader; return ``{key: (expected, actual)}`` mismatches."""
    exp = dict(entry.expected)
    if not exp:
        return {}
    fmt = entry.format
    act: dict = {}
    if fmt == "alist":
        from nimbus.problems.formats.alist import read_alist

        h = read_alist(path)
        act = {"n": h.n, "m": h.m, "nnz": h.nnz, "var_degree_max": int(h.var_degrees.max()),
               "check_degree_max": int(h.check_degrees.max()), "zero_padded": h.zero_padded}
        if "rank" in exp or "k" in exp:
            act["rank"] = h.rank()
            act["k"] = h.n - h.rank()
    elif fmt == "text":
        text = path.read_text(encoding="utf-8", errors="replace")
        act = {"contains": [s for s in exp.get("contains", []) if s in text]}
    return _cmp(exp, act)


# ---------------------------------------------------------------------------------------------
# SOURCES.md
# ---------------------------------------------------------------------------------------------
def write_sources_md(data_root: str | os.PathLike, manifest: Manifest | None = None) -> Path:
    """Write ``<data_root>/SOURCES.md`` listing provenance, licenses and checksums of local files."""
    man = manifest or load_manifest()
    root = Path(data_root)
    lock = _read_lock(root)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Benchmark data: sources and checksums",
        "",
        f"Written by `nimbus.experiments.fetch` on {now} from `benchmarks/manifest.yaml`.",
        "Files are stored exactly as downloaded (compressed files stay compressed); the sha256 is of the stored bytes.",
        "Do not edit files here by hand; re-run `python -m nimbus.experiments.fetch --set <set>` instead.",
        "",
    ]
    by_source: dict[str, list[ManifestEntry]] = {}
    for e in man.entries:
        if (root / e.path).exists():
            by_source.setdefault(e.source, []).append(e)
    for key, ents in by_source.items():
        s = man.sources.get(key, {})
        lines.append(f"## {s.get('name', key)}")
        lines.append("")
        for label, field_ in (("Homepage", "homepage"), ("Default branch", "default_branch"), ("Pinned revision", "revision"),
                              ("Revision date", "revision_date"), ("License", "license"), ("Citation", "citation"),
                              ("Documentation", "documentation"), ("Notes", "notes")):
            if s.get(field_):
                lines.append(f"- {label}: {' '.join(str(s[field_]).split())}")
        lines.append("")
        lines.append("| id | path | sets | bytes | sha256 | url |")
        lines.append("|---|---|---|---:|---|---|")
        for e in ents:
            sha, n = _sha256_file(root / e.path)
            pinned = e.sha256 or lock["entries"].get(e.id, {}).get("sha256")
            flag = "" if pinned == sha else " (UNVERIFIED)" if pinned is None else " (MISMATCH)"
            lines.append(f"| {e.id} | {e.path} | {', '.join(e.sets)} | {n} | `{sha}`{flag} | {e.url} |")
        lines.append("")
    p = root / SOURCES_NAME
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, p)
    return p


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nimbus.experiments.fetch", description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", dest="set_name", help="benchmark set to fetch (e.g. matching)")
    ap.add_argument("--id", dest="ids", action="append", help="explicit entry id (repeatable)")
    ap.add_argument("--data-root", help=f"destination (default ${paths.ENV_DATA_ROOT} or <repo>/data)")
    ap.add_argument("--manifest", help="manifest path (default benchmarks/manifest.yaml)")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--list", action="store_true", help="list entries and exit")
    ap.add_argument("--verify", action="store_true", help="offline check of the local copy")
    ap.add_argument("--no-check", action="store_true", help="skip parsing/expected-metadata checks")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    ap.add_argument("--time-limit", type=float, default=900.0, help="overall time limit in seconds")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="size limit for entries without 'bytes'")
    args = ap.parse_args(argv)
    try:
        man = load_manifest(args.manifest)
        if args.list:
            for e in (man.select(args.set_name, args.ids) if (args.set_name or args.ids) else man.entries):
                print(f"{e.id:55s} {','.join(e.sets):28s} {e.format:16s} {e.bytes if e.bytes is not None else '?':>8} {e.path}")
            return 0
        if not (args.set_name or args.ids):
            ap.error("give --set or --id (see --list)")
        if args.verify:
            rep = verify(args.set_name, args.data_root, ids=args.ids, manifest=man, check=not args.no_check)
            for k, v in rep.items():
                print(f"{k:55s} {v}")
            return 0 if all(v == "ok" for v in rep.values()) else 1
        res = fetch(args.set_name, args.data_root, args.force, ids=args.ids, manifest=man, timeout_s=args.timeout,
                    time_limit_s=args.time_limit, max_bytes=args.max_bytes, check=not args.no_check, log=print)
        for r in res:
            print(f"{r.id:55s} {r.status:10s} {r.bytes:>9d} {r.sha256}")
        return 0
    except (NimbusError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

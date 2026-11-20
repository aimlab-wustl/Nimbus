"""Benchmark manifest, fetcher and data-root paths (offline: file:// URLs in temporary manifests)."""

import hashlib
import json
import re
import textwrap
from pathlib import Path

import pytest

from nimbus.core.errors import ConfigError
from nimbus.experiments import fetch as F
from nimbus.experiments import paths

pytestmark = pytest.mark.quick

FIX = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]


# ---- paths ------------------------------------------------------------------------------------
def test_default_roots_are_inside_the_repo(monkeypatch):
    monkeypatch.delenv(paths.ENV_DATA_ROOT, raising=False)
    monkeypatch.delenv(paths.ENV_RUNS_ROOT, raising=False)
    monkeypatch.delenv(paths.ENV_REPO_ROOT, raising=False)
    assert paths.repo_root() == REPO
    assert paths.data_root() == REPO / "data" and paths.runs_root() == REPO / "local_runs"
    assert paths.manifest_path() == REPO / "benchmarks" / "manifest.yaml"


def test_env_overrides_with_variable_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_ROOT_TEST", str(tmp_path))
    monkeypatch.setenv(paths.ENV_DATA_ROOT, "$STORAGE_ROOT_TEST/data")
    monkeypatch.setenv(paths.ENV_RUNS_ROOT, "${STORAGE_ROOT_TEST}/runs")
    assert paths.data_root() == tmp_path / "data" and paths.runs_root() == tmp_path / "runs"
    monkeypatch.setenv(paths.ENV_DATA_ROOT, "$UNSET_VARIABLE_FOR_NIMBUS_TEST/data")
    with pytest.raises(ConfigError, match="unset environment variable"):
        paths.data_root()


def test_no_hardcoded_storage_roots():
    src = REPO / "src" / "nimbus"
    for p in [src / "experiments" / "paths.py", src / "experiments" / "fetch.py", *(src / "problems" / "formats").glob("*.py")]:
        text = p.read_text(encoding="utf-8")
        assert "/storage" not in text, p


# ---- committed manifest -------------------------------------------------------------------------
def test_committed_manifest_is_valid_and_pinned():
    man = F.load_manifest()
    assert man.sets == ["matching"]
    for e in man.entries:
        assert e.sha256 and re.fullmatch(r"[0-9a-f]{64}", e.sha256), e.id
        assert isinstance(e.bytes, int) and e.bytes > 0, e.id
        if "raw.githubusercontent.com" in e.url:
            assert man.sources[e.source]["revision"] in e.url, e.id
        assert e.expected, e.id
    willow = man.entry("willow/objectclass.zip")
    assert willow.id in {e.id for e in man.select("matching")}
    assert willow.source == "willow_objectclass" and willow.format == "text"


# ---- temporary manifests -----------------------------------------------------------------------
HEADER = """\
schema: nimbus.benchmark_manifest/1
sources:
  local: {name: local test files, license: test}
  gh: {name: github test, revision: 0123456789abcdef0123456789abcdef01234567}
entries:
"""


def _entry(eid, url, path, fmt="text", sha=None, nbytes=None, expected=None, sets=("t",), source="local"):
    exp = json.dumps(expected or {})
    return textwrap.indent(textwrap.dedent(f"""\
        - id: {eid}
          sets: {json.dumps(list(sets))}
          source: {source}
          url: {url}
          path: {path}
          format: {fmt}
          sha256: {json.dumps(sha)}
          bytes: {json.dumps(nbytes)}
          expected: {exp}
        """), "  ")


def _manifest(tmp_path, *entries) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(HEADER + "".join(entries), encoding="utf-8")
    return p


def _src(tmp_path, name, content: bytes) -> tuple[str, str, int]:
    d = tmp_path / "remote"
    d.mkdir(exist_ok=True)
    f = d / name
    f.write_bytes(content)
    return f.as_uri(), hashlib.sha256(content).hexdigest(), len(content)


def test_first_fetch_records_then_verifies(tmp_path):
    url, sha, n = _src(tmp_path, "a.txt", b"hello benchmark\n")
    man = _manifest(tmp_path, _entry("a", url, "x/a.txt", expected={"contains": ["benchmark"]}))
    root = tmp_path / "data"
    r1 = F.fetch("t", root, manifest=man)
    assert [(r.status, r.sha256, r.bytes) for r in r1] == [("recorded", sha, n)]
    lock = json.loads((root / F.LOCK_NAME).read_text())
    assert lock["entries"]["a"]["sha256"] == sha
    assert F.fetch("t", root, manifest=man)[0].status == "cached"
    (root / "x" / "a.txt").write_bytes(b"corrupted")
    assert F.fetch("t", root, manifest=man)[0].status == "replaced"  # verified against the lock
    assert (root / "x" / "a.txt").read_bytes() == b"hello benchmark\n"
    assert F.fetch("t", root, force=True, manifest=man)[0].status == "replaced"
    sources = (root / "SOURCES.md").read_text()
    assert "local test files" in sources and sha in sources
    assert F.verify("t", root, manifest=man) == {"a": "ok"}


def test_pinned_checksum_and_size_are_enforced(tmp_path):
    url, sha, n = _src(tmp_path, "b.txt", b"0123456789")
    root = tmp_path / "data"
    bad = _manifest(tmp_path, _entry("b", url, "b.txt", sha="0" * 64, nbytes=n))
    with pytest.raises(F.ChecksumMismatch):
        F.fetch("t", root, manifest=bad)
    assert not (root / "b.txt").exists() and not list(root.glob("*.part"))
    small = _manifest(tmp_path, _entry("b", url, "b.txt", sha=sha, nbytes=5))
    with pytest.raises(F.FetchError, match="size limit"):
        F.fetch("t", root, manifest=small)
    unknown = _manifest(tmp_path, _entry("b", url, "b.txt"))
    with pytest.raises(F.FetchError, match="size limit of 4 bytes"):
        F.fetch("t", root, manifest=unknown, max_bytes=4)
    good = _manifest(tmp_path, _entry("b", url, "b.txt", sha=sha, nbytes=n))
    assert F.fetch("t", root, manifest=good)[0].status == "downloaded"
    (root / "b.txt").write_bytes(b"xxxxxxxxxx")
    assert F.verify("t", root, manifest=good)["b"].startswith("sha256 mismatch")


def test_expected_metadata_is_checked_through_the_parser(tmp_path):
    alist = FIX / "alist" / "toy_10_4.alist"
    url, sha, n = _src(tmp_path, "toy.alist", alist.read_bytes())
    ok = _manifest(tmp_path, _entry("toy", url, "toy.alist", fmt="alist", sha=sha, nbytes=n,
                                    expected={"n": 10, "m": 4, "rank": 3, "k": 7}))
    root = tmp_path / "data"
    assert F.fetch("t", root, manifest=ok)[0].status == "downloaded"
    wrong = _manifest(tmp_path, _entry("toy", url, "toy.alist", fmt="alist", sha=sha, nbytes=n,
                                       expected={"n": 10, "rank": 4}))
    with pytest.raises(F.FetchError, match=r"'rank': \(4, 3\)"):
        F.fetch("t", root, manifest=wrong)
    assert F.verify("t", root, manifest=wrong)["toy"].startswith("metadata mismatch")


@pytest.mark.parametrize("mutation, fragment", [
    (lambda e: e.replace("url: file:///X", "url: https://raw.githubusercontent.com/o/r/main/f.txt"),
     "must pin a 40-hex commit"),
    (lambda e: e.replace("url: file:///X", "url: https://raw.githubusercontent.com/o/r/" + "f" * 40 + "/f.txt"),
     "differs from sources.gh.revision"),
    (lambda e: e.replace("path: p.txt", "path: ../escape.txt"), "relative POSIX path"),
    (lambda e: e.replace("format: text", "format: pdf"), "unknown format"),
    (lambda e: e + "    color: red\n", "unknown field(s) ['color']"),
    (lambda e: e.replace("url: file:///X", "url: ftp://example.org/f"), "http(s):// or file://"),
    (lambda e: e.replace('sha256: null', 'sha256: "abc"'), "64 lowercase hex"),
])
def test_invalid_manifests(tmp_path, mutation, fragment):
    entry = mutation(_entry("e", "file:///X", "p.txt", source="gh"))
    p = _manifest(tmp_path, entry)
    with pytest.raises(ConfigError, match=re.escape(fragment)):
        F.load_manifest(p)


def test_unknown_set_and_duplicate_ids(tmp_path):
    url, _, _ = _src(tmp_path, "c.txt", b"c")
    man = F.load_manifest(_manifest(tmp_path, _entry("c", url, "c.txt")))
    with pytest.raises(ConfigError, match="unknown benchmark set"):
        man.select("nope")
    with pytest.raises(ConfigError, match="duplicate entry id"):
        F.load_manifest(_manifest(tmp_path, _entry("c", url, "c.txt"), _entry("c", url, "d.txt")))


def test_cli_list_fetch_verify(tmp_path, capsys):
    url, sha, n = _src(tmp_path, "d.txt", b"cli\n")
    man = _manifest(tmp_path, _entry("d", url, "d.txt", sha=sha, nbytes=n))
    root = tmp_path / "data"
    assert F.main(["--manifest", str(man), "--list"]) == 0
    assert "d.txt" in capsys.readouterr().out
    assert F.main(["--manifest", str(man), "--set", "t", "--data-root", str(root), "--verify"]) == 1  # missing
    assert F.main(["--manifest", str(man), "--set", "t", "--data-root", str(root)]) == 0
    assert F.main(["--manifest", str(man), "--set", "t", "--data-root", str(root), "--verify"]) == 0
    assert F.main(["--manifest", str(man), "--set", "zzz", "--data-root", str(root)]) == 2
    assert F.entry_path("d", root, F.load_manifest(man)) == root / "d.txt"

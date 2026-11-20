"""Paths outside a source checkout (plain pip install): roots fall back to the working directory and
the manifest to the copy shipped in the package, which must equal the committed manifest."""

from pathlib import Path

from nimbus.experiments import paths

REPO = Path(__file__).resolve().parent.parent


def test_packaged_manifest_matches_committed():
    shipped = REPO / "src" / "nimbus" / "data" / "benchmark_manifest.yaml"
    assert shipped.read_bytes() == (REPO / "benchmarks" / "manifest.yaml").read_bytes()


def test_roots_without_repo(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ENV_DATA_ROOT, raising=False)
    monkeypatch.delenv(paths.ENV_RUNS_ROOT, raising=False)
    monkeypatch.delenv(paths.ENV_REPO_ROOT, raising=False)
    monkeypatch.setattr(paths, "repo_root", lambda required=True: None)
    monkeypatch.chdir(tmp_path)
    assert paths.data_root() == tmp_path / "data"
    assert paths.runs_root() == tmp_path / "local_runs"
    m = paths.manifest_path()
    assert m.name == "benchmark_manifest.yaml" and m.is_file()

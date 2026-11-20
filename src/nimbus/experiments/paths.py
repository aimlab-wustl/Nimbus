"""Location of the repository, the data root and the runs root.

The code never hardcodes a storage location. Two environment variables select the roots:

``NIMBUS_DATA_ROOT``
    Benchmark instances downloaded by :mod:`nimbus.experiments.fetch`. Default: ``<repo>/data``
    (gitignored).
``NIMBUS_RUNS_ROOT``
    Run outputs. Default: ``<repo>/local_runs`` (gitignored).

Values may contain ``~`` and ``$VAR``/``${VAR}`` references (expanded here), so a cluster job can
set, for example, ``NIMBUS_DATA_ROOT=$SHARED_STORAGE/nimbus/data``.

The repository root is found by walking up from this file (editable/source checkout) and, if that
fails (e.g. a wheel install), from the current working directory; a directory qualifies when it
contains ``pyproject.toml`` declaring the ``nimbus`` project. ``NIMBUS_REPO_ROOT`` overrides the
search.

Outside a source checkout (a plain ``pip install``), the data and runs roots default to ``./data`` and
``./local_runs`` under the current working directory, and the benchmark manifest is read from the
copy shipped inside the package (``nimbus/data/benchmark_manifest.yaml``, kept identical to
``benchmarks/manifest.yaml`` by ``tests/test_paths_install.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

from nimbus.core.errors import ConfigError

ENV_DATA_ROOT = "NIMBUS_DATA_ROOT"
ENV_RUNS_ROOT = "NIMBUS_RUNS_ROOT"
ENV_REPO_ROOT = "NIMBUS_REPO_ROOT"

__all__ = [
    "ENV_DATA_ROOT",
    "ENV_RUNS_ROOT",
    "ENV_REPO_ROOT",
    "expand_path",
    "repo_root",
    "data_root",
    "runs_root",
    "manifest_path",
    "fixtures_root",
]


def expand_path(value: str | os.PathLike) -> Path:
    """Expand ``~`` and environment variables; refuse unresolved ``$VAR`` references."""
    text = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    if "$" in text:
        raise ConfigError(f"path {value!r} contains an unset environment variable (expanded: {text!r})")
    return Path(text)


def _is_repo_root(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        text = pyproject.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return 'name = "nimbus-ising"' in text


def _search_up(start: Path) -> Path | None:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if _is_repo_root(candidate):
            return candidate
    return None


def repo_root(required: bool = True) -> Path | None:
    """Root of the nimbus source checkout (directory holding ``pyproject.toml``)."""
    env = os.environ.get(ENV_REPO_ROOT)
    if env:
        root = expand_path(env)
        if not _is_repo_root(root):
            raise ConfigError(f"{ENV_REPO_ROOT}={env!r} does not contain the nimbus pyproject.toml")
        return root.resolve()
    found = _search_up(Path(__file__).parent) or _search_up(Path.cwd())
    if found is None and required:
        raise ConfigError(
            "cannot locate the nimbus repository root (no pyproject.toml found above the package "
            f"or the working directory); set {ENV_REPO_ROOT}, {ENV_DATA_ROOT} and {ENV_RUNS_ROOT}"
        )
    return found


def _root_from_env(env_name: str, default_subdir: str) -> Path:
    env = os.environ.get(env_name)
    if env:
        return expand_path(env)
    root = repo_root(required=False)
    return (root if root is not None else Path.cwd()) / default_subdir


def data_root() -> Path:
    """Benchmark data root: ``$NIMBUS_DATA_ROOT`` or ``<repo>/data``. The directory may not exist yet."""
    return _root_from_env(ENV_DATA_ROOT, "data")


def runs_root() -> Path:
    """Run output root: ``$NIMBUS_RUNS_ROOT`` or ``<repo>/local_runs``."""
    return _root_from_env(ENV_RUNS_ROOT, "local_runs")


def manifest_path() -> Path:
    """The committed benchmark manifest, ``<repo>/benchmarks/manifest.yaml``; outside a checkout,
    the copy shipped in the package."""
    root = repo_root(required=False)
    if root is not None:
        return root / "benchmarks" / "manifest.yaml"
    return Path(__file__).resolve().parent.parent / "data" / "benchmark_manifest.yaml"


def fixtures_root() -> Path:
    """Committed tiny test fixtures, ``<repo>/tests/fixtures``."""
    return repo_root(required=True) / "tests" / "fixtures"

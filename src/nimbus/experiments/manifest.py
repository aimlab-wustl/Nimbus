"""The run manifest document: everything needed to interpret or reproduce a run.

:func:`build_manifest` assembles the JSON-able mapping from the pieces the runner has at hand;
:func:`write_manifest` / :func:`read_manifest` persist it (atomically) under ``manifest.json``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nimbus.experiments.results import atomic_write_text

__all__ = ["SCHEMA", "git_info", "build_manifest", "write_manifest", "read_manifest"]

SCHEMA = "nimbus.manifest/1"


def _jsonable(x):
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def git_info(repo_dir=None) -> dict:
    """``{commit, dirty}``; ``commit`` is ``None`` outside a git checkout (never fabricated)."""
    cwd = str(repo_dir) if repo_dir is not None else None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                                 timeout=10, check=True).stdout.strip()
    except Exception:
        return {"commit": None, "dirty": None}
    try:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True,
                                 timeout=10, check=True).stdout
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _process_block(cfg):
    """Proposal process, clock, cohort and self-loop rule of the run."""
    try:
        from nimbus.experiments.config import process_spec_of

        return process_spec_of(cfg.data.get("annealing") or {}).describe()
    except Exception:  # a failed/unsupported run may not have a valid annealing section
        return None


def build_manifest(cfg, *, problem=None, model=None, bindings=None, backend_plan=None, policy=None,
                    calibration=None, timings=None, status: str = "ok", message: str | None = None,
                    instances=(), run_id: str | None = None, extra: Mapping | None = None) -> dict:
    """Assemble the manifest document (a fixed list of fields). Any piece may be ``None`` if unused
    (e.g. a failed run that never reached compilation)."""
    from nimbus.experiments.paths import repo_root

    d: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": run_id,
        "status": status,
        "message": message,
        "git": git_info(repo_root(required=False)),
        "config": {"trajectory_hash": cfg.trajectory_hash(), "full_hash": cfg.full_hash(),
                   "source": cfg.source, "name": cfg.name},
        "problem": problem.metadata() if problem is not None else None,
        "input_sources": getattr(problem, "sources", lambda: {})() if problem is not None else {},
        "compile": model.describe() if model is not None else None,
        "weights": [{"instance_id": iid, "weight_hash": b.weight_hash, "slot_names": list(b.slot_names)}
                    for iid, b in (bindings or [])],
        "backend": backend_plan.describe() if backend_plan is not None else None,
        "policy": policy.to_dict() if policy is not None else None,
        "process": _process_block(cfg),
        "calibration": calibration.to_dict() if hasattr(calibration, "to_dict") else calibration,
        "instances": list(instances),
        "timings": dict(timings or {}),
        "host": {"python": sys.version.split()[0], "platform": platform.platform(),
                 "numpy": np.__version__},
        "recording": cfg.data.get("recording"),
        "output": cfg.data.get("output"),
        "baseline": cfg.data.get("baseline"),
        "split": cfg.data.get("split"),
    }
    if extra:
        d.update(dict(extra))
    return _jsonable(d)


def write_manifest(run_dir, manifest: Mapping) -> Path:
    p = Path(run_dir) / "manifest.json"
    atomic_write_text(p, json.dumps(_jsonable(manifest), indent=1, sort_keys=False))
    return p


def read_manifest(run_dir) -> dict:
    p = Path(run_dir) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))

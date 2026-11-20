"""Versioned result schema ``nimbus.result/1`` and atomic I/O helpers.

A run directory holds:

``config.resolved.yaml``
    Written by :meth:`nimbus.experiments.config.RunConfig.write_resolved`.
``manifest.json``
    Built by :mod:`nimbus.experiments.manifest` (provenance, hashes, schedule, backend, timings).
``results.parquet``
    One row per (instance, replica): :class:`ReplicaRow` fields, written with :func:`write_shard`.
``checkpoint.npz`` (only while incomplete, or if ``output.keep_checkpoint``)
    Written by :mod:`nimbus.experiments.checkpoint`.

Numbers that a backend/problem does not support are stored as ``None`` (-> parquet null), never as
0, so "not measured" is distinguishable from "measured as zero".
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from nimbus.core.errors import ConfigError

__all__ = ["SCHEMA", "ReplicaRow", "atomic_write_text", "atomic_write_bytes", "atomic_savez",
           "write_shard", "read_shard", "consolidate"]

SCHEMA = "nimbus.result/1"

_ROW_FIELDS = (
    "run_id", "instance_id", "replica_id", "replica_position", "seed",
    "best_feasible_objective", "best_feasible_native", "best_feasible_attempt",
    "best_energy", "best_attempt", "final_energy", "final_feasible",
    "attempts", "accepted", "accepted_changed", "self_loops", "rejected", "uphill_accepted",
    "termination", "target_hit", "target_hit_attempt",
    "timings_json", "error",
    "process", "steps", "candidate_evaluations",
)


@dataclass
class ReplicaRow:
    """One result row (one replica of one instance). Unsupported metrics are ``None``, not 0."""

    run_id: str
    instance_id: int
    replica_id: int
    replica_position: int
    seed: int
    best_feasible_objective: float | None = None
    best_feasible_native: float | None = None
    best_feasible_attempt: int | None = None
    best_energy: float | None = None
    best_attempt: int | None = None
    final_energy: float | None = None
    final_feasible: bool | None = None
    attempts: int | None = None
    accepted: int | None = None
    accepted_changed: int | None = None
    self_loops: int | None = None
    rejected: int | None = None
    uphill_accepted: int | None = None
    termination: str | None = None
    target_hit: bool | None = None
    target_hit_attempt: int | None = None
    timings_json: str = "{}"
    error: str | None = None
    # proposal process: name, parallel rounds and candidate evaluations (None = not measured,
    # e.g. baselines or backends without these counters). ``attempts`` is in the process's clock units.
    process: str | None = None
    steps: int | None = None
    candidate_evaluations: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================================
# atomic I/O
# ============================================================================================
def _atomic_write(path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-" + path.name + "-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_text(path, text: str) -> Path:
    _atomic_write(path, text.encode("utf-8"))
    return Path(path)


def atomic_write_bytes(path, data: bytes) -> Path:
    _atomic_write(path, data)
    return Path(path)


def atomic_savez(path, **arrays) -> Path:
    """``np.savez_compressed`` written atomically (temp file + rename).

    The temp file is given a ``.npz`` suffix: ``numpy.savez_compressed`` silently appends ``.npz``
    to any path that lacks it, which would write the data to a different path than the one renamed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-" + path.name + "-", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


# ============================================================================================
# shards (parquet, falls back to a single-file npz-backed json if pyarrow is unavailable)
# ============================================================================================
def _rows_to_table(rows: Sequence[ReplicaRow | Mapping]):
    import pyarrow as pa

    dicts = [r.to_dict() if isinstance(r, ReplicaRow) else dict(r) for r in rows]
    cols = {name: [d.get(name) for d in dicts] for name in _ROW_FIELDS}
    return pa.table(cols)


def write_shard(path, rows: Sequence[ReplicaRow | Mapping]) -> Path:
    """Write one compact shard (one file per batch of trials, not one file per trial)."""
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _rows_to_table(rows)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-" + path.name + "-")
    os.close(fd)
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def read_shard(path) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    return table.to_pylist()


def consolidate(run_dirs: Sequence[str], out_path, *, schema=SCHEMA) -> Path:
    """Merge ``results.parquet`` of several run directories into one shard.

    Raises :class:`ConfigError` on duplicate ``(run_id, instance_id, replica_id)`` triples (a run
    counted twice), matching the "unique-run checks" requirement.
    """
    rows: list[dict] = []
    seen = set()
    for d in run_dirs:
        p = Path(d) / "results.parquet"
        if not p.exists():
            continue
        for row in read_shard(p):
            key = (row.get("run_id"), row.get("instance_id"), row.get("replica_id"))
            if key in seen:
                raise ConfigError(f"consolidate: duplicate run/instance/replica {key} from {p}")
            seen.add(key)
            rows.append(row)
    write_shard(out_path, rows)
    return Path(out_path)

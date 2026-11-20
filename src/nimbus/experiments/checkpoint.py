"""Atomic checkpoint/resume.

One checkpoint file (``checkpoint.npz``) holds everything needed to continue a run exactly:
the exported :class:`~nimbus.backends.base.ReplicaStates` of the instance currently in progress,
the index of that instance within the run's selected-instance order, the list of already-finished
instance ids (whose rows are already in ``results.parquet``), the resolved schedule and seed, and
hashes of the configuration/topology/weights it was made for. Only the latest checkpoint per run is
kept (overwrite in place, so a run never accumulates checkpoint files): :func:`save_checkpoint` always
writes to the same path.

:func:`resume` rejects a checkpoint made for a different topology, weights or (trajectory-affecting)
configuration with :class:`~nimbus.core.errors.IncompatibleCheckpoint`, so a resumed run can never
silently diverge from an uninterrupted one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nimbus.backends.base import ReplicaStates
from nimbus.core.errors import IncompatibleCheckpoint
from nimbus.experiments.recording import RecordingData
from nimbus.experiments.results import atomic_savez

__all__ = ["SCHEMA", "CheckpointMeta", "checkpoint_path", "save_checkpoint", "load_checkpoint",
           "check_resumable", "remove_checkpoint"]

SCHEMA = "nimbus.checkpoint/1"


@dataclass
class CheckpointMeta:
    """Everything besides the replica-state arrays and (optional) recording data."""

    run_id: str
    trajectory_hash: str
    topology_hash: str
    weight_hashes: tuple
    schedule: dict
    seed: int
    instance_order: tuple  # instance ids, in run order
    instance_index: int  # position of the in-progress instance within instance_order
    finished_instances: tuple  # instance ids already consolidated into results.parquet
    chunk_attempts_done: int  # attempts executed so far for the in-progress instance (per replica, min)
    n_attempts_total: int
    started_utc: str
    updated_utc: str

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["schema"] = SCHEMA
        d["weight_hashes"] = list(self.weight_hashes)
        d["instance_order"] = list(self.instance_order)
        d["finished_instances"] = list(self.finished_instances)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointMeta":
        d = dict(d)
        d.pop("schema", None)
        d["weight_hashes"] = tuple(d["weight_hashes"])
        d["instance_order"] = tuple(d["instance_order"])
        d["finished_instances"] = tuple(d["finished_instances"])
        return cls(**d)


def checkpoint_path(run_dir) -> Path:
    return Path(run_dir) / "checkpoint.npz"


def save_checkpoint(run_dir, meta: CheckpointMeta, state: ReplicaStates, recording: RecordingData | None = None):
    """Atomically overwrite the single checkpoint file for ``run_dir``."""
    arrays = {"checkpoint.meta_json": np.asarray(json.dumps(meta.to_dict()))}
    arrays.update(state.to_arrays())
    if recording is not None:
        arrays.update({"rec." + k: v for k, v in recording.to_arrays().items()})
    return atomic_savez(checkpoint_path(run_dir), **arrays)


def load_checkpoint(run_dir) -> tuple[CheckpointMeta, ReplicaStates, RecordingData | None]:
    p = checkpoint_path(run_dir)
    if not p.exists():
        raise IncompatibleCheckpoint(f"no checkpoint at {p}")
    with np.load(p, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    try:
        meta = CheckpointMeta.from_dict(json.loads(str(np.asarray(a["checkpoint.meta_json"]))))
    except (KeyError, ValueError) as exc:
        raise IncompatibleCheckpoint(f"{p}: not a nimbus checkpoint ({exc})") from exc
    state = ReplicaStates.from_arrays(a)
    rec_arrays = {k[len("rec."):]: v for k, v in a.items() if k.startswith("rec.")}
    recording = RecordingData.from_arrays(rec_arrays) if rec_arrays else None
    return meta, state, recording


def check_resumable(meta: CheckpointMeta, *, trajectory_hash: str, topology_hash: str, weight_hashes):
    """Raise :class:`IncompatibleCheckpoint` unless the checkpoint matches the current run."""
    if meta.trajectory_hash != trajectory_hash:
        raise IncompatibleCheckpoint("checkpoint was made for a different (trajectory-affecting) configuration")
    if meta.topology_hash != topology_hash:
        raise IncompatibleCheckpoint("checkpoint was made for a different compiled topology")
    if tuple(meta.weight_hashes) != tuple(weight_hashes):
        raise IncompatibleCheckpoint("checkpoint was made with different instance weights")


def remove_checkpoint(run_dir):
    p = checkpoint_path(run_dir)
    if p.exists():
        p.unlink()

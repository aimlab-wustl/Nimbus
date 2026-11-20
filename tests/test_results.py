"""nimbus.experiments.results: atomic I/O, shards, consolidation (owned by experiments-cli)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.quick

import numpy as np

from nimbus.core.errors import ConfigError
from nimbus.experiments.results import (
    ReplicaRow,
    atomic_savez,
    atomic_write_text,
    consolidate,
    read_shard,
    write_shard,
)


def test_atomic_write_text_roundtrip(tmp_path):
    p = tmp_path / "a" / "b.txt"
    atomic_write_text(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"
    assert not list(p.parent.glob(".tmp-*"))  # no leftover temp files


def test_atomic_savez_roundtrip(tmp_path):
    p = tmp_path / "state.npz"
    atomic_savez(p, x=np.arange(5), y=np.asarray("hi"))
    with np.load(p, allow_pickle=False) as z:
        assert list(z["x"]) == [0, 1, 2, 3, 4]
    assert not list(p.parent.glob(".tmp-*"))


def test_write_shard_roundtrip(tmp_path):
    rows = [ReplicaRow(run_id="r1", instance_id=0, replica_id=0, replica_position=0, seed=1,
                       best_feasible_objective=-3.5, attempts=100),
           ReplicaRow(run_id="r1", instance_id=0, replica_id=1, replica_position=1, seed=1,
                       best_feasible_objective=None, attempts=100)]
    p = write_shard(tmp_path / "shard.parquet", rows)
    got = read_shard(p)
    assert len(got) == 2
    assert got[0]["best_feasible_objective"] == -3.5
    assert got[1]["best_feasible_objective"] is None  # unsupported/absent, never 0


def test_write_shard_empty(tmp_path):
    p = write_shard(tmp_path / "empty.parquet", [])
    assert read_shard(p) == []


def test_consolidate_merges_shards(tmp_path):
    for i in range(2):
        run_dir = tmp_path / f"run{i}"
        run_dir.mkdir()
        write_shard(run_dir / "results.parquet",
                   [ReplicaRow(run_id=f"run{i}", instance_id=0, replica_id=0, replica_position=0, seed=1)])
    out = consolidate([tmp_path / "run0", tmp_path / "run1"], tmp_path / "all.parquet")
    rows = read_shard(out)
    assert {r["run_id"] for r in rows} == {"run0", "run1"}


def test_consolidate_rejects_duplicate_run_instance_replica(tmp_path):
    run_dir = tmp_path / "run0"
    run_dir.mkdir()
    write_shard(run_dir / "results.parquet",
               [ReplicaRow(run_id="run0", instance_id=0, replica_id=0, replica_position=0, seed=1)])
    with pytest.raises(ConfigError, match="duplicate"):
        consolidate([run_dir, run_dir], tmp_path / "all.parquet")


def test_consolidate_skips_missing_shard(tmp_path):
    (tmp_path / "empty_run").mkdir()
    out = consolidate([tmp_path / "empty_run"], tmp_path / "all.parquet")
    assert read_shard(out) == []

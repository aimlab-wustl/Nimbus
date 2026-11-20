"""nimbus.experiments.checkpoint: save/load round-trip and resume-compatibility checks."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.quick

import time

import numpy as np

from nimbus.core.errors import IncompatibleCheckpoint
from nimbus.experiments import checkpoint as ckpt
from nimbus.problems.toy_cardinality import ToyCardinalityProblem
from nimbus.compiler import compile_model
from nimbus.core.model import bind_weights
from nimbus.backends import get_backend


def _toy_state():
    problem = ToyCardinalityProblem()
    lib = problem.transition_library("pair_exchanges")
    model = compile_model(problem.structure(), lib)
    binding = bind_weights(model, problem.term_weights(0), problem.const(0))
    backend = get_backend("emulation")
    plan = backend.prepare(model)
    spins0 = problem.random_feasible(1, 2, 0)
    native = backend.init_state(plan, binding, spins0, is_feasible=problem.is_feasible)
    return backend, plan, model, binding, native


def _meta(model, binding, **over):
    base = dict(run_id="r", trajectory_hash="TRAJ", topology_hash=model.topology_hash,
               weight_hashes=(binding.weight_hash,), schedule={"kind": "constant", "p": [0.5, 0, 0, 0], "n_total": 10},
               seed=1, instance_order=(0,), instance_index=0, finished_instances=(),
               chunk_attempts_done=0, n_attempts_total=10, started_utc="now", updated_utc="now")
    base.update(over)
    return ckpt.CheckpointMeta(**base)


def test_save_and_load_roundtrip(tmp_path):
    backend, plan, model, binding, native = _toy_state()
    exported = backend.export_state(native)
    meta = _meta(model, binding, chunk_attempts_done=5)
    ckpt.save_checkpoint(tmp_path, meta, exported)
    meta2, state2, rec2 = ckpt.load_checkpoint(tmp_path)
    assert meta2.trajectory_hash == meta.trajectory_hash
    assert meta2.chunk_attempts_done == 5
    assert rec2 is None
    np.testing.assert_array_equal(state2.spins, exported.spins)
    np.testing.assert_array_equal(state2.T, exported.T)


def test_checkpoint_overwrites_in_place(tmp_path):
    backend, plan, model, binding, native = _toy_state()
    exported = backend.export_state(native)
    ckpt.save_checkpoint(tmp_path, _meta(model, binding, chunk_attempts_done=1), exported)
    ckpt.save_checkpoint(tmp_path, _meta(model, binding, chunk_attempts_done=2), exported)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1  # only the latest checkpoint is kept
    meta, _, _ = ckpt.load_checkpoint(tmp_path)
    assert meta.chunk_attempts_done == 2


def test_load_missing_checkpoint_raises(tmp_path):
    with pytest.raises(IncompatibleCheckpoint):
        ckpt.load_checkpoint(tmp_path)


def test_check_resumable_rejects_mismatched_trajectory(tmp_path):
    backend, plan, model, binding, native = _toy_state()
    meta = _meta(model, binding)
    with pytest.raises(IncompatibleCheckpoint):
        ckpt.check_resumable(meta, trajectory_hash="OTHER", topology_hash=model.topology_hash,
                             weight_hashes=(binding.weight_hash,))


def test_check_resumable_rejects_mismatched_topology(tmp_path):
    backend, plan, model, binding, native = _toy_state()
    meta = _meta(model, binding)
    with pytest.raises(IncompatibleCheckpoint):
        ckpt.check_resumable(meta, trajectory_hash=meta.trajectory_hash, topology_hash="OTHER",
                             weight_hashes=(binding.weight_hash,))


def test_check_resumable_rejects_mismatched_weights(tmp_path):
    backend, plan, model, binding, native = _toy_state()
    meta = _meta(model, binding)
    with pytest.raises(IncompatibleCheckpoint):
        ckpt.check_resumable(meta, trajectory_hash=meta.trajectory_hash, topology_hash=model.topology_hash,
                             weight_hashes=("other",))


def test_check_resumable_accepts_match(tmp_path):
    backend, plan, model, binding, native = _toy_state()
    meta = _meta(model, binding)
    ckpt.check_resumable(meta, trajectory_hash=meta.trajectory_hash, topology_hash=model.topology_hash,
                         weight_hashes=(binding.weight_hash,))  # no raise


def test_remove_checkpoint(tmp_path):
    backend, plan, model, binding, native = _toy_state()
    exported = backend.export_state(native)
    ckpt.save_checkpoint(tmp_path, _meta(model, binding), exported)
    assert ckpt.checkpoint_path(tmp_path).exists()
    ckpt.remove_checkpoint(tmp_path)
    assert not ckpt.checkpoint_path(tmp_path).exists()
    ckpt.remove_checkpoint(tmp_path)  # idempotent

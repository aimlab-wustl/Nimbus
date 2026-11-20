"""Backend base: registry, shared initialization helpers (Supplementary Information, Exact local maintenance of move-neuron inputs), replica-state
container persistence, run-window rules, acceptance law and small numeric helpers."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from test_replay_core import case_cardinality_random, case_cardinality_toy, case_section29
from nimbus.annealing import schedule as S
from nimbus.backends import Backend, ReplicaStates, available_backends, get_backend, register_backend
from nimbus.backends import base
from nimbus.backends.base import (
    COUNTER_NAMES,
    SlotLayout,
    apply_targets,
    check_initial_feasibility,
    clause_signs,
    coerce_replicas,
    csr_gather,
    csr_rows,
    encoder_inputs,
    initial_replica_states,
    metropolis_accept,
    metropolis_threshold,
    resolve_window,
    route_switches,
    slot_energies,
)
from nimbus.core import rng
from nimbus.core.errors import BackendUnavailable, ConfigError, IncompatibleCheckpoint, InfeasibleState
from nimbus.core.model import bind_weights


# ============================================================================================
# registry
# ============================================================================================
def test_registry_known_backends_and_aliases():
    names = available_backends()
    for n in ("oracle", "emulation", "cpu", "gpu"):
        assert n in names
    assert get_backend("oracle").name == "oracle"
    assert get_backend("emulation").name == "emulation"
    assert get_backend("emulator").name == "emulation"
    with pytest.raises(ConfigError):
        get_backend("no-such-backend")
    for n in ("cpu", "gpu"):  # implemented by other modules; never silently replaced
        try:
            b = get_backend(n)
        except BackendUnavailable:
            continue
        assert isinstance(b, Backend) and b.name == n


def test_register_custom_backend_and_lazy_failure(monkeypatch):
    class Dummy(base.Backend):
        name = "dummy-test"

        def prepare(self, model, **kw):
            return None

        def init_state(self, *a, **k):
            return None

        def run(self, *a, **k):
            return None

        def export_state(self, state):
            return None

        def load_state(self, *a, **k):
            return None

        def evaluate_inputs(self, state, groups=None):
            return np.zeros((0, 0))

    monkeypatch.setattr(base, "_REGISTRY", dict(base._REGISTRY))
    register_backend("dummy-test", Dummy)
    assert isinstance(get_backend("dummy-test"), Dummy)
    with pytest.raises(ConfigError):
        register_backend("dummy-test", lambda: Dummy())
    register_backend("dummy-test", Dummy, replace=True)
    monkeypatch.setitem(base._LAZY_MODULES, "phantom", "nimbus.backends.does_not_exist_xyz")
    with pytest.raises(BackendUnavailable, match="cannot be imported"):
        get_backend("phantom")
    monkeypatch.setitem(base._LAZY_MODULES, "silent", "nimbus.backends.base")
    with pytest.raises(BackendUnavailable, match="did not register"):
        get_backend("silent")
    d = get_backend("oracle").describe()
    assert d["backend"] == "oracle" and "numpy" in d and "python" in d


# ============================================================================================
# shared initialization helpers equal the model's reference definitions
# ============================================================================================
@pytest.mark.parametrize("builder", [case_cardinality_toy, case_cardinality_random, case_section29])
def test_init_helpers_match_reference_definitions(builder):
    case = builder()
    model = case.model
    X = case.states
    T = clause_signs(model, X)
    assert T.dtype == np.int8 and np.array_equal(T, model.clause_values(X))
    al = route_switches(model, T)
    assert al.dtype == np.int8 and np.array_equal(al, model.route_alpha(T))
    sub = np.arange(0, model.n_routes, 3)
    assert np.array_equal(route_switches(model, T, sub), al[:, sub])
    for b in case.bindings:
        J0 = np.broadcast_to(b.J[0], (X.shape[0], model.n_clauses))
        v = encoder_inputs(model, J0, T)
        assert np.allclose(v, model.inputs(T, b), rtol=0, atol=1e-12)
        groups = [model.n_groups - 1, 0, 2]
        assert np.allclose(encoder_inputs(model, J0, T, groups=groups), v[:, groups], rtol=0, atol=0)
        assert np.array_equal(encoder_inputs(model, J0, T, alpha=al), v)
        # Supplementary Information, Exact local maintenance of move-neuron inputs equals the energy differences on the feasible states
        E = slot_energies([b], np.zeros(X.shape[0], np.int32), T)
        assert np.allclose(E[:, 0], model.energy(X, b))
    single = clause_signs(model, X[0])
    assert single.shape == (1, model.n_clauses)


def test_slot_energies_multislot_and_layout():
    case = case_cardinality_toy()
    model = case.model
    w = case.weights[0]
    b = bind_weights(model, w, case.consts[0], extra_slots={"objective": (w, case.consts[0]),
                                                             "violation": (np.zeros_like(w), 0.0)})
    T = clause_signs(model, case.states)
    E = slot_energies([b], np.zeros(len(case.states), np.int32), T)
    assert np.allclose(E[:, 0], E[:, 1]) and np.allclose(E[:, 2], 0.0)
    lay = SlotLayout.from_bindings([b])
    assert lay.slot_names == ("run", "objective", "violation") and lay.score_slot == 1 and lay.violation_slot == 2
    assert lay.penalty and lay.scale.shape == (1, 3) and np.all(lay.scale >= 1.0)
    plain = SlotLayout.from_bindings(case.bindings)
    assert plain.score_slot == 0 and plain.violation_slot is None and not plain.penalty
    assert plain.scale[0, 0] == pytest.approx(max(1.0, np.abs(case.bindings[0].J[0]).sum() + abs(case.bindings[0].const[0])))


def test_coerce_replicas_and_feasibility():
    case = case_cardinality_toy()
    model = case.model
    s, rb, rid, iid = coerce_replicas(model, 2, case.states[:3].astype(np.int64), [0, 1, 1])
    assert s.dtype == np.int8 and rb.dtype == np.int32 and rid.tolist() == [0, 1, 2] and iid.tolist() == [0, 1, 1]
    with pytest.raises(ConfigError):
        coerce_replicas(model, 1, case.states[:2], [0, 1])  # binding index out of range
    with pytest.raises(ConfigError):
        coerce_replicas(model, 1, np.ones((2, 5), np.int8))
    with pytest.raises(ConfigError):
        coerce_replicas(model, 1, case.states[:2], replica_id=[-1, 0])
    with pytest.raises(ConfigError):
        coerce_replicas(model, 1, case.states[:2], replica_id=[2 ** 32, 0])
    check_initial_feasibility(case.states[:4], case.is_feasible)
    check_initial_feasibility(np.ones((1, 6), np.int8), None)
    with pytest.raises(InfeasibleState, match="replica 1"):
        check_initial_feasibility(np.stack([case.states[0], np.ones(6, np.int8)]), case.is_feasible)


# ============================================================================================
# replica states container
# ============================================================================================
def _states(case):
    e = get_backend("emulation")
    plan = e.prepare(case.model, capacity=4)
    st = e.init_state(plan, case.bindings, case.initial(4, 1), [0, 0, 1, 1])
    e.run(plan, st, S.constant(1.0, 300), seed=3, stop=150)
    return e, plan, e.export_state(st)


def test_replica_states_roundtrip(tmp_path):
    case = case_cardinality_toy()
    e, plan, rs = _states(case)
    assert rs.n_replicas == 4 and set(COUNTER_NAMES) <= set(rs.counters)
    assert rs.weight_hashes == tuple(b.weight_hash for b in case.bindings)
    p = rs.save(tmp_path / "state")
    back = ReplicaStates.load(p)
    for name in ("spins", "T", "energies", "best_energy", "best_spins", "best_attempt", "feasible_best_energy",
                 "feasible_best_attempt", "next_attempt", "replica_binding", "replica_id", "instance_id", "stopped"):
        assert np.array_equal(getattr(back, name), getattr(rs, name)), name
    assert np.array_equal(back.v, rs.v, equal_nan=True)
    assert set(back.extras) == set(rs.extras)
    assert back.topology_hash == rs.topology_hash and back.slot_names == rs.slot_names
    c = rs.copy()
    c.T[0, 0] *= -1
    c.counters["attempts"][0] += 1
    assert rs.T[0, 0] != c.T[0, 0] and rs.counters["attempts"][0] != c.counters["attempts"][0]
    with pytest.raises(ValueError):  # spins are a read-only readout of the singleton clauses
        c.spins[0, 0] = 1
    back.check_compatible(case.model, case.bindings)
    with pytest.raises(IncompatibleCheckpoint):
        back.check_compatible(case.model, case.bindings[::-1])
    a = rs.to_arrays()
    a["state.header_json"] = np.asarray(json.dumps({"schema_version": "other/9"}))
    with pytest.raises(IncompatibleCheckpoint):
        ReplicaStates.from_arrays(a)
    with pytest.raises(IncompatibleCheckpoint):
        ReplicaStates.from_arrays({})
    # cross-backend import: the oracle accepts an emulation snapshot (spins/T/counters/best)
    o = get_backend("oracle")
    ost = o.load_state(o.prepare(case.model, library=case.library), back, bindings=case.bindings)
    assert np.array_equal(ost.rs.spins, rs.spins) and np.array_equal(ost.rs.next_attempt, rs.next_attempt)
    # and the emulation rebuilds its caches from an oracle snapshot
    est = e.load_state(plan, o.export_state(ost))
    assert np.allclose(e.evaluate_inputs(est), o.evaluate_inputs(ost), atol=1e-12)


def test_initial_states_and_targets():
    case = case_cardinality_toy()
    model = case.model
    s, rb, rid, iid = coerce_replicas(model, 2, case.states[:3], [0, 1, 0])
    T = clause_signs(model, s)
    E = slot_energies(case.bindings, rb, T)
    rs = initial_replica_states("x", model, case.bindings, s, rb, rid, iid, T, E)
    assert np.all(rs.best_attempt == 0) and np.all(rs.counters["target_hit_attempt"] == -1)
    assert np.array_equal(rs.best_energy, E[:, 0]) and np.array_equal(rs.feasible_best_energy, E[:, 0])
    lay = SlotLayout.from_bindings(case.bindings)
    rs.next_attempt[:] = 17
    apply_targets(rs, lay, [E[0, 0], E[1, 0] - 1.0, np.nan], stop_on_target=True)
    assert rs.counters["target_hit_attempt"].tolist() == [17, -1, -1]
    assert rs.stopped.tolist() == [True, False, False]
    apply_targets(rs, lay, None)
    assert rs.counters["target_hit_attempt"].tolist() == [17, -1, -1]


def test_resolve_window_rules():
    case = case_cardinality_toy()
    e = get_backend("emulation")
    plan = e.prepare(case.model)
    st = e.init_state(plan, case.bindings[0], case.initial(2))
    sched = S.constant(1.0, 100)
    assert resolve_window(st.rs, sched) == (100, 0, 100)
    assert resolve_window(st.rs, sched, 100, 0, 40) == (100, 0, 40)
    with pytest.raises(ConfigError):
        resolve_window(st.rs, sched, n_attempts_total=99)
    with pytest.raises(ConfigError):
        resolve_window(st.rs, sched, start=5)
    with pytest.raises(ConfigError):
        resolve_window(st.rs, sched, stop=101)
    st.rs.next_attempt[1] = 3
    with pytest.raises(ConfigError, match="repeat or skip"):
        resolve_window(st.rs, sched)
    st.rs.stopped[1] = True  # stopped replicas do not constrain the window
    assert resolve_window(st.rs, sched) == (100, 0, 100)


# ============================================================================================
# acceptance law and helpers
# ============================================================================================
def test_metropolis_rule():
    assert metropolis_threshold(0.0, 0.3) == 0.0
    assert metropolis_accept(0.0, 0.0, 0.999) and not metropolis_accept(1e-300, 0.0, 1e-9)
    assert metropolis_threshold(2.0, 0.5) == -(2.0 / 2.0) * math.log(0.5)
    # exp(-dE/tau) acceptance frequency from the solver-stream uniforms (deterministic sample)
    _, u = rng.solver_draws(9, np.arange(40000), 0, 0, 7)
    tau, dE = 1.3, 0.9
    p = np.mean([metropolis_accept(dE / 2.0, tau, x) for x in u.tolist()])
    assert abs(p - math.exp(-dE / tau)) < 4 * math.sqrt(p * (1 - p) / u.size)
    # downhill and neutral always accepted
    assert all(metropolis_accept(-0.5, tau, x) and metropolis_accept(0.0, tau, x) for x in u[:1000].tolist())


def test_csr_helpers():
    rng_ = np.random.default_rng(0)
    lens = rng_.integers(0, 5, size=30)
    ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    data = rng_.integers(0, 1000, size=int(ptr[-1])).astype(np.int64)
    for rows in (np.array([], np.int64), np.array([3, 0, 29, 3]), rng_.integers(0, 30, size=12)):
        want = [x for r in rows.tolist() for x in data[ptr[r]:ptr[r + 1]].tolist()]
        want_row = [j for j, r in enumerate(rows.tolist()) for _ in range(int(lens[r]))]
        got, row = csr_gather(ptr, data, rows)
        assert got.tolist() == want and row.tolist() == want_row
        assert csr_rows(ptr, data, rows).tolist() == want


def test_step_record_and_summary_are_serializable():
    case = case_cardinality_toy()
    e = get_backend("emulation")
    plan = e.prepare(case.model)
    st = e.init_state(plan, case.bindings[0], case.initial(2))
    res = e.run(plan, st, S.constant(0.5, 50), seed=1)
    s = res.summary()
    json.dumps(s)
    assert s["attempts"] == [50, 50] and res.termination == ("budget", "budget")
    rec = base.StepRecord.empty(4, 0.5, 3)
    assert rec.group.tolist() == [-1, -1, -1] and not rec.active.any()
    assert plan.describe()["backend"] == "emulation"

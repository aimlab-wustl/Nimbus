"""Optimized CPU architecture backend (``nimbus.backends.cpu``): lock-step replay against the
oracle and the emulator for every strategy/layout, continuation, thread independence, zero
temperature, multi-slot (penalty-style) bindings and corruption detection.

Case builders are shared with the emulator/oracle lock-step suite
(``tests/test_replay_core.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from helpers_core import random_polynomial
from test_replay_core import CASE_BUILDERS, Case, make_pair, mixing_schedule

from nimbus.annealing import schedule as S
from nimbus.backends import get_backend
from nimbus.backends.base import ArchitectureValidationError
from nimbus.backends.cpu import LAYOUTS, STRATEGIES, choose_auto
from nimbus.core.model import bind_weights

N_ATTEMPTS = 300
STRAT_LAYOUT = [(s, l) for s in STRATEGIES for l in LAYOUTS]


def _cpu_pair(case: Case, strategy: str, layout: str, n_rep_per_binding=2, seed=0):
    nb = len(case.bindings)
    n = nb * n_rep_per_binding
    spins0 = case.initial(n, seed)
    rb = np.repeat(np.arange(nb), n_rep_per_binding)
    rid = np.tile(np.arange(n_rep_per_binding), nb)
    o = get_backend("oracle")
    c = get_backend("cpu")
    po = o.prepare(case.model, library=case.library)
    pc = c.prepare(case.model, strategy=strategy, layout=layout)
    so = o.init_state(po, case.bindings, spins0, rb, rid, rb, is_feasible=case.is_feasible)
    sc = c.init_state(pc, case.bindings, spins0, rb, rid, rb, is_feasible=case.is_feasible)
    return o, so, c, pc, sc


def _assert_matches_oracle(case: Case, so, sc, cpu_backend):
    assert np.array_equal(so.rs.spins, sc.rs.spins)
    assert np.array_equal(so.rs.T, sc.rs.T)
    assert sc.rs.T.shape[1] == case.model.n_clauses  # no dummy controller slot leaks into ReplicaStates
    assert np.all(sc.T_full[:, case.model.n_clauses] == 1)  # dummy controller slot pinned to +1 internally
    scale = float(sc.layout.scale.max())
    assert np.allclose(so.rs.energies, sc.rs.energies, atol=1e-9 * scale)
    for k in so.rs.counters:
        assert np.array_equal(so.rs.counters[k], sc.rs.counters[k]), k


# ============================================================================================
# lock-step replay: cpu.step() vs oracle.step(), every family x strategy x layout
# ============================================================================================
@pytest.mark.parametrize("strategy,layout", STRAT_LAYOUT)
@pytest.mark.parametrize("name", list(CASE_BUILDERS))
def test_cpu_lockstep_vs_oracle(name, strategy, layout):
    case = CASE_BUILDERS[name]()
    sched = mixing_schedule(case, N_ATTEMPTS)
    o, so, c, pc, sc = _cpu_pair(case, strategy, layout, n_rep_per_binding=2, seed=7)
    seed = 4242
    tot_rej = tot_chg = tot_slf = 0
    for n in range(N_ATTEMPTS):
        ro = o.step(so, n, sched, seed)
        rc = c.step(sc, n, sched, seed)
        assert np.array_equal(ro.group, rc.group), (name, strategy, layout, n)
        assert np.array_equal(ro.u, rc.u) and np.array_equal(ro.xi, rc.xi), n
        assert np.array_equal(ro.accepted, rc.accepted), (name, strategy, layout, n, ro.v, rc.v)
        assert np.array_equal(ro.changed, rc.changed), n
        tot_rej += int(np.count_nonzero(~ro.accepted))
        tot_chg += int(np.count_nonzero(ro.changed))
        tot_slf += int(np.count_nonzero(ro.accepted & ~ro.changed))
    _assert_matches_oracle(case, so, sc, c)
    assert tot_rej > 0 and tot_chg > 0
    if case.has_selfloops:
        assert tot_slf > 0
    assert all(case.is_feasible(s) for s in sc.rs.spins)


@pytest.mark.parametrize("strategy,layout", STRAT_LAYOUT)
def test_cpu_lockstep_vs_emulator(strategy, layout):
    """One representative family (gates, readout, PERM, CONDMASK all exercised): section29."""
    case = CASE_BUILDERS["section29"]()
    sched = mixing_schedule(case, N_ATTEMPTS)
    _, _, e, pe, se = make_pair(case, n_rep_per_binding=2, seed=3, validate="full")
    c = get_backend("cpu")
    pc = c.prepare(case.model, strategy=strategy, layout=layout)
    spins0 = se.rs.spins.copy()
    sc = c.init_state(pc, case.bindings, spins0, se.rs.replica_binding, se.rs.replica_id, se.rs.instance_id,
                      is_feasible=case.is_feasible)
    seed = 909
    for n in range(N_ATTEMPTS):
        re = e.step(se, n, sched, seed)
        rc = c.step(sc, n, sched, seed)
        assert np.array_equal(re.group, rc.group)
        assert np.array_equal(re.accepted, rc.accepted)
        assert np.array_equal(re.changed, rc.changed)
    assert np.array_equal(se.rs.spins, sc.rs.spins)
    assert np.array_equal(se.rs.T, sc.rs.T)
    scale = float(sc.layout.scale.max())
    assert np.allclose(se.rs.energies, sc.rs.energies, atol=1e-9 * scale)


# ============================================================================================
# run() in chunks (settled boundaries every 50-100 attempts) equals the uninterrupted run,
# bitwise, for every strategy/layout; also checks run() agrees with step()
# ============================================================================================
@pytest.mark.parametrize("strategy,layout", STRAT_LAYOUT)
def test_cpu_run_equals_step_and_chunked_continuation(strategy, layout):
    case = CASE_BUILDERS["onehot_P5"]()
    sched = mixing_schedule(case, 600)
    o, so, c, pc, sc = _cpu_pair(case, strategy, layout, n_rep_per_binding=2, seed=11)
    seed = 55
    for n in range(600):
        o.step(so, n, sched, seed)
    res = c.run(pc, sc, sched, seed=seed)
    assert np.array_equal(res.state.spins, so.rs.spins)
    assert np.array_equal(res.state.T, so.rs.T)
    for k in so.rs.counters:
        assert np.array_equal(res.state.counters[k], so.rs.counters[k]), k

    # chunked: same model/bindings/initial spins, settle at boundaries of 50-100 attempts
    c2 = get_backend("cpu")
    pc2 = c2.prepare(case.model, strategy=strategy, layout=layout)
    nb = len(case.bindings)
    spins0 = case.initial(nb * 2, 11)
    rb = np.repeat(np.arange(nb), 2)
    rid = np.tile(np.arange(2), nb)
    sc2 = c2.init_state(pc2, case.bindings, spins0, rb, rid, rb, is_feasible=case.is_feasible)
    boundaries = list(range(0, 601, 73))
    if boundaries[-1] != 600:
        boundaries.append(600)
    chunked = None
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        chunked = c2.run(pc2, sc2, sched, n_attempts_total=600, start=a, stop=b, seed=seed)
        sc2 = c2.load_state(pc2, chunked.state, case.bindings)
    assert np.array_equal(chunked.state.spins, so.rs.spins)
    assert np.array_equal(chunked.state.T, so.rs.T)
    for k in so.rs.counters:
        assert np.array_equal(chunked.state.counters[k], so.rs.counters[k]), k
    assert np.array_equal(chunked.state.best_energy, so.rs.best_energy)
    assert np.array_equal(chunked.state.best_attempt, so.rs.best_attempt)


# ============================================================================================
# thread-count independence: 1 thread vs many threads give bitwise-identical trajectories
# ============================================================================================
@pytest.mark.parametrize("strategy,layout", STRAT_LAYOUT)
def test_cpu_thread_count_independence(strategy, layout):
    case = CASE_BUILDERS["cardinality_random"]()
    sched = mixing_schedule(case, 400)
    nb = len(case.bindings)
    spins0 = case.initial(nb * 4, 2)
    rb = np.repeat(np.arange(nb), 4)
    rid = np.tile(np.arange(4), nb)
    results = {}
    for threads in (1, 4):
        c = get_backend("cpu")
        pc = c.prepare(case.model, strategy=strategy, layout=layout, threads=threads)
        sc = c.init_state(pc, case.bindings, spins0, rb, rid, rb, is_feasible=case.is_feasible)
        res = c.run(pc, sc, sched, seed=321)
        results[threads] = res
    r1, r4 = results[1], results[4]
    assert np.array_equal(r1.state.spins, r4.state.spins)
    assert np.array_equal(r1.state.T, r4.state.T)
    assert np.array_equal(r1.state.best_energy, r4.state.best_energy)
    for k in r1.state.counters:
        assert np.array_equal(r1.state.counters[k], r4.state.counters[k]), k


# ============================================================================================
# zero temperature: only non-uphill moves accepted, matches the oracle
# ============================================================================================
@pytest.mark.parametrize("strategy,layout", STRAT_LAYOUT)
def test_cpu_zero_temperature(strategy, layout):
    case = CASE_BUILDERS["cardinality_toy"]()
    sched = S.zero(300)
    o, so, c, pc, sc = _cpu_pair(case, strategy, layout, n_rep_per_binding=3, seed=1)
    seed = 17
    for n in range(300):
        ro = o.step(so, n, sched, seed)
        rc = c.step(sc, n, sched, seed)
        assert np.array_equal(ro.accepted, rc.accepted)
        assert np.array_equal(ro.v <= 0.0, ro.accepted)  # tau == 0: accept iff v <= 0
    assert np.array_equal(so.rs.counters["uphill_accepted"], np.zeros(so.rs.n_replicas, dtype=np.int64))
    assert np.array_equal(sc.rs.counters["uphill_accepted"], np.zeros(sc.rs.n_replicas, dtype=np.int64))
    _assert_matches_oracle(case, so, sc, c)


# ============================================================================================
# penalty-style multi-slot bindings (extra "objective"/"violation" slots, score_structures path)
# ============================================================================================
@pytest.mark.parametrize("strategy,layout", STRAT_LAYOUT)
def test_cpu_multi_slot_bindings(strategy, layout):
    case = CASE_BUILDERS["cardinality_random"]()
    structure, w_run, c_run = random_polynomial(case.model.n_spins, len(case.structure.terms), 1, seed=99,
                                                duplicates=False, spins=range(case.model.n_spins))
    w_obj = np.asarray(case.weights[0])
    rng = np.random.default_rng(99)
    w_viol = np.round(rng.normal(size=w_obj.shape), 3)
    binding = bind_weights(case.model, w_obj, case.consts[0],
                           extra_slots={"objective": (w_obj, case.consts[0]), "violation": (w_viol, 0.5)})
    assert binding.slot_names == ("run", "objective", "violation")
    o, c = get_backend("oracle"), get_backend("cpu")
    po, pc = o.prepare(case.model, library=case.library), c.prepare(case.model, strategy=strategy, layout=layout)
    spins0 = case.initial(3, 2)
    so = o.init_state(po, binding, spins0, is_feasible=case.is_feasible)
    sc = c.init_state(pc, binding, spins0, is_feasible=case.is_feasible)
    sched = mixing_schedule(case, 300)
    seed = 71
    for n in range(300):
        ro = o.step(so, n, sched, seed)
        rc = c.step(sc, n, sched, seed)
        assert np.array_equal(ro.accepted, rc.accepted)
        assert np.array_equal(ro.changed, rc.changed)
    scale = float(sc.layout.scale.max())
    assert np.allclose(so.rs.energies, sc.rs.energies, atol=1e-9 * scale)
    assert np.allclose(so.rs.feasible_best_energy, sc.rs.feasible_best_energy, atol=1e-9 * scale, equal_nan=True)
    assert np.array_equal(so.rs.feasible_best_attempt, sc.rs.feasible_best_attempt)


# ============================================================================================
# corruption detection (audit helper)
# ============================================================================================
def test_cpu_validate_state_detects_corruption():
    case = CASE_BUILDERS["cardinality_random"]()
    c = get_backend("cpu")
    pc = c.prepare(case.model, strategy="incremental", layout="flat")
    spins0 = case.initial(2, 0)
    sc = c.init_state(pc, case.bindings[0], spins0, is_feasible=case.is_feasible)
    c.validate_state(sc)  # clean state: no error
    sched = mixing_schedule(case, 50)
    c.run(pc, sc, sched, seed=3)
    c.validate_state(sc)
    bad_T = sc.rs.T.copy()
    bad_T[0, 0] *= -1
    sc.rs.T = bad_T
    with pytest.raises(ArchitectureValidationError):
        c.validate_state(sc)
    sc.rs.T = bad_T  # undo via a fresh corruption of v instead, on a clean state
    sc.rs.T[0, 0] *= -1
    if sc.rs.v is not None:
        sc.rs.v[0, 0] = sc.rs.v[0, 0] + 10.0
        with pytest.raises(ArchitectureValidationError):
            c.validate_state(sc)


# ============================================================================================
# auto policy: deterministic, returns known strategy/layout names
# ============================================================================================
def test_cpu_auto_policy_is_reproducible():
    case = CASE_BUILDERS["section29"]()
    s1, l1 = choose_auto(case.model)
    s2, l2 = choose_auto(case.model)
    assert (s1, l1) == (s2, l2)
    assert s1 in STRATEGIES and l1 in LAYOUTS
    c = get_backend("cpu")
    pc = c.prepare(case.model, strategy="auto", layout="auto")
    assert pc.strategy == s1 and pc.layout == l1


def test_cpu_describe_and_resources():
    case = CASE_BUILDERS["cardinality_toy"]()
    c = get_backend("cpu")
    pc = c.prepare(case.model, strategy="recompute", layout="flat", threads=2)
    d = pc.describe()
    assert d["backend"] == "cpu" and d["strategy"] == "recompute" and d["layout"] == "flat"
    assert "summation_order" in d["options"]
    res = pc.resources()
    assert res["R"] == case.model.n_groups
    bd = c.describe()
    assert bd["backend"] == "cpu" and "numba" in bd

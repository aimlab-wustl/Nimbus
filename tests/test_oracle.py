"""Direct recomputation oracle: independent inputs, spin actions, acceptance law, bookkeeping and
penalty-run (feasible-best) semantics; lock-step with the emulation on a penalty baseline."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from helpers_core import all_states, direct_energy, toy_structure
from test_replay_core import CASE_BUILDERS, case_cardinality_toy, case_table_kempe
from nimbus.annealing import schedule as S
from nimbus.backends import get_backend
from nimbus.backends.oracle import direct_action, direct_clause_signs, _padded_supports
from nimbus.compiler import compile_model
from nimbus.core.errors import ConfigError
from nimbus.core.model import KIND_READOUT, ObjectiveStructure, bind_weights
from nimbus.transitions import builders as B


@pytest.mark.parametrize("name", list(CASE_BUILDERS))
def test_oracle_inputs_equal_direct_energy_differences(name):
    """v_r = (E(mu_r(s)) - E(s))/2 at every feasible state, with E from the raw term list."""
    case = CASE_BUILDERS[name]()
    o = get_backend("oracle")
    plan = o.prepare(case.model, library=case.library)
    for b in range(len(case.bindings)):
        st = o.init_state(plan, case.bindings[b], case.states)
        v = o.evaluate_inputs(st)
        E0 = case.energy(b, case.states)
        for r in range(case.model.n_groups):
            mu = case.library.groups[r].apply(case.states)
            dE = case.energy(b, mu) - E0
            assert np.allclose(2 * v[:, r], dE, rtol=0, atol=1e-9 * max(1.0, np.abs(E0).max()))
        # initial energies are direct recomputations from the spins (binding constants included)
        assert np.allclose(st.rs.energies[:, 0], E0, atol=1e-9)


@pytest.mark.parametrize("name", [n for n in CASE_BUILDERS if not n.startswith("table")])
def test_direct_action_matches_descriptors(name):
    case = CASE_BUILDERS[name]()
    model = case.model
    sup = _padded_supports(model)
    assert np.array_equal(direct_clause_signs(sup, case.states), model.clause_values(case.states))
    for r in range(model.n_groups):
        want = case.library.groups[r].apply(case.states)
        got = np.stack([direct_action(model, r, s) for s in case.states])
        assert np.array_equal(got, want), r
    # the oracle without the library gives the same inputs
    o = get_backend("oracle")
    v_lib = o.evaluate_inputs(o.init_state(o.prepare(model, library=case.library), case.bindings[0], case.states))
    v_arr = o.evaluate_inputs(o.init_state(o.prepare(model), case.bindings[0], case.states))
    assert np.array_equal(v_lib, v_arr)


def test_readout_groups_need_the_library():
    case = case_table_kempe()
    assert np.all(case.model.act_kind == KIND_READOUT)
    o = get_backend("oracle")
    with pytest.raises(ConfigError, match="library"):
        o.prepare(case.model)
    with pytest.raises(ConfigError):
        direct_action(case.model, 0, case.states[0])
    other = case_cardinality_toy()
    with pytest.raises(ConfigError):
        o.prepare(case.model, library=other.library)
    with pytest.raises(ConfigError):
        o.prepare(case.model, library=case.library, strategy="recompute")


def test_oracle_step_acceptance_law_and_counters():
    case = case_cardinality_toy()
    o = get_backend("oracle")
    plan = o.prepare(case.model, library=case.library)
    st = o.init_state(plan, case.bindings, case.initial(4, 2), [0, 0, 1, 1])
    sched = S.geometric(3.0, 0.05, 1500)
    seen_uphill = 0
    for n in range(1500):
        prev = st.rs.spins.copy()
        rec = o.step(st, n, sched, 4)
        tau = sched.tau(n)
        assert np.array_equal(rec.accepted, rec.v <= -(tau / 2) * np.log(rec.u))
        assert np.array_equal(rec.changed, np.any(prev != st.rs.spins, axis=1))
        assert not np.any(rec.changed & ~rec.accepted)
        seen_uphill += int(np.count_nonzero(rec.accepted & rec.changed & (rec.v > 1e-9)))
    c = st.rs.counters
    assert np.array_equal(c["attempts"], c["accepted"] + c["rejected"])
    assert np.array_equal(c["accepted"], c["accepted_changed"] + c["self_loops"])
    assert int(c["uphill_accepted"].sum()) == seen_uphill > 0
    for i in range(4):
        b = int(st.rs.replica_binding[i])
        assert st.rs.best_energy[i] == pytest.approx(case.energy(b, st.rs.best_spins[i:i + 1])[0])
        assert st.rs.best_energy[i] >= case.energy(b, case.states).min() - 1e-12
    with pytest.raises(ConfigError):
        o.step(st, 7, sched, 4)  # attempts cannot be repeated


def test_zero_temperature_oracle_is_monotone():
    case = CASE_BUILDERS["section29"]()
    o = get_backend("oracle")
    plan = o.prepare(case.model, library=case.library)
    st = o.init_state(plan, case.bindings[0], case.initial(3, 1))
    res = o.run(plan, st, S.zero(800), seed=2, recording={"profile": "trace", "trace_snapshots": ("spins",)})
    tr = res.recording.trace.columns()
    assert np.all(tr["v"][tr["accepted"]] <= 0.0) and np.all(tr["xi"] == 0.0)
    for i in range(3):
        e = tr["energy"][tr["replica"] == i]
        assert np.all(np.diff(e) <= 1e-12)
    assert np.all(res.state.counters["uphill_accepted"] == 0)
    assert res.recording.trace.columns()["snap_spins"].shape == (3 * 800, case.model.n_spins)
    assert res.recording.events is None
    res2 = o.run(plan, o.init_state(plan, case.bindings[0], case.initial(1)), S.zero(10), seed=2, recording="events")
    assert "not applicable" in res2.recording.notes["events"]
    json.dumps(res.summary())


# ============================================================================================
# penalty baseline on the same infrastructure
# ============================================================================================
def penalty_problem(lam: float = 2.0):
    """Toy objective + lam * (sum_i x_i - 3)^2 on 6 unconstrained spins with single flips.

    With x = (1 + s)/2 and m = sum_i s_i: (sum x - 3)^2 = m^2 / 4 = (6 + 2 sum_{i<j} s_i s_j) / 4.
    Slots: run = objective + penalty, objective = toy objective, violation = penalty (0 iff feasible).
    """
    obj, w_obj, c_obj = toy_structure()
    pairs = list(itertools.combinations(range(6), 2))
    pen = ObjectiveStructure(n_spins=6, terms=pairs)
    w_pen = np.full(len(pairs), -lam / 2.0)  # E_pen = lam*6/4 - sum J T with J = -lam/2
    c_pen = lam * 6 / 4.0
    run = ObjectiveStructure(n_spins=6, terms=list(obj.terms) + pairs)
    w_run = np.concatenate([w_obj, w_pen])
    model = compile_model(run, B.single_flips(6), score_structures={"objective": obj, "violation": pen})
    bnd = bind_weights(model, w_run, c_obj + c_pen,
                       extra_slots={"objective": (w_obj, c_obj), "violation": (w_pen, c_pen)})
    return obj, w_obj, c_obj, pen, w_pen, c_pen, model, bnd


def test_penalty_slots_are_consistent():
    obj, w_obj, c_obj, pen, w_pen, c_pen, model, bnd = penalty_problem()
    X = all_states(6)
    E_obj = direct_energy(obj, w_obj, c_obj, X)
    E_pen = direct_energy(pen, w_pen, c_pen, X)
    k = np.count_nonzero(X == 1, axis=1)
    assert np.allclose(E_pen, 2.0 * (k - 3) ** 2)
    assert np.allclose(model.energy(X, bnd, "objective"), E_obj)
    assert np.allclose(model.energy(X, bnd, "violation"), E_pen)
    assert np.allclose(model.energy(X, bnd, "run"), E_obj + E_pen)


def test_penalty_run_feasible_best_lockstep():
    obj, w_obj, c_obj, pen, w_pen, c_pen, model, bnd = penalty_problem()
    lib = B.single_flips(6)
    o, e = get_backend("oracle"), get_backend("emulation")
    po = o.prepare(model, library=lib)
    pe = e.prepare(model, validate="full", library=lib)
    X = all_states(6)
    rng = np.random.default_rng(0)
    s0 = X[rng.integers(0, 64, size=4)]
    E_obj_all = direct_energy(obj, w_obj, c_obj, X)
    feas = np.count_nonzero(X == 1, axis=1) == 3
    target = float(E_obj_all[feas].min())  # best feasible objective
    so = o.init_state(po, bnd, s0, targets=target)
    se = e.init_state(pe, bnd, s0, targets=target)
    sched = S.geometric(4.0, 0.2, 3000)
    visited_feasible_best = np.full(4, np.inf)
    first_hit = np.full(4, -1)
    for i in range(4):
        if np.count_nonzero(s0[i] == 1) == 3:
            visited_feasible_best[i] = direct_energy(obj, w_obj, c_obj, s0[i:i + 1])[0]
            if visited_feasible_best[i] <= target + 1e-9:
                first_hit[i] = 0
    for n in range(3000):
        ro = o.step(so, n, sched, 13)
        re = e.step(se, n, sched, 13)
        assert np.array_equal(ro.group, re.group) and np.array_equal(ro.accepted, re.accepted)
        assert np.array_equal(so.rs.spins, se.rs.spins)
        for i in np.flatnonzero(ro.changed):
            s = so.rs.spins[i]
            if np.count_nonzero(s == 1) == 3:
                eo = direct_energy(obj, w_obj, c_obj, s[None, :])[0]
                visited_feasible_best[i] = min(visited_feasible_best[i], eo)
                if first_hit[i] < 0 and eo <= target + 1e-9:
                    first_hit[i] = n + 1
    for st in (so, se):
        rs = st.rs
        assert np.allclose(rs.feasible_best_energy, visited_feasible_best)
        for i in range(4):
            if np.isfinite(rs.feasible_best_energy[i]):
                fs = rs.feasible_best_spins[i]
                assert np.count_nonzero(fs == 1) == 3  # feasible-best state has zero violation
                assert direct_energy(obj, w_obj, c_obj, fs[None, :])[0] == pytest.approx(rs.feasible_best_energy[i])
        assert np.array_equal(rs.counters["target_hit_attempt"], first_hit)
        # slot energies: run = objective + violation
        assert np.allclose(rs.energies[:, 0], rs.energies[:, 1] + rs.energies[:, 2])
    assert np.array_equal(so.rs.feasible_best_attempt, se.rs.feasible_best_attempt)
    assert np.any(first_hit > 0)

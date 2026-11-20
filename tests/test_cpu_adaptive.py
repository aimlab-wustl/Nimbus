"""CPU backend ``strategy='adaptive'`` (``nimbus.backends.cpu``): correctness against the
``'recompute'`` reference (same final T, energies, counters, best values) and evidence that
sub-window switching actually happens.

A backend plan may change how and when v_r is computed, never v_r, sigma or the toggles
themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_replay_core import CASE_BUILDERS, Case, mixing_schedule

from nimbus.annealing import schedule as S
from nimbus.backends import cpu as cpu_mod
from nimbus.backends import get_backend


def _cpu_pair_adaptive(case: Case, n_rep_per_binding=2, seed=0, layout="gate_blocked"):
    """(recompute-plan state, adaptive-plan state) on identical replicas/bindings."""
    nb = len(case.bindings)
    n = nb * n_rep_per_binding
    spins0 = case.initial(n, seed)
    rb = np.repeat(np.arange(nb), n_rep_per_binding)
    rid = np.tile(np.arange(n_rep_per_binding), nb)
    c = get_backend("cpu")
    p_ref = c.prepare(case.model, strategy="recompute", layout=layout)
    p_ada = c.prepare(case.model, strategy="adaptive", layout=layout)
    s_ref = c.init_state(p_ref, case.bindings, spins0, rb, rid, rb, is_feasible=case.is_feasible)
    s_ada = c.init_state(p_ada, case.bindings, spins0, rb, rid, rb, is_feasible=case.is_feasible)
    return c, p_ref, s_ref, p_ada, s_ada


@pytest.fixture
def tiny_subwindow(monkeypatch):
    """Force many sub-windows within a test-sized run (production default is 4096+)."""
    monkeypatch.setattr(cpu_mod, "ADAPTIVE_MIN_SUBWINDOW", 64)
    monkeypatch.setattr(cpu_mod, "ADAPTIVE_SUBWINDOW_DIVISOR", 1000000)


# ============================================================================================
# correctness: adaptive matches the 'recompute' reference, every replay family
# ============================================================================================
@pytest.mark.parametrize("name", list(CASE_BUILDERS))
def test_cpu_adaptive_matches_recompute(name, tiny_subwindow):
    case = CASE_BUILDERS[name]()
    sched = mixing_schedule(case, 3000)
    c, p_ref, s_ref, p_ada, s_ada = _cpu_pair_adaptive(case, n_rep_per_binding=2, seed=5)
    seed = 2024
    res_ref = c.run(p_ref, s_ref, sched, seed=seed)
    res_ada = c.run(p_ada, s_ada, sched, seed=seed)
    assert np.array_equal(res_ref.state.spins, res_ada.state.spins)
    assert np.array_equal(res_ref.state.T, res_ada.state.T)
    scale = float(s_ref.layout.scale.max())
    assert np.allclose(res_ref.state.energies, res_ada.state.energies, atol=1e-9 * scale)
    for k in res_ref.state.counters:
        assert np.array_equal(res_ref.state.counters[k], res_ada.state.counters[k]), (name, k)
    assert np.array_equal(res_ref.state.best_energy, res_ada.state.best_energy)
    assert np.array_equal(res_ref.state.best_attempt, res_ada.state.best_attempt)
    assert np.array_equal(res_ref.state.best_spins, res_ada.state.best_spins)
    # a strategy schedule was recorded and the plan is tagged 'adaptive'
    assert "strategy_schedule" in res_ada.metadata
    assert len(res_ada.metadata["strategy_schedule"]) >= 1
    assert res_ada.metadata["strategy_schedule"][0]["strategy"] == "recompute"  # forced first window


def test_cpu_adaptive_matches_recompute_flat_layout(tiny_subwindow):
    """Same correctness check with the explicit 'flat' layout (adaptive defaults to gate_blocked)."""
    case = CASE_BUILDERS["section29"]()
    sched = mixing_schedule(case, 3000)
    c, p_ref, s_ref, p_ada, s_ada = _cpu_pair_adaptive(case, n_rep_per_binding=2, seed=1, layout="flat")
    res_ref = c.run(p_ref, s_ref, sched, seed=77)
    res_ada = c.run(p_ada, s_ada, sched, seed=77)
    assert np.array_equal(res_ref.state.spins, res_ada.state.spins)
    assert np.array_equal(res_ref.state.T, res_ada.state.T)
    for k in res_ref.state.counters:
        assert np.array_equal(res_ref.state.counters[k], res_ada.state.counters[k])


# ============================================================================================
# a switch actually happens: a schedule running from high to low acceptance
# ============================================================================================
def test_cpu_adaptive_switches_strategy(tiny_subwindow):
    case = CASE_BUILDERS["onehot_P5"]()
    c = get_backend("cpu")
    plan = c.prepare(case.model, strategy="adaptive")
    spins0 = case.initial(2, 3)
    st = c.init_state(plan, case.bindings[0], spins0, is_feasible=case.is_feasible)
    # oracle-measured |dE| of proposed moves bounds a geometric schedule from (almost always
    # accept) to (almost never accept), the same construction mixing_schedule uses internally.
    o = get_backend("oracle")
    po = o.prepare(case.model, library=case.library)
    pilot = o.init_state(po, case.bindings[0], case.states[: min(40, case.states.shape[0])])
    dE = 2.0 * np.abs(o.evaluate_inputs(pilot)).ravel()
    pos = dE[dE > 1e-12]
    tau_hi, tau_lo = 50.0 * float(pos.max()), 0.01 * float(np.median(pos))
    sched = S.geometric(tau_hi, tau_lo, 40000)
    res = c.run(plan, st, sched, seed=13)
    strategies = {w["strategy"] for w in res.metadata["strategy_schedule"]}
    assert strategies == {"recompute", "incremental"}, res.metadata["strategy_schedule"]
    assert len(res.metadata["strategy_schedule"]) > 1


# ============================================================================================
# plan/state plumbing: layout defaults, strategy_code mutation, v allocation, describe()
# ============================================================================================
def test_cpu_adaptive_plan_defaults_and_describe():
    case = CASE_BUILDERS["cardinality_toy"]()
    c = get_backend("cpu")
    plan = c.prepare(case.model, strategy="adaptive")
    assert plan.strategy == "adaptive"
    assert plan.layout == "gate_blocked"  # adaptive default, independent of choose_auto
    assert plan.is_adaptive
    d = plan.describe()
    assert d["strategy"] == "adaptive"
    st = c.init_state(plan, case.bindings[0], case.initial(2, 0), is_feasible=case.is_feasible)
    assert st.rs.v is not None  # allocated even though the first sub-window is 'recompute'


def test_cpu_adaptive_unknown_strategy_still_rejected():
    case = CASE_BUILDERS["cardinality_toy"]()
    c = get_backend("cpu")
    with pytest.raises(Exception):
        c.prepare(case.model, strategy="not_a_strategy")


# ============================================================================================
# works through the diagnostics (sequential) path too, not just the prange summary path
# ============================================================================================
def test_cpu_adaptive_with_trace_recording_does_not_crash(tiny_subwindow):
    case = CASE_BUILDERS["cardinality_random"]()
    sched = mixing_schedule(case, 1500)
    c = get_backend("cpu")
    plan = c.prepare(case.model, strategy="adaptive")
    st = c.init_state(plan, case.bindings[0], case.initial(2, 2), is_feasible=case.is_feasible)
    res = c.run(plan, st, sched, seed=5, recording={"profile": "trace"})
    assert res.recording.trace is not None
    assert res.recording.trace.n_records > 0
    # the trace spans the whole requested window, not just the last internal sub-window
    attempts = res.recording.trace.columns()["attempt"]
    assert attempts.min() < 200
    assert attempts.max() > 1000

"""Work-queue replica scheduling, CPU backend. Goal: early-stopped replicas (stop_on_target, or an nfold freeze) free their prange
thread for the next replica instead of leaving it idle until the slowest thread's static chunk
finishes.

Semantics must not change: a replica's RNG is keyed by its
replica_id/instance_id and every replica's array row is independent, so the choice of which numba
thread executes which replica, and when, cannot affect the per-replica trajectory. This file checks
that ``scheduling="static"`` and ``scheduling="queue"`` give bit-identical results for every strategy
x layout, and for every proposal process, with per-replica targets staggered so replicas actually stop
at very different attempt counts (stop_on_target) -- exercising the early-exit/re-claim path, not the
no-op case where every replica runs to the same budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_replay_core import CASE_BUILDERS, mixing_schedule
from nimbus.annealing import schedule as S
from nimbus.backends import get_backend
from nimbus.backends.cpu import LAYOUTS, SCHEDULINGS, STRATEGIES, resolve_cpu_scheduling
from nimbus.experiments.recording import RecordingSpec

PROCESSES = ["select", "shared", "sweep", "parallel_trial", "nfold"]
CPU_EXTRA = {"parallel_trial": {"strategy": "incremental"}, "nfold": {"strategy": "incremental"}}


# ============================================================================================
# host-only: the pure auto-scheduling decision
# ============================================================================================
@pytest.mark.parametrize("scheduling", SCHEDULINGS)
def test_resolve_cpu_scheduling_values(scheduling):
    assert resolve_cpu_scheduling(scheduling, stop_on_target=False, process_name="select") in ("static", "queue")


def test_resolve_cpu_scheduling_auto_picks_queue_on_stop_on_target_or_nfold():
    assert resolve_cpu_scheduling("auto", stop_on_target=True, process_name="select") == "queue"
    assert resolve_cpu_scheduling("auto", stop_on_target=False, process_name="nfold") == "queue"
    assert resolve_cpu_scheduling("auto", stop_on_target=False, process_name="select") == "static"


def test_resolve_cpu_scheduling_explicit_overrides_auto():
    assert resolve_cpu_scheduling("static", stop_on_target=True, process_name="nfold") == "static"
    assert resolve_cpu_scheduling("queue", stop_on_target=False, process_name="select") == "queue"


def test_resolve_cpu_scheduling_rejects_unknown():
    with pytest.raises(Exception):
        resolve_cpu_scheduling("bogus", stop_on_target=False, process_name="select")


def test_cpu_plan_scheduling_rejects_unknown():
    cpu = get_backend("cpu")
    case = CASE_BUILDERS["cardinality_toy"]()
    with pytest.raises(Exception):
        cpu.prepare(case.model, scheduling="bogus")


# ============================================================================================
# bit-identical results: static vs queue, staggered targets so stop_on_target stops replicas at
# very different attempts
# ============================================================================================
def _dyadic_replicas(case, n, seed=1):
    spins0 = case.initial(n, seed)
    return spins0, np.ones(n, dtype=np.int64), np.arange(n)


def _staggered_targets(model, bindings, spins0, rb, rid, sched, n_att, n_rep, process, extra):
    """Replica i targets the best energy ``process`` reached (on 'static' scheduling, seed 11) by
    attempt stops[i % len(stops)] -- a spread of checkpoints, so stop_on_target stops replicas at very
    different attempts instead of all at the same one."""
    stops = sorted({max(1, n_att // k) for k in (2, 3, 5, 8, 13)})
    cpu = get_backend("cpu")
    plan = cpu.prepare(model, process=process, scheduling="static", **extra)
    st = cpu.init_state(plan, bindings, spins0, rb, rid, rb)
    spec = RecordingSpec(profile="anytime", anytime_checkpoints=stops)
    res = cpu.run(plan, st, sched, seed=11, stop=n_att, recording=spec)
    best = res.recording.anytime.best  # [n_rep, n_checkpoints]
    return np.array([best[rep, rep % len(stops)] for rep in range(n_rep)])


def _run_both_schedulings(model, bindings, spins0, rb, rid, sched, n_att, n_rep, process, extra):
    targets = _staggered_targets(model, bindings, spins0, rb, rid, sched, n_att, n_rep, process, extra)
    results = {}
    for scheduling in ("static", "queue"):
        cpu = get_backend("cpu")
        plan = cpu.prepare(model, process=process, scheduling=scheduling, **extra)
        st = cpu.init_state(plan, bindings, spins0, rb, rid, rb)
        results[scheduling] = cpu.run(plan, st, sched, seed=11, stop=n_att, targets=targets, stop_on_target=True)
    return results


def _assert_bit_identical(rs, rq, ctx):
    assert np.array_equal(rs.state.T, rq.state.T), ctx
    assert np.array_equal(rs.state.spins, rq.state.spins), ctx
    for k in rs.state.counters:
        assert np.array_equal(rs.state.counters[k], rq.state.counters[k]), (ctx, k)
    assert np.array_equal(rs.state.next_attempt, rq.state.next_attempt), ctx
    assert np.array_equal(rs.state.best_attempt, rq.state.best_attempt), ctx
    assert np.array_equal(rs.state.best_spins, rq.state.best_spins), ctx
    assert np.array_equal(rs.state.stopped, rq.state.stopped), ctx
    assert np.allclose(rs.state.energies, rq.state.energies, atol=1e-9), ctx
    assert rs.termination == rq.termination, ctx
    if rs.state.v is not None:
        np.testing.assert_array_equal(rs.state.v, rq.state.v)
    stopped_attempts = rs.state.next_attempt[rs.state.stopped]
    assert stopped_attempts.size == 0 or len(set(stopped_attempts.tolist())) > 1, ctx


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("case_name", ["cardinality_random", "onehot_P4", "parity_masks", "section29"])
def test_cpu_scheduling_static_vs_queue_bit_identical_select(case_name, strategy, layout):
    case = CASE_BUILDERS[case_name]()
    n_rep, n_att = 32, 300
    spins0, rb, rid = _dyadic_replicas(case, n_rep)
    sched = mixing_schedule(case, n_att)
    results = _run_both_schedulings(case.model, case.bindings, spins0, rb, rid, sched, n_att, n_rep, "select",
                                    {"strategy": strategy, "layout": layout})
    _assert_bit_identical(results["static"], results["queue"], (case_name, strategy, layout))


@pytest.mark.parametrize("process", PROCESSES)
@pytest.mark.parametrize("case_name", ["cardinality_random", "onehot_P4"])
def test_cpu_scheduling_static_vs_queue_bit_identical_processes(case_name, process):
    case = CASE_BUILDERS[case_name]()
    n_rep, n_att = 32, 300
    spins0, rb, rid = _dyadic_replicas(case, n_rep)
    sched = mixing_schedule(case, n_att)
    results = _run_both_schedulings(case.model, case.bindings, spins0, rb, rid, sched, n_att, n_rep, process,
                                    CPU_EXTRA.get(process, {}))
    _assert_bit_identical(results["static"], results["queue"], (case_name, process))


@pytest.mark.parametrize("shared_impl", ["lockstep", "per_replica"])
def test_cpu_scheduling_shared_lockstep_is_always_static(shared_impl):
    """The lock-step shared implementation processes several replicas per SIMD-like lane group, not
    independent prange-over-replicas work; it must always run (and agree) with 'static', whatever
    'scheduling' is requested."""
    case = CASE_BUILDERS["cardinality_random"]()
    n_rep, n_att = 16, 200
    spins0, rb, rid = _dyadic_replicas(case, n_rep)
    sched = mixing_schedule(case, n_att)
    results = {}
    for scheduling in ("static", "queue"):
        cpu = get_backend("cpu")
        plan = cpu.prepare(case.model, process="shared", shared_impl=shared_impl, scheduling=scheduling)
        st = cpu.init_state(plan, case.bindings, spins0, rb, rid, rb)
        results[scheduling] = cpu.run(plan, st, sched, seed=7, stop=n_att)
    _assert_bit_identical(results["static"], results["queue"], shared_impl)
    if shared_impl == "lockstep":
        assert results["queue"].metadata["plan"]["options"]["process_impl"] == "shared_lockstep"


def test_cpu_scheduling_nfold_auto_is_queue_without_targets():
    """nfold can freeze early even with no target at all (tau = 0 local minimum): 'auto' must still
    pick 'queue' and results must match 'static' bit for bit."""
    case = CASE_BUILDERS["cardinality_toy"]()
    n_rep = 16
    spins0, rb, rid = _dyadic_replicas(case, n_rep)
    sched = S.constant(0.0, 400)
    results = {}
    for scheduling in ("static", "queue", "auto"):
        cpu = get_backend("cpu")
        plan = cpu.prepare(case.model, process="nfold", strategy="incremental", scheduling=scheduling)
        st = cpu.init_state(plan, case.bindings, spins0, rb, rid, rb)
        results[scheduling] = cpu.run(plan, st, sched, seed=3)
    assert results["auto"].metadata["scheduling"] == "queue"
    _assert_bit_identical(results["static"], results["queue"], "nfold-auto")
    _assert_bit_identical(results["static"], results["auto"], "nfold-auto-vs-auto")
    assert "frozen" in results["static"].termination


def test_cpu_scheduling_adaptive_static_vs_queue_bit_identical():
    case = CASE_BUILDERS["cardinality_random"]()
    n_rep, n_att = 32, 2000
    spins0, rb, rid = _dyadic_replicas(case, n_rep)
    sched = mixing_schedule(case, n_att)
    results = _run_both_schedulings(case.model, case.bindings, spins0, rb, rid, sched, n_att, n_rep, "select",
                                    {"strategy": "adaptive"})
    _assert_bit_identical(results["static"], results["queue"], "adaptive")

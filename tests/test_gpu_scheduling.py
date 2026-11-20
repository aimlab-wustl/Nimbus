"""Work-queue (persistent) replica scheduling, GPU backend. Goal: early-stopped replicas (stop_on_target, or an nfold freeze) free their worker for
the next replica instead of the static one-replica-per-worker launch leaving it idle.

Semantics must not change: a replica's RNG is keyed by its
replica_id/instance_id, so its trajectory, counters, best/feasible-best values and termination are
independent of which worker executes it and when. This file checks that ``scheduling="static"`` and
``scheduling="queue"`` give bit-identical results for every kernel variant, both strategies and every
proposal process, with per-replica targets staggered so replicas actually stop at very different
attempt counts within the run (exercising the early-exit/re-claim path, not just the no-op case where
every replica runs to the budget).

``@pytest.mark.gpu``: executed only with a CUDA device (skipped otherwise).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from test_replay_core import CASE_BUILDERS, mixing_schedule
from nimbus.annealing import schedule as S
from nimbus.backends import get_backend
from nimbus.backends.gpu import SCHEDULINGS, cuda_available, resolve_scheduling
from nimbus.compiler import compile_model
from nimbus.core.model import bind_weights

HAVE_GPU = cuda_available()
gpu_only = pytest.mark.skipif(not HAVE_GPU, reason="not executed: no CUDA device")

GPU_CFGS = {
    "thread-rec": dict(kernel_variant="thread", strategy="recompute"),
    "thread-inc": dict(kernel_variant="thread", strategy="incremental"),
    "warp-rec": dict(kernel_variant="warp", strategy="recompute"),
    "warp-inc": dict(kernel_variant="warp", strategy="incremental", t_memory="shared"),
    "block-rec": dict(kernel_variant="block", strategy="recompute", block_threads=64),
    "block-inc": dict(kernel_variant="block", strategy="incremental"),
}
PROCESSES = ["select", "shared", "sweep", "parallel_trial", "nfold"]


# ============================================================================================
# host-only: the pure auto-scheduling decision (no device needed)
# ============================================================================================
@pytest.mark.parametrize("scheduling", SCHEDULINGS)
def test_resolve_scheduling_values(scheduling):
    assert resolve_scheduling(scheduling, stop_on_target=False, process="select") in ("static", "queue")


def test_resolve_scheduling_auto_picks_queue_on_stop_on_target_or_nfold():
    assert resolve_scheduling("auto", stop_on_target=True, process="select") == "queue"
    assert resolve_scheduling("auto", stop_on_target=False, process="nfold") == "queue"
    assert resolve_scheduling("auto", stop_on_target=False, process="select") == "static"
    assert resolve_scheduling("auto", stop_on_target=True, process="nfold") == "queue"


def test_resolve_scheduling_explicit_overrides_auto_heuristic():
    assert resolve_scheduling("static", stop_on_target=True, process="nfold") == "static"
    assert resolve_scheduling("queue", stop_on_target=False, process="select") == "queue"


def test_resolve_scheduling_rejects_unknown():
    with pytest.raises(Exception):
        resolve_scheduling("bogus", stop_on_target=False, process="select")


# ============================================================================================
# device: staggered per-replica targets so stop_on_target ends replicas at very different attempts
# ============================================================================================
def _cpu_anytime_best(model, bindings, spins0, rb, rid, sched, n_att, checkpoints, process, opts, cpu_extra):
    from nimbus.experiments.recording import RecordingSpec

    cpu = get_backend("cpu")
    plan = cpu.prepare(model, process=process, process_options=opts, **cpu_extra)
    st = cpu.init_state(plan, bindings, spins0, rb, rid, rb)
    spec = RecordingSpec(profile="anytime", anytime_checkpoints=checkpoints)
    res = cpu.run(plan, st, sched, seed=11, stop=n_att, recording=spec)
    return res.recording.anytime.best  # [n_rep, n_checkpoints]


def _dyadic_replicas(case, n, seed=1):
    spins0 = case.initial(n, seed)
    return spins0, np.ones(n, dtype=np.int64), np.arange(n)


CPU_EXTRA = {"parallel_trial": {"strategy": "incremental"}, "nfold": {"strategy": "incremental"}}


def _staggered_targets(model, bindings, spins0, rb, rid, sched, n_att, n_rep, process, opts):
    """One target per replica, cycling over a spread of attempt counts (reached via the CPU's own
    anytime-recorded best energy for this exact process): replica i targets the best energy that
    process reached by attempt stops[i % len(stops)], so, with stop_on_target, replicas stop at very
    different attempts instead of all running to the same budget."""
    stops = sorted({max(1, n_att // k) for k in (2, 3, 5, 8, 13)})
    best = _cpu_anytime_best(model, bindings, spins0, rb, rid, sched, n_att, stops, process, opts,
                             CPU_EXTRA.get(process, {}))
    return np.array([best[rep, rep % len(stops)] for rep in range(n_rep)])


def _run_both_schedulings(model, bindings, spins0, rb, rid, sched, n_att, n_rep, gcfg, process, opts):
    """Runs the GPU backend with scheduling='static' then 'queue' from identical fresh states, with
    per-replica targets staggered so stop_on_target stops replicas at very different attempts."""
    targets = _staggered_targets(model, bindings, spins0, rb, rid, sched, n_att, n_rep, process, opts)
    results = {}
    for scheduling in ("static", "queue"):
        g = get_backend("gpu")
        p = g.prepare(model, process=process, process_options=opts, scheduling=scheduling, **gcfg)
        st = g.init_state(p, bindings, spins0, rb, rid, rb)
        results[scheduling] = g.run(p, st, sched, seed=11, stop=n_att, targets=targets, stop_on_target=True)
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
    # the point of the test: stop_on_target really did stop replicas at different attempts (otherwise
    # this would not exercise the early re-claim path at all)
    stopped_attempts = rs.state.next_attempt[rs.state.stopped]
    assert stopped_attempts.size == 0 or len(set(stopped_attempts.tolist())) > 1, ctx


@pytest.mark.gpu
@gpu_only
@pytest.mark.parametrize("gcfg", list(GPU_CFGS))
@pytest.mark.parametrize("process", PROCESSES)
@pytest.mark.parametrize("case_name", ["cardinality_random", "onehot_P4", "parity_masks"])
def test_gpu_scheduling_static_vs_queue_bit_identical(case_name, process, gcfg):
    case = CASE_BUILDERS[case_name]()
    n_rep, n_att = 24, 300
    spins0, rb, rid = _dyadic_replicas(case, n_rep)
    sched = mixing_schedule(case, n_att)
    results = _run_both_schedulings(case.model, case.bindings, spins0, rb, rid, sched, n_att, n_rep,
                                    GPU_CFGS[gcfg], process, None)
    assert results["static"].metadata["kernel"]["scheduling"] == "static"
    assert results["queue"].metadata["kernel"]["scheduling"] == "queue"
    assert results["queue"].metadata["kernel"]["persistent"] is True
    _assert_bit_identical(results["static"], results["queue"], (case_name, process, gcfg))


def _matching_small():
    from nimbus.problems.factory import make_problem
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    problem = make_problem({"kind": "matching", "tensor": str(fixtures / "matching" / "five.npz")})
    lib = problem.transition_library("complete")
    model = compile_model(problem.structure(), lib, identities=tuple(problem.identities()))
    return problem, model, bind_weights(model, problem.term_weights(0), problem.const(0))


@pytest.mark.gpu
@gpu_only
@pytest.mark.parametrize("gcfg", ["thread-rec", "warp-inc", "block-rec"])
@pytest.mark.parametrize("process", ["select", "nfold"])
def test_gpu_scheduling_static_vs_queue_bit_identical_matching(process, gcfg):
    problem, model, binding = _matching_small()
    n_rep, n_att = 32, 600
    spins0 = problem.random_feasible(40, n_rep, 0)
    rb = np.zeros(n_rep, dtype=np.int64)
    rid = np.arange(n_rep)
    cpu = get_backend("cpu")
    plan0 = cpu.prepare(model)
    st0 = cpu.init_state(plan0, [binding], spins0, rb, rid, rb)
    dE = 2 * np.abs(cpu.evaluate_inputs(st0)).ravel()
    med = float(np.median(dE[dE > 0]))
    sched = S.fn_endpoints(med / math.log(1 / 0.8), med / math.log(100), 0.1, n_att)
    results = _run_both_schedulings(model, [binding], spins0, rb, rid, sched, n_att, n_rep, GPU_CFGS[gcfg],
                                    process, None)
    _assert_bit_identical(results["static"], results["queue"], (process, gcfg, "matching"))


@pytest.mark.gpu
@gpu_only
def test_gpu_scheduling_nfold_auto_is_queue_without_targets():
    """nfold can freeze early even with no target at all (tau = 0 local minimum): 'auto' must still
    pick 'queue' (resolve_scheduling), and results must match 'static' bit for bit."""
    case = CASE_BUILDERS["cardinality_toy"]()
    n_rep = 16
    spins0, rb, rid = _dyadic_replicas(case, n_rep)
    sched = S.constant(0.0, 400)
    results = {}
    for scheduling in ("static", "queue", "auto"):
        g = get_backend("gpu")
        p = g.prepare(case.model, process="nfold", strategy="incremental", scheduling=scheduling)
        st = g.init_state(p, case.bindings, spins0, rb, rid, rb)
        results[scheduling] = g.run(p, st, sched, seed=3)
    assert results["auto"].metadata["kernel"]["scheduling"] == "queue"
    _assert_bit_identical(results["static"], results["queue"], "nfold-auto")
    _assert_bit_identical(results["static"], results["auto"], "nfold-auto-vs-auto")
    assert "frozen" in results["static"].termination

"""Proposal processes on the GPU backend.

* Host-only: transcriptions of the kernel's process logic (kernels/anneal.cu) against the CPU
  kernels and the RNG references: the sweep Fisher-Yates word layout (Philox blocks, word pairs
  (2t, 2t+1)), the parallel_trial arbiter over a fired bitmask (popcount, j-th set bit), and the nfold
  sequential prefix sums with the "last non-empty interval" fallback and the geometric clock.
* ``@pytest.mark.gpu``: every process (and its options) x kernel variant x input strategy, through
  chunked runs, equals the CPU backend exactly (clause states, every counter including steps /
  candidate_evaluations / frozen_attempt, clocks, best assignments, terminations) on the replay families
  with their dyadic binding and on a small matching instance.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from test_replay_core import CASE_BUILDERS, mixing_schedule
from nimbus.annealing import schedule as S
from nimbus.backends import _numba_common as nc
from nimbus.backends import _numba_process as PK
from nimbus.backends import get_backend
from nimbus.backends.gpu import cuda_available
from nimbus.compiler import compile_model
from nimbus.core import rng
from nimbus.core.model import bind_weights

HAVE_GPU = cuda_available()
gpu_only = pytest.mark.skipif(not HAVE_GPU, reason="not executed: no CUDA device")
FIXTURES = Path(__file__).parent / "fixtures"


# ============================================================================================
# host transcriptions of the kernel's process logic
# ============================================================================================
def _kernel_sweep_perm(seed, s, cid, iid, R):
    """Kernel: Philox block t >> 1 of counter (s, t >> 1, cohort, instance); t even -> (x0, x1), odd -> (x2, x3)."""
    perm = list(range(R))
    x = None
    for t in range(R - 1):
        if t % 2 == 0:
            x = rng.philox4x32_scalar(s, t >> 1, cid, iid, seed, rng.SWEEP)
        i = R - 1 - t
        j = rng.group_index(x[2], x[3], i + 1) if t % 2 else rng.group_index(x[0], x[1], i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


@pytest.mark.parametrize("R", [1, 2, 3, 7, 136, 501])
def test_kernel_sweep_layout_equals_reference(R):
    for s, cid, iid in [(0, 0, 0), (5, 3, 1), (2 ** 32 - 1, 7, 2)]:
        assert _kernel_sweep_perm(99, s, cid, iid, R) == rng.sweep_permutation(99, s, cid, iid, R).tolist()


def _kernel_pt_decide(seed, n, rid, iid, R, tau, v, cand):
    """Kernel phase A/B of parallel_trial: fired bitmask words, popcount, arbiter, j-th set bit."""
    fw = (R + 31) // 32
    fb = [0] * fw
    n_eval = 0
    for r in range(R):
        if not cand[r]:
            continue
        n_eval += 1
        fired = v[r] <= 0.0 or (tau > 0.0 and v[r] <= -(tau * 0.5) * math.log(rng.pt_uniform(seed, n, r, rid, iid)))
        if fired:
            fb[r >> 5] |= 1 << (r & 31)
    nf = sum(bin(w).count("1") for w in fb)
    if nf == 0:
        return -1, n_eval, 0
    j = rng.pt_arbiter(seed, n, rid, iid, nf)
    for w in range(fw):
        c = bin(fb[w]).count("1")
        if j >= c:
            j -= c
            continue
        bits = fb[w]
        for _ in range(j):
            bits &= bits - 1
        return w * 32 + ((bits & -bits).bit_length() - 1), n_eval, nf


def _kernel_nfold_decide(seed, k, n, rid, iid, R, tau, v, cand, B, attempts_clock):
    """Kernel phase A/B of nfold: a_r (-1 = non-candidate), two sequential passes, fallback, W."""
    a = [(-1.0 if not cand[r] else (1.0 if v[r] <= 0.0 else (math.exp((-2.0 * v[r]) / tau) if tau > 0.0 else 0.0)))
         for r in range(R)]
    A, m = 0.0, 0
    for r in range(R):
        if a[r] >= 0.0:
            A += a[r]
            m += 1
    if m == 0 or A == 0.0:
        return PK.OUT_FROZEN, -1, 0, m
    u_sel, u_w = rng.nfold_uniforms(seed, k, rid, iid)
    target = u_sel * A
    S_, S_prev, last_inc, dec = 0.0, 0.0, -1, -1
    for r in range(R):
        if a[r] < 0.0:
            continue
        S_ += a[r]
        if S_ > target:
            dec = r
            break
        if S_ > S_prev or last_inc < 0:
            last_inc = r
        S_prev = S_
    if dec < 0:
        dec = last_inc
    if not attempts_clock:
        return PK.OUT_COMMIT, dec, 1, m
    p = A / R
    if p >= 1.0:
        return PK.OUT_COMMIT, dec, 1, m
    q = math.log(u_w) / math.log1p(-p)
    if q >= float(B - n):
        return PK.OUT_OVERSHOOT, dec, B - n, m
    return PK.OUT_COMMIT, dec, int(math.floor(q)) + 1, m


def _random_inputs(g, R):
    v = g.choice([-2.0, -0.5, 0.0, 0.25, 0.5, 1.0, 3.0, 40.0], size=R)
    cand = g.random(R) < 0.8
    return v, cand


def test_kernel_pt_decide_equals_cpu():
    g = np.random.default_rng(1)
    for trial in range(400):
        R = int(g.integers(1, 90))
        v, cand = _random_inputs(g, R)
        tau = float(g.choice([0.0, 0.3, 1.0, 5.0]))
        n, rid, iid = int(g.integers(0, 2 ** 32)), int(g.integers(0, 100)), int(g.integers(0, 4))
        cb = np.zeros(R, dtype=np.int64)
        oc = cand.astype(np.int64)
        want = PK.pt_decide(7, n, rid, iid, R, tau, v.astype(np.float64), oc, True, cb)
        assert _kernel_pt_decide(7, n, rid, iid, R, tau, v, cand) == tuple(int(x) for x in want), trial


def test_kernel_nfold_decide_equals_cpu():
    g = np.random.default_rng(2)
    for trial in range(400):
        R = int(g.integers(1, 90))
        v, cand = _random_inputs(g, R)
        if trial % 7 == 0:  # tiny weights: rounding at the end of the prefix sums, zero intervals
            v = np.where(g.random(R) < 0.5, 800.0, v)
        tau = float(g.choice([0.0, 0.3, 1.0, 5.0]))
        k, n, rid, iid = int(g.integers(0, 2 ** 32)), int(g.integers(0, 1000)), int(g.integers(0, 100)), 0
        B = int(n + g.integers(1, 5000))
        for clock in (True, False):
            cb = np.zeros(R, dtype=np.int64)
            pb = np.zeros(R, dtype=np.float64)
            want = PK.nfold_decide(7, k, n, rid, iid, R, tau, v.astype(np.float64), cand.astype(np.int64), cb, pb,
                                   B, clock)
            got = _kernel_nfold_decide(7, k, n, rid, iid, R, tau, v, cand, B, clock)
            assert got == tuple(int(x) for x in want), (trial, clock)


def test_kernel_shared_draw_equals_reference():
    g = np.random.default_rng(3)
    for _ in range(300):
        n, cid, iid, R = int(g.integers(0, 2 ** 40)), int(g.integers(0, 50)), int(g.integers(0, 4)), int(g.integers(1, 600))
        x = rng.philox4x32_scalar(n & 0xFFFFFFFF, n >> 32, cid, iid, 5, rng.SHARED)
        assert rng.group_index(x[0], x[1], R) == rng.shared_group(5, n, cid, iid, R) == nc.shared_group(5, n, cid, iid, R)


# ============================================================================================
# device replay vs the CPU backend
# ============================================================================================
PROCESS_VARIANTS = [
    ("shared", None), ("shared", {"cohort_size": 2}), ("sweep", None), ("sweep", {"cohort_size": 2}),
    ("parallel_trial", None), ("parallel_trial", {"pt_exclude_self_loops": False}),
    ("nfold", None), ("nfold", {"nfold_clock": "steps"}), ("select", None),
]
GPU_OPTS = {
    "thread-rec": dict(kernel_variant="thread", strategy="recompute"),
    "thread-inc": dict(kernel_variant="thread", strategy="incremental"),
    "warp-rec": dict(kernel_variant="warp", strategy="recompute"),
    "warp-inc": dict(kernel_variant="warp", strategy="incremental", t_memory="shared"),
    "block-rec": dict(kernel_variant="block", strategy="recompute", block_threads=64),
    "block-inc": dict(kernel_variant="block", strategy="incremental"),
    "thread-minor": dict(kernel_variant="thread", strategy="recompute", t_layout="minor"),
    "thread-minor-smem-inc": dict(kernel_variant="thread", strategy="incremental", t_layout="minor",
                                  t_memory="shared"),
}
CPU_EXTRA = {"parallel_trial": {"strategy": "incremental"}, "nfold": {"strategy": "incremental"}}
_CPU_REF: dict = {}
STOPS = (60, 150, 151, 300)


def _dyadic_replicas(case, n=6, seed=1):
    spins0 = case.initial(n, seed)
    return spins0, np.ones(n, dtype=np.int64), np.arange(n)  # binding 1: dyadic in every family


def _cpu_reference(case_name, proc, opts):
    key = (case_name, proc, repr(opts))
    if key not in _CPU_REF:
        case = CASE_BUILDERS[case_name]()
        sched = mixing_schedule(case, 300)
        spins0, rb, rid = _dyadic_replicas(case)
        c = get_backend("cpu")
        pc = c.prepare(case.model, process=proc, process_options=opts, **CPU_EXTRA.get(proc, {}))
        sc = c.init_state(pc, case.bindings, spins0, rb, rid, rb)
        _CPU_REF[key] = [c.run(pc, sc, sched, seed=5, stop=stop) for stop in STOPS]
    return _CPU_REF[key]


def _assert_same(rg, rc, ctx):
    assert np.array_equal(rg.state.T, rc.state.T), ctx
    for k in rc.state.counters:
        assert np.array_equal(rg.state.counters[k], rc.state.counters[k]), (ctx, k)
    assert np.array_equal(rg.state.next_attempt, rc.state.next_attempt), ctx
    assert np.array_equal(rg.state.best_attempt, rc.state.best_attempt), ctx
    assert np.array_equal(rg.state.best_spins, rc.state.best_spins), ctx
    assert np.allclose(rg.state.energies, rc.state.energies, atol=1e-9), ctx
    assert rg.termination == rc.termination, ctx


@pytest.mark.gpu
@gpu_only
@pytest.mark.parametrize("gcfg", list(GPU_OPTS))
@pytest.mark.parametrize("proc,opts", PROCESS_VARIANTS, ids=[f"{p}-{o}" if o else p for p, o in PROCESS_VARIANTS])
@pytest.mark.parametrize("case_name", list(CASE_BUILDERS))
def test_gpu_process_equals_cpu(case_name, proc, opts, gcfg):
    gopts = dict(GPU_OPTS[gcfg])
    if gopts.get("t_layout") == "minor" and proc not in ("shared", "select", "sweep"):
        gopts.pop("t_layout")
    case = CASE_BUILDERS[case_name]()
    sched = mixing_schedule(case, 300)
    spins0, rb, rid = _dyadic_replicas(case)
    refs = _cpu_reference(case_name, proc, opts)
    g = get_backend("gpu")
    pg = g.prepare(case.model, process=proc, process_options=opts, **gopts)
    sg = g.init_state(pg, case.bindings, spins0, rb, rid, rb)
    for stop, rc in zip(STOPS, refs):
        rg = g.run(pg, sg, sched, seed=5, stop=stop)
        assert rg.metadata["process"] == rc.metadata["process"]
        _assert_same(rg, rc, (case_name, proc, opts, gcfg, stop))


def _matching_small():
    from nimbus.problems.factory import make_problem

    problem = make_problem({"kind": "matching", "tensor": str(FIXTURES / "matching" / "five.npz")})
    lib = problem.transition_library("complete")
    model = compile_model(problem.structure(), lib, identities=tuple(problem.identities()))
    return problem, model, bind_weights(model, problem.term_weights(0), problem.const(0))


@pytest.mark.gpu
@gpu_only
@pytest.mark.parametrize("gcfg", ["thread-rec", "warp-inc", "block-rec", "block-inc", "thread-minor"])
@pytest.mark.parametrize("proc,opts", PROCESS_VARIANTS, ids=[f"{p}-{o}" if o else p for p, o in PROCESS_VARIANTS])
def test_gpu_process_equals_cpu_matching(proc, opts, gcfg):
    gopts = dict(GPU_OPTS[gcfg])
    if gopts.get("t_layout") == "minor" and proc not in ("shared", "select", "sweep"):
        gopts.pop("t_layout")
    problem, model, binding = _matching_small()
    spins0 = problem.random_feasible(3, 40, 0)
    c, g = get_backend("cpu"), get_backend("gpu")
    pc = c.prepare(model, process=proc, process_options=opts, **CPU_EXTRA.get(proc, {}))
    pg = g.prepare(model, process=proc, process_options=opts, **gopts)
    sc, sg = c.init_state(pc, binding, spins0), g.init_state(pg, binding, spins0)
    dE = 2 * np.abs(c.evaluate_inputs(sc)).ravel()
    med = float(np.median(dE[dE > 0]))
    sched = S.fn_endpoints(med / math.log(1 / 0.8), med / math.log(100), 0.1, 2000)
    for stop in (500, 1311, 2000):
        rc = c.run(pc, sc, sched, seed=9, stop=stop)
        rg = g.run(pg, sg, sched, seed=9, stop=stop)
        _assert_same(rg, rc, (proc, opts, gcfg, stop))
    assert all(problem.is_feasible(s) for s in rg.state.spins)


@pytest.mark.gpu
@gpu_only
@pytest.mark.parametrize("variant", ["thread", "warp", "block"])
def test_gpu_nfold_frozen_and_stop_on_target(variant):
    """tau = 0: nfold freezes at a local minimum (clock jumps to B, termination 'frozen'), as on the CPU."""
    case = CASE_BUILDERS["cardinality_toy"]()
    spins0, rb, rid = _dyadic_replicas(case)
    sched = S.constant(0.0, 400)
    out = {}
    for be, extra in (("cpu", {"strategy": "incremental"}), ("gpu", {"kernel_variant": variant})):
        b = get_backend(be)
        p = b.prepare(case.model, process="nfold", **extra)
        st = b.init_state(p, case.bindings, spins0, rb, rid, rb)
        out[be] = b.run(p, st, sched, seed=3)
    _assert_same(out["gpu"], out["cpu"], variant)
    assert "frozen" in out["gpu"].termination

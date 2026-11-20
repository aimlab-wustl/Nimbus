"""Proposal processes on the CPU backend.

* RNG streams 8-11: Numba versions bit for bit against the pure-Python references.
* ``select`` unchanged: covered by the existing replay/CPU tests (tests/test_replay_core.py,
  tests/test_cpu_backend.py); here the process counters added for select are checked.
* Exact trajectories: every process on the CPU equals an independent spin-level reference driver
  (library descriptors, direct energies, the RNG references) and the event-driven emulation (kept as a
  test-only reference), on the replay families and on a small matching instance, through chunked runs.
* Statistics on tiny enumerated problems at a fixed temperature (fixed seeds, chi-square):
  nfold and sweep stationary distributions are Boltzmann (and nfold matches select); the nfold
  attempt clock W is Geometric(A / R) and its embedded jump chain equals select's (exact matrix);
  parallel_trial one-step probabilities equal a_r E[1 / (1 + K_{-r})] (exact, by enumerating fired
  subsets), and its stationary law on the 2-spin ferromagnet is the non-Boltzmann one.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from helpers_core import direct_energy
from test_replay_core import CASE_BUILDERS, Case, case_cardinality_toy, case_table_kempe, mixing_schedule
from nimbus.annealing import schedule as S
from nimbus.backends import get_backend
from nimbus.backends import _numba_common as nc
from nimbus.backends.base import ProcessSpec
from nimbus.compiler import compile_model
from nimbus.core import rng
from nimbus.core.errors import ConfigError
from nimbus.core.model import ObjectiveStructure, bind_weights
from nimbus.transitions import builders as B

FIXTURES = Path(__file__).parent / "fixtures"
P_MIN = 1e-3  # chi-square acceptance threshold (fixed seeds: deterministic outcomes)

PROCESS_VARIANTS = [
    ("shared", None, {}), ("shared", {"cohort_size": 2}, {"shared_impl": "lockstep"}),
    ("sweep", None, {}), ("sweep", {"cohort_size": 2}, {}),
    ("parallel_trial", None, {"strategy": "incremental"}), ("parallel_trial", None, {"strategy": "recompute"}),
    ("parallel_trial", {"pt_exclude_self_loops": False}, {}),
    ("nfold", None, {"strategy": "incremental"}), ("nfold", None, {"strategy": "recompute"}),
    ("nfold", {"nfold_clock": "steps"}, {}),
]


# ============================================================================================
# RNG streams
# ============================================================================================
def test_process_streams_numba_vs_reference():
    g = np.random.default_rng(0)
    for _ in range(500):
        seed, c, i = (int(x) for x in g.integers(0, 2 ** 32, 3))
        n = int(g.integers(0, 2 ** 40))
        R = int(g.integers(1, 2 ** 31))
        k = int(g.integers(0, 2 ** 32))
        r = int(g.integers(0, R))
        assert rng.shared_group(seed, n, c, i, R) == nc.shared_group(seed, n, c, i, R)
        assert rng.pt_uniform(seed, k, r, c, i) == nc.pt_uniform(seed, k, r, c, i)
        assert rng.pt_arbiter(seed, k, c, i, R) == nc.pt_arbiter(seed, k, c, i, R)
        assert rng.nfold_uniforms(seed, k, c, i) == nc.nfold_uniforms(seed, k, c, i)
        Rs = int(g.integers(1, 200))
        perm = np.empty(Rs, np.int64)
        nc.sweep_permutation_into(seed, k, c, i, perm)
        ref = rng.sweep_permutation(seed, k, c, i, Rs)
        assert np.array_equal(perm, ref) and np.array_equal(np.sort(ref), np.arange(Rs))
        assert nc.solver_uniform(seed, n, c, i) == rng.solver_draw(seed, n, c, i, 7)[1]


def test_process_stream_layouts():
    """Counter layouts of the proposal processes, checked against raw Philox calls."""
    seed, R = 12345, 136
    x = rng.philox4x32_scalar(7, 0, 3, 5, seed, rng.SHARED)
    assert rng.shared_group(seed, 7, 3, 5, R) == rng.group_index(x[0], x[1], R)
    x = rng.philox4x32_scalar(9, 17, 2, 5, seed, rng.PT)
    assert rng.pt_uniform(seed, 9, 17, 2, 5) == rng.uniform52(x[0], x[1])
    x = rng.philox4x32_scalar(9, 0xFFFFFFFF, 2, 5, seed, rng.PT)
    assert rng.pt_arbiter(seed, 9, 2, 5, 11) == rng.group_index(x[0], x[1], 11)
    x = rng.philox4x32_scalar(4, 0, 2, 5, seed, rng.NFOLD)
    assert rng.nfold_uniforms(seed, 4, 2, 5) == (rng.uniform52(x[0], x[1]), rng.uniform52(x[2], x[3]))
    w = rng.sweep_words(seed, 6, 1, 5, 12)
    for b in range(3):
        assert tuple(int(v) for v in w[4 * b:4 * b + 4]) == rng.philox4x32_scalar(6, b, 1, 5, seed, rng.SWEEP)
    assert (rng.SHARED, rng.PT, rng.NFOLD, rng.SWEEP) == (8, 9, 10, 11)
    # sweep permutations: Fisher-Yates over the word pairs, uniform over permutations of 3 elements
    counts = {}
    for s in range(6000):
        p = tuple(rng.sweep_permutation(1, s, 0, 0, 3).tolist())
        counts[p] = counts.get(p, 0) + 1
    assert len(counts) == 6
    assert stats.chisquare(list(counts.values())).pvalue > P_MIN


def test_process_spec_validation():
    assert ProcessSpec.coerce().name == "select"
    assert ProcessSpec.coerce("nfold", {"nfold_clock": "steps"}).clock == "steps"
    assert ProcessSpec.coerce("parallel_trial").clock == "steps"
    with pytest.raises(ConfigError):
        ProcessSpec.coerce("bogus")
    with pytest.raises(ConfigError):
        ProcessSpec.coerce("nfold", {"cohort_size": 2})
    with pytest.raises(ConfigError):
        ProcessSpec.coerce("shared", {"cohort_size": 0})
    with pytest.raises(ConfigError):
        ProcessSpec.coerce("nfold", {"nfold_clock": "hours"})
    case = case_cardinality_toy()
    with pytest.raises(ConfigError):  # stepwise processes need the full library
        get_backend("emulation").prepare(case.model, capacity=4, process="nfold")
    with pytest.raises(ConfigError):
        get_backend("cpu").prepare(case.model, process="parallel_trial", strategy="lazy")
    with pytest.raises(ConfigError):
        get_backend("cpu").prepare(case.model, process="shared", strategy="adaptive")
    with pytest.raises(ConfigError):
        get_backend("oracle").prepare(case.model, process="shared")


# ============================================================================================
# independent spin-level reference driver
# ============================================================================================
class RefChain:
    """Process semantics on explicit spins: library descriptors mu_r, direct energies, RNG references."""

    def __init__(self, case: Case, binding: int):
        self.case, self.b = case, binding
        self.lib = case.library
        self.R = self.lib.n_groups

    def energy(self, s):
        return float(self.case.energy(self.b, np.asarray(s)[None, :])[0])

    def moves(self, s):
        """(mu_r(s) for every r, v_r = (E(mu_r s) - E(s)) / 2, state-changing mask)."""
        mus = [np.asarray(g.apply(s), dtype=np.int8) for g in self.lib.groups]
        E0 = self.energy(s)
        v = np.array([(self.energy(m) - E0) / 2.0 for m in mus])
        chg = np.array([not np.array_equal(m, s) for m in mus])
        return mus, v, chg

    def run(self, s0, proc: ProcessSpec, sched, seed, rid, iid, cid):
        """List of (clock after the step, spins) for every step until the budget."""
        s = np.asarray(s0, dtype=np.int8).copy()
        B = sched.n_total
        out = []
        n = k = 0
        while n < B:
            tau = sched.tau(n)
            mus, v, chg = self.moves(s)
            W = 1
            if proc.name in ("shared", "sweep"):
                u = rng.solver_draw(seed, n, rid, iid, self.R)[1]
                r = (rng.shared_group(seed, n, cid, iid, self.R) if proc.name == "shared"
                     else int(rng.sweep_groups(seed, [n], cid, iid, self.R)[0]))
                acc = v[r] <= (-(tau / 2.0) * math.log(u) if tau > 0 else 0.0)
                if acc:
                    s = mus[r]
            elif proc.name == "parallel_trial":
                cand = [r for r in range(self.R) if chg[r] or not proc.pt_exclude_self_loops]
                F = [r for r in cand if v[r] <= 0.0 or (tau > 0.0 and v[r] <= -(tau / 2.0) * math.log(
                    rng.pt_uniform(seed, n, r, rid, iid)))]
                if F:
                    s = mus[F[rng.pt_arbiter(seed, n, rid, iid, len(F))]]
            else:  # nfold
                cand = [r for r in range(self.R) if chg[r]]
                a = [1.0 if v[r] <= 0.0 else (math.exp(-2.0 * v[r] / tau) if tau > 0.0 else 0.0) for r in cand]
                A = sum(a) if a else 0.0
                if A == 0.0:
                    out.append((B, s.copy()))
                    break
                us, uw = rng.nfold_uniforms(seed, k, rid, iid)
                cum = np.cumsum(a)
                i = int(np.searchsorted(cum, us * A, side="right"))
                i = min(i, len(cand) - 1)
                r = cand[i]
                if proc.nfold_clock == "attempts":
                    p = A / self.R
                    if p < 1.0:
                        q = math.log(uw) / math.log1p(-p)
                        if q >= B - n:
                            out.append((B, s.copy()))
                            break
                        W = int(math.floor(q)) + 1
                s = mus[r]
            n += W
            k += 1
            out.append((n, s.copy()))
        return out


def _replicas(case: Case, per_binding: int, seed: int):
    nb = len(case.bindings)
    n = nb * per_binding
    return (case.initial(n, seed), np.repeat(np.arange(nb), per_binding), np.tile(np.arange(per_binding), nb))


@pytest.mark.parametrize("proc_name,opts,extra", PROCESS_VARIANTS)
@pytest.mark.parametrize("case_name", ["cardinality_toy", "table_kempe", "section29"])
def test_cpu_equals_reference_driver(case_name, proc_name, opts, extra):
    """CPU trajectories (chunked runs) equal the spin-level reference chain of every replica (dyadic
    weights, so architecture inputs and direct energy differences are exactly equal)."""
    case = CASE_BUILDERS[case_name]()
    b = 1  # dyadic weights in every family's second binding
    proc = ProcessSpec.coerce(proc_name, opts)
    sched = mixing_schedule(case, 240)
    spins0 = case.initial(4, 3)
    rid = np.arange(4)
    cpu = get_backend("cpu")
    plan = cpu.prepare(case.model, process=proc_name, process_options=opts, **extra)
    st = cpu.init_state(plan, case.bindings, spins0, np.full(4, b), rid, np.full(4, b))
    seed = 77
    ref = RefChain(case, b)
    from nimbus.backends.base import cohort_ids

    cids = cohort_ids(proc, rid)
    trajs = [ref.run(spins0[i], proc, sched, seed, int(rid[i]), b, int(cids[i])) for i in range(4)]
    for stop in (37, 120, 240):
        res = cpu.run(plan, st, sched, seed=seed, stop=stop)
        for i in range(4):
            # a run window executes every step that starts before ``stop`` (an nfold step may end beyond it)
            starts = [0] + [x[0] for x in trajs[i][:-1]]
            done = [x for x, n0 in zip(trajs[i], starts) if n0 < stop]
            want_n, want_s = done[-1] if done else (0, spins0[i])
            assert int(res.state.next_attempt[i]) == want_n, (case_name, proc_name, stop, i)
            assert np.array_equal(res.state.spins[i], want_s), (case_name, proc_name, stop, i)
            assert int(res.state.counters["steps"][i]) == len(done), (case_name, proc_name, stop, i)


@pytest.mark.parametrize("proc_name,opts,extra", PROCESS_VARIANTS)
@pytest.mark.parametrize("case_name", list(CASE_BUILDERS))
def test_cpu_equals_emulation(case_name, proc_name, opts, extra):
    """CPU and the event-driven emulation (test-only reference) agree exactly: clause states, counters,
    best bookkeeping and clocks after every chunk."""
    case = CASE_BUILDERS[case_name]()
    sched = mixing_schedule(case, 300)
    spins0, rb, rid = _replicas(case, 3, 1)
    e, c = get_backend("emulation"), get_backend("cpu")
    pe = e.prepare(case.model, process=proc_name, process_options=opts)
    pc = c.prepare(case.model, process=proc_name, process_options=opts, **extra)
    se = e.init_state(pe, case.bindings, spins0, rb, rid, rb, is_feasible=case.is_feasible)
    sc = c.init_state(pc, case.bindings, spins0, rb, rid, rb)
    for stop in (60, 150, 151, 300):
        re = e.run(pe, se, sched, seed=5, stop=stop)
        rc = c.run(pc, sc, sched, seed=5, stop=stop)
        assert np.array_equal(re.state.T, rc.state.T)
        for k in re.state.counters:
            assert np.array_equal(re.state.counters[k], rc.state.counters[k]), k
        assert np.array_equal(re.state.next_attempt, rc.state.next_attempt)
        assert np.array_equal(re.state.best_attempt, rc.state.best_attempt)
        assert np.array_equal(re.state.best_spins, rc.state.best_spins)
        assert re.termination == rc.termination
    assert all(case.is_feasible(s) for s in rc.state.spins)
    c.validate_state(sc)


def _matching_small():
    from nimbus.problems.factory import make_problem

    problem = make_problem({"kind": "matching", "tensor": str(FIXTURES / "matching" / "five.npz")})
    lib = problem.transition_library("complete")
    model = compile_model(problem.structure(), lib, identities=tuple(problem.identities()))
    return problem, model, bind_weights(model, problem.term_weights(0), problem.const(0))


@pytest.mark.parametrize("proc_name,opts,extra", PROCESS_VARIANTS)
def test_cpu_equals_emulation_matching(proc_name, opts, extra):
    problem, model, binding = _matching_small()
    spins0 = problem.random_feasible(3, 6, 0)
    e, c = get_backend("emulation"), get_backend("cpu")
    pe = e.prepare(model, process=proc_name, process_options=opts)
    pc = c.prepare(model, process=proc_name, process_options=opts, **extra)
    se = e.init_state(pe, binding, spins0)
    sc = c.init_state(pc, binding, spins0)
    dE = 2 * np.abs(c.evaluate_inputs(sc)).ravel()
    med = float(np.median(dE[dE > 0]))
    sched = S.fn_endpoints(med / math.log(1 / 0.8), med / math.log(100), 0.1, 2000)
    for stop in (500, 1311, 2000):
        re = e.run(pe, se, sched, seed=9, stop=stop)
        rc = c.run(pc, sc, sched, seed=9, stop=stop)
        assert np.array_equal(re.state.T, rc.state.T)
        for k in re.state.counters:
            assert np.array_equal(re.state.counters[k], rc.state.counters[k]), k
        assert np.array_equal(re.state.best_attempt, rc.state.best_attempt)
    assert all(problem.is_feasible(s) for s in rc.state.spins)


def test_cpu_recording_path_equals_fast_path():
    """The RunBook (anytime/trace) path of the processes reproduces the fast path."""
    case = case_cardinality_toy()
    sched = mixing_schedule(case, 400)
    spins0, rb, rid = _replicas(case, 2, 2)
    c = get_backend("cpu")
    for proc_name, opts, extra in PROCESS_VARIANTS:
        pc = c.prepare(case.model, process=proc_name, process_options=opts, **extra)
        s1 = c.init_state(pc, case.bindings, spins0, rb, rid, rb)
        s2 = c.init_state(pc, case.bindings, spins0, rb, rid, rb)
        r1 = c.run(pc, s1, sched, seed=3)
        r2 = c.run(pc, s2, sched, seed=3, recording={"profile": "anytime", "anytime_points": 8})
        assert np.array_equal(r1.state.T, r2.state.T), proc_name
        for k in r1.state.counters:
            assert np.array_equal(r1.state.counters[k], r2.state.counters[k]), (proc_name, k)
        assert np.array_equal(r1.state.best_attempt, r2.state.best_attempt)
        assert r2.recording.anytime is not None


def test_select_process_counters_and_resume():
    """select: steps = candidate_evaluations = attempts; checkpoints without the new counters still load."""
    case = case_cardinality_toy()
    sched = mixing_schedule(case, 200)
    spins0, rb, rid = _replicas(case, 2, 2)
    for name in ("cpu", "emulation"):
        be = get_backend(name)
        plan = be.prepare(case.model)
        st = be.init_state(plan, case.bindings, spins0, rb, rid, rb)
        res = be.run(plan, st, sched, seed=1, stop=120)
        c = res.state.counters
        assert np.array_equal(c["steps"], c["attempts"]) and np.array_equal(c["candidate_evaluations"], c["attempts"])
        assert np.all(c["frozen_attempt"] == -1)
        old = res.state.copy()
        for k in ("steps", "candidate_evaluations", "frozen_attempt"):
            del old.counters[k]
        st2 = be.load_state(plan, old)
        r2 = be.run(plan, st2, sched, seed=1)
        assert np.array_equal(r2.state.counters["steps"], r2.state.counters["attempts"])


def test_shared_cohorts_see_one_group_sequence():
    """Replicas of one cohort get identical r_n (rng.shared_group) and their own acceptance uniforms."""
    case = case_cardinality_toy()
    sched = mixing_schedule(case, 50)
    spins0 = case.initial(6, 4)
    c = get_backend("cpu")
    plan = c.prepare(case.model, process="shared", process_options={"cohort_size": 3})
    st = c.init_state(plan, case.bindings[0], spins0)
    for n in range(50):
        rec = c.step(st, n, sched, 11)
        for rep in range(6):
            assert rec.group[rep] == rng.shared_group(11, n, rep // 3, 0, case.model.n_groups)
            assert rec.u[rep] == rng.solver_draw(11, n, rep, 0, case.model.n_groups)[1]
        assert len(set(rec.u.tolist())) == 6
    # sweep: every R consecutive attempts visit every group once (per replica permutation)
    plan = c.prepare(case.model, process="sweep")
    st = c.init_state(plan, case.bindings[0], spins0)
    R = case.model.n_groups
    groups = np.array([c.step(st, n, sched, 11).group for n in range(2 * R)])
    for rep in range(6):
        for sw in range(2):
            assert sorted(groups[sw * R:(sw + 1) * R, rep].tolist()) == list(range(R))
    assert not np.array_equal(groups[:R, 0], groups[:R, 1])


# ============================================================================================
# statistics on tiny enumerated problems (fixed tau)
# ============================================================================================
def _enumerate(case: Case, b: int, tau: float):
    """States, Boltzmann law, select jump matrix and per-state move data at temperature tau."""
    states = [tuple(s) for s in case.states.tolist()]
    index = {s: i for i, s in enumerate(states)}
    E = case.energy(b, case.states)
    boltz = np.exp(-(E - E.min()) / tau)
    boltz /= boltz.sum()
    R = case.library.n_groups
    jump = np.zeros((len(states), len(states)))
    moves = []
    for i, s in enumerate(states):
        dst = [index[tuple(np.asarray(g.apply(np.array(s, np.int8))).tolist())] for g in case.library.groups]
        a = np.array([0.0 if d == i else min(1.0, math.exp(-(E[d] - E[i]) / tau)) for d in dst])
        A = a.sum()
        for r, d in enumerate(dst):
            if d != i:
                jump[i, d] += a[r] / A
        moves.append((dst, a, A))
    return states, index, E, boltz, jump, moves, R


def _final_counts(be_name, case, b, proc, opts, tau, n_rep, budget, seed, **extra):
    be = get_backend(be_name)
    plan = be.prepare(case.model, process=proc, process_options=opts, **extra)
    spins0 = case.initial(n_rep, seed)
    st = be.init_state(plan, case.bindings[b], spins0)
    res = be.run(plan, st, S.constant(tau, budget), seed=seed)
    idx = {tuple(s): i for i, s in enumerate(case.states.tolist())}
    return np.bincount([idx[tuple(s)] for s in res.state.spins.tolist()], minlength=case.states.shape[0]), res


def _chi2_p(counts, probs, min_expected=5.0):
    """Chi-square goodness-of-fit p-value; cells expected below ``min_expected`` are pooled into one."""
    counts = np.asarray(counts, float)
    probs = np.asarray(probs, float) / np.sum(probs)
    exp = probs * counts.sum()
    assert np.all(counts[exp == 0] == 0), "observed an outcome of probability 0"
    small = exp < min_expected
    obs_c = np.append(counts[~small], counts[small].sum()) if small.any() else counts
    exp_c = np.append(exp[~small], exp[small].sum()) if small.any() else exp
    m = exp_c > 0
    return float(stats.chisquare(obs_c[m], exp_c[m]).pvalue)


TAU = 1.5


@pytest.mark.parametrize("proc,opts", [("nfold", None), ("sweep", None), ("select", None), ("shared", {"cohort_size": 1})])
def test_stationary_boltzmann(proc, opts):
    """Final states of independent replicas after a long fixed-tau run follow Boltzmann (chi-square), and a
    wrong temperature is rejected (the test has power)."""
    case = case_cardinality_toy()
    states, _, E, boltz, *_ = _enumerate(case, 0, TAU)
    counts, _ = _final_counts("cpu", case, 0, proc, opts, TAU, 6000, 3000, seed=21)
    p = _chi2_p(counts, boltz)
    print(f"[stat] {proc} stationary vs Boltzmann: p = {p:.4g}")
    assert p > P_MIN, (proc, p)
    wrong = np.exp(-(E - E.min()) / (1.5 * TAU))
    assert _chi2_p(counts, wrong / wrong.sum()) < 1e-6


def test_nfold_matches_select_distribution():
    case = case_cardinality_toy()
    c1, _ = _final_counts("cpu", case, 0, "nfold", None, TAU, 6000, 3000, seed=5)
    c2, _ = _final_counts("cpu", case, 0, "select", None, TAU, 6000, 3000, seed=6)
    m = (c1 + c2) > 0
    p = stats.chi2_contingency(np.stack([c1[m], c2[m]])).pvalue
    print(f"[stat] nfold vs select final states (contingency): p = {p:.4g}")
    assert p > P_MIN


def test_nfold_waiting_time_and_choice():
    """From one state: W ~ Geometric(A / R) on {1, 2, ...} and the committed move is r w.p. a_r / A."""
    case = case_cardinality_toy()
    states, index, E, boltz, jump, moves, R = _enumerate(case, 0, TAU)
    i0 = int(np.argmin([abs(m[2] / R - 0.3) for m in moves]))  # A / R near 0.3
    dst, a, A = moves[i0]
    p = A / R
    assert 0.05 < p < 0.95
    n_rep = 200000
    cpu = get_backend("cpu")
    plan = cpu.prepare(case.model, process="nfold")
    st = cpu.init_state(plan, case.bindings[0], np.repeat(case.states[i0:i0 + 1], n_rep, 0))
    res = cpu.run(plan, st, S.constant(TAU, 10 ** 9), seed=8, stop=1)  # exactly one step per replica
    W = res.state.next_attempt
    assert np.all(res.state.counters["steps"] == 1) and np.all(W >= 1)
    kmax = int(np.ceil(np.log(1e-4) / np.log(1 - p)))
    ks = np.arange(1, kmax + 1)
    pk = (1 - p) ** (ks - 1) * p
    obs = np.bincount(np.minimum(W, kmax + 1), minlength=kmax + 2)[1:]
    exp_p = np.append(pk, 1 - pk.sum())
    p_w = _chi2_p(obs, exp_p)
    print(f"[stat] nfold W vs Geometric(A/R = {p:.4f}): p = {p_w:.4g}, mean W = {W.mean():.4f} (1/p = {1 / p:.4f})")
    assert p_w > P_MIN, p_w
    assert abs(W.mean() - 1 / p) < 4 * math.sqrt((1 - p) / p ** 2 / n_rep)
    # wrong rate rejected
    q = min(0.99, 1.3 * p)
    assert _chi2_p(obs, np.append((1 - q) ** (ks - 1) * q, max(0.0, 1 - ((1 - q) ** (ks - 1) * q).sum()))) < 1e-6
    # choice of the next state = select's jump row
    nxt = np.bincount([index[tuple(s)] for s in res.state.spins.tolist()], minlength=len(states))
    p_c = _chi2_p(nxt, jump[i0])
    print(f"[stat] nfold committed move vs a_r/A: p = {p_c:.4g}")
    assert p_c > P_MIN, p_c
    assert np.all(res.state.counters["rejected"] == W - 1)


def test_nfold_embedded_chain_equals_select_jump_chain():
    """Transitions of the nfold embedded chain (steps clock, one step per chunk) vs the exact select jump
    matrix computed on the enumerated space: summed chi-square over the visited rows."""
    case = case_cardinality_toy()
    states, index, E, boltz, jump, moves, R = _enumerate(case, 0, TAU)
    n_rep, n_steps = 2000, 60
    cpu = get_backend("cpu")
    plan = cpu.prepare(case.model, process="nfold", process_options={"nfold_clock": "steps"})
    st = cpu.init_state(plan, case.bindings[0], case.initial(n_rep, 2))
    sched = S.constant(TAU, n_steps)
    N = np.zeros((len(states), len(states)))
    prev = [index[tuple(s)] for s in st.rs.spins.tolist()]
    for k in range(1, n_steps + 1):
        res = cpu.run(plan, st, sched, seed=4, stop=k)
        cur = [index[tuple(s)] for s in res.state.spins.tolist()]
        for a, b in zip(prev, cur):
            N[a, b] += 1
        prev = cur
    assert np.trace(N) == 0  # nfold never stays (no self-loops, no rejections)
    chi, dof = 0.0, 0
    for i in range(len(states)):
        n_i = N[i].sum()
        m = jump[i] > 0
        assert np.all(N[i][~m] == 0)
        if n_i < 50:
            continue
        exp = jump[i][m] * n_i
        chi += float(((N[i][m] - exp) ** 2 / exp).sum())
        dof += int(m.sum()) - 1
    p = float(stats.chi2.sf(chi, dof))
    print(f"[stat] nfold embedded chain vs select jump matrix: chi2 = {chi:.1f}, dof = {dof}, p = {p:.4g}")
    assert dof > 50 and p > P_MIN, (chi, dof, p)


def _pt_exact_row(a: np.ndarray, dst, n_states: int):
    """parallel_trial one-step law from fired-subset enumeration: commit r w.p. sum_F prob(F) / |F|."""
    out = np.zeros(n_states + 1)  # last entry: nothing fired
    cand = [r for r in range(len(a)) if a[r] >= 0]
    for bits in itertools.product((0, 1), repeat=len(cand)):
        pr = 1.0
        F = []
        for r, f in zip(cand, bits):
            pr *= a[r] if f else 1 - a[r]
            if f:
                F.append(r)
        if pr == 0.0:
            continue
        if not F:
            out[-1] += pr
        for r in F:
            out[dst[r]] += pr / len(F)
    return out


def test_parallel_trial_one_step_law():
    """From each of several states: the CPU's next-state frequencies equal the exact law
    a_r E[1 / (1 + K_{-r})] obtained by enumerating fired subsets; also compared with the closed form."""
    case = case_cardinality_toy()
    states, index, E, boltz, jump, moves, R = _enumerate(case, 0, TAU)
    cpu = get_backend("cpu")
    plan = cpu.prepare(case.model, process="parallel_trial")
    n_rep = 20000
    for i0 in (0, int(np.argmax(boltz)), int(np.argmin(boltz))):
        dst, a, A = moves[i0]
        ac = np.where(np.array(dst) == i0, -1.0, a)  # self-loops excluded from the competition
        exact = _pt_exact_row(ac, dst, len(states))
        # closed form a_r E[1/(1+K_{-r})] with K_{-r} Poisson-binomial over the other candidates
        cf = np.zeros(len(states) + 1)
        cand = [r for r in range(R) if ac[r] >= 0]
        for r in cand:
            others = [ac[q] for q in cand if q != r]
            pk = np.array([1.0])
            for q in others:
                pk = np.convolve(pk, [1 - q, q])
            cf[dst[r]] += ac[r] * float((pk / np.arange(1, pk.size + 1)).sum())
        cf[-1] = np.prod([1 - x for x in ac if x >= 0])
        assert np.allclose(cf, exact, atol=1e-12)
        st = cpu.init_state(plan, case.bindings[0], np.repeat(case.states[i0:i0 + 1], n_rep, 0))
        res = cpu.run(plan, st, S.constant(TAU, 100), seed=31, stop=1)
        obs = np.zeros(len(states) + 1)
        for s, acc in zip(res.state.spins.tolist(), res.state.counters["accepted"].tolist()):
            obs[index[tuple(s)] if acc else -1] += 1
        p = _chi2_p(obs, exact)
        print(f"[stat] parallel_trial one-step law from state {i0}: p = {p:.4g}")
        assert p > P_MIN, (i0, p)
        assert np.all(res.state.counters["candidate_evaluations"] == int((ac >= 0).sum()))


def _ferro2():
    """Two spins, E = -J s0 s1 - h s0, single-flip masks (arXiv:2303.08392's two-spin example)."""
    structure = ObjectiveStructure(n_spins=2, terms=[(0, 1), (0,)])
    lib = B.single_flips(2)
    model = compile_model(structure, lib)
    return structure, lib, model


def test_parallel_trial_non_boltzmann_two_spin_ferromagnet():
    """PT's stationary law is not Boltzmann: exact chain from the enumerated one-step law, CPU agrees with
    it and rejects Boltzmann."""
    structure, lib, model = _ferro2()
    J, h, tau = 1.0, 0.25, 1.2
    binding = bind_weights(model, np.array([J, h]), 0.0)
    states = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], np.int8)
    E = direct_energy(structure, [J, h], 0.0, states)
    n = 4
    P = np.zeros((n, n))
    for i, s in enumerate(states):
        dst, a = [], []
        for g in lib.groups:
            m = np.asarray(g.apply(s), np.int8)
            d = int(np.flatnonzero(np.all(states == m, axis=1))[0])
            dst.append(d)
            a.append(min(1.0, math.exp(-(E[d] - E[i]) / tau)))
        row = _pt_exact_row(np.array(a), dst, n)
        P[i] = row[:n]
        P[i, i] += row[n]
    w, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmin(np.abs(w - 1))])
    pi /= pi.sum()
    boltz = np.exp(-(E - E.min()) / tau)
    boltz /= boltz.sum()
    assert np.max(np.abs(pi - boltz)) > 0.05  # genuinely different
    cpu = get_backend("cpu")
    plan = cpu.prepare(model, process="parallel_trial")
    n_rep = 8000
    st = cpu.init_state(plan, binding, states[np.random.default_rng(0).integers(0, 4, n_rep)])
    res = cpu.run(plan, st, S.constant(tau, 200), seed=17)
    counts = np.array([np.sum(np.all(res.state.spins == s, axis=1)) for s in states])
    p_pi, p_b = _chi2_p(counts, pi), _chi2_p(counts, boltz)
    print(f"[stat] PT 2-spin ferromagnet: exact PT stationary {np.round(pi, 4).tolist()}, Boltzmann "
          f"{np.round(boltz, 4).tolist()}, observed {np.round(counts / counts.sum(), 4).tolist()}; "
          f"p(PT law) = {p_pi:.4g}, p(Boltzmann) = {p_b:.3g}")
    assert p_pi > P_MIN
    assert p_b < 1e-6
    # select on the same model is Boltzmann
    plan = cpu.prepare(model)
    st = cpu.init_state(plan, binding, states[np.random.default_rng(1).integers(0, 4, n_rep)])
    res = cpu.run(plan, st, S.constant(tau, 400), seed=18)
    counts = np.array([np.sum(np.all(res.state.spins == s, axis=1)) for s in states])
    assert _chi2_p(counts, boltz) > P_MIN


def test_nfold_frozen_at_zero_temperature():
    """tau = 0 and no downhill/neutral move: the replica stops with termination 'frozen', clock = B."""
    structure, lib, model = _ferro2()
    binding = bind_weights(model, np.array([1.0, 0.25]), 0.0)
    cpu = get_backend("cpu")
    e = get_backend("emulation")
    for be in (cpu, e):
        plan = be.prepare(model, process="nfold")
        st = be.init_state(plan, binding, np.array([[1, -1], [1, 1]], np.int8))  # ++ is the ground state
        res = be.run(plan, st, S.zero(1000), seed=1, stop=10)
        assert res.termination == ("frozen", "frozen")
        assert res.state.next_attempt.tolist() == [1000, 1000]
        assert res.state.counters["frozen_attempt"].tolist() == [1, 0]
        # replica 0 takes one downhill step into a local minimum of the library (++ or --), then freezes
        assert res.state.spins.tolist()[0] in ([1, 1], [-1, -1]) and res.state.spins.tolist()[1] == [1, 1]
        assert res.state.counters["attempts"].tolist() == [1000, 1000]
        res = be.run(plan, be.load_state(plan, res.state), S.zero(1000), seed=1)
        assert res.termination == ("frozen", "frozen")


def test_runner_process_end_to_end(tmp_path):
    """Config -> runner -> results rows and manifest carry the process; select hashes are unchanged."""
    from nimbus.experiments.config import load_config
    from nimbus.experiments.results import read_shard
    from nimbus.experiments.runner import solve

    base = {"schema": "nimbus.run/1", "name": "proc_e2e",
            "problem": {"kind": "matching", "tensor": str(FIXTURES / "matching" / "five.npz")},
            "transitions": {"library": "complete"},
            "annealing": {"schedule": "fn_endpoints", "params": {"tau_start": 50.0, "tau_end": 1.0, "eta": 0.1},
                          "attempts": 3000},
            "run": {"seed": 2, "replicas": 3, "chunk_attempts": 700}, "backend": {"name": "cpu"}}
    h0 = load_config(base).trajectory_hash()
    explicit = {**base, "annealing": {**base["annealing"], "process": "select", "process_options": {}}}
    assert load_config(explicit).trajectory_hash() == h0
    nf = {**base, "annealing": {**base["annealing"], "process": "nfold"}}
    assert load_config(nf).trajectory_hash() != h0
    with pytest.raises(ConfigError):
        load_config({**base, "annealing": {**base["annealing"], "process": "nfold",
                                            "process_options": {"cohort_size": 4}}})
    with pytest.raises(ConfigError):
        load_config({**base, "annealing": {**base["annealing"], "process": "parallel_trial",
                                            "batching": {"capacity": 3}}})
    with pytest.raises(ConfigError):
        load_config({**nf, "backend": {"name": "oracle"}})
    out = solve(nf, output_dir=tmp_path / "nf")
    assert out.status in ("ok", "target_hit")
    rows = read_shard(Path(out.run_dir) / "results.parquet")
    assert all(r["process"] == "nfold" and r["steps"] >= 1 and r["candidate_evaluations"] >= r["steps"]
               for r in rows)
    assert all(r["attempts"] == 3000 for r in rows)
    assert out.manifest["process"]["process"] == "nfold" and out.manifest["process"]["clock"] == "attempts"
    # chunked runner result equals one uninterrupted CPU run
    pt = {**base, "annealing": {**base["annealing"], "process": "parallel_trial"}}
    out = solve(pt, output_dir=tmp_path / "pt")
    rows = read_shard(Path(out.run_dir) / "results.parquet")
    assert all(r["process"] == "parallel_trial" and r["steps"] == r["attempts"] == 3000 for r in rows)
    assert out.manifest["process"]["clock"] == "steps"
    sel = solve(base, output_dir=tmp_path / "sel")
    rows = read_shard(Path(sel.run_dir) / "results.parquet")
    assert all(r["process"] == "select" and r["steps"] == r["attempts"] for r in rows)

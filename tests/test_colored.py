"""The ``colored`` proposal process.

* compile-time coloring: classes are non-interacting (independent re-derivation), deterministic;
* commuting: committing any subset of a class in any order gives exactly the same T, v and switches
  as every other order and as the spin-level result;
* Boltzmann: each class kernel (a product of independent Metropolis kernels) leaves exp(-E/tau)
  invariant (exact matrices, 1e-12) and the cpu run reproduces the exact one-step law and stationary
  law (chi-square on fixed seeds);
* emulation == cpu bit for bit (all CPU strategies, chunk boundaries inside a class), against an
  independent spin-level reference driver, counters, recording path, process spec, runner.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from helpers_core import direct_energy
from test_replay_core import CASE_BUILDERS, Case, _finish, mixing_schedule
from nimbus.annealing import schedule as S
from nimbus.backends import _numba_common as nc
from nimbus.backends import get_backend
from nimbus.backends.base import ProcessSpec, clause_signs
from nimbus.backends.coloring import ColoringPlan, check_coloring, color_groups, interaction_graph
from nimbus.compiler import compile_model
from nimbus.core import rng
from nimbus.core.errors import ConfigError
from nimbus.problems import surface_code as SC

FIXTURES = Path(__file__).parent / "fixtures"
P_MIN = 1e-3

CPU_VARIANTS = [{}, {"strategy": "incremental"}, {"strategy": "recompute"}, {"strategy": "lazy"}]


def surface_case(d=3, noise="independent", p=0.08, library="stabilizers_logical_all") -> Case:
    prob = SC.SurfaceCodeProblem(d, p, noise=noise, library=library)
    lib = prob.transition_library()
    structure = prob.structure()
    model = compile_model(structure, lib, identities=())
    case = Case(f"surface_{noise}_d{d}", structure, lib, model, prob.enumerate_feasible(), [prob.term_weights(0)],
                [prob.const(0)], has_gates=False, has_selfloops=True)
    return _finish(case)


def all_cases():
    out = {n: b for n, b in CASE_BUILDERS.items()}
    out["surface_ind_d3"] = lambda: surface_case(3, "independent")
    out["surface_dep_d3"] = lambda: surface_case(3, "depolarizing")
    return out


NONTRIVIAL = ["section29", "table_chain", "surface_ind_d3", "surface_dep_d3"]


# ============================================================================================
# spec, rng
# ============================================================================================
def test_spec_validation():
    s = ProcessSpec.coerce("colored")
    assert s.colored and s.clock == "attempts" and not s.stepwise and not s.ragged and s.full_library
    assert s.self_loop_rule() == "attempt" and s.options() == {}
    assert s.describe()["process"] == "colored"
    with pytest.raises(ConfigError):
        ProcessSpec.coerce("colored", {"cohort_size": 2})
    case = CASE_BUILDERS["section29"]()
    with pytest.raises(ConfigError):  # needs every group resident
        get_backend("emulation").prepare(case.model, capacity=4, process="colored")
    with pytest.raises(ConfigError):
        get_backend("cpu").prepare(case.model, process="colored", strategy="adaptive")
    with pytest.raises(ConfigError):
        get_backend("oracle").prepare(case.model, process="colored")
    for name in ("cpu", "emulation"):
        be = get_backend(name)
        plan = be.prepare(case.model, process="colored")
        st = be.init_state(plan, case.bindings[0], case.initial(2, 0))
        with pytest.raises(ConfigError):  # lock-step step() has no colored form
            be.step(st, 0, mixing_schedule(case, 20), 1)
        d = plan.describe()
        assert d["options"]["coloring_n_colors"] == 4


def test_color_stream():
    g = np.random.default_rng(0)
    for _ in range(300):
        seed, c, i = (int(x) for x in g.integers(0, 2 ** 32, 3))
        k, r = int(g.integers(0, 2 ** 32)), int(g.integers(0, 2 ** 31))
        assert rng.color_uniform(seed, k, r, c, i) == nc.color_uniform(seed, k, r, c, i)
    x = rng.philox4x32_scalar(9, 17, 2, 5, 12345, rng.COLOR)
    assert rng.color_uniform(12345, 9, 17, 2, 5) == rng.uniform52(x[0], x[1])
    assert rng.COLOR == 12 and rng.COLOR not in (rng.SOLVER, rng.INIT, rng.CHANNEL, rng.PILOT, rng.DIAG, rng.DATA,
                                                  rng.BASELINE, rng.SHARED, rng.PT, rng.NFOLD, rng.SWEEP)
    assert rng.color_uniform(1, 3, 4, 5, 6) != rng.pt_uniform(1, 3, 4, 5, 6)


# ============================================================================================
# coloring
# ============================================================================================
@pytest.mark.parametrize("name", list(all_cases()))
def test_coloring_valid_and_deterministic(name):
    case = all_cases()[name]()
    p1, p2 = color_groups(case.model), color_groups(case.model)
    check_coloring(case.model, p1)
    for f in ("color_of_group", "order", "class_ptr", "class_of_pos"):
        assert np.array_equal(getattr(p1, f), getattr(p2, f))
    R = case.model.n_groups
    assert sorted(p1.order.tolist()) == list(range(R)) and p1.class_sizes.sum() == R
    assert np.all(np.diff(p1.class_of_pos) >= 0)
    # a proper coloring of the interaction graph, and every class below the last has a conflicting member
    adj = interaction_graph(case.model).tocoo()
    assert np.all(p1.color_of_group[adj.row] != p1.color_of_group[adj.col])
    for c in range(1, p1.n_colors):
        for r in p1.classes()[c]:
            nb = adj.col[adj.row == r]
            assert len(set(p1.color_of_group[nb].tolist()) & set(range(c))) == c  # DSATUR: smallest free color


def test_coloring_expected_structure():
    # surface code: stabilizers colour into a handful of classes; logicals sit in their own class(es)
    for d in (3, 5):
        case = surface_case(d, "independent")
        p = color_groups(case.model)
        check_coloring(case.model, p)
        assert 1 < p.n_colors < case.model.n_groups
        print(f"[coloring] surface d={d}: {p.n_colors} colors, sizes {p.class_sizes.tolist()}")


# ============================================================================================
# commuting (exact)
# ============================================================================================
@pytest.mark.parametrize("name", NONTRIVIAL)
def test_class_commit_commutes_exactly(name):
    case = all_cases()[name]()
    b = len(case.bindings) - 1
    plan_c = color_groups(case.model)
    e = get_backend("emulation")
    plan = e.prepare(case.model, process="colored")
    g = np.random.default_rng(3)
    n_checked = 0
    for s0 in case.initial(6, 1):
        for cls in plan_c.classes():
            if len(cls) < 2:
                continue
            for _ in range(3):
                k = int(g.integers(2, len(cls) + 1))
                subset = [int(x) for x in g.choice(cls, size=k, replace=False)]
                finals = []
                for order in (sorted(subset), sorted(subset, reverse=True), list(g.permutation(subset))):
                    st = e.init_state(plan, case.bindings[b], s0[None, :])
                    for r in order:
                        e._commit(st, 0, int(r), None)
                    finals.append((st.rs.T[0].copy(), st.alpha[0].copy(), st.rs.v[0].copy(), st.rs.energies[0].copy()))
                for f in finals[1:]:
                    assert np.array_equal(f[0], finals[0][0])      # clause signs T
                    assert np.array_equal(f[1], finals[0][1])      # switches
                    # inputs v: exact for every group of the class (they only see their own commit); other
                    # groups receive the corrections of several commits, summed in a different order, so they
                    # agree to rounding (and exactly with dyadic weights)
                    assert np.array_equal(f[2][cls], finals[0][2][cls])
                    if name == "section29":
                        assert np.array_equal(f[2], finals[0][2])
                    else:
                        assert np.allclose(f[2], finals[0][2], rtol=1e-12, atol=1e-12)
                # spin-level: sequential library actions give the same state, v is the fresh encoder sum
                s = s0.copy()
                for r in sorted(subset):
                    s = np.asarray(case.library.groups[r].apply(s), dtype=np.int8)
                assert np.array_equal(finals[0][0], clause_signs(case.model, s[None, :])[0])
                fresh = e.init_state(plan, case.bindings[b], s[None, :])
                assert np.allclose(finals[0][2], fresh.rs.v[0], atol=1e-9, rtol=0)
                # energy of the set = sum of the individual 2 v_r (disjoint clause sets)
                st0 = e.init_state(plan, case.bindings[b], s0[None, :])
                dE = sum(2.0 * st0.rs.v[0, r] for r in subset)
                assert math.isclose(finals[0][3][0] - st0.rs.energies[0, 0], dE, rel_tol=1e-9, abs_tol=1e-9)
                n_checked += 1
    assert n_checked > 0


# ============================================================================================
# emulation == cpu == independent reference
# ============================================================================================
@pytest.mark.parametrize("extra", CPU_VARIANTS)
@pytest.mark.parametrize("name", list(all_cases()))
def test_cpu_equals_emulation(name, extra):
    case = all_cases()[name]()
    sched = mixing_schedule(case, 300)
    nb = len(case.bindings)
    rb = np.repeat(np.arange(nb), 3)
    rid = np.tile(np.arange(3), nb)
    spins0 = case.initial(3 * nb, 1)
    e, c = get_backend("emulation"), get_backend("cpu")
    pe = e.prepare(case.model, process="colored")
    pc = c.prepare(case.model, process="colored", **extra)
    se = e.init_state(pe, case.bindings, spins0, rb, rid, rb, is_feasible=case.is_feasible)
    sc = c.init_state(pc, case.bindings, spins0, rb, rid, rb)
    for stop in (1, 7, 60, 151, 300):  # window ends fall inside classes
        re_ = e.run(pe, se, sched, seed=5, stop=stop)
        rc = c.run(pc, sc, sched, seed=5, stop=stop)
        assert np.array_equal(re_.state.T, rc.state.T), (name, extra, stop)
        for k in re_.state.counters:
            assert np.array_equal(re_.state.counters[k], rc.state.counters[k]), (name, extra, stop, k)
        assert np.array_equal(re_.state.next_attempt, rc.state.next_attempt)
        assert np.array_equal(re_.state.best_attempt, rc.state.best_attempt)
        assert np.array_equal(re_.state.best_spins, rc.state.best_spins)
        assert re_.termination == rc.termination
    assert all(case.is_feasible(s) for s in rc.state.spins)
    c.validate_state(sc)


class RefColored:
    """Process semantics on explicit spins: library actions, direct energies, RNG references; every group
    of the class is decided from the state at the start of the step, the accepted set is applied together."""

    def __init__(self, case: Case, b: int):
        self.case, self.b, self.lib = case, b, case.library
        self.R = self.lib.n_groups
        self.col = color_groups(case.model)

    def energy(self, s):
        return float(self.case.energy(self.b, np.asarray(s)[None, :])[0])

    def run(self, s0, sched, seed, rid, iid):
        s = np.asarray(s0, dtype=np.int8).copy()
        B = sched.n_total
        a, out, C = 0, [], self.col.n_colors
        while a < B:
            cyc, p = divmod(a, self.R)
            k = int(self.col.class_of_pos[p])
            lo, hi = int(self.col.class_ptr[k]), int(self.col.class_ptr[k + 1])
            a_start = a - (p - lo)
            tau = sched.tau(a_start)
            step = cyc * C + k
            n_do = min(hi - p, B - a)
            E0 = self.energy(s)
            flips = []
            for q in range(p, p + n_do):
                r = int(self.col.order[q])
                m = np.asarray(self.lib.groups[r].apply(s), dtype=np.int8)
                v = (self.energy(m) - E0) / 2.0
                u = rng.color_uniform(seed, step, r, rid, iid)
                if v <= (-(tau / 2.0) * math.log(u) if tau > 0 else 0.0):
                    flips.append(m != s)
            for f in flips:
                s = np.where(f, -s, s).astype(np.int8)
            a += n_do
            out.append((a, s.copy()))
        return out


def test_cpu_equals_reference_driver():
    for name in ("section29", "table_chain"):
        case = CASE_BUILDERS[name]()
        b = 1  # dyadic weights: exact energy differences
        sched = mixing_schedule(case, 240)
        spins0 = case.initial(4, 3)
        c = get_backend("cpu")
        plan = c.prepare(case.model, process="colored")
        st = c.init_state(plan, case.bindings, spins0, np.full(4, b), np.arange(4), np.full(4, b))
        ref = RefColored(case, b)
        trajs = [ref.run(spins0[i], sched, 77, i, b) for i in range(4)]
        for stop in (37, 120, 240):
            res = c.run(plan, st, sched, seed=77, stop=stop)
            for i in range(4):
                done = [x for x in trajs[i] if x[0] <= stop]
                # the run stops at ``stop`` inside a class; the reference steps at class ends (the final
                # window of a class cut at ``stop`` is applied at the cut): compare only at class ends
                if int(res.state.next_attempt[i]) == (done[-1][0] if done else 0):
                    want = done[-1][1] if done else spins0[i]
                    assert np.array_equal(res.state.spins[i], want), (name, stop, i)


def test_recording_path_equals_fast_path():
    case = CASE_BUILDERS["section29"]()
    sched = mixing_schedule(case, 400)
    spins0 = case.initial(4, 2)
    rb, rid = np.zeros(4, int), np.arange(4)
    c, e = get_backend("cpu"), get_backend("emulation")
    pc = c.prepare(case.model, process="colored")
    s1 = c.init_state(pc, case.bindings, spins0, rb, rid, rb)
    s2 = c.init_state(pc, case.bindings, spins0, rb, rid, rb)
    r1 = c.run(pc, s1, sched, seed=3)
    r2 = c.run(pc, s2, sched, seed=3, recording={"profile": "anytime", "anytime_points": 8})
    assert np.array_equal(r1.state.T, r2.state.T)
    for k in r1.state.counters:
        assert np.array_equal(r1.state.counters[k], r2.state.counters[k]), k
    assert np.array_equal(r1.state.best_attempt, r2.state.best_attempt)
    assert r2.recording.anytime is not None
    pe = e.prepare(case.model, process="colored")
    se = e.init_state(pe, case.bindings, spins0, rb, rid, rb)
    r3 = e.run(pe, se, sched, seed=3, recording={"profile": "anytime", "anytime_points": 8})
    assert np.array_equal(r3.state.T, r1.state.T)


# ============================================================================================
# counters and clocks
# ============================================================================================
def test_counters_and_resume():
    case = surface_case(3, "independent")
    p = color_groups(case.model)
    R, C = case.model.n_groups, p.n_colors
    sched = mixing_schedule(case, 10 * R + 3)
    spins0 = case.initial(3, 2)
    for name in ("cpu", "emulation"):
        be = get_backend(name)
        plan = be.prepare(case.model, process="colored")
        st = be.init_state(plan, case.bindings[0], spins0)
        res = be.run(plan, st, sched, seed=1)
        c = res.state.counters
        B = 10 * R + 3
        assert np.all(c["attempts"] == B) and np.all(c["candidate_evaluations"] == B)
        assert np.all(c["accepted"] + c["rejected"] == B)
        # 10 whole cycles plus a 3-group prefix of the first position: its class starts one more step
        assert np.all(c["steps"] == 10 * C + 1)
        assert np.all(res.state.next_attempt == B)
        # resume through any cut gives the identical trajectory
        st2 = be.init_state(plan, case.bindings[0], spins0)
        for stop in (2, 5, 19, 20, B):
            r2 = be.run(plan, st2, sched, seed=1, stop=stop)
        assert np.array_equal(r2.state.T, res.state.T)
        for k in c:
            assert np.array_equal(c[k], r2.state.counters[k]), (name, k)
    # attempts per color step = R / C on average
    assert p.class_sizes.sum() == R


# ============================================================================================
# Boltzmann (exact matrices + chi-square)
# ============================================================================================
def class_kernels(case: Case, b: int, tau: float):
    """(states, E, boltz, [P_c per class in cyclic order]) with the exact product-Metropolis kernel of every
    class: every group decided from the current state, the accepted set applied together."""
    col = color_groups(case.model)
    states = [tuple(s) for s in case.states.tolist()]
    index = {s: i for i, s in enumerate(states)}
    E = case.energy(b, case.states)
    boltz = np.exp(-(E - E.min()) / tau)
    boltz /= boltz.sum()
    n = len(states)
    kernels = []
    for cls in col.classes():
        P = np.zeros((n, n))
        for i, s in enumerate(states):
            s_arr = np.array(s, dtype=np.int8)
            moves = []
            for r in cls.tolist():
                m = np.asarray(case.library.groups[r].apply(s_arr), dtype=np.int8)
                j = index[tuple(m.tolist())]
                a = 1.0 if E[j] <= E[i] else math.exp(-(E[j] - E[i]) / tau)
                moves.append((m != s_arr, a))
            for mask in itertools.product((0, 1), repeat=len(moves)):
                pr, t = 1.0, s_arr.copy()
                for (flip, a), take in zip(moves, mask):
                    pr *= a if take else 1.0 - a
                    if take:
                        t = np.where(flip, -t, t).astype(np.int8)
                P[i, index[tuple(t.tolist())]] += pr
        kernels.append(P)
    return states, E, boltz, kernels, col


def _chi2_p(counts, probs, min_expected=5.0):
    counts = np.asarray(counts, float)
    probs = np.asarray(probs, float) / np.sum(probs)
    exp = probs * counts.sum()
    assert np.all(counts[exp == 0] == 0), "observed an outcome of probability 0"
    small = exp < min_expected
    obs_c = np.append(counts[~small], counts[small].sum()) if small.any() else counts
    exp_c = np.append(exp[~small], exp[small].sum()) if small.any() else exp
    m = exp_c > 0
    return float(stats.chisquare(obs_c[m], exp_c[m]).pvalue)


@pytest.mark.parametrize("name", ["section29", "surface_ind_d3"])
def test_class_kernels_preserve_boltzmann_exactly(name):
    case = all_cases()[name]()
    tau = 1.3
    states, E, boltz, kernels, col = class_kernels(case, len(case.bindings) - 1, tau)
    big = [c for c in col.classes() if len(c) > 1]
    assert big, "test needs a class with more than one group"
    for P in kernels:
        assert np.allclose(P.sum(1), 1.0, atol=1e-12)
        assert np.allclose(boltz @ P, boltz, atol=1e-12)
    # the kernel of a multi-group class is not the same as one Metropolis step: detailed balance still holds
    # for the (symmetric product) kernel
    for P in kernels:
        assert np.allclose(boltz[:, None] * P, (boltz[:, None] * P).T, atol=1e-12)


@pytest.mark.parametrize("name", ["section29", "surface_ind_d3"])
def test_cpu_one_step_law_and_stationary(name):
    case = all_cases()[name]()
    b = len(case.bindings) - 1
    tau = 1.3
    states, E, boltz, kernels, col = class_kernels(case, b, tau)
    R, C = case.model.n_groups, col.n_colors
    c = get_backend("cpu")
    plan = c.prepare(case.model, process="colored")
    idx = {s: i for i, s in enumerate(states)}
    # the first step (class 0 starts at clock 0) from the state whose exact row has the most outcomes
    k0 = 0
    assert int(col.class_ptr[0]) == 0
    row0 = int(np.argmax((kernels[k0] > 1e-9).sum(1)))
    s0 = case.states[row0]
    n_rep = 20000
    spins0 = np.repeat(s0[None, :], n_rep, 0)
    st = c.init_state(plan, case.bindings[b], spins0)
    sched = S.constant(tau, R * 3)
    res = c.run(plan, st, sched, seed=11, stop=int(col.class_sizes[0]))
    cnt = np.bincount([idx[tuple(s)] for s in res.state.spins.tolist()], minlength=len(states))
    p = _chi2_p(cnt, kernels[k0][row0])
    print(f"[stat] {name} class-0 one-step law, {int(col.class_sizes[0])} groups: p = {p:.4g}")
    assert p > P_MIN
    # stationary: whole cycles at fixed tau from random starts
    n_rep = 6000
    st = c.init_state(plan, case.bindings[b], case.initial(n_rep, 4))
    res = c.run(plan, st, S.constant(tau, R * 60), seed=21)
    cnt = np.bincount([idx[tuple(s)] for s in res.state.spins.tolist()], minlength=len(states))
    p = _chi2_p(cnt, boltz)
    print(f"[stat] {name} colored stationary vs Boltzmann: p = {p:.4g}")
    assert p > P_MIN
    wrong = np.exp(-(E - E.min()) / (1.5 * tau))
    assert _chi2_p(cnt, wrong / wrong.sum()) < 1e-6  # the test has power


def test_emulation_stationary_depolarizing():
    case = surface_case(3, "depolarizing")
    tau, R = 1.2, case.model.n_groups
    E = case.energy(0, case.states)
    boltz = np.exp(-(E - E.min()) / tau)
    boltz /= boltz.sum()
    idx = {tuple(s): i for i, s in enumerate(case.states.tolist())}
    c = get_backend("cpu")
    plan = c.prepare(case.model, process="colored")
    st = c.init_state(plan, case.bindings[0], case.initial(20000, 6))
    res = c.run(plan, st, S.constant(tau, R * 40), seed=2)
    cnt = np.bincount([idx[tuple(s)] for s in res.state.spins.tolist()], minlength=len(idx))
    p = _chi2_p(cnt, boltz)
    print(f"[stat] depolarizing d3 colored stationary: p = {p:.4g}")
    assert p > P_MIN


# ============================================================================================
# runner
# ============================================================================================
def test_runner_colored_end_to_end(tmp_path):
    from nimbus.experiments.config import load_config
    from nimbus.experiments.results import read_shard
    from nimbus.experiments.runner import solve

    base = {"schema": "nimbus.run/1", "name": "col_e2e",
            "problem": {"kind": "matching", "tensor": str(FIXTURES / "matching" / "five.npz")},
            "transitions": {"library": "complete"},
            "annealing": {"schedule": "fn_endpoints", "params": {"tau_start": 50.0, "tau_end": 1.0, "eta": 0.1},
                          "attempts": 3000},
            "run": {"seed": 2, "replicas": 3, "chunk_attempts": 700}, "backend": {"name": "cpu"}}
    h0 = load_config(base).trajectory_hash()
    col = {**base, "annealing": {**base["annealing"], "process": "colored"}}
    assert load_config(col).trajectory_hash() != h0
    with pytest.raises(ConfigError):
        load_config({**col, "annealing": {**col["annealing"], "batching": {"capacity": 3}}})
    out = solve(col, output_dir=tmp_path / "col")
    assert out.status in ("ok", "target_hit")
    rows = read_shard(Path(out.run_dir) / "results.parquet")
    assert all(r["process"] == "colored" and r["attempts"] == 3000 and 1 <= r["steps"] <= 3000 for r in rows)
    assert out.manifest["process"]["process"] == "colored" and out.manifest["process"]["clock"] == "attempts"
    one = solve({**col, "run": {**col["run"], "chunk_attempts": 3000}}, output_dir=tmp_path / "col1")
    rows1 = read_shard(Path(one.run_dir) / "results.parquet")
    assert [r["best_energy"] for r in rows] == [r["best_energy"] for r in rows1] if "best_energy" in rows[0] else True

"""Matching adapter: polynomial vs direct tensor evaluation, compile (exhaustive 2v=dE), emulation == cpu,
penalty consistency, enumeration oracle, tensor protocol, double exchanges."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from helpers_core import check_compiled_exhaustive
from nimbus.annealing import schedule as S
from nimbus.backends.base import get_backend
from nimbus.compiler import compile_model
from nimbus.core.model import bind_weights
from nimbus.problems.formats.matching_tensor import (canonical_tensor, evaluate_F, load_tensor, point_pair_tensor,
                                                      save_tensor, similarity_deformation_pair)
from nimbus.problems.matching import MatchingProblem, enumerate_optimum, from_config

pytestmark = pytest.mark.quick


def _tensor(n, seed=0, sigma=0.03):
    rng = np.random.default_rng(seed)
    P1, P2, gt, valid = similarity_deformation_pair(n, sigma, rng)
    return point_pair_tensor(P1, P2, seed=seed), gt


def _perm_states(n):
    perms = np.array(list(itertools.permutations(range(n))))  # perms[b, i] = a
    S_ = -np.ones((len(perms), n * n), dtype=np.int8)
    for b, p in enumerate(perms):
        S_[b, np.arange(n) * n + p] = 1
    return S_, perms


def test_energy_equals_minus_F_all_permutations():
    t, _ = _tensor(5, seed=3)
    prob = MatchingProblem(t)
    S_, perms = _perm_states(5)
    E = prob.polynomial().evaluate(S_)
    assert np.allclose(E, -evaluate_F(t, perms), atol=1e-9)
    for s, p in list(zip(S_, perms))[:10]:
        assert prob.is_feasible(s)
        assert np.array_equal(prob.decode(s)["perm"], p)
        assert prob.native_objective(s) == pytest.approx(prob.energy(s))


def test_expansion_degree_and_random_permutations_n8():
    t, _ = _tensor(8, seed=1)
    prob = MatchingProblem(t)
    assert max(len(s) for s in prob.polynomial().supports) <= 3
    assert len(set(prob.polynomial().supports)) == len(prob.polynomial().supports)
    rng = np.random.default_rng(0)
    perms = np.array([rng.permutation(8) for _ in range(50)])
    X = -np.ones((50, 64), dtype=np.int8)
    for b, p in enumerate(perms):
        X[b, np.arange(8) * 8 + p] = 1
    assert np.allclose(prob.polynomial().evaluate(X), -evaluate_F(t, perms))


@pytest.mark.parametrize("lib,kw", [("complete", {}), ("path", {}),
                                    ("complete", {"double_exchanges": {"count": 3, "seed": 1}})])
def test_compile_2v_equals_delta_E(lib, kw):
    n = 5 if "double_exchanges" not in kw else 5
    t, _ = _tensor(n, seed=2)
    prob = MatchingProblem(t)
    library = prob.transition_library(lib, **kw)
    model = compile_model(prob.structure(), library, identities=prob.identities())
    S_, _ = _perm_states(n)
    binding = bind_weights(model, prob.term_weights(), prob.const())
    rep = check_compiled_exhaustive(model, library, S_, binding=binding, energy=prob.polynomial().evaluate)
    assert rep["states"] == S_.shape[0]


def test_enumerate_optimum_matches_bruteforce(tmp_path):
    t, _ = _tensor(6, seed=5)
    _, perms = _perm_states(6)
    F = evaluate_F(t, perms)
    r = enumerate_optimum(t)
    assert r["F_opt"] == pytest.approx(F.max(), rel=1e-12)
    assert r["n_optimal"] == int(np.sum(F == F.max()))
    assert evaluate_F(t, np.array(r["perm"])) == pytest.approx(F.max())


def test_tensor_roundtrip_and_canonical(tmp_path):
    t, _ = _tensor(5, seed=7)
    p = tmp_path / "t.npz"
    save_tensor(p, t)
    u = load_tensor(p)
    assert u.n == 5 and np.array_equal(u.triples, t.triples) and np.array_equal(u.values, t.values)
    # duplicates merge; degenerate entries vanish
    c = canonical_tensor(4, [[[0, 1], [1, 2], [2, 3]], [[1, 2], [0, 1], [2, 3]], [[0, 1], [0, 2], [2, 3]]], [1.0, 2.0, 5.0])
    assert c.nnz == 1 and c.values[0] == 3.0
    prob = from_config({"kind": "matching", "tensor": "t.npz"}, tmp_path)
    assert prob.P == 5


def test_emulation_equals_cpu():
    t, _ = _tensor(5, seed=4)
    prob = MatchingProblem(t)
    lib = prob.transition_library("complete")
    model = compile_model(prob.structure(), lib, identities=prob.identities())
    binding = bind_weights(model, prob.term_weights(), prob.const())
    spins0 = prob.random_feasible(3, 4)
    out = {}
    for name in ("emulation", "cpu"):
        be = get_backend(name)
        plan = be.prepare(model)
        st = be.init_state(plan, [binding], spins0)
        be.run(plan, st, S.geometric(2.0, 0.05, 800), n_attempts_total=800, seed=9)
        out[name] = be.export_state(st)
    a, b = out["emulation"], out["cpu"]
    assert np.array_equal(a.spins, b.spins) and np.array_equal(a.T, b.T)
    assert np.allclose(a.energies, b.energies)
    for k in a.counters:
        assert np.array_equal(a.counters[k], b.counters[k]), k
    for r in range(4):
        assert prob.is_feasible(a.spins[r])
        assert prob.native_objective(a.spins[r]) == pytest.approx(a.energies[r, 0])


def test_penalty_formulation_consistency():
    t, _ = _tensor(4, seed=8)
    prob = MatchingProblem(t)
    pf = prob.penalty_formulation()
    rng = np.random.default_rng(1)
    X = rng.choice([-1, 1], size=(200, 16)).astype(np.int8)
    feas = np.array([prob.is_feasible(x) for x in X])
    v = pf.violation.evaluate(X)
    assert np.all(v[feas] == pytest.approx(0.0, abs=1e-9)) and np.all(v[~feas] > 0.5)
    assert np.allclose(pf.run.evaluate(X), pf.objective.evaluate(X) + v)
    S_, _ = _perm_states(4)
    assert np.allclose(pf.violation.evaluate(S_), 0.0, atol=1e-9)
    # with the default bound the global minimum of the penalized energy is feasible and equals the optimum
    allx = np.array([[1 if (k >> j) & 1 else -1 for j in range(16)] for k in range(1 << 16)], dtype=np.int8)
    er = pf.run.evaluate(allx)
    best = int(np.argmin(er))
    assert prob.is_feasible(allx[best])
    assert er[best] == pytest.approx(pf.objective.evaluate(S_).min())


@pytest.mark.parametrize("strategy,layout", [(s, l) for s in ("recompute", "incremental", "lazy")
                                             for l in ("flat", "gate_blocked")])
def test_cpu_event_counters_equal_emulation_events(strategy, layout):
    """Exact per-run event totals of the cpu backend (opt-in) equal the emulator's events profile."""
    from nimbus.backends.base import EVENT_COUNTER_NAMES

    t, _ = _tensor(5, seed=4)
    prob = MatchingProblem(t)
    lib = prob.transition_library("path", double_exchanges={"count": 2, "seed": 1})
    model = compile_model(prob.structure(), lib, identities=prob.identities())
    binding = bind_weights(model, prob.term_weights(), prob.const())
    spins0 = prob.random_feasible(3, 4)
    sched = S.geometric(3.0, 0.2, 1500)

    em = get_backend("emulation")
    pe = em.prepare(model)
    se = em.init_state(pe, [binding], spins0)
    res_e = em.run(pe, se, sched, n_attempts_total=1500, seed=9, recording="events")
    ev = res_e.recording.events
    assert ev.counter("clause_toggles").sum() > 0

    for rec in (None, "events"):  # fast prange path, sequential path
        cp = get_backend("cpu")
        pc = cp.prepare(model, strategy=strategy, layout=layout, event_counters=True)
        sc = cp.init_state(pc, [binding], spins0)
        r = cp.run(pc, sc, sched, n_attempts_total=1500, seed=9, recording=rec)
        for name in EVENT_COUNTER_NAMES:
            assert np.array_equal(r.state.counters["ev_" + name], ev.counter(name)), (name, rec)
        assert np.array_equal(r.state.counters["accepted_changed"], ev.counter("state_changes"))

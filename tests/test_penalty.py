"""Generic penalty baseline (nimbus.baselines.penalty): exhaustive equivalence on self-contained
polynomials, and end-to-end runs through the real problem adapters (toy, matching)."""

from __future__ import annotations

import numpy as np
import pytest

from helpers_core import all_states
from nimbus.annealing import schedule as S
from nimbus.baselines import penalty as P
from nimbus.problems import toy_cardinality as toy
from nimbus.problems.base import PenaltyFormulation, SpinPolynomial
from nimbus.problems.formats.matching_tensor import point_pair_tensor, similarity_deformation_pair
from nimbus.problems.matching import MatchingProblem

pytestmark = pytest.mark.quick


# ============================================================================================
# self-contained polynomial: exact penalized-energy identity
# ============================================================================================
def _cardinality_pf(n: int, k: int, lam: float) -> PenaltyFormulation:
    """objective = sum s_i (minimize number of +1 spins); violation = (n_+ - k)^2, exactly, as a
    SpinPolynomial over the pairs (expand (sum x_i - k)^2 with x_i=(1+s_i)/2)."""
    obj = SpinPolynomial(n_spins=n, supports=tuple((i,) for i in range(n)),
                         coeffs=np.ones(n), const=0.0)
    # (n+ - k)^2 = sum_i x_i + 2 sum_{i<j} x_i x_j - 2k sum_i x_i + k^2, x_i = (1+s_i)/2
    singles = {}
    pairs = {}
    const = float(k * k)
    for i in range(n):
        singles[i] = singles.get(i, 0.0) + (1.0 - 2.0 * k) * 0.5
        const += (1.0 - 2.0 * k) * 0.5
    for i in range(n):
        for j in range(i + 1, n):
            pairs[(i, j)] = pairs.get((i, j), 0.0) + 2.0 * 0.25
            singles[i] = singles.get(i, 0.0) + 2.0 * 0.25
            singles[j] = singles.get(j, 0.0) + 2.0 * 0.25
            const += 2.0 * 0.25
    supports = tuple((i,) for i in range(n)) + tuple(pairs.keys())
    coeffs = np.array([singles[i] for i in range(n)] + [pairs[p] for p in pairs], dtype=np.float64)
    viol = SpinPolynomial(n_spins=n, supports=supports, coeffs=coeffs, const=const)
    run_supports = obj.supports + viol.supports
    run_coeffs = np.concatenate([obj.coeffs, lam * viol.coeffs])
    run = SpinPolynomial(n_spins=n, supports=run_supports, coeffs=run_coeffs, const=obj.const + lam * viol.const)
    return PenaltyFormulation(run=run, objective=obj, violation=viol, n_physical=n, ledger={"lam": lam, "k": k})


def test_self_contained_cardinality_penalty_exhaustive():
    n, k, lam = 5, 2, 3.0
    pf = _cardinality_pf(n, k, lam)
    S_ = all_states(n)
    n_plus = (S_.astype(np.float64) + 1).sum(axis=1) / 2.0
    viol_direct = (n_plus - k) ** 2
    assert np.allclose(pf.violation.evaluate(S_), viol_direct)
    run_direct = pf.objective.evaluate(S_) + lam * viol_direct
    assert np.allclose(pf.run.evaluate(S_), run_direct)
    for s in S_:
        feasible = (np.sum((s.astype(np.float64) + 1) / 2.0) == k)
        assert (abs(float(pf.violation.evaluate(s))) < 1e-9) == feasible


def test_build_and_bind_penalty_model_matches_direct_evaluation():
    pf = _cardinality_pf(4, 2, 5.0)
    model, lib = P.build_penalty_model(pf)
    binding = P.bind_penalty(model, pf)
    assert binding.slot_names == ("run", "objective", "violation")
    from nimbus.backends.base import clause_signs, slot_energies

    S_ = all_states(4)
    T = clause_signs(model, S_)
    E = slot_energies([binding], np.zeros(S_.shape[0], dtype=np.int64), T)
    assert np.allclose(E[:, 0], pf.run.evaluate(S_), atol=1e-9)
    assert np.allclose(E[:, 1], pf.objective.evaluate(S_), atol=1e-9)
    assert np.allclose(E[:, 2], pf.violation.evaluate(S_), atol=1e-9)


# ============================================================================================
# run_penalty on the real problem adapters
# ============================================================================================
def test_run_penalty_toy_cardinality_reports_feasible_best():
    prob = toy.ToyCardinalityProblem()
    sched = S.geometric(2.0, 0.05, 4000)
    res = P.run_penalty(prob, strengths={"lam": 6.0}, backend="emulation", schedule=sched, seed=7,
                        n_replicas=6, feasible_start=True)
    assert res.n_physical == 6 and res.n_aux == 0
    assert np.all(res.feasible)  # feasible_start starts feasible and lam=6 keeps a feasible-best from tick 0
    assert np.nanmin(res.best_feasible_objective) == pytest.approx(-7.0, abs=1e-9)
    assert all(st == "ok" for st in res.status)


def test_run_penalty_matching_small_feasible_start_is_zero_violation():
    P1, P2, _, _ = similarity_deformation_pair(5, 0.03, np.random.default_rng(3))
    prob = MatchingProblem(point_pair_tensor(P1, P2, seed=3))
    pf = prob.penalty_formulation()
    model, lib = P.build_penalty_model(pf)
    binding = P.bind_penalty(model, pf)
    start = prob.random_feasible(seed=1, n=3)
    from nimbus.backends.base import clause_signs, slot_energies

    T = clause_signs(model, start)
    E = slot_energies([binding], np.zeros(3, dtype=np.int64), T)
    assert np.allclose(E[:, 2], 0.0, atol=1e-9)  # violation is 0 on feasible permutations
    assert np.allclose(E[:, 1], [prob.energy(row) for row in start], atol=1e-6)


def test_run_penalty_never_feasible_status_is_honest():
    pf = _cardinality_pf(4, 2, 1.0)
    model, lib = P.build_penalty_model(pf)
    binding = P.bind_penalty(model, pf)
    from nimbus.backends.base import get_backend

    be = get_backend("emulation")
    plan = be.prepare(model, library=lib)
    infeasible = -np.ones((1, 4), dtype=np.int8)  # 0 spins +1, k=2 required: infeasible
    state = be.init_state(plan, [binding], infeasible, [0], [0], [0])
    res = be.run(plan, state, S.constant(1.0, 1), seed=3)
    assert res.state.feasible_best_attempt[0] < 0  # never feasible yet


def test_conservative_strength_bound_and_sweep():
    prob = toy.ToyCardinalityProblem()
    obj = prob.polynomial()
    bound = P.conservative_strength_bound(obj, min_positive_violation=1.0, margin=1.5)
    j_sum = float(np.sum(np.abs(obj.clause_weights())))
    assert bound == pytest.approx(1.5 * 2.0 * j_sum)
    sched = S.geometric(2.0, 0.1, 500)
    rows = P.strength_sweep(prob, [1.0, 6.0, 12.0], backend="emulation", schedule=sched, seed=11, n_replicas=4)
    assert len(rows) == 3
    for row in rows:
        assert 0.0 <= row["feasible_fraction"] <= 1.0

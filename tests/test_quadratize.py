"""Rosenberg degree reduction (nimbus.baselines.quadratize): exhaustive recovery of the original
energy (including constants) after minimizing over auxiliaries, on random cubic/quartic
polynomials, a quartic N=6 autocorrelation polynomial, and a parity-check penalty."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from helpers_core import all_states, random_parity_code
from nimbus.baselines import quadratize as Q
from nimbus.problems.base import SpinPolynomial

pytestmark = pytest.mark.quick


def _random_poly(n: int, degree: int, n_terms: int, seed: int) -> SpinPolynomial:
    rng = np.random.default_rng(seed)
    supports, coeffs = [], []
    for _ in range(n_terms):
        sup = tuple(sorted(rng.choice(n, size=degree, replace=False).tolist()))
        supports.append(sup)
        coeffs.append(float(rng.integers(-5, 6)))
    supports.append(())
    coeffs.append(float(rng.integers(-3, 4)))
    return SpinPolynomial(n_spins=n, supports=tuple(supports), coeffs=np.asarray(coeffs, dtype=np.float64),
                          const=float(rng.integers(-3, 4)))


def _autocorrelation_n6() -> SpinPolynomial:
    """Aperiodic autocorrelation energy E = sum_{k=1}^{N-1} C_k^2, C_k = sum_i s_i s_{i+k}:
    C_k^2 = sum_{i,j} s_i s_{i+k} s_j s_{j+k}; i == j contributes a
    constant 1, i != j contributes a (possibly degree < 4, via s^2 = 1) spin monomial."""
    N = 6
    from collections import defaultdict

    agg = defaultdict(float)
    const = 0.0
    for k in range(1, N):
        idx = list(range(N - k))
        for i in idx:
            for j in idx:
                support = (i, i + k, j, j + k)
                cnt = {}
                for v in support:
                    cnt[v] = cnt.get(v, 0) + 1
                reduced = tuple(sorted(v for v, c in cnt.items() if c % 2))
                if reduced:
                    agg[reduced] += 1.0
                else:
                    const += 1.0
    supports = tuple(agg.keys())
    coeffs = np.asarray([agg[s] for s in supports], dtype=np.float64)
    return SpinPolynomial(n_spins=N, supports=supports, coeffs=coeffs, const=const)


def _exhaustive_check(poly: SpinPolynomial, *, strength=None):
    qr = Q.quadratize(poly, strength=strength)
    assert all(len(k) <= 2 for k in [tuple(s) for s in qr.poly.supports])
    for s in all_states(poly.n_spins):
        direct = float(poly.evaluate(s))
        best_e, best_aux = Q.recover_energy(qr, s)
        assert best_e == pytest.approx(direct, abs=1e-6), (s, direct, best_e)
        init_aux = Q.auxiliary_init(qr, s)
        full = np.concatenate([s.astype(np.int8), init_aux])
        e_init = float(qr.poly.evaluate(full))
        assert e_init == pytest.approx(direct, abs=1e-6), (s, direct, e_init, init_aux, best_aux)
    return qr


def test_quadratize_random_cubic_polynomials():
    for seed in range(5):
        poly = _random_poly(5, 3, 4, seed)
        qr = _exhaustive_check(poly)
        assert qr.n_aux >= 1


def test_quadratize_random_quartic_polynomial_multi_round():
    poly = _random_poly(5, 4, 3, seed=42)
    qr = _exhaustive_check(poly)
    assert qr.n_aux >= 2  # a degree-4 term needs at least two substitutions


def test_quadratize_autocorrelation_n6():
    poly = _autocorrelation_n6()
    assert max(len(s) for s in poly.supports) == 4
    _exhaustive_check(poly)


def test_quadratize_parity_check_penalty():
    n, row_weight = 8, 4
    c_mat = random_parity_code(n, 1, seed=9, row_weight=row_weight)
    row = np.flatnonzero(c_mat[0]).tolist()
    # violation = 1 - sign * T_row, T_row = prod_{i in row} s_i (zero iff the check is satisfied
    # with target sign +1); already degree == row_weight == 4, a good quadratization stress test.
    poly = SpinPolynomial(n_spins=n, supports=(tuple(row),), coeffs=np.array([-1.0]), const=1.0)
    _exhaustive_check(poly)


def test_quadratize_already_quadratic_is_a_no_op():
    poly = SpinPolynomial(n_spins=4, supports=((0, 1), (2,)), coeffs=np.array([2.0, -1.0]), const=0.5)
    qr = Q.quadratize(poly)
    assert qr.n_aux == 0
    assert qr.substitutions == []
    for s in all_states(4):
        assert float(qr.poly.evaluate(s)) == pytest.approx(float(poly.evaluate(s)), abs=1e-9)


def test_quadratize_penalty_wrapper_runs_through_penalty_module():
    from nimbus.baselines import penalty as P
    from nimbus.problems import toy_cardinality as toy

    prob = toy.ToyCardinalityProblem()
    pf = prob.penalty_formulation(lam=6.0)
    # toy's run polynomial is already degree <= 3 (triangle rewards); force at least one
    # reduction round by adding a synthetic quartic coupling on top for this test only.
    extra_support = (0, 1, 2, 3)
    run2 = SpinPolynomial(n_spins=pf.run.n_spins, supports=pf.run.supports + (extra_support,),
                          coeffs=np.concatenate([pf.run.coeffs, [0.3]]), const=pf.run.const)
    pf2 = pf.__class__(run=run2, objective=pf.objective, violation=pf.violation, n_physical=pf.n_physical,
                       ledger=dict(pf.ledger))
    qpf = Q.quadratize_penalty(pf2)
    assert qpf.run.n_spins > pf2.n_physical
    model, lib = P.build_penalty_model(qpf)
    binding = P.bind_penalty(model, qpf)
    assert binding.slot_names == ("run", "objective", "violation")

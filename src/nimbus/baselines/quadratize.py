"""Degree reduction of a spin polynomial to quadratic via Rosenberg substitution.

Rosenberg's trick replaces a Boolean product ``x_a x_b`` (``x in {0,1}``) that appears together with
a third variable by a fresh auxiliary ``z`` and a penalty ``M*(x_a x_b - 2 x_a z - 2 x_b z + 3 z)``
whose unique minimiser over ``z`` (for a large enough ``M``) is ``z = x_a AND x_b``; substituting
``z`` for the pair everywhere it occurs together reduces the maximum degree by (at least) one.

Spin polynomials are converted to the Boolean basis termwise (``s_i = 2 x_i - 1``, expanded only
over each term's own support, so the work is bounded by the term's degree, not by ``n_spins``),
reduced to quadratic in the Boolean basis by repeated substitution (greedy: at each round, the pair
appearing together in the most still-too-high-degree monomials is substituted first, so aux
variables are reused across terms), then mapped back to a quadratic ``SpinPolynomial`` over the
physical + auxiliary spins (``x_i = (1 + s_i) / 2``).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from nimbus.problems.base import PenaltyFormulation, SpinPolynomial

__all__ = ["QuadraticReduction", "quadratize", "quadratize_penalty", "recover_energy", "auxiliary_init"]


# ============================================================================================
# spin <-> Boolean basis conversion (termwise, no full n-variable transform)
# ============================================================================================
def _term_to_boolean(support: tuple, coeff: float) -> dict:
    """prod_{i in support} s_i = prod_{i in support} (2 x_i - 1), expanded over the term's support."""
    out: dict = {}
    k = len(support)
    for r in range(k + 1):
        for subset in itertools.combinations(support, r):
            c = coeff * (2.0 ** r) * ((-1.0) ** (k - r))
            key = frozenset(subset)
            out[key] = out.get(key, 0.0) + c
    return out


def _poly_to_boolean(poly: SpinPolynomial) -> dict:
    agg = {frozenset(): float(poly.const)}
    for sup, c in zip(poly.supports, poly.coeffs):
        for key, val in _term_to_boolean(tuple(sup), float(c)).items():
            agg[key] = agg.get(key, 0.0) + val
    return {k: v for k, v in agg.items() if abs(v) > 0 or not k}


def _boolean_to_spin(mono: dict, n_spins: int) -> SpinPolynomial:
    """x_i = (1 + s_i) / 2 for every Boolean variable (degree <= 2 monomials only)."""
    acc = {(): 0.0}

    def add(sup: tuple, val: float):
        acc[sup] = acc.get(sup, 0.0) + val

    for key, val in mono.items():
        items = sorted(key)
        if len(items) == 0:
            add((), val)
        elif len(items) == 1:
            i = items[0]
            add((), val / 2.0)
            add((i,), val / 2.0)
        elif len(items) == 2:
            i, j = items
            add((), val / 4.0)
            add((i,), val / 4.0)
            add((j,), val / 4.0)
            add((i, j), val / 4.0)
        else:
            raise ValueError(f"monomial {key} has degree {len(items)} > 2; reduction did not finish")
    const = acc.pop((), 0.0)
    supports = tuple(acc.keys())
    coeffs = np.asarray([acc[s] for s in supports], dtype=np.float64)
    return SpinPolynomial(n_spins=n_spins, supports=supports, coeffs=coeffs, const=float(const))


# ============================================================================================
# reduction
# ============================================================================================
@dataclass
class QuadraticReduction:
    """A quadratic ``SpinPolynomial`` (``poly``, over ``n_physical + n_aux`` spins) equivalent to
    the original higher-order polynomial after minimizing over the auxiliaries.

    ``substitutions``: ordered list of ``{"aux": z, "pair": (a, b), "strength": M}`` (``a``, ``b``
    may themselves be earlier auxiliaries), the ledger needed to recover the original energy.
    """

    poly: SpinPolynomial
    n_physical: int
    n_aux: int
    substitutions: list = field(default_factory=list)
    original_const: float = 0.0


def quadratize(poly: SpinPolynomial, *, strength: float | None = None) -> QuadraticReduction:
    """Reduce ``poly`` (any degree) to a quadratic ``SpinPolynomial`` over physical + auxiliary spins.

    ``strength``: fixed penalty ``M`` for every substitution if given; otherwise chosen fresh at each
    substitution round as ``1 + 2 * sum(|coeff| for monomials touching the pair)`` (reported per
    substitution in the ledger), which dominates every term the substitution can affect so the
    minimiser over the fresh auxiliary is exactly the Boolean AND of the pair.
    """
    n_phys = int(poly.n_spins)
    mono = _poly_to_boolean(poly)
    next_var = n_phys
    subs = []
    while True:
        pair_counts: Counter = Counter()
        pair_weight: Counter = Counter()
        for key, val in mono.items():
            if len(key) >= 3:
                items = sorted(key)
                for a, b in itertools.combinations(items, 2):
                    pair_counts[(a, b)] += 1
                    pair_weight[(a, b)] += abs(val)
        if not pair_counts:
            break
        (a, b), _ = pair_counts.most_common(1)[0]
        z = next_var
        next_var += 1
        M = float(strength) if strength is not None else 1.0 + 2.0 * pair_weight[(a, b)]
        new_mono: dict = {}
        for key, val in mono.items():
            if a in key and b in key:
                newkey = frozenset((key - {a, b}) | {z})
            else:
                newkey = key
            new_mono[newkey] = new_mono.get(newkey, 0.0) + val
        for key, val in ((frozenset((a, b)), M), (frozenset((a, z)), -2.0 * M), (frozenset((b, z)), -2.0 * M),
                         (frozenset((z,)), 3.0 * M)):
            new_mono[key] = new_mono.get(key, 0.0) + val
        mono = {k: v for k, v in new_mono.items() if abs(v) > 1e-15 or not k}
        subs.append({"aux": z, "pair": (a, b), "strength": M})
    n_aux = next_var - n_phys
    quad = _boolean_to_spin(mono, n_phys + n_aux)
    return QuadraticReduction(poly=quad, n_physical=n_phys, n_aux=n_aux, substitutions=subs,
                              original_const=float(poly.const))


def auxiliary_init(qr: QuadraticReduction, physical_spins) -> np.ndarray:
    """Auxiliary spins consistent with the Rosenberg relations (``z = AND(a, b)``) given a physical
    start, computed by walking the substitution ledger in order (each pair may reference an earlier
    auxiliary). Returns int8[n_aux]."""
    s = np.asarray(physical_spins, dtype=np.int64)
    full = np.concatenate([s, np.zeros(qr.n_aux, dtype=np.int64)])
    for sub in qr.substitutions:
        a, b, z = sub["pair"][0], sub["pair"][1], sub["aux"]
        xa, xb = (1 + full[a]) // 2, (1 + full[b]) // 2
        full[z] = 2 * (xa & xb) - 1
    return full[qr.n_physical:qr.n_physical + qr.n_aux].astype(np.int8)


def recover_energy(qr: QuadraticReduction, physical_spins) -> tuple:
    """Exhaustively minimize ``qr.poly`` over all ``2**n_aux`` auxiliary assignments for one physical
    state; returns ``(min_energy, best_aux)``. Exhaustive -- intended for small ``n_aux`` (tests)."""
    s = np.asarray(physical_spins, dtype=np.int8).reshape(-1)
    if qr.n_aux == 0:
        return float(qr.poly.evaluate(s)), np.zeros(0, dtype=np.int8)
    best_e, best_aux = math.inf, None
    for bits in itertools.product((-1, 1), repeat=qr.n_aux):
        full = np.concatenate([s, np.asarray(bits, dtype=np.int8)])
        e = float(qr.poly.evaluate(full))
        if e < best_e:
            best_e, best_aux = e, np.asarray(bits, dtype=np.int8)
    return best_e, best_aux


# ============================================================================================
# penalty-formulation wrapper (runnable through nimbus.baselines.penalty)
# ============================================================================================
def quadratize_penalty(pf: PenaltyFormulation, *, strength: float | None = None) -> PenaltyFormulation:
    """Quadratize ``pf.run``; ``objective``/``violation`` are untouched SCORE clauses over the
    physical spins only (``pf.n_physical`` is unchanged; the returned formulation has
    ``run.n_spins == pf.n_physical + total_aux``)."""
    qr = quadratize(pf.run, strength=strength)
    ledger = dict(pf.ledger)
    ledger.update({"quadratization": "rosenberg", "n_aux": qr.n_aux, "n_physical": qr.n_physical,
                  "substitutions": list(qr.substitutions)})
    return PenaltyFormulation(run=qr.poly, objective=pf.objective, violation=pf.violation,
                              n_physical=pf.n_physical, ledger=ledger)

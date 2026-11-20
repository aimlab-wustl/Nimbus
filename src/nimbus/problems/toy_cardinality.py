"""The manuscript's six-spin exact-cardinality toy (Results, Visualizing operating principle using a toy problem; Supplementary
Methods, An example toy problem).

Six spins (0-based ``0..5``, manuscript vertices 1..6). Exactly three are selected (``s_i = +1``).
Selecting all of triangle ``{0,1,2}`` (manuscript ``{1,2,3}``) rewards 4; selecting all of triangle
``{2,3,4}`` (manuscript ``{3,4,5}``) rewards 7 (vertex 5 / manuscript 6 completes neither)::

    E0(s) = -4 * prod_{i in {0,1,2}} (1+s_i)/2 - 7 * prod_{i in {2,3,4}} (1+s_i)/2

under the manuscript convention ``E = E_const - sum_k J_k T_k``. Expanding the two triangle
indicators independently (``x_i = (1+s_i)/2``, inclusion-exclusion over each triangle's power set,
dropping the empty subset into the constant) and merging the one shared singleton (spin 2, the
vertex common to both triangles) gives 13 nonconstant clauses: 5 singletons, 6 pairs, 2 triples;
constant -11/8; the shared singleton has ``J = 1/2 + 7/8 = 11/8``. This module performs that
expansion independently of ``tests/helpers_core.toy_structure`` (which exists only for compiler/
backend tests) and cross-checks it in ``tests/test_problem_toy_cardinality.py`` against the project
schema fixture ``tests/fixtures/spinpoly/six_spin_toy.yaml``.

Feasible set: the 20 states with exactly 3 spins at +1 (``exact_cardinality`` family). Starting
selection ``{1,2,3}`` (1-based, manuscript) has E = -4; the unique optimum ``{3,4,5}`` has E = -7.

Transition libraries (Methods, An example toy problem):

- ``pair_exchanges``: all C(6,2) = 15 pairwise exchanges among the six spins (involutions,
  preserve cardinality).
- ``augmented``: the same 15 exchanges plus one extra group that exchanges the two disjoint pairs
  (0,3)+(1,4) (manuscript (1,4)+(2,5)) *simultaneously*, compiled from the net product
  (Methods, An example toy problem), not as two separate moves.

``penalty_formulation`` builds the single-spin-flip penalty baseline ``E0 + lambda*(n+ - 3)^2``
(default ``lambda = 6``, the manuscript's example strength) with ``violation = (n+ - 3)^2``, which
expands to ``const=3/2`` plus ``1/2`` on every one of the 15 pairs (derived below).

INIT-stream convention for :meth:`ToyCardinalityProblem.random_feasible`: one ``stream_integers``
draw per sample, counter ``(block, 0, instance, sample_index)``, indexing the 20
``itertools.combinations(range(6), 3)`` states in ascending order (the same order as
:meth:`enumerate_feasible`).
"""

from __future__ import annotations

import itertools

import numpy as np

from nimbus.core.rng import INIT, stream_integers
from nimbus.problems.base import PenaltyFormulation, Problem, SpinPolynomial
from nimbus.transitions import builders as B
from nimbus.transitions.library import TransitionLibrary

PROBLEM_KIND = "toy_cardinality"

N_SPINS = 6
K_SELECTED = 3
TRIANGLES = ((4.0, (0, 1, 2)), (7.0, (2, 3, 4)))
#: 1-based manuscript vertex sets, for documentation/tests only.
START_SELECTION = (1, 2, 3)
OPTIMAL_SELECTION = (3, 4, 5)

_LIBRARY_NAMES = ("pair_exchanges", "augmented")


def spins_from_selection(selected_1based) -> np.ndarray:
    """6-spin int8 state with the given 1-based manuscript vertices at +1."""
    s = -np.ones(N_SPINS, dtype=np.int8)
    for v in selected_1based:
        s[int(v) - 1] = 1
    return s


def _toy_polynomial() -> SpinPolynomial:
    """Own analytic expansion (inclusion-exclusion over each rewarded triangle)."""
    terms: dict[tuple, float] = {}
    const = 0.0
    for reward, tri in TRIANGLES:
        for r in range(len(tri) + 1):
            for sub in itertools.combinations(tri, r):
                c = -reward / 8.0
                if not sub:
                    const += c
                else:
                    terms[sub] = terms.get(sub, 0.0) + c
    supports = tuple(sorted(terms))
    coeffs = np.array([terms[s] for s in supports], dtype=np.float64)
    return SpinPolynomial(n_spins=N_SPINS, supports=supports, coeffs=coeffs, const=const)


def _cardinality_violation_polynomial(n_spins: int, block, k: int) -> SpinPolynomial:
    """(n+ - k)^2 for a block of spins requiring exactly k selected, expanded in spins.

    n+ - k = (m/2 - k) + (sum_i s_i)/2 with m = |block|; squaring and using s_i^2 = 1 gives::

        const = (m/2-k)^2 + m/4
        singleton_i = (m/2-k)      (i in block)
        pair_ij = 1/2              (i < j in block)
    """
    block = [int(i) for i in block]
    m = len(block)
    a = m / 2.0 - k
    const = a * a + m / 4.0
    terms: dict[tuple, float] = {}
    if a != 0.0:
        for i in block:
            terms[(i,)] = a
    for i, j in itertools.combinations(sorted(block), 2):
        terms[(i, j)] = 0.5
    supports = tuple(sorted(terms))
    coeffs = np.array([terms[s] for s in supports], dtype=np.float64)
    return SpinPolynomial(n_spins=n_spins, supports=supports, coeffs=coeffs, const=const)


def _add_polynomials(n_spins: int, parts) -> SpinPolynomial:
    """Sum of (poly, scale) pairs, merging equal supports."""
    terms: dict[tuple, float] = {}
    const = 0.0
    for poly, scale in parts:
        const += scale * poly.const
        for sup, c in zip(poly.supports, poly.coeffs):
            terms[sup] = terms.get(sup, 0.0) + scale * float(c)
    supports = tuple(sorted(terms))
    coeffs = np.array([terms[s] for s in supports], dtype=np.float64)
    return SpinPolynomial(n_spins=n_spins, supports=supports, coeffs=coeffs, const=const)


class ToyCardinalityProblem(Problem):
    """The paper's six-spin exact-cardinality toy. See module docstring.

    Parameters
    ----------
    library: default transition library name used by :meth:`transition_library` with no name
        override is not used; ``library_names()`` lists the supported names.
    """

    family = "exact_cardinality"
    name = "toy_cardinality"
    n_spins = N_SPINS
    n_instances = 1

    def __init__(self):
        self._poly = _toy_polynomial()

    # ---- objective and constraints -------------------------------------------------------------
    def polynomial(self, instance: int = 0) -> SpinPolynomial:
        if instance != 0:
            raise ValueError("toy_cardinality has a single instance")
        return self._poly

    def identities(self):
        sign = -1 if (N_SPINS - K_SELECTED) % 2 else 1
        return [(tuple(range(N_SPINS)), sign)]

    def is_feasible(self, spins) -> bool:
        s = np.asarray(spins)
        return int(np.count_nonzero(s == 1)) == K_SELECTED

    # ---- independent evaluation -------------------------------------------------------------
    def native_objective(self, spins, instance: int = 0) -> float:
        """Direct reward sum from triangle membership (independent of the polynomial)."""
        s = np.asarray(spins)
        e = 0.0
        for reward, tri in TRIANGLES:
            if all(s[i] == 1 for i in tri):
                e -= reward
        return float(e)

    def decode(self, spins) -> dict:
        s = np.asarray(spins)
        selected = tuple(int(i) + 1 for i in range(N_SPINS) if s[i] == 1)  # 1-based manuscript ids
        return {"selected": selected}

    # ---- initialization, transitions ---------------------------------------------------------
    def random_feasible(self, seed: int, n: int, instance: int = 0) -> np.ndarray:
        combos = list(itertools.combinations(range(N_SPINS), K_SELECTED))
        out = np.full((int(n), N_SPINS), -1, dtype=np.int8)
        for m in range(int(n)):
            idx = int(stream_integers(seed, INIT, 1, len(combos), c2=int(instance), c3=m)[0])
            out[m, list(combos[idx])] = 1
        return out

    def library_names(self) -> list:
        return list(_LIBRARY_NAMES)

    def transition_library(self, name: str, **options) -> TransitionLibrary:
        if name == "pair_exchanges":
            return B.pair_exchanges(range(N_SPINS), pattern="complete", name="toy_pair_exchanges")
        if name == "augmented":
            base = B.pair_exchanges(range(N_SPINS), pattern="complete", name="toy_pair_exchanges")
            extra = B.simultaneous_exchanges([(0, 3), (1, 4)])
            double = TransitionLibrary(N_SPINS, [extra], ["swap(0,3)+swap(1,4)"],
                                       name="toy_double_exchange", template="simultaneous_exchange",
                                       params={"pairs": [[0, 3], [1, 4]]})
            return TransitionLibrary.concat(base, double, name="toy_augmented", template="union")
        raise ValueError(f"unknown toy_cardinality library {name!r}; known: {_LIBRARY_NAMES}")

    def enumerate_feasible(self) -> np.ndarray:
        combos = list(itertools.combinations(range(N_SPINS), K_SELECTED))
        out = np.full((len(combos), N_SPINS), -1, dtype=np.int8)
        for row, combo in enumerate(combos):
            out[row, list(combo)] = 1
        return out

    # ---- baselines -----------------------------------------------------------------------------
    def penalty_formulation(self, instance: int = 0, lam: float = 6.0, **strengths) -> PenaltyFormulation:
        if strengths:
            raise ValueError(f"unknown penalty strength keys: {sorted(strengths)}")
        objective = self.polynomial(instance)
        violation = _cardinality_violation_polynomial(N_SPINS, range(N_SPINS), K_SELECTED)
        run = _add_polynomials(N_SPINS, [(objective, 1.0), (violation, float(lam))])
        return PenaltyFormulation(run=run, objective=objective, violation=violation, n_physical=N_SPINS,
                                  ledger={"lambda": float(lam), "block": list(range(N_SPINS)), "k": K_SELECTED})


def from_config(cfg: dict, base_dir) -> ToyCardinalityProblem:
    """``cfg`` fields (all optional besides ``kind``): ``kind: "toy_cardinality"``.

    No other fields are read at construction time (the instance is fixed by the manuscript); the
    transition library and penalty strength are chosen per-run via :meth:`transition_library` and
    :meth:`penalty_formulation`, not at adapter construction. Unknown fields raise ``ConfigError``.
    """
    from nimbus.core.errors import ConfigError

    cfg = dict(cfg)
    kind = cfg.pop("kind", None)
    if kind != PROBLEM_KIND:
        raise ConfigError(f"toy_cardinality.from_config: expected kind={PROBLEM_KIND!r}, got {kind!r}")
    if cfg:
        raise ConfigError(f"toy_cardinality: unknown config fields {sorted(cfg)}")
    return ToyCardinalityProblem()

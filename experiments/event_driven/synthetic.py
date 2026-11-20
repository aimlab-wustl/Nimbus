"""Synthetic problems of the event-driven measurements (Fig. S9b-d) and the 2D +-J spin glass (Fig. 6, Fig. S9a).

Random spin polynomials with a chosen interaction order, variable degree and move type, built as a
``ConstrainedBlocksProblem`` (or a permutation problem with row exchanges) and compiled like every other workload.

Parameters of :func:`make_problem`
  k        interaction order (number of spins per term): 2, 3, 4
  degree   mean number of terms that contain a given spin; m = round(n * degree / k) terms
  move     'flip'          every spin free, single spin flips
           'onehot'        spins in one-hot groups of size q (exactly one +1 per group); all pair exchanges inside each
                           group (q (q - 1) / 2 moves per group)
           'onehot_path'   the same groups, adjacent exchanges only (q - 1 moves per group)
           'rowswap'       q x q permutation matrix, all exchanges of two rows
           'rowswap_path'  q x q permutation matrix, exchanges of adjacent rows only (q - 1 moves)
  coupling 'gauss' (N(0, 1) coefficients, no exactly degenerate moves) or 'pm1' (+-1)
For one-hot groups and permutations each term takes its k spins from k distinct groups (rows). The generator is
numpy's PCG64 seeded from (seed, k, degree, move, q) only, so the same arguments give the same problem.
:func:`exact_optimum` enumerates the feasible set when it has at most 2^24 states.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from nimbus.problems.constrained_blocks import ConstrainedBlocksProblem

MOVES = ("flip", "onehot", "onehot_path", "rowswap", "rowswap_path")

#: the six settings of the synthetic family (Fig. S9b-d)
SETTINGS = {
    "syn_k2_flip": dict(n_vars=24, k=2, degree=3.0, move="flip"),
    "syn_k3_flip": dict(n_vars=24, k=3, degree=3.0, move="flip"),
    "syn_k4_flip": dict(n_vars=24, k=4, degree=4.0, move="flip"),
    "syn_k2_onehot": dict(n_vars=40, k=2, degree=3.0, move="onehot", q=5),
    "syn_k2_rowswap_path_q9": dict(k=2, degree=6.0, move="rowswap_path", q=9),
    "syn_k2_rowswap_path_q16": dict(k=2, degree=6.0, move="rowswap_path", q=16),
}


class PermutationProblem(ConstrainedBlocksProblem):
    """q x q permutation matrix (spin a*q + p = row a in column p) moved by exchanging whole rows, with a random
    objective: the structure of the graph-matching row-exchange libraries. Only the row constraints are declared as
    blocks; every row exchange keeps the column constraints, which ``is_feasible`` checks."""

    def __init__(self, q, supports, coeffs, pattern, name):
        self.q = int(q)
        self.pattern = pattern
        rows = [list(range(a * q, (a + 1) * q)) for a in range(q)]
        super().__init__(q * q, supports, coeffs, 0.0, exact_cardinality=[(r, 1) for r in rows], name=name)
        self.rows = rows

    def is_feasible(self, spins) -> bool:
        x = (np.asarray(spins).reshape(self.q, self.q) > 0)
        return bool((x.sum(axis=0) == 1).all() and (x.sum(axis=1) == 1).all())

    def random_feasible(self, seed: int, n: int, instance: int = 0) -> np.ndarray:
        out = -np.ones((int(n), self.n_spins), dtype=np.int8)
        for m in range(int(n)):
            perm = np.random.default_rng([int(seed), int(instance), m, 7]).permutation(self.q)
            out[m, np.arange(self.q) * self.q + perm] = 1
        return out

    def transition_library(self, name: str = "default", **options):
        from nimbus.transitions import builders as TB
        return TB.block_exchanges(self.rows, pattern=self.pattern, n_spins=self.n_spins)


def make_problem(n_vars=24, k=2, degree=3.0, move="flip", q=5, coupling="gauss", seed=0):
    """``(problem, library_options, info)``. ``n_vars`` is rounded down to a multiple of q for one-hot moves."""
    if move not in MOVES:
        raise ValueError(move)
    rng = np.random.default_rng([int(seed), int(k), int(round(degree * 100)), MOVES.index(move), int(q)])
    if move.startswith("rowswap"):
        q = int(q)
        n = q * q
        blocks = [list(range(b * q, (b + 1) * q)) for b in range(q)]
    elif move == "flip":
        n = int(n_vars)
        blocks = []
    else:
        nb = int(n_vars) // q
        n = nb * q
        blocks = [list(range(b * q, (b + 1) * q)) for b in range(nb)]
        if nb < k:
            raise ValueError("need at least k groups")
    m = int(round(n * degree / k))
    terms = _balanced_terms(rng, n, k, m, int(round(degree)), q if blocks else None)
    supports = sorted(terms)
    coeffs = rng.standard_normal(len(supports)) if coupling == "gauss" else rng.choice([-1.0, 1.0], size=len(supports))
    name = f"synth_k{k}_d{degree:g}_{move}_s{seed}"
    if move.startswith("rowswap"):
        prob = PermutationProblem(q, supports, coeffs, "path" if move == "rowswap_path" else "complete", name)
        lib_opts = {}
    else:
        prob = ConstrainedBlocksProblem(n, supports, coeffs, 0.0, exact_cardinality=[(b, 1) for b in blocks], name=name)
        lib_opts = {"cardinality_pattern": "path" if move == "onehot_path" else "complete"}
    info = dict(n=n, k=k, degree=degree, move=move, q=q if blocks else None, coupling=coupling, seed=seed,
                terms=len(supports), realised_degree=float(k * len(supports) / n))
    return prob, lib_opts, info


def _balanced_terms(rng, n, k, m, d, q):
    """m distinct k-subsets with every spin in d of them where possible (no isolated spin): each term draws its spins
    with probability proportional to their remaining degree; with groups (q) the k spins come from k distinct
    groups. Restarts on a dead end."""
    for _ in range(200):
        rem = np.full(n, d, dtype=np.int64)
        terms = set()
        ok = True
        for _t in range(m):
            for _try in range(50):
                w = rem.astype(np.float64)
                if q is not None:
                    pick, used = [], set()
                    for _j in range(k):
                        ww = w.copy()
                        for b in used:
                            ww[b * q:(b + 1) * q] = 0
                        if ww.sum() <= 0:
                            break
                        i = int(rng.choice(n, p=ww / ww.sum()))
                        pick.append(i)
                        used.add(i // q)
                        w[i] = 0
                else:
                    if np.count_nonzero(w) < k:
                        break
                    pick = rng.choice(n, size=k, replace=False, p=w / w.sum()).tolist()
                sup = tuple(sorted(int(i) for i in pick))
                if len(sup) == k and sup not in terms:
                    terms.add(sup)
                    rem[list(sup)] -= 1
                    break
            else:
                ok = False
                break
        if ok and len(terms) == m:
            return terms
    raise RuntimeError("could not draw a degree-balanced term set")


def exact_optimum(prob, max_states=1 << 24, chunk=1 << 18):
    """Minimum energy over the feasible set by enumeration (None if the set has more than ``max_states`` states)."""
    poly = prob.polynomial(0)
    sup = [np.asarray(s, dtype=np.int64) for s in poly.supports]
    c = np.asarray(poly.coeffs, dtype=np.float64)
    blocks = [list(sp) for sp, _ in prob.exact_cardinality]
    n = prob.n_spins
    if isinstance(prob, PermutationProblem):
        q = prob.q
        if math.factorial(q) > max_states:
            return None
        perms = np.array(list(itertools.permutations(range(q))), dtype=np.int64)
        best = np.inf
        for start in range(0, perms.shape[0], chunk):
            P = perms[start:start + chunk]
            S = -np.ones((P.shape[0], n), dtype=np.int8)
            S[np.arange(P.shape[0])[:, None], np.arange(q)[None, :] * q + P] = 1
            best = min(best, float(_energy(S, sup, c, poly.const).min()))
        return best
    if blocks:
        q = len(blocks[0])
        total = q ** len(blocks)
        if total > max_states:
            return None
        best = np.inf
        for start in range(0, total, chunk):
            idx = np.arange(start, min(total, start + chunk), dtype=np.int64)
            S = -np.ones((idx.size, n), dtype=np.int8)
            x = idx.copy()
            for b in blocks:
                S[np.arange(idx.size), np.asarray(b)[x % q]] = 1
                x //= q
            best = min(best, float(_energy(S, sup, c, poly.const).min()))
        return best
    total = 1 << n
    if total > max_states:
        return None
    best = np.inf
    bitpos = np.arange(n, dtype=np.int64)
    for start in range(0, total, chunk):
        idx = np.arange(start, min(total, start + chunk), dtype=np.int64)
        S = (1 - 2 * ((idx[:, None] >> bitpos[None, :]) & 1)).astype(np.int8)
        best = min(best, float(_energy(S, sup, c, poly.const).min()))
    return best


def _energy(S, sup, c, const):
    e = np.full(S.shape[0], float(const))
    for s, w in zip(sup, c):
        e += w * np.prod(S[:, s], axis=1, dtype=np.int64)
    return e


def make_spin_glass(L=16, seed=0):
    """2D +-J spin glass on an L x L torus (degree 4): every spin free, single spin flips, couplings +1 or -1 with
    equal probability. The +-J couplings allow moves with zero energy change."""
    rng = np.random.default_rng([int(seed), 2, 2024, int(L)])
    n = L * L
    sup = []
    for x in range(L):
        for y in range(L):
            i = x * L + y
            sup.append(tuple(sorted((i, ((x + 1) % L) * L + y))))
            sup.append(tuple(sorted((i, x * L + (y + 1) % L))))
    coeffs = rng.choice([-1.0, 1.0], size=len(sup))
    prob = ConstrainedBlocksProblem(n, sup, coeffs, 0.0, name=f"spin_glass_L{L}_s{seed}")
    info = dict(n=n, k=2, degree=4.0, move="flip", q=None, coupling="pm1", seed=seed, terms=len(sup),
                realised_degree=4.0, lattice=f"{L}x{L} torus")
    return prob, {"cardinality_pattern": "complete"}, info


if __name__ == "__main__":  # structural summary of the synthetic family
    from nimbus.compiler import compile_model
    from nimbus.core.model import structural_resources
    for name, kw in SETTINGS.items():
        p, lo, info = make_problem(**kw)
        mdl = compile_model(p.structure(), p.transition_library(**lo), identities=())
        print(name, info, "R", mdl.n_groups, "routes", mdl.n_routes, "fan_in", structural_resources(mdl)["fan_in_mean"])

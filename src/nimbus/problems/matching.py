"""Graph matching with third-order (triangle) similarity terms as a one-hot ``assignment_permutation`` problem
with a cubic objective.

Encoding: source ``i`` matched to target ``a`` is spin ``i*n + a`` (``transitions.builders.onehot_index(i,
a, n)``, row-major); ``x = (1+s)/2``. Rows are sources, so a row exchange
swaps the targets of two sources. The objective is ``E = -F`` with the sparse third-order tensor
``F(X) = sum_m v_m X[i1,a1] X[i2,a2] X[i3,a3]`` (``formats/matching_tensor.py``). Expanding every product
``x_p x_q x_r = (1 + s_p)(1 + s_q)(1 + s_r)/8`` gives clauses of degree <= 3 (all subsets of the three
spins), consolidated per support (the compiler needs unique supports).

Move libraries (row exchanges):

* ``path``: the backbone of the ``n - 1`` exchanges of consecutive rows (path library);
* ``complete``: all ``C(n,2)`` row exchanges (the backbone with all shortcuts; complete library);
* :func:`shortcut_library`: the backbone plus ``K`` informed or random shortcuts, optionally with a fixed
  backbone proposal share;
* ``tree``: deprecated alias of ``path``, also accepting ``pattern="star"|"random_tree"`` for other spanning trees;
* ``double_exchanges={...}`` option: composite groups exchanging two disjoint row pairs at once.

``penalty_formulation`` adds row and column one-hot penalties (cubic objective is kept: no
quadratization). ``enumerate_optimum`` certifies the optimum for ``n <= 10`` by enumerating all ``n!``
permutations (numba, parallel).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from nimbus.core.rng import INIT, group_index, stream_words
from nimbus.problems.base import PenaltyFormulation, Problem, SpinPolynomial
from nimbus.problems.formats.matching_tensor import MatchingTensor, evaluate_F, load_tensor
from nimbus.transitions import builders as B
from nimbus.transitions.library import TransitionLibrary

PROBLEM_KIND = "matching"

__all__ = ["PROBLEM_KIND", "MatchingProblem", "enumerate_optimum", "double_exchange_library", "from_config",
           "backbone_edges", "informed_pairs", "random_pairs", "shortcut_library", "describe_library"]


def _cardinality_violation_polynomial(n_spins: int, block, k: int) -> SpinPolynomial:
    """(n+ - k)^2 over ``block`` (see ``toy_cardinality``'s identical derivation)."""
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
    terms: dict[tuple, float] = {}
    const = 0.0
    for poly, scale in parts:
        const += scale * poly.const
        for sup, c in zip(poly.supports, poly.coeffs):
            terms[sup] = terms.get(sup, 0.0) + scale * float(c)
    supports = tuple(sorted(terms))
    coeffs = np.array([terms[s] for s in supports], dtype=np.float64)
    return SpinPolynomial(n_spins=n_spins, supports=supports, coeffs=coeffs, const=const)


def random_permutation(seed: int, instance: int, sample_index: int, P: int) -> list:
    """Uniform random permutation of ``range(P)`` from the INIT stream: Fisher-Yates over the words
    ``stream_words(seed, INIT, 2 * (P - 1), c2=instance, c3=sample_index)``, step ``i`` drawing
    ``j = i + group_index(w[2i], w[2i + 1], P - i)``."""
    if P <= 1:
        return list(range(P))
    w = stream_words(seed, INIT, 2 * (P - 1), c2=int(instance), c3=int(sample_index))
    perm = list(range(P))
    for i in range(P - 1):
        j = i + group_index(int(w[2 * i]), int(w[2 * i + 1]), P - i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _expand_objective(t: MatchingTensor) -> SpinPolynomial:
    n = t.n
    N = n * n
    spin = (t.triples[:, :, 0].astype(np.int64) * n + t.triples[:, :, 1].astype(np.int64))
    spin.sort(axis=1)
    c = -t.values / 8.0  # E = -F
    const = float(c.sum())
    keys, coefs = [], []
    # base-(N+1) code of a support padded with the sentinel N (sorted, so sentinels trail)
    S = N
    for r in (1, 2, 3):
        for sub in itertools.combinations(range(3), r):
            cols = [spin[:, j] for j in sub] + [np.full(len(spin), S, dtype=np.int64)] * (3 - r)
            keys.append((cols[0] * (S + 1) + cols[1]) * (S + 1) + cols[2])
            coefs.append(c)
    key = np.concatenate(keys)
    cf = np.concatenate(coefs)
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, weights=cf, minlength=uk.size)
    supports = []
    for k in uk.tolist():
        a, b, cc = k // ((S + 1) ** 2), (k // (S + 1)) % (S + 1), k % (S + 1)
        supports.append(tuple(x for x in (a, b, cc) if x != S))
    order = sorted(range(len(supports)), key=lambda i: supports[i])
    return SpinPolynomial(n_spins=N, supports=tuple(supports[i] for i in order),
                          coeffs=np.array([w[i] for i in order], dtype=np.float64), const=const)


# ---------------------------------------------------------------------------------------------
# exact enumeration
# ---------------------------------------------------------------------------------------------
def _enum_kernel_factory():
    import numba as nb

    @nb.njit(cache=True, nogil=True)
    def _best_for_prefix(first, n, ti, ta, tv, order_ptr, order_idx):
        # permutations with perm[0] = first, all others by lexicographic Heap-free recursion (iterative next_permutation)
        perm = np.empty(n, np.int64)
        perm[0] = first
        k = 1
        for a in range(n):
            if a != first:
                perm[k] = a
                k += 1
        best = -1e300
        count = 0
        best_perm = perm.copy()
        while True:
            f = 0.0
            for m in range(tv.shape[0]):
                if perm[ti[m, 0]] == ta[m, 0] and perm[ti[m, 1]] == ta[m, 1] and perm[ti[m, 2]] == ta[m, 2]:
                    f += tv[m]
            if f > best:
                best = f
                count = 1
                best_perm[:] = perm
            elif f == best:
                count += 1
            # next permutation of perm[1:]
            i = n - 2
            while i >= 1 and perm[i] >= perm[i + 1]:
                i -= 1
            if i < 1:
                break
            j = n - 1
            while perm[j] <= perm[i]:
                j -= 1
            tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp
            lo, hi = i + 1, n - 1
            while lo < hi:
                tmp = perm[lo]; perm[lo] = perm[hi]; perm[hi] = tmp
                lo += 1; hi -= 1
        return best, count, best_perm

    @nb.njit(cache=True, parallel=True)
    def _enum(n, ti, ta, tv):
        bests = np.empty(n)
        counts = np.empty(n, np.int64)
        perms = np.empty((n, n), np.int64)
        dummy = np.zeros(1, np.int64)
        for f0 in nb.prange(n):
            b, c, p = _best_for_prefix(f0, n, ti, ta, tv, dummy, dummy)
            bests[f0] = b
            counts[f0] = c
            perms[f0] = p
        return bests, counts, perms

    return _enum


_ENUM = None


def enumerate_optimum(tensor: MatchingTensor, max_n: int = 11) -> dict:
    """Certified optimum of ``F`` over all ``n!`` permutations (n <= ``max_n``; n = 10 is 3.6 M, seconds).

    Returns ``{"F_opt", "n_optimal", "perm" (one optimizer, perm[i] = a), "n_perms"}``. Optimal
    permutations are counted by exact equality of the float sum, which is evaluated in the same
    entry order for every permutation (so ties are exact up to identical summation order).
    """
    global _ENUM
    n = tensor.n
    if n > max_n:
        raise ValueError(f"enumerate_optimum: n={n} > max_n={max_n}")
    if _ENUM is None:
        _ENUM = _enum_kernel_factory()
    ti = np.ascontiguousarray(tensor.triples[:, :, 0].astype(np.int64))
    ta = np.ascontiguousarray(tensor.triples[:, :, 1].astype(np.int64))
    bests, counts, perms = _ENUM(n, ti, ta, np.ascontiguousarray(tensor.values))
    top = float(bests.max())
    sel = np.flatnonzero(bests == top)
    return {"F_opt": top, "n_optimal": int(counts[sel].sum()), "perm": perms[sel[0]].astype(int).tolist(),
            "n_perms": int(np.prod(np.arange(1, n + 1, dtype=np.float64)))}


# ---------------------------------------------------------------------------------------------
# libraries
# ---------------------------------------------------------------------------------------------
def double_exchange_library(n: int, pairs_of_pairs) -> TransitionLibrary:
    """Groups exchanging two disjoint row pairs ``((a,b),(c,d))`` simultaneously (an involution)."""
    rows = B.onehot_rows(n)
    groups, labels = [], []
    for (a, b), (c, d) in pairs_of_pairs:
        if len({a, b, c, d}) != 4:
            raise ValueError("double exchange needs four distinct rows")
        pairs = list(zip(rows[a], rows[b])) + list(zip(rows[c], rows[d]))
        groups.append(B.simultaneous_exchanges(pairs))
        labels.append(f"row_swap2({a},{b};{c},{d})")
    return TransitionLibrary(n * n, groups, labels, name=f"double_row_exchange_P{n}", template="double_row_exchange",
                             params={"P": n, "pairs_of_pairs": [[list(p), list(q)] for p, q in pairs_of_pairs],
                                     "one_hot": True})


class MatchingProblem(Problem):
    family = "assignment_permutation"
    n_instances = 1

    def __init__(self, tensor: MatchingTensor, *, name: str = "matching"):
        self.tensor = tensor
        self.P = int(tensor.n)
        self.n_spins = self.P * self.P
        self.name = name
        self._poly = _expand_objective(tensor)

    def polynomial(self, instance: int = 0) -> SpinPolynomial:
        if instance != 0:
            raise ValueError("MatchingProblem has a single instance")
        return self._poly

    def identities(self):
        sign = -1 if (self.P - 1) % 2 else 1
        return ([(tuple(r), sign) for r in B.onehot_rows(self.P)] +
                [(tuple(c), sign) for c in B.onehot_columns(self.P)])

    def is_feasible(self, spins) -> bool:
        x = (np.asarray(spins).reshape(self.P, self.P).astype(np.int64) + 1) // 2
        return bool(np.all(x.sum(axis=1) == 1) and np.all(x.sum(axis=0) == 1))

    def native_objective(self, spins, instance: int = 0) -> float:
        """Native objective ``-F`` evaluated directly from the tensor on the decoded permutation."""
        return -float(evaluate_F(self.tensor, self.decode(spins)["perm"]))

    def decode(self, spins) -> dict:
        from nimbus.core.errors import InfeasibleState

        if not self.is_feasible(spins):
            raise InfeasibleState("MatchingProblem.decode: state is not a one-hot permutation")
        x = (np.asarray(spins).reshape(self.P, self.P).astype(np.int64) + 1) // 2
        return {"perm": np.argmax(x, axis=1).astype(np.int64)}  # perm[i] = target of source i

    def random_feasible(self, seed: int, n: int, instance: int = 0) -> np.ndarray:
        P = self.P
        out = np.full((int(n), self.n_spins), -1, dtype=np.int8)
        for m in range(int(n)):
            perm = random_permutation(seed, instance, m, P)  # perm[p] = a (INIT stream, Fisher-Yates)
            for i, a in enumerate(perm):
                out[m, B.onehot_index(i, a, P)] = 1
        return out

    def library_names(self) -> list:
        return ["complete", "path"]

    def transition_library(self, name: str, *, pattern: str | None = None, seed: int | None = None,
                           double_exchanges: dict | None = None) -> TransitionLibrary:
        """``name``: ``"path"`` (backbone alone), ``"complete"`` (backbone with all shortcuts) or the deprecated
        ``"tree"`` (``pattern`` ``"path"`` (default), ``"star"`` or ``"random_tree"``, seeded)."""
        P = self.P
        if name == "complete":
            lib = B.row_exchanges(P, pattern="complete")
        elif name == "path":
            if pattern not in (None, "path"):
                raise ValueError("the path library has no 'pattern' option; use name='tree' for other spanning trees")
            lib = B.row_exchanges(P, pattern="path", seed=seed)
        elif name == "tree":
            import warnings

            warnings.warn("matching library 'tree' is deprecated; use 'path' (the backbone)", DeprecationWarning,
                          stacklevel=2)
            lib = B.row_exchanges(P, pattern=pattern or "path", seed=seed)
        else:
            raise ValueError(f"unknown matching library {name!r}; known: {self.library_names()}")
        if double_exchanges:
            opts = dict(double_exchanges)
            count, rseed = int(opts.pop("count")), int(opts.pop("seed", 0))
            if opts:
                raise ValueError(f"unknown double_exchanges options {sorted(opts)}")
            rng = np.random.default_rng(rseed)
            seen, chosen = set(), []
            while len(chosen) < count:
                a, b, c, d = rng.choice(P, size=4, replace=False)
                key = tuple(sorted([tuple(sorted((int(a), int(b)))), tuple(sorted((int(c), int(d))))]))
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(key)
            lib = TransitionLibrary.concat(lib, double_exchange_library(P, chosen), name=f"{lib.name}+double")
        return lib

    def penalty_formulation(self, instance: int = 0, lambda_r: float | None = None,
                            lambda_c: float | None = None, **strengths) -> PenaltyFormulation:
        if strengths:
            raise ValueError(f"unknown penalty strength keys: {sorted(strengths)}")
        P = self.P
        objective = self.polynomial(instance)
        row_viol = _add_polynomials(self.n_spins, [(_cardinality_violation_polynomial(self.n_spins, blk, 1), 1.0)
                                                   for blk in B.onehot_rows(P)])
        col_viol = _add_polynomials(self.n_spins, [(_cardinality_violation_polynomial(self.n_spins, blk, 1), 1.0)
                                                   for blk in B.onehot_columns(P)])
        if lambda_r is None or lambda_c is None:
            j_sum = float(np.sum(np.abs(objective.clause_weights())))
            default = 1.5 * 2.0 * j_sum
            lambda_r = default if lambda_r is None else lambda_r
            lambda_c = default if lambda_c is None else lambda_c
        violation = _add_polynomials(self.n_spins, [(row_viol, float(lambda_r)), (col_viol, float(lambda_c))])
        run = _add_polynomials(self.n_spins, [(objective, 1.0), (violation, 1.0)])
        return PenaltyFormulation(run=run, objective=objective, violation=violation, n_physical=self.n_spins,
                                  ledger={"lambda_r": float(lambda_r), "lambda_c": float(lambda_c),
                                          "bound": "margin(1.5) * 2*sum|J_objective| / min_positive_violation(1)",
                                          "quadratic_already": False, "degree": 3})

    def metadata(self):
        md = super().metadata()
        md.update({"P": self.P, "tensor_nnz": self.tensor.nnz})
        return md


# shortcut libraries
# ---------------------------------------------------------------------------------------------
def backbone_edges(n: int) -> list:
    """Row pairs of the path backbone: ``(i, i + 1)`` for ``i = 0 .. n - 2``."""
    return [(i, i + 1) for i in range(n - 1)]


def informed_pairs(tensor: MatchingTensor, k: int, exclude: set) -> list:
    """The ``k`` informed shortcuts: row pairs chosen from the tensor alone (no optimum, no ground truth).

    For every source row ``i`` the candidate-target distribution is ``p_i(a) = U[i,a] / sum_a U[i,a]``, with
    ``U[i,a]`` the summed ``|value|`` of all tensor entries that contain the assignment ``(i, a)``. The overlap of rows
    ``i`` and ``j`` is ``sum_a min(p_i(a), p_j(a))`` (rows that compete for the same targets). The ``k`` pairs with the
    largest overlap that are not in ``exclude`` are returned (ties: lowest indices). Cost ``O(nnz + n^3)``.
    """
    n = int(tensor.n)
    U = np.zeros((n, n))
    absv = np.abs(tensor.values)
    for pos in range(3):
        np.add.at(U, (tensor.triples[:, pos, 0], tensor.triples[:, pos, 1]), absv)
    s = U.sum(axis=1, keepdims=True)
    Pm = U / np.where(s > 0, s, 1.0)
    cand = []
    for i in range(n):
        ov = np.minimum(Pm[i][None, :], Pm[i + 1:]).sum(axis=1)     # overlap with rows j > i
        for off, o in enumerate(ov):
            j = i + 1 + off
            if (i, j) not in exclude:
                cand.append((-o, i, j))
    cand.sort()
    return [(i, j) for _, i, j in cand[:k]]


def random_pairs(n: int, k: int, exclude: set, seed: int) -> list:
    """``k`` uniformly random row pairs outside ``exclude`` (``numpy.random.default_rng(seed)``), sorted."""
    allp = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in exclude]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(allp), size=min(k, len(allp)), replace=False)
    return [allp[t] for t in sorted(idx.tolist())]


def shortcut_library(problem: "MatchingProblem", spec: dict, tensor: MatchingTensor | None = None):
    """Move library from a specification dict; returns ``(TransitionLibrary, info)``.

    ``spec``:

    * ``{"base": "complete"}``: all row exchanges (complete library);
    * ``{"base": "path"}``: the backbone of ``n - 1`` consecutive-row exchanges (path library);
    * ``{"base": "path", "extra": "informed" | "random", "k": K, "seed": s, "share": f}``: the backbone plus ``K``
      shortcuts (:func:`informed_pairs`, or :func:`random_pairs` with seed ``s``). ``share`` fixes the backbone's
      proposal share: proposals are uniform over the group list, so the backbone is repeated
      ``c = max(1, round(f / (1 - f) * K / (n - 1)))`` times. ``share`` absent or ``None``: uniform proposals over
      the union.

    ``info``: ``base``, ``k``, ``copies`` (``c``), ``select_s`` (time to choose the shortcuts), ``pairs`` (the added
    row pairs) and ``R_nodup`` (number of distinct exchanges).
    """
    import time

    n = int(problem.P)
    base = spec["base"]
    info = {"base": base, "k": int(spec.get("k", 0)), "copies": 1, "select_s": 0.0}
    if base == "complete":
        lib = B.row_exchanges(n, pattern="complete")
        info["R_nodup"] = lib.n_groups
        return lib, info
    if base != "path":
        raise ValueError(f"unknown base library {base!r}")
    bb = B.row_exchanges(n, pattern="path")
    extra = spec.get("extra")
    if not extra:
        info["R_nodup"] = bb.n_groups
        return bb, info
    k = int(spec["k"])
    excl = set(backbone_edges(n))
    t0 = time.perf_counter()
    if extra == "informed":
        pairs = informed_pairs(tensor if tensor is not None else problem.tensor, k, excl)
    elif extra == "random":
        pairs = random_pairs(n, k, excl, int(spec.get("seed", 0)))
    else:
        raise ValueError(f"unknown shortcut kind {extra!r}")
    info["select_s"] = time.perf_counter() - t0
    info["pairs"] = pairs
    add = B.row_exchanges(n, pattern="edges", edges=pairs)
    share = spec.get("share")
    c = 1
    if share is not None:
        c = max(1, int(round(share / (1.0 - share) * len(pairs) / (n - 1))))
    info["copies"] = c
    info["R_nodup"] = bb.n_groups + add.n_groups
    lib = TransitionLibrary.concat(*([bb] * c + [add]), name=f"path+{extra}{k}x{c}")
    return lib, info


def describe_library(spec: dict) -> str:
    """Short label of a library specification, e.g. ``complete``, ``path``, ``path+informed40@share0.5``."""
    if spec["base"] != "path" or not spec.get("extra"):
        return spec["base"]
    sh = spec.get("share")
    return f"path+{spec['extra']}{spec['k']}" + (f"@share{sh:g}" if sh is not None else "")


def from_config(cfg: dict, base_dir) -> MatchingProblem:
    """``cfg``: ``kind: "matching"`` and ``tensor: <path to .npz>`` (relative to ``base_dir``)."""
    from nimbus.core.errors import ConfigError

    cfg = dict(cfg)
    kind = cfg.pop("kind", None)
    if kind != PROBLEM_KIND:
        raise ConfigError(f"matching.from_config: expected kind={PROBLEM_KIND!r}, got {kind!r}")
    path = cfg.pop("tensor", None)
    if path is None:
        raise ConfigError("matching: 'tensor' (npz path) is required")
    if cfg:
        raise ConfigError(f"matching: unknown config fields {sorted(cfg)}")
    p = Path(base_dir) / path
    return MatchingProblem(load_tensor(p), name=p.stem)

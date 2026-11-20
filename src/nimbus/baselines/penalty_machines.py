"""Penalty-machine baselines for permutation (graph-matching) problems: N-HOIM style and NeuroSA style.

Both machines solve the same penalized polynomial: the matching objective plus row and column one-hot penalty terms of
strength ``lam`` (:meth:`nimbus.problems.matching.MatchingProblem.penalty_formulation`). They differ in the order of the
interactions and in the update rule.

``kind="nhoim"`` (N-HOIM style)
    The third-order objective terms are kept. Every tick draws a fresh noise threshold for every spin; the spins whose
    flip energy passes their threshold form the candidate set, and one of them, chosen uniformly, flips
    (rejection-free single-spin dynamics of the higher-order neuromorphic Ising machine).
``kind="neurosa"`` (NeuroSA style)
    The penalized polynomial is first quadratized exactly (Rosenberg reduction: every cubic Boolean monomial
    ``x_a x_b x_c`` becomes ``z_ab x_c`` with the penalty ``M (x_a x_b - 2 x_a z - 2 x_b z + 3 z)``, one auxiliary
    spin per used pair). Each iteration then picks one spin uniformly among the physical and auxiliary spins and
    accepts its flip when ``dE < tau * Exp(u)``.

Iteration accounting: one iteration is one single-spin update test (one flip-energy evaluation against a threshold).
An N-HOIM tick tests all ``N_total`` spins, so a budget of ``B`` iterations is ``B // N_total`` ticks.

Acceptance: ``dE < T * (-ln(u + 1e-6))`` with ``u ~ U[0, 1)``, so ``P(accept) = exp(-dE / T)``.
Schedule (Fowler-Nordheim shape): ``T(k) = T_end ln 2 / ln(1 + q0 + (1 - q0) k / (K - 1))`` with
``q0 = 2 ** (1 / r) - 1`` and ``r = T_start / T_end``, so ``T(0) = T_start`` and ``T(K - 1) = T_end``.

Start state: uniformly random +-1 physical spins (not feasible in general); auxiliary spins are set to the AND of
their pair, so the quadratized energy equals the penalized energy at the start.

Per replica the machines record the best objective seen in a feasible state (NaN if no feasible state was visited),
the iteration index at which it was reached, the feasibility of the final state and the number of accepted flips.
The objective is maintained incrementally and recomputed exactly at the end (``drift``).
"""
from __future__ import annotations

import itertools
import math
import time
from collections import Counter

import numba as nb
import numpy as np
from numba import njit, prange

from nimbus.baselines.penalty import random_physical_start
from nimbus.baselines.quadratize import _boolean_to_spin, _poly_to_boolean

__all__ = ["KINDS", "PenaltyMachine", "quadratize_pairs", "csr_terms", "aux_from_phys", "fn_temperature"]

KINDS = ("nhoim", "neurosa")
EPS = 1e-6
LN2 = math.log(2.0)


# --------------------------------------------------------------------------------------------- structures
def csr_terms(supports, coeffs, n_spins):
    """Term list -> ``(t_ptr, t_idx, coef, s_ptr, s_term)``. Empty supports (constants) are skipped."""
    keep = [i for i, s in enumerate(supports) if len(s) > 0]
    t_ptr = np.zeros(len(keep) + 1, dtype=np.int64)
    flat = []
    for k, i in enumerate(keep):
        flat.extend(supports[i])
        t_ptr[k + 1] = t_ptr[k] + len(supports[i])
    t_idx = np.asarray(flat, dtype=np.int64)
    c = np.asarray([coeffs[i] for i in keep], dtype=np.float64)
    deg = np.zeros(n_spins, dtype=np.int64)
    np.add.at(deg, t_idx, 1)
    s_ptr = np.zeros(n_spins + 1, dtype=np.int64)
    s_ptr[1:] = np.cumsum(deg)
    s_term = np.empty(len(t_idx), dtype=np.int64)
    fill = s_ptr[:-1].copy()
    for t in range(len(keep)):
        for p in range(t_ptr[t], t_ptr[t + 1]):
            j = t_idx[p]
            s_term[fill[j]] = t
            fill[j] += 1
    return t_ptr, t_idx, c, s_ptr, s_term


def quadratize_pairs(poly, mult=1.0):
    """Exact Rosenberg quadratization with one auxiliary spin per used pair (static frequency greedy, one pass).

    Each Boolean monomial of degree 3 (``x_a x_b x_c``) picks the pair of its three spins with the largest global
    count (ties: lowest indices) and becomes ``z_ab x_c``. ``M_ab = mult * (1 + sum |coefficients of the monomials
    that use z_ab|)``; ``mult >= 1`` suffices for exactness (a wrong ``z`` gains at most that sum).
    Returns ``(quadratic SpinPolynomial, pairs, n_aux)``.
    """
    n_phys = int(poly.n_spins)
    mono = _poly_to_boolean(poly)
    cnt = Counter()
    for key in mono:
        if len(key) == 3:
            for a, b in itertools.combinations(sorted(key), 2):
                cnt[(a, b)] += 1
        elif len(key) > 3:
            raise ValueError("degree > 3")
    new, used, weight = {}, {}, Counter()
    for key, val in mono.items():
        if len(key) == 3:
            it = sorted(key)
            pr = max(itertools.combinations(it, 2), key=lambda p: (cnt[p], -p[0], -p[1]))
            c = [x for x in it if x not in pr][0]
            if pr not in used:
                used[pr] = n_phys + len(used)
            weight[pr] += abs(val)
            nk = frozenset((used[pr], c))
            new[nk] = new.get(nk, 0.0) + val
        else:
            new[key] = new.get(key, 0.0) + val
    pairs = [None] * len(used)
    for (a, b), z in used.items():
        M = float(mult) * (1.0 + weight[(a, b)])
        pairs[z - n_phys] = (a, b)
        for k, v in ((frozenset((a, b)), M), (frozenset((a, z)), -2.0 * M), (frozenset((b, z)), -2.0 * M),
                     (frozenset((z,)), 3.0 * M)):
            new[k] = new.get(k, 0.0) + v
    new = {k: v for k, v in new.items() if abs(v) > 1e-15 or not k}
    return _boolean_to_spin(new, n_phys + len(used)), pairs, len(used)


def aux_from_phys(spins, pairs, n_phys):
    """Auxiliary spins ``z = AND(x_a, x_b)`` in the +-1 convention, in pair order."""
    s = np.asarray(spins)
    out = np.empty((s.shape[0], len(pairs)), dtype=np.int8)
    for k, (a, b) in enumerate(pairs):
        out[:, k] = np.where((s[:, a] == 1) & (s[:, b] == 1), 1, -1)
    return out


def fn_temperature(k, K, T_start, T_end):
    """Temperature of iteration ``k`` of ``K`` (the schedule of the kernels, for plotting and checks)."""
    r = max(T_start / T_end, 1.0 + 1e-9)
    q0 = 2.0 ** (1.0 / r) - 1.0
    frac = k / (K - 1.0) if K > 1 else 0.0
    return T_end * LN2 / np.log1p(q0 + (1.0 - q0) * frac)


# --------------------------------------------------------------------------------------------- kernel
@njit(cache=True)
def _flip(c, s, T, G, cr, t_ptr, t_idx, s_ptr, s_term):
    s[c] = -s[c]
    for q in range(s_ptr[c], s_ptr[c + 1]):
        t = s_term[q]
        d = -2.0 * cr[t] * T[t]
        for p in range(t_ptr[t], t_ptr[t + 1]):
            G[t_idx[p]] += d
        T[t] = -T[t]


@njit(cache=True)
def _one(mode, K, Ts, Te, q0, nphys, n, s, t_ptr, t_idx, cr, s_ptr, s_term, o_tptr, o_tidx, oc, o_sptr, o_sterm,
         o_const, seed):
    np.random.seed(seed)
    ntot = s.shape[0]
    nt = cr.shape[0]
    T = np.empty(nt)
    G = np.zeros(ntot)
    for t in range(nt):
        pr = 1.0
        for p in range(t_ptr[t], t_ptr[t + 1]):
            pr *= s[t_idx[p]]
        T[t] = pr
        for p in range(t_ptr[t], t_ptr[t + 1]):
            G[t_idx[p]] += cr[t] * pr
    no = oc.shape[0]
    To = np.empty(no)
    Eo = o_const
    for t in range(no):
        pr = 1.0
        for p in range(o_tptr[t], o_tptr[t + 1]):
            pr *= s[o_tidx[p]]
        To[t] = pr
        Eo += oc[t] * pr
    rowc = np.zeros(n, dtype=np.int64)
    colc = np.zeros(n, dtype=np.int64)
    for c in range(nphys):
        if s[c] == 1:
            rowc[c // n] += 1
            colc[c % n] += 1
    nbad = 0
    for i in range(n):
        if rowc[i] != 1:
            nbad += 1
        if colc[i] != 1:
            nbad += 1
    bestE = np.inf
    bestk = -1
    if nbad == 0:
        bestE = Eo
        bestk = 0
    acc = 0
    lnq = LN2
    for k in range(K):
        frac = k / (K - 1.0) if K > 1 else 0.0
        tau = Te * lnq / np.log1p(q0 + (1.0 - q0) * frac)
        chosen = -1
        if mode == 0:
            cnt = 0
            for j in range(ntot):
                u = np.random.random()
                if -2.0 * G[j] < -tau * np.log(u + 1e-6):
                    cnt += 1
                    if np.random.random() * cnt < 1.0:
                        chosen = j
        else:
            j = int(np.random.random() * ntot)
            u = np.random.random()
            if -2.0 * G[j] < -tau * np.log(u + 1e-6):
                chosen = j
        if chosen < 0:
            continue
        c = chosen
        acc += 1
        if c < nphys:                           # objective energy and feasibility bookkeeping (before the spin changes)
            for q in range(o_sptr[c], o_sptr[c + 1]):
                t = o_sterm[q]
                Eo += -2.0 * oc[t] * To[t]
                To[t] = -To[t]
            d = 1 if s[c] == -1 else -1
            i = c // n
            j2 = c % n
            before = (1 if rowc[i] != 1 else 0) + (1 if colc[j2] != 1 else 0)
            rowc[i] += d
            colc[j2] += d
            after = (1 if rowc[i] != 1 else 0) + (1 if colc[j2] != 1 else 0)
            nbad += after - before
        _flip(c, s, T, G, cr, t_ptr, t_idx, s_ptr, s_term)
        if nbad == 0 and Eo < bestE:
            bestE = Eo
            bestk = k + 1
    Ec = o_const                                # exact recomputation of the objective at the end (drift check)
    for t in range(no):
        pr = 1.0
        for p in range(o_tptr[t], o_tptr[t + 1]):
            pr *= s[o_tidx[p]]
        Ec += oc[t] * pr
    return bestE, bestk, nbad, acc, Eo, Ec


@njit(parallel=True, cache=True)
def _run_all(mode, K, Ts, Te, q0, nphys, n, spins0, t_ptr, t_idx, cr, s_ptr, s_term, o_tptr, o_tidx, oc, o_sptr, o_sterm,
             o_const, seeds):
    R = spins0.shape[0]
    bestE = np.empty(R)
    bestk = np.empty(R, dtype=np.int64)
    nbad = np.empty(R, dtype=np.int64)
    acc = np.empty(R, dtype=np.int64)
    Eo = np.empty(R)
    Ec = np.empty(R)
    for r in prange(R):
        s = spins0[r].copy()
        a, b, c, d, e, f = _one(mode, K, Ts, Te, q0, nphys, n, s, t_ptr, t_idx, cr, s_ptr, s_term, o_tptr, o_tidx, oc,
                                o_sptr, o_sterm, o_const, seeds[r])
        bestE[r] = a
        bestk[r] = b
        nbad[r] = c
        acc[r] = d
        Eo[r] = e
        Ec[r] = f
    return bestE, bestk, nbad, acc, Eo, Ec


# --------------------------------------------------------------------------------------------- machine
class PenaltyMachine:
    """A penalty machine for one matching problem.

    Parameters
    ----------
    problem : MatchingProblem
    kind    : ``"nhoim"`` or ``"neurosa"``.
    lam     : penalty strength (same for the row and column one-hot penalties), in objective units.
    mult    : quadratization strength multiplier (NeuroSA style only; ``>= 1`` keeps the reduction exact).
    """

    def __init__(self, problem, kind: str, lam: float, mult: float = 1.0):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        pf = problem.penalty_formulation(lambda_r=lam, lambda_c=lam)
        self.kind, self.n, self.nphys = kind, int(problem.P), int(pf.n_physical)
        self.lam, self.mult = float(lam), float(mult)
        self.pairs, self.n_aux = [], 0
        t0 = time.perf_counter()
        poly = pf.run
        if kind == "neurosa":
            poly, self.pairs, self.n_aux = quadratize_pairs(pf.run, mult)
        self.t_quad = time.perf_counter() - t0
        self.ntot = int(poly.n_spins)
        self.run_poly = poly
        self.run_terms = csr_terms(poly.supports, poly.coeffs, self.ntot)
        o = pf.objective
        self.obj_terms = csr_terms(o.supports, o.coeffs, self.nphys)
        self.o_const = float(o.const)
        self.objective = o

    def start(self, seed: int, replicas: int) -> np.ndarray:
        """Start states ``int8[replicas, ntot]``: random physical spins, auxiliary spins consistent with them."""
        phys = random_physical_start(self.nphys, seed, replicas)
        if self.n_aux:
            return np.concatenate([phys, aux_from_phys(phys, self.pairs, self.nphys)], axis=1).astype(np.int8)
        return phys

    def iterations_to_steps(self, budget: int) -> int:
        """Kernel steps for a budget of ``budget`` iterations (ticks for N-HOIM style, single tests for NeuroSA)."""
        return max(1, int(budget) // self.ntot) if self.kind == "nhoim" else int(budget)

    def run(self, budget: int, T_start: float, T_end: float, replicas: int, seed: int, threads: int | None = None):
        """Anneal ``replicas`` independent replicas for ``budget`` iterations each.

        Returns a dict with ``bestF`` (best feasible objective per replica, NaN if never feasible), ``best_event``
        (iteration index of the best, -1 if none), ``final_feasible``, ``accepted``, ``K`` (kernel steps),
        ``ticks``, ``tests`` (iterations actually spent), ``drift`` and ``wall_s``.
        Replica ``r`` uses the NumPy legacy random stream seeded with ``(seed * 1000003 + r) mod (2^31 - 1)``.
        """
        if threads:
            nb.set_num_threads(int(threads))
        mode = 0 if self.kind == "nhoim" else 1
        K = self.iterations_to_steps(budget)
        r = max(T_start / T_end, 1.0 + 1e-9)
        q0 = 2.0 ** (1.0 / r) - 1.0
        spins0 = self.start(seed, replicas)
        seeds = (np.int64(seed) * 1000003 + np.arange(replicas, dtype=np.int64)) % (2 ** 31 - 1)
        t0 = time.perf_counter()
        out = _run_all(mode, int(K), float(T_start), float(T_end), float(q0), self.nphys, self.n, spins0,
                       *self.run_terms, *self.obj_terms, self.o_const, seeds)
        wall = time.perf_counter() - t0
        bestE, bestk, nbad, acc, Eo, Ec = out
        return dict(bestF=np.where(np.isfinite(bestE), -bestE, np.nan), best_event=bestk, final_feasible=(nbad == 0),
                    accepted=acc, K=int(K), ticks=int(K) if mode == 0 else 0, drift=float(np.max(np.abs(Eo - Ec))),
                    wall_s=wall, tests=int(K) * (self.ntot if mode == 0 else 1))

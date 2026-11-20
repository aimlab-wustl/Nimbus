"""Shared tiny-model builders and exhaustive architecture checkers (for compiler, emulator,
oracle and backend tests).

Everything here is independent of the compiler's internals: energies are evaluated directly from
term supports and weights, feasible sets are enumerated from the constraint definitions, and the
clause-change check recomputes T_k(mu_r(s)) T_k(s) by applying the library descriptor to a copy of
every feasible state.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from nimbus.core.model import ObjectiveStructure
from nimbus.transitions import builders as B
from nimbus.transitions.library import TransitionLibrary


# ============================================================================================
# generic
# ============================================================================================
def all_states(n: int) -> np.ndarray:
    """All 2^n spin states (int8, 2^n x n), lexicographic with -1 before +1."""
    return np.array(list(itertools.product((-1, 1), repeat=n)), dtype=np.int8).reshape(-1, n)


def direct_products(states, supports) -> np.ndarray:
    """T for each support by np.prod (independent of nimbus.core.algebra); (S, K) int64."""
    s = np.asarray(states, dtype=np.int64)
    out = np.ones((s.shape[0], len(supports)), dtype=np.int64)
    for k, sup in enumerate(supports):
        if len(sup):
            out[:, k] = np.prod(s[:, list(sup)], axis=1)
    return out


def direct_energy(structure: ObjectiveStructure, weights, const, states) -> np.ndarray:
    """E(s) = const - sum_t w_t prod_{i in support_t} s_i, straight from the term list."""
    T = direct_products(states, structure.terms)
    return float(const) - T.astype(np.float64) @ np.asarray(weights, dtype=np.float64)


def random_polynomial(n: int, n_terms: int, max_degree: int, seed: int, *, duplicates: bool = True,
                      constant: bool = True, integer: bool = False, spins=None):
    """Random spin polynomial (structure, term weights J, const) over ``spins`` (default all n).

    With ``duplicates`` some supports repeat (and some raw index lists contain repeated indices), so
    exact-support merging and parity canonicalization are exercised.
    """
    rng = np.random.default_rng(seed)
    pool = list(range(n)) if spins is None else list(spins)
    terms = []
    for _ in range(n_terms):
        d = int(rng.integers(1, max_degree + 1))
        sup = list(rng.choice(pool, size=min(d, len(pool)), replace=False))
        if duplicates and rng.random() < 0.15:
            sup = sup + [sup[0], sup[0]]  # s_i^2 = 1
        terms.append(tuple(int(x) for x in sup))
    if duplicates and terms:
        terms.append(terms[0])
    if constant:
        terms.append(())
    if integer:
        w = rng.integers(-5, 6, size=len(terms)).astype(np.float64)
    else:
        w = np.round(rng.normal(size=len(terms)), 3)
    return ObjectiveStructure(n_spins=n, terms=terms), w, float(rng.normal())


# ============================================================================================
# exact cardinality
# ============================================================================================
def exact_cardinality_states(n: int, k: int) -> np.ndarray:
    """All states with exactly k spins at +1."""
    rows = []
    for sel in itertools.combinations(range(n), k):
        s = -np.ones(n, dtype=np.int8)
        s[list(sel)] = 1
        rows.append(s)
    return np.array(rows, dtype=np.int8).reshape(-1, n)


def cardinality_identities(n: int, k: int):
    """prod_i s_i = (-1)^(n-k) on exact-k states."""
    return [(tuple(range(n)), -1 if (n - k) % 2 else 1)]


def toy_structure():
    """The paper's six-spin exact-cardinality toy (rewards 4 for {1,2,3}, 7 for {3,4,5}; 0-based spins).

    E = -4 x0 x1 x2 - 7 x2 x3 x4 with x = (1 + s)/2, expanded under E = E_const - sum J T.
    Returns (structure, J, const). Expected: 13 nonconstant clauses, const -11/8, J(s_2) = 11/8.
    """
    terms, w = [], []
    const = 0.0
    for reward, tri in ((4.0, (0, 1, 2)), (7.0, (2, 3, 4))):
        for r in range(0, 4):
            for sub in itertools.combinations(tri, r):
                c = -reward / 8.0  # E contribution of the monomial prod_{sub} s
                if not sub:
                    const += c
                else:
                    terms.append(sub)
                    w.append(-c)  # J = -coefficient
    return ObjectiveStructure(n_spins=6, terms=terms), np.asarray(w), const


# ============================================================================================
# parity codes
# ============================================================================================
def random_parity_code(n: int, r: int, seed: int, row_weight: int | None = None) -> np.ndarray:
    """Random r x n binary parity-check matrix (rows may be dependent), uint8."""
    rng = np.random.default_rng(seed)
    C = np.zeros((r, n), dtype=np.uint8)
    for i in range(r):
        w = row_weight or int(rng.integers(2, max(3, n // 2) + 1))
        C[i, rng.choice(n, size=min(w, n), replace=False)] = 1
    return C


def codewords(C, syndrome=None) -> np.ndarray:
    """All spin states s (y = (1 - s)/2) with C y = syndrome (mod 2), by brute force over 2^n."""
    C = np.asarray(C, dtype=np.int64)
    n = C.shape[1]
    b = np.zeros(C.shape[0], dtype=np.int64) if syndrome is None else np.asarray(syndrome, dtype=np.int64) % 2
    S = all_states(n)
    Y = (S < 0).astype(np.int64)
    ok = np.all((Y @ C.T) % 2 == b[None, :], axis=1)
    return S[ok]


def code_identities(C, syndrome=None):
    """Check-row identities prod_{i in row} s_i = (-1)^{b_row}."""
    C = np.asarray(C)
    b = np.zeros(C.shape[0], dtype=np.int64) if syndrome is None else np.asarray(syndrome) % 2
    return [(tuple(int(i) for i in np.flatnonzero(C[a])), -1 if b[a] else 1)
            for a in range(C.shape[0]) if np.any(C[a])]


# ============================================================================================
# one-hot assignment (permutation matrices)
# ============================================================================================
def onehot_identities(P: int):
    """Row and column products of a P x P one-hot permutation matrix: (-1)^(P-1) each."""
    sign = -1 if (P - 1) % 2 else 1
    return [(tuple(r), sign) for r in B.onehot_rows(P)] + [(tuple(c), sign) for c in B.onehot_columns(P)]


def permutation_states(P: int):
    """All P! tours as one-hot spin states; tour[p] = city at position p. Returns (states, tours)."""
    tours = list(itertools.permutations(range(P)))
    S = -np.ones((len(tours), P * P), dtype=np.int8)
    for t, tour in enumerate(tours):
        for p, a in enumerate(tour):
            S[t, B.onehot_index(a, p, P)] = 1
    return S, np.asarray(tours, dtype=np.int64)


def random_distances(P: int, seed: int, symmetric: bool = True, high: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = rng.integers(1, high, size=(P, P)).astype(np.float64)
    if symmetric:
        d = np.triu(d, 1)
        d = d + d.T
    np.fill_diagonal(d, 0.0)
    return d


def tour_length(d, tour) -> float:
    P = len(tour)
    return float(sum(d[tour[p], tour[(p + 1) % P]] for p in range(P)))


def cyclic_assignment_objective(d):
    """Local expansion of E = sum_p sum_{a != b} d_ab x_{a,p} x_{b,p+1} (x = (1+s)/2) into spin terms.

    Emits, per (p, a, b), the three nonconstant monomials s_{a,p}, s_{b,p+1}, s_{a,p} s_{b,p+1} with
    coefficient d_ab/4 (so singletons repeat and are merged by the compiler). Returns
    (structure, J, const) with J = -coefficient (E = E_const - sum J T).
    """
    d = np.asarray(d, dtype=np.float64)
    P = d.shape[0]
    terms, w = [], []
    const = 0.0
    for p in range(P):
        q = (p + 1) % P
        for a in range(P):
            for b in range(P):
                if a == b or d[a, b] == 0.0:
                    continue
                c = d[a, b] / 4.0
                i, j = B.onehot_index(a, p, P), B.onehot_index(b, q, P)
                const += c
                for sup in ((i,), (j,), (i, j)):
                    terms.append(sup)
                    w.append(-c)
    return ObjectiveStructure(n_spins=P * P, terms=terms), np.asarray(w), const


# ============================================================================================
# block-constraint families (chains, OR blocks, AMO blocks)
# ============================================================================================
def chain_states(m: int) -> np.ndarray:
    """Prefix-closed states of one chain: +1 ... +1 -1 ... -1 (m + 1 states)."""
    return np.array([[1] * k + [-1] * (m - k) for k in range(m + 1)], dtype=np.int8).reshape(-1, m)


def or_states(signs) -> np.ndarray:
    """States of one OR block with at least one satisfied literal (sign_j * s_j == +1)."""
    m = len(signs)
    S = all_states(m)
    sat = (S * np.asarray(signs, dtype=np.int8)[None, :]) == 1
    return S[np.any(sat, axis=1)]


def amo_states(m: int) -> np.ndarray:
    """States with at most one +1."""
    S = all_states(m)
    return S[np.count_nonzero(S == 1, axis=1) <= 1]


def product_states(blocks_states) -> np.ndarray:
    """Cartesian product of per-block state arrays (blocks occupy consecutive spin ranges)."""
    out = np.zeros((1, 0), dtype=np.int8)
    for bs in blocks_states:
        out = np.concatenate([np.repeat(out, bs.shape[0], axis=0), np.tile(bs, (out.shape[0], 1))], axis=1)
    return out


def section29_problem(blocks, tree: str = "path"):
    """Disjoint constraint blocks on consecutive spins.

    ``blocks``: list of ("chain", m) | ("or", [signs]) | ("amo", m). Returns
    (n_spins, library, states, is_feasible, block_spins) where library concatenates the block
    templates (``require_disjoint``) and states is the full feasible product set.
    """
    libs, per_block, spins_of = [], [], []
    off = 0
    sizes = []
    for kind, spec in blocks:
        m = len(spec) if kind == "or" else int(spec)
        sizes.append(m)
    n = sum(sizes)
    for (kind, spec), m in zip(blocks, sizes):
        sp_ = list(range(off, off + m))
        spins_of.append((kind, sp_, spec))
        if kind == "chain":
            libs.append(B.chain_precedence(sp_, n_spins=n))
            per_block.append(chain_states(m))
        elif kind == "or":
            lits = [(i, int(sg)) for i, sg in zip(sp_, spec)]
            if m > 1:
                libs.append(B.or_block(lits, tree=tree, n_spins=n))
            per_block.append(or_states(spec))
        elif kind == "amo":
            libs.append(B.amo_block(sp_, n_spins=n))
            per_block.append(amo_states(m))
        else:
            raise ValueError(kind)
        off += m
    lib = TransitionLibrary.concat(*libs, name="section29", require_disjoint=True)
    states = product_states(per_block)

    def is_feasible(s):
        s = np.asarray(s)
        for kind, sp_, spec in spins_of:
            x = s[sp_]
            if kind == "chain":
                if any(x[t] < x[t + 1] for t in range(len(x) - 1)):
                    return False
            elif kind == "or":
                if not np.any(x * np.asarray(spec) == 1):
                    return False
            elif np.count_nonzero(x == 1) > 1:
                return False
        return True

    return n, lib, states, is_feasible, spins_of


# ============================================================================================
# exhaustive architecture check
# ============================================================================================
def check_compiled_exhaustive(model, library, states, *, energy=None, binding=None, slot=0,
                              require_live_routes: bool = False, rtol=1e-9, atol=1e-9) -> dict:
    """Check the compiled arrays against direct recomputation on every feasible state.

    For every state s, group r and stored clause k:
      * compiled alpha (route arrays, controller signs T(s)) == (1 - T_k(mu_r(s)) T_k(s))/2, where mu_r(s)
        comes from the library descriptor and T from np.prod over the support (missing route => 0);
      * the decoder sigma (``decoder_toggles``) equals that alpha;
      * the compiled spin action (``apply_action``) equals the descriptor action;
      * if ``binding`` and ``energy`` (callable states -> E) are given: 2 v_r(s) == E(mu_r(s)) - E(s).
    ``require_live_routes``: every route must have alpha = 1 for at least one feasible state (exact Pi).
    Returns counts.
    """
    S = np.asarray(states, dtype=np.int8)
    supports = model.clause_supports()
    T = direct_products(S, supports).astype(np.int8)
    assert np.array_equal(model.clause_values(S), T), "clause_values disagree with direct products"
    alpha = model.route_alpha(T)
    v_all = model.inputs(T, binding, slot) if binding is not None else None
    E0 = energy(S) if energy is not None else None
    live = np.zeros(model.n_routes, dtype=bool)
    checks = 0
    for r in range(model.n_groups):
        mu = library.groups[r].apply(S)
        mu_c = model.apply_action(r, S, T)
        assert np.array_equal(mu, mu_c), f"compiled spin action of group {r} differs from the descriptor"
        Tm = direct_products(mu, supports)
        direct = (Tm != T).astype(np.uint8)
        a, b = int(model.route_ptr[r]), int(model.route_ptr[r + 1])
        compiled = np.zeros_like(direct)
        compiled[:, model.route_clause[a:b]] = alpha[:, a:b]
        if not np.array_equal(direct, compiled):
            st, k = np.argwhere(direct != compiled)[0]
            raise AssertionError(f"alpha mismatch: group {r} ({library.labels[r]}), clause {k} {supports[k]}, "
                                 f"state {S[st].tolist()}: direct {direct[st, k]} compiled {compiled[st, k]}")
        sigma = model.decoder_toggles(r, T)
        assert np.array_equal(sigma, compiled), f"decoder sigma differs from alpha for group {r}"
        live[a:b] |= np.any(alpha[:, a:b] == 1, axis=0)
        if v_all is not None and E0 is not None:
            dE = energy(mu) - E0
            if not np.allclose(2.0 * v_all[:, r], dE, rtol=rtol, atol=atol):
                st = int(np.flatnonzero(~np.isclose(2.0 * v_all[:, r], dE, rtol=rtol, atol=atol))[0])
                raise AssertionError(f"2v != dE for group {r} at state {S[st].tolist()}: "
                                     f"2v={2 * v_all[st, r]} dE={dE[st]}")
        checks += S.shape[0] * model.n_clauses
    if require_live_routes:
        assert live.all(), f"{int((~live).sum())} routes never switch on over the feasible states"
    return {"states": int(S.shape[0]), "groups": int(model.n_groups), "clauses": int(model.n_clauses),
            "clause_group_state_checks": int(checks), "dead_routes": int((~live).sum())}


def gate_only_supports(model):
    from nimbus.core.model import ROLE_GATE, ROLE_OBJECTIVE

    sup = model.clause_supports()
    return [sup[k] for k in range(model.n_clauses)
            if (model.clause_role[k] & ROLE_GATE) and not (model.clause_role[k] & ROLE_OBJECTIVE)]


def n_choose(n, k):
    return math.comb(n, k)

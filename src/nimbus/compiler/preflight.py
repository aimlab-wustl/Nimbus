"""Preflight resource estimation without materializing the architecture.

``preflight(structure, library, readout)`` returns *upper bounds* on the stored clauses, routes
and array memory that ``compile_model`` would materialize, using the class-specific gate-clause
bounds of Supplementary Information, Number of auxiliary clauses added by compilation where the library structure is recognized:

* only fixed masks: M_add = 0 (Supplementary Information, Number of auxiliary clauses added by compilation (parity constraints: fixed reversals));
* only fixed and conditional masks: every requested switch product is one of the groups'
  condition products, so M_add <= number of distinct nonempty conditions (block-constraint templates:
  at most one product per chain position, the singleton family of an OR block, at most m
  products per AMO block);
* only single pair exchanges: M_add <= |E| (Supplementary Information, Number of auxiliary clauses added by compilation (exact cardinality: pair exchanges));
* only exchanges of equal-sized blocks of a declared block partition (library params ``blocks``,
  e.g. one-hot row exchanges): M_add <= |E| |S_0| (Supplementary Information, Number of auxiliary clauses added by compilation (row exchanges)), with S_0 computed
  from the initial clause supports (objective, score and requested readout singletons), merging
  complementary position sets when every block product is a declared identity (one-hot
  permutations, Supplementary Information, Number of auxiliary clauses added by compilation (row exchanges)). For cyclic assignment objectives (adjacent positions on a cycle) |S_0| <= 2P (Supplementary Information, Number of auxiliary clauses added by compilation (row exchanges)).

Libraries outside these classes (general permutations such as segment reversals, mixed
templates, finite tables) get no structural gate bound: the report says so explicitly
(``M_add_upper = None``) instead of inventing one. Route bounds: an initial clause has at most one
route per group touching its support (exact count); a gate of support size <= g has at most
min(R, g * max spin degree) routes.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from nimbus.core.algebra import canonical_identities, canonical_support
from nimbus.transitions.descriptors import (
    KIND_CONDMASK,
    KIND_MASK,
    KIND_PERM,
    ConditionalMask,
    SignedPermutation,
)

__all__ = ["preflight", "estimate_model_bytes", "row_exchange_B"]


def estimate_model_bytes(*, n_spins: int, n_clauses: int, support_storage: int, n_routes: int,
                         n_weighted_routes: int, n_ctrl_routes: int, n_groups: int, act_nnz: int,
                         cond_nnz: int, n_terms: int) -> int:
    """Bytes of the CompiledModel arrays for the given counts."""
    M, R = int(n_clauses), int(n_groups)
    b = 8 * (M + 1) + 4 * int(support_storage) + M  # clause_ptr, clause_spins, clause_role
    b += 8 * (R + 1) + 8 * R + 9 * int(n_routes)  # route_ptr, route_wend, route_clause/ctrl/pol
    b += R + 8 * (R + 1) + 9 * int(act_nnz) + 8 * (R + 1) + 4 * int(cond_nnz) + R  # actions
    b += 4 * int(n_spins)  # readout_clause_of_spin
    b += 3 * 8 * (M + 1) + 8 * int(n_weighted_routes) + 8 * int(n_ctrl_routes) \
        + 8 * min(int(n_weighted_routes), int(n_ctrl_routes))  # dependency routes
    b += 4 * int(n_terms)
    return int(b)


def row_exchange_B(P: int, d: int) -> int:
    """B(P, d) of Supplementary Information, Number of auxiliary clauses added by compilation (row exchanges): position subsets of size <= d up to complements (one-hot rows)."""
    total = sum(math.comb(P, j) for j in range(1, min(d, (P - 1) // 2) + 1))
    if P % 2 == 0 and d >= P // 2:
        total += math.comb(P, P // 2) // 2
    return total


def _readout_list(readout, n: int) -> list[int]:
    # every spin is a stored singleton clause; None and 'all' are equivalent
    if readout is None:
        return list(range(n))
    if isinstance(readout, str):
        if readout != "all":
            raise ValueError(f"readout must be None, 'all' or a list of spins, got {readout!r}")
        return list(range(n))
    out = sorted({int(i) for i in readout})
    if out and (out[0] < 0 or out[-1] >= n):
        raise ValueError("readout spin index out of range")
    return out


def _is_transposition(g) -> tuple[int, int] | None:
    if isinstance(g, SignedPermutation) and len(g.dests) == 2 and g.signs == (1, 1):
        a, b = g.dests
        if g.srcs == (b, a):
            return (a, b)
    return None


def _block_pair(g, where: dict, blocks: list) -> tuple[int, int] | None:
    """(X, Y) if g exchanges complete blocks X and Y position-wise (signs +1), else None."""
    if not isinstance(g, SignedPermutation) or any(e != 1 for e in g.signs):
        return None
    d0 = g.dests[0]
    if d0 not in where:
        return None
    X, _p = where[d0]
    src_block = where.get(g.srcs[0])
    if src_block is None:
        return None
    Y = src_block[0]
    if X == Y or len(blocks[X]) != len(blocks[Y]):
        return None
    expect = {}
    for p, (u, v) in enumerate(zip(blocks[X], blocks[Y])):
        if u != v:
            expect[u] = v
            expect[v] = u
    got = dict(zip(g.dests, g.srcs))
    if got != expect:
        return None
    return (min(X, Y), max(X, Y))


def _gate_bound(library, initial: list[tuple[int, ...]], identities) -> dict:
    groups = library.groups
    kinds = {g.kind for g in groups}
    if kinds <= {KIND_MASK}:
        return {"M_add_upper": 0, "gate_support_max": 0, "source": "Supplementary Information, Number of auxiliary clauses added by compilation (parity constraints: fixed reversals)"}
    if kinds <= {KIND_MASK, KIND_CONDMASK}:
        conds = {g.cond for g in groups if isinstance(g, ConditionalMask) and g.cond}
        gmax = max((len(c) for c in conds), default=0)
        return {"M_add_upper": len(conds), "gate_support_max": gmax,
                "source": "conditional masks: every requested product is a group condition"}
    if kinds == {KIND_PERM}:
        pairs = [_is_transposition(g) for g in groups]
        if all(p is not None for p in pairs):
            return {"M_add_upper": len(set(pairs)), "gate_support_max": 2,
                    "source": "Supplementary Information, Number of auxiliary clauses added by compilation (exact cardinality: pair exchanges)"}
        blocks = library.params.get("blocks")
        if blocks:
            blocks = [list(map(int, b)) for b in blocks]
            where = {}
            for bi, blk in enumerate(blocks):
                for p, i in enumerate(blk):
                    where[i] = (bi, p)
            bp = [_block_pair(g, where, blocks) for g in groups]
            if all(p is not None for p in bp):
                E = sorted(set(bp))
                Eset = set(E)
                ids = {s for s, _c in canonical_identities(identities)}
                complement = all(tuple(sorted(b)) in ids for b in blocks)
                deg: dict[int, int] = {}
                for X, Y in E:
                    deg[X] = deg.get(X, 0) + 1
                    deg[Y] = deg.get(Y, 0) + 1
                S0 = set()
                seen_s: set = set()
                smax = 0

                def add(S: frozenset, P: int):
                    nonlocal smax
                    if not S or S in seen_s:
                        return
                    seen_s.add(S)
                    if complement:
                        if len(S) == P:
                            return  # product of two complete rows: constant on Omega
                        a = tuple(sorted(S))
                        b = tuple(sorted(set(range(P)) - S))
                        key = min(a, b, key=lambda t: (len(t), t))
                    else:
                        key = tuple(sorted(S))
                    S0.add(key)
                    smax = max(smax, len(S))

                for sup in initial:
                    rows: dict[int, set] = {}
                    for i in sup:
                        w = where.get(i)
                        if w is not None:
                            rows.setdefault(w[0], set()).add(w[1])
                    keys = list(rows)
                    for X in keys:
                        dX = deg.get(X, 0)
                        if not dX:
                            continue
                        n_in = 0
                        for Y in keys:
                            if Y != X and (min(X, Y), max(X, Y)) in Eset:
                                n_in += 1
                                if Y > X:
                                    add(frozenset(rows[X] ^ rows[Y]), len(blocks[X]))
                        if n_in < dX:  # some exchange partner Y holds none of the clause's spins: S = pos(X)
                            add(frozenset(rows[X]), len(blocks[X]))
                out = {"M_add_upper": len(E) * len(S0), "gate_support_max": 2 * smax, "n_exchanges": len(E),
                       "S0": len(S0), "complement_merging": complement,
                       "source": "Supplementary Information, Number of auxiliary clauses added by compilation (row exchanges), M_add <= |E| |S_0|"}
                sizes = {len(b) for b in blocks}
                if complement and len(sizes) == 1:
                    P = sizes.pop()
                    d = max((len(s) for s in initial), default=0)
                    out["closed_form_bound"] = len(E) * row_exchange_B(P, d)
                    out["closed_form_source"] = "Supplementary Information, Number of auxiliary clauses added by compilation (row exchanges): |E| B(P, d)"
                return out
    return {"M_add_upper": None, "gate_support_max": None,
            "source": "no structural gate-clause bound known for this library (general permutation, mixed "
                      "templates or finite table); materialize under CompileLimits"}


def preflight(structure, library, readout=None, *, identities: Sequence = (),
              score_structures: Mapping | None = None) -> dict:
    """Upper-bound resource estimates for compiling ``structure`` with ``library`` (no materialization)."""
    N = int(structure.n_spins)
    if library.n_spins != N:
        raise ValueError("structure and library have different numbers of spins")
    R = library.n_groups
    obj = structure.unique_supports()
    seen = set(obj)
    score = []
    for s in (score_structures or {}).values():
        for t in s.terms:
            if t and t not in seen:
                seen.add(t)
                score.append(t)
    readout_spins = _readout_list(readout, N)
    read = [(i,) for i in readout_spins if (i,) not in seen]
    initial = obj + score + read
    inc = library.touched_incidence()
    deg = [len(x) for x in inc]
    maxdeg = max(deg, default=0)

    def touch(sup):
        if len(sup) == 1:
            return deg[sup[0]]
        u = set()
        for i in sup:
            u.update(inc[i])
        return len(u)

    w_routes = sum(touch(s) for s in obj)
    init_routes = w_routes + sum(touch(s) for s in score) + sum(touch(s) for s in read)
    gb = _gate_bound(library, [canonical_support(s) for s in initial], identities)
    m_add = gb["M_add_upper"]
    storage0 = sum(len(s) for s in initial)
    act_nnz = sum(len(g.touched()) for g in library.groups if g.kind in (KIND_MASK, KIND_CONDMASK, KIND_PERM))
    cond_nnz = sum(len(g.cond) for g in library.groups if isinstance(g, ConditionalMask))
    out = {
        "estimated": True, "N": N, "R": R, "M": len(obj), "M_score_upper": len(score),
        "M_read_upper": len(read), "M_add_upper": m_add, "gate_bound_source": gb["source"],
        "gate_bound_detail": {k: v for k, v in gb.items() if k not in ("M_add_upper", "source")},
        "routes_weighted_upper": w_routes, "routes_initial_upper": init_routes,
        "max_spin_degree": maxdeg, "n_terms": structure.n_terms,
    }
    if m_add is None:
        out.update({"M_total_upper": None, "routes_decoder_upper": None, "support_storage_upper": None,
                    "memory_bytes_upper": None,
                    "memory_bytes_initial": estimate_model_bytes(
                        n_spins=N, n_clauses=len(initial), support_storage=storage0, n_routes=init_routes,
                        n_weighted_routes=w_routes, n_ctrl_routes=init_routes, n_groups=R, act_nnz=act_nnz,
                        cond_nnz=cond_nnz, n_terms=structure.n_terms)})
        return out
    gsize = int(gb["gate_support_max"] or 0)
    per_gate = min(R, gsize * maxdeg)
    routes = init_routes + m_add * per_gate
    total = len(initial) + m_add
    storage = storage0 + m_add * gsize
    out.update({
        "M_total_upper": total, "routes_decoder_upper": routes, "support_storage_upper": storage,
        "memory_bytes_upper": estimate_model_bytes(
            n_spins=N, n_clauses=total, support_storage=storage, n_routes=routes, n_weighted_routes=w_routes,
            n_ctrl_routes=routes, n_groups=R, act_nnz=act_nnz, cond_nnz=cond_nnz, n_terms=structure.n_terms),
    })
    return out

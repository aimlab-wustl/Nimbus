"""Common compiler: objective structure + transition library -> CompiledModel.

Implements Supplementary Algorithm 1 (compilation of a move library) for the descriptor kinds of
``nimbus.transitions.descriptors``:

1. Objective clauses: one stored clause per distinct canonical support (exact-support merging;
   ``term_to_clause`` maps every term, constant terms to -1). Objective clauses are never
   rewritten by feasible-domain identities.
2. Score clauses (baselines): supports of the optional score structures that are not already
   stored get role SCORE (run weight 0).
3. Spin singletons: every spin s_i is represented by a stored singleton clause ``{i}`` (an existing
   singleton is reused and gains the READOUT role; a missing one is added with role READOUT), seeded
   *before* closure. The singletons update through the same compiled decoder as every other clause,
   sigma_k = sum_r Pi_kr alpha_kr q_r and T_k <- (1 - 2 sigma_k) T_k, so the physical transformation
   mu^(r) is executed by the decoder and the backends need no separate spin-update path. Transformation descriptors are compiler inputs only.
4. Closure worklist: every stored clause A is compiled against every group that touches its
   support (spin -> group incidence; other groups leave T_A unchanged, alpha = 0). For each
   (A, r) the switch alpha_{A,r}(s) = (1 - T_A(mu_r(s)) T_A(s))/2 (Methods, Derivation of the transition-programmable architecture) is derived:

   * MASK m: alpha = |A ∩ m| mod 2, constant (Supplementary Information, Number of auxiliary clauses added by compilation (parity constraints: fixed reversals)).
   * CONDMASK (m, G, c): |A ∩ m| even -> 0; else alpha = (1 + c prod_G s)/2.
   * PERM new s_i = eps_i s_pi(i): T_A(mu) T_A = (prod_{i in A∩D} eps_i) prod_{j in A Δ pi(A)} s_j,
     alpha = (1 - ratio)/2 (Supplementary Information, Number of auxiliary clauses added by compilation (exact cardinality: pair exchanges) and (row exchanges), Methods, An example toy problem).
   * AFFINE new s_d = e_d prod_{j in src_d} s_j: as PERM with the product over src_d in place of
     s_pi(d); executed by the decoder (READOUT action), like a table.
   * TABLE: GF(2) fit over the enumerated domain (``compiler.table_fit``).

   A nonconstant switch (1 + pol T_P)/2 needs the sign of product P. P is reduced modulo the
   parity identities (``core.algebra.IdentityReducer``): a constant folds into a constant route or
   no route; an equivalent stored clause is reused with the corrected polarity (first stored
   clause with that key wins); otherwise P (as its low-weight best representative) is appended
   as a zero-weight GATE clause and enqueued, so its own switches are compiled too.
5. Routes are sorted per group (weighted first by clause index, then unweighted), dependency
   arrays are derived, and the resource report (Supplementary Information, Size of the compiled machine) is attached.

Limits (``CompileLimits``) raise ``ResourceLimitExceeded`` with a partial report; nothing is
truncated silently.
"""

from __future__ import annotations

import datetime as _dt
import platform
import socket
import sys
import time
from array import array
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np

from nimbus.compiler.preflight import estimate_model_bytes, preflight
from nimbus.compiler.table_fit import AffineFitter
from nimbus.core.algebra import IdentityReducer, canonical_identities, support_to_bits
from nimbus.core.errors import ConfigError, ResourceLimitExceeded, UnsupportedTransition
from nimbus.core.gf2 import bits_to_indices
from nimbus.core.model import (
    KIND_CONDMASK,
    KIND_MASK,
    KIND_PERM,
    KIND_READOUT,
    ROLE_GATE,
    ROLE_OBJECTIVE,
    ROLE_READOUT,
    ROLE_SCORE,
    CompiledModel,
    ObjectiveStructure,
    derive_dependencies,
    structural_resources,
)
from nimbus.transitions.descriptors import (
    AffineMove,
    ConditionalMask,
    FixedMask,
    SignedPermutation,
    TableTransition,
)
from nimbus.transitions.library import TransitionLibrary

__all__ = ["CompileLimits", "CompileOptions", "compile_model", "COMPILER_VERSION"]

COMPILER_VERSION = "nimbus.compiler/1"
_KIND_TABLE = 4  # internal dispatch code (compiled to KIND_READOUT)


@dataclass(frozen=True)
class CompileLimits:
    """Resource limits enforced during materialization (None = unlimited).

    ``max_memory_bytes`` bounds the estimated bytes of the model arrays (not Python temporaries).
    """

    max_clauses: int | None = None
    max_routes: int | None = None
    max_memory_bytes: int | None = 8 * 2 ** 30
    max_closure_rounds: int | None = None
    time_limit_s: float | None = None


@dataclass(frozen=True)
class CompileOptions:
    """Compiler options (recorded in the model metadata).

    use_identities: reuse stored signs modulo declared/derived parity identities (feasible-domain
        simplification of *gate lookup only*; objective clauses are never rewritten).
    gate_representative: 'best' stores a new gate as the greedy low-weight representative of the
        requested product (``IdentityReducer.best_representative``); 'requested' stores it as requested.
    allow_non_involutive: accept groups whose action is not an involution (the default proposal
        symmetry argument then no longer holds).
    table_max_states: largest enumerated domain accepted by the finite-table route.
    run_preflight: attach preflight upper-bound estimates to the resource report.
    family / metadata: free-form provenance stored in ``model.metadata``.
    """

    use_identities: bool = True
    gate_representative: str = "best"
    allow_non_involutive: bool = False
    table_max_states: int = 1 << 16
    run_preflight: bool = True
    family: str | None = None
    metadata: dict = field(default_factory=dict)


# ============================================================================================
def _readout_spins(readout, n: int) -> list[int]:
    """Every spin gets a stored singleton clause (spin states are read out from them). ``readout`` is accepted for
    backward compatibility: None, 'all' or a sequence naming every spin; a partial request is an error,
    because a spin without a singleton clause could not be updated by the decoder."""
    if readout is None:
        return list(range(n))
    if isinstance(readout, str):
        if readout != "all":
            raise ConfigError(f"readout must be None or 'all', got {readout!r}")
        return list(range(n))
    out = sorted({int(i) for i in readout})
    if out and (out[0] < 0 or out[-1] >= n):
        raise ConfigError("readout spin index out of range")
    if out != list(range(n)):
        raise ConfigError("every spin must be stored as a singleton clause (spin states are read out from "
                          "their singleton clauses); a partial readout request is not supported -- pass "
                          "readout=None or 'all'")
    return out


def _score_structures(score_structures, n: int) -> dict:
    if not score_structures:
        return {}
    if not isinstance(score_structures, Mapping):
        raise ConfigError("score_structures must be a mapping slot_name -> ObjectiveStructure")
    out = {}
    for name, s in score_structures.items():
        name = str(name)
        if name == "run":
            raise ConfigError("score slot name 'run' is reserved for the acceptance objective")
        if not isinstance(s, ObjectiveStructure):
            raise ConfigError(f"score structure {name!r} is not an ObjectiveStructure")
        if s.n_spins != n:
            raise ConfigError(f"score structure {name!r} has n_spins={s.n_spins}, expected {n}")
        out[name] = s
    return out


class _Limiter:
    __slots__ = ("lim", "t0", "N", "R", "act_nnz", "cond_nnz", "n_terms")

    def __init__(self, lim: CompileLimits, t0: float, N, R, act_nnz, cond_nnz, n_terms):
        self.lim, self.t0 = lim, t0
        self.N, self.R, self.act_nnz, self.cond_nnz, self.n_terms = N, R, act_nnz, cond_nnz, n_terms

    def memory(self, n_clauses, storage, n_routes, n_wroutes, n_ctrl):
        return estimate_model_bytes(n_spins=self.N, n_clauses=n_clauses, support_storage=storage, n_routes=n_routes,
                                    n_weighted_routes=n_wroutes, n_ctrl_routes=n_ctrl, n_groups=self.R,
                                    act_nnz=self.act_nnz, cond_nnz=self.cond_nnz, n_terms=self.n_terms)

    def check(self, *, n_clauses, storage, n_routes, n_wroutes, n_ctrl, rounds, processed, roles):
        lim = self.lim
        reason = None
        if lim.max_clauses is not None and n_clauses > lim.max_clauses:
            reason = f"stored clauses {n_clauses} > max_clauses={lim.max_clauses}"
        elif lim.max_routes is not None and n_routes > lim.max_routes:
            reason = f"routes {n_routes} > max_routes={lim.max_routes}"
        elif lim.max_closure_rounds is not None and rounds > lim.max_closure_rounds:
            reason = f"closure rounds {rounds} > max_closure_rounds={lim.max_closure_rounds}"
        elif lim.time_limit_s is not None and time.perf_counter() - self.t0 > lim.time_limit_s:
            reason = f"compile time exceeded time_limit_s={lim.time_limit_s}"
        else:
            mem = self.memory(n_clauses, storage, n_routes, n_wroutes, n_ctrl)
            if lim.max_memory_bytes is not None and mem > lim.max_memory_bytes:
                reason = f"estimated model memory {mem} B > max_memory_bytes={lim.max_memory_bytes}"
        if reason is None:
            return
        rl = np.frombuffer(bytes(roles), dtype=np.uint8)
        report = {
            "partial": True, "reason": reason, "n_clauses": int(n_clauses), "n_routes": int(n_routes),
            "n_weighted_routes": int(n_wroutes), "n_gate_controlled_routes": int(n_ctrl),
            "closure_rounds": int(rounds), "clauses_processed": int(processed), "support_storage": int(storage),
            "M": int(np.count_nonzero(rl & ROLE_OBJECTIVE)),
            "M_add_so_far": int(np.count_nonzero((rl & ROLE_GATE) & ~(rl & ROLE_OBJECTIVE))),
            "memory_bytes_estimate": self.memory(n_clauses, storage, n_routes, n_wroutes, n_ctrl),
            "elapsed_s": time.perf_counter() - self.t0, "limits": asdict(lim),
        }
        raise ResourceLimitExceeded("compile limit exceeded: " + reason, report)


# ============================================================================================
def compile_model(structure: ObjectiveStructure, library: TransitionLibrary, *, readout=None,
                  identities: Sequence = (), limits: CompileLimits | None = None,
                  score_structures: Mapping | None = None, options: CompileOptions | None = None) -> CompiledModel:
    """Compile an objective structure and a verified transition library into a CompiledModel.

    Parameters
    ----------
    structure: objective supports (weights are bound later with ``bind_weights``).
    library: grouped transitions (one transition neuron each).
    readout: None or 'all' (equivalent): every spin is stored as a singleton clause, seeded before closure.
    identities: feasible-domain parity identities [(support, sign)] used for gate reuse.
    limits: CompileLimits; exceeding one raises ResourceLimitExceeded with a partial report.
    score_structures: optional {slot_name: ObjectiveStructure} for baseline score slots.
    options: CompileOptions.
    """
    t0 = time.perf_counter()
    wall0 = _dt.datetime.now(_dt.timezone.utc)
    opts = options or CompileOptions()
    lim = limits or CompileLimits()
    if not isinstance(structure, ObjectiveStructure):
        raise ConfigError("structure must be an ObjectiveStructure")
    if not isinstance(library, TransitionLibrary):
        raise ConfigError("library must be a TransitionLibrary")
    N = structure.n_spins
    if library.n_spins != N:
        raise ConfigError(f"library has n_spins={library.n_spins}, structure has {N}")
    groups = library.groups
    R = len(groups)
    if R == 0:
        raise ConfigError("the transition library is empty")
    if opts.gate_representative not in ("best", "requested"):
        raise ConfigError("gate_representative must be 'best' or 'requested'")
    if not opts.allow_non_involutive:
        bad = library.missing_reverses()
        if bad:
            raise UnsupportedTransition(
                f"groups {bad[:5]} are not their own inverse and their reverse moves are not in the library; "
                "add them with TransitionLibrary.with_reverses(), or pass "
                "CompileOptions(allow_non_involutive=True) only with an explicit reverse-proposal treatment")
    readout_spins = _readout_spins(readout, N)
    scores = _score_structures(score_structures, N)

    # ---- finite-table groups: common domain, derived identities, change patterns ---------------
    table_idx = [r for r, g in enumerate(groups) if isinstance(g, TableTransition)]
    fitter = None
    derived_ids: list = []
    Z_of: dict[int, np.ndarray] = {}
    if table_idx:
        t_first = groups[table_idx[0]]
        D = t_first.domain_states
        if D.shape[0] > opts.table_max_states:
            raise ResourceLimitExceeded(f"table domain has {D.shape[0]} states > table_max_states={opts.table_max_states}",
                                        {"partial": True, "table_states": int(D.shape[0])})
        dom_set = set(t_first.domain)
        for r in table_idx[1:]:
            if groups[r].domain is not t_first.domain and set(groups[r].domain) != dom_set:
                raise ConfigError("all finite-table groups of one library must share the same enumerated domain")
        fitter = AffineFitter(D)
        derived_ids = fitter.identities()
        if identities:
            chk = IdentityReducer(N, identities)
            if not chk.holds_on(D):
                raise ConfigError("a declared identity does not hold on the finite-table domain")
        for r in table_idx:
            Z_of[r] = (D != groups[r].apply(D)).astype(np.uint8)

    decl_ids = canonical_identities(identities, N)
    all_ids = canonical_identities(list(decl_ids) + list(derived_ids), N)
    reducer = IdentityReducer(N, all_ids if opts.use_identities else ())
    use_red = reducer.active
    best_rep = reducer.best_representative_bits if (use_red and opts.gate_representative == "best") else None
    reduce_bits = reducer.reduce_bits

    # ---- per-group dispatch data --------------------------------------------------------------
    gkind: list[int] = []
    gdata: list = []
    act_nnz = 0
    cond_nnz = 0
    for r, g in enumerate(groups):
        if isinstance(g, FixedMask):
            gkind.append(KIND_MASK)
            gdata.append(support_to_bits(g.mask))
            act_nnz += len(g.mask)
        elif isinstance(g, ConditionalMask):
            gkind.append(KIND_CONDMASK)
            gdata.append((support_to_bits(g.mask), support_to_bits(g.cond), g.pol))
            act_nnz += len(g.mask)
            cond_nnz += len(g.cond)
        elif isinstance(g, SignedPermutation):
            gkind.append(KIND_PERM)
            gdata.append({d: ((1 << d) ^ (1 << s), e) for d, s, e in zip(g.dests, g.srcs, g.signs)})
            act_nnz += len(g.dests)
        elif isinstance(g, AffineMove):
            # same rule as a permutation: new s_d = e * prod(sources) contributes the product
            # s_d * prod(sources) to the change of every clause that contains d
            gkind.append(KIND_PERM)
            gdata.append({d: ((1 << d) ^ support_to_bits(src), e) for d, src, e in zip(g.dests, g.sources, g.signs)})
            act_nnz += sum(len(src) for src in g.sources)
        elif isinstance(g, TableTransition):
            gkind.append(_KIND_TABLE)
            gdata.append(Z_of[r])
        else:
            raise UnsupportedTransition(f"group {r}: unsupported descriptor type {type(g).__name__}")
    inc = library.touched_incidence()

    estimated = None
    t_pf = time.perf_counter()
    if opts.run_preflight:
        try:
            estimated = preflight(structure, library, readout, identities=all_ids, score_structures=scores)
        except Exception as exc:  # pragma: no cover - estimation must never block compilation
            estimated = {"error": repr(exc)}
    t_pf = time.perf_counter() - t_pf

    limiter = _Limiter(lim, t0, N, R, act_nnz, cond_nnz, structure.n_terms)

    # ---- stored clauses -----------------------------------------------------------------------
    supports: list[tuple] = []
    sbits: list[int] = []
    roles = bytearray()
    exact: dict[int, int] = {}
    storage = 0

    def add_clause(sup: tuple, bits: int, role: int) -> int:
        nonlocal storage
        k = len(supports)
        supports.append(sup)
        sbits.append(bits)
        roles.append(role)
        exact[bits] = k
        storage += len(sup)
        return k

    ttc = np.empty(structure.n_terms, dtype=np.int32)
    for t, sup in enumerate(structure.terms):
        if not sup:
            ttc[t] = -1
            continue
        b = support_to_bits(sup)
        k = exact.get(b)
        if k is None:
            k = add_clause(sup, b, ROLE_OBJECTIVE)
        ttc[t] = k
    score_ttc: dict[str, np.ndarray] = {}
    for name, s in scores.items():
        m = np.empty(s.n_terms, dtype=np.int32)
        for t, sup in enumerate(s.terms):
            if not sup:
                m[t] = -1
                continue
            b = support_to_bits(sup)
            k = exact.get(b)
            if k is None:
                k = add_clause(sup, b, ROLE_SCORE)
            m[t] = k
        score_ttc[name] = m
    for i in readout_spins:
        b = 1 << i
        k = exact.get(b)
        if k is None:
            add_clause((i,), b, ROLE_READOUT)
        else:
            roles[k] |= ROLE_READOUT
    n_initial = len(supports)
    limiter.check(n_clauses=n_initial, storage=storage, n_routes=0, n_wroutes=0, n_ctrl=0, rounds=1,
                  processed=0, roles=roles)

    # ---- key table (first stored clause with a key wins) --------------------------------------
    keytab: dict[int, tuple[int, int]] = {}  # reduced bits -> (clause, c) with T_clause = c * T_key
    for k in range(n_initial):
        red, sg = reduce_bits(sbits[k])
        if red and red not in keytab:
            keytab[red] = (k, sg)

    gate_cache: dict[int, tuple[int, int]] = {}  # requested product bits -> (clause or -1, c): T_P = c * T_clause
    used_ids = False
    n_gates_created = 0

    def resolve(pb: int) -> tuple[int, int]:
        nonlocal used_ids, n_gates_created
        red, sg = reduce_bits(pb)
        if red == 0:
            res = (-1, sg)
            used_ids = True  # pb != 0 is constant only through the identities
        else:
            e = keytab.get(red)
            if e is not None:
                g, cg = e
                res = (g, sg * cg)
                if sbits[g] != pb:
                    used_ids = True
            else:
                if best_rep is not None:
                    rb, rs = best_rep(pb)
                else:
                    rb, rs = pb, 1
                if rb != pb:
                    used_ids = True
                g = add_clause(tuple(bits_to_indices(rb)), rb, ROLE_GATE)
                n_gates_created += 1
                keytab[red] = (g, rs * sg)
                res = (g, rs)
                limiter.check(n_clauses=len(supports), storage=storage, n_routes=len(rt_r), n_wroutes=n_w[0],
                              n_ctrl=n_c[0], rounds=rounds, processed=pos, roles=roles)
        gate_cache[pb] = res
        return res

    # ---- closure worklist ---------------------------------------------------------------------
    rt_r, rt_k, rt_c = array("i"), array("i"), array("i")
    rt_p = array("b")
    n_w = [0]  # weighted routes so far
    n_c = [0]  # gate-controlled routes so far
    pos = 0
    rounds = 1
    round_end = n_initial
    check_every = 64
    t_cl = time.perf_counter()
    while pos < len(supports):
        if pos == round_end:
            rounds += 1
            round_end = len(supports)
        k = pos
        pos += 1
        if pos % check_every == 0:
            limiter.check(n_clauses=len(supports), storage=storage, n_routes=len(rt_r), n_wroutes=n_w[0],
                          n_ctrl=n_c[0], rounds=rounds, processed=pos, roles=roles)
        sup = supports[k]
        kb = sbits[k]
        weighted = roles[k] & ROLE_OBJECTIVE
        if len(sup) == 1:
            touching = inc[sup[0]]
        else:
            u = set()
            for i in sup:
                u.update(inc[i])
            touching = sorted(u)
        for r in touching:
            kind = gkind[r]
            if kind == KIND_PERM:
                smap = gdata[r]
                pb = 0
                neg = False
                for i in sup:
                    e = smap.get(i)
                    if e is not None:
                        pb ^= e[0]
                        if e[1] < 0:
                            neg = not neg
                # ratio = (-1 if neg) * T_pb ; alpha = (1 - ratio)/2
                if pb == 0:
                    if neg:
                        ctrl, pol = -1, 1
                    else:
                        continue
                else:
                    pol = 1 if neg else -1
                    hit = gate_cache.get(pb)
                    if hit is None:
                        hit = resolve(pb)
                    ctrl, c = hit
                    if ctrl < 0:
                        if pol * c > 0:
                            pol = 1
                        else:
                            continue
                    else:
                        pol *= c
            elif kind == KIND_MASK:
                if (kb & gdata[r]).bit_count() & 1:
                    ctrl, pol = -1, 1
                else:
                    continue
            elif kind == KIND_CONDMASK:
                mb, cb, cp = gdata[r]
                if not ((kb & mb).bit_count() & 1):
                    continue
                if cb == 0:
                    if cp > 0:
                        ctrl, pol = -1, 1
                    else:
                        continue
                else:
                    hit = gate_cache.get(cb)
                    if hit is None:
                        hit = resolve(cb)
                    ctrl, c = hit
                    if ctrl < 0:
                        if cp * c > 0:
                            pol = 1
                        else:
                            continue
                    else:
                        pol = cp * c
            else:  # finite table
                Zr = gdata[r]
                bvec = Zr[:, list(sup)].sum(axis=1, dtype=np.int64) & 1
                nz = int(np.count_nonzero(bvec))
                if nz == 0:
                    continue
                if nz == bvec.size:
                    ctrl, pol = -1, 1
                else:
                    gbits, cbit = fitter.fit(bvec, context=f" (clause support {sup}, group {r} {library.labels[r]!r})")
                    # alpha = <g,y> xor c ; <g,y> = 1 iff T_g = -1
                    pol = 1 if cbit else -1
                    hit = gate_cache.get(gbits)
                    if hit is None:
                        hit = resolve(gbits)
                    ctrl, c = hit
                    if ctrl < 0:
                        if pol * c > 0:
                            pol = 1
                        else:
                            continue
                    else:
                        pol *= c
            rt_r.append(r)
            rt_k.append(k)
            rt_c.append(ctrl)
            rt_p.append(pol)
            if weighted:
                n_w[0] += 1
            if ctrl >= 0:
                n_c[0] += 1
                roles[ctrl] |= ROLE_GATE
    limiter.check(n_clauses=len(supports), storage=storage, n_routes=len(rt_r), n_wroutes=n_w[0], n_ctrl=n_c[0],
                  rounds=rounds, processed=pos, roles=roles)
    t_closure = time.perf_counter() - t_cl

    # ---- assemble arrays ----------------------------------------------------------------------
    M = len(supports)
    clause_ptr = np.zeros(M + 1, dtype=np.int64)
    np.cumsum([len(s) for s in supports], out=clause_ptr[1:])
    clause_spins = np.fromiter((i for s in supports for i in s), dtype=np.int32, count=int(clause_ptr[-1]))
    clause_role = np.frombuffer(bytes(roles), dtype=np.uint8).copy()

    idt = np.dtype(f"i{rt_r.itemsize}")  # C int of the array module (4 bytes on all supported platforms)
    r_arr = np.frombuffer(rt_r, dtype=idt).astype(np.int64) if len(rt_r) else np.zeros(0, np.int64)
    k_arr = np.frombuffer(rt_k, dtype=idt).astype(np.int32) if len(rt_k) else np.zeros(0, np.int32)
    c_arr = np.frombuffer(rt_c, dtype=idt).astype(np.int32) if len(rt_c) else np.zeros(0, np.int32)
    p_arr = np.frombuffer(rt_p, dtype=np.int8) if len(rt_p) else np.zeros(0, np.int8)
    wflag = (clause_role[k_arr] & ROLE_OBJECTIVE) != 0
    order = np.lexsort((k_arr, ~wflag, r_arr))
    route_clause = k_arr[order].astype(np.int32)
    route_ctrl = c_arr[order].astype(np.int32)
    route_pol = p_arr[order].astype(np.int8)
    r_sorted = r_arr[order]
    w_sorted = wflag[order]
    counts = np.bincount(r_sorted, minlength=R).astype(np.int64)
    route_ptr = np.zeros(R + 1, dtype=np.int64)
    np.cumsum(counts, out=route_ptr[1:])
    wcounts = np.bincount(r_sorted[w_sorted], minlength=R).astype(np.int64)
    route_wend = route_ptr[:-1] + wcounts
    dep = derive_dependencies(M, route_clause, route_ctrl, w_sorted)

    act_kind = np.empty(R, dtype=np.int8)
    act_ptr = np.zeros(R + 1, dtype=np.int64)
    cond_ptr = np.zeros(R + 1, dtype=np.int64)
    cond_pol = np.ones(R, dtype=np.int8)
    a_idx, a_src, a_sgn, c_idx = [], [], [], []
    for r, g in enumerate(groups):
        kind = gkind[r]
        if kind == KIND_MASK:
            act_kind[r] = KIND_MASK
            a_idx.extend(g.mask)
            a_src.extend([-1] * len(g.mask))
            a_sgn.extend([1] * len(g.mask))
        elif kind == KIND_CONDMASK:
            act_kind[r] = KIND_CONDMASK
            a_idx.extend(g.mask)
            a_src.extend([-1] * len(g.mask))
            a_sgn.extend([1] * len(g.mask))
            c_idx.extend(g.cond)
            cond_pol[r] = g.pol
        elif kind == KIND_PERM and isinstance(g, SignedPermutation):
            act_kind[r] = KIND_PERM
            a_idx.extend(g.dests)
            a_src.extend(g.srcs)
            a_sgn.extend(g.signs)
        else:
            act_kind[r] = KIND_READOUT
        act_ptr[r + 1] = len(a_idx)
        cond_ptr[r + 1] = len(c_idx)
    readout_clause_of_spin = np.array([exact.get(1 << i, -1) for i in range(N)], dtype=np.int32)

    lib_meta = library.metadata
    meta = {
        "compiler": COMPILER_VERSION,
        "family": opts.family,
        "library": {k: lib_meta[k] for k in ("name", "template", "params", "components", "n_groups", "kinds",
                                              "all_involutions", "hash")},
        "structure_hash": structure.hash,
        "score_structure_hashes": {n: s.hash for n, s in scores.items()},
        "readout": "all",
        "options": {k: v for k, v in asdict(opts).items() if k != "metadata"},
        "n_declared_identities": len(decl_ids),
        "n_derived_identities": len(derived_ids),
        "user": dict(opts.metadata),
    }
    model = CompiledModel(
        n_spins=N, n_clauses=M, n_groups=R, group_label=library.labels,
        clause_ptr=clause_ptr, clause_spins=clause_spins, clause_role=clause_role,
        route_ptr=route_ptr, route_wend=route_wend, route_clause=route_clause, route_ctrl=route_ctrl,
        route_pol=route_pol, act_kind=act_kind, act_ptr=act_ptr,
        act_idx=np.asarray(a_idx, dtype=np.int32), act_src=np.asarray(a_src, dtype=np.int32),
        act_sgn=np.asarray(a_sgn, dtype=np.int8), cond_ptr=cond_ptr, cond_idx=np.asarray(c_idx, dtype=np.int32),
        cond_pol=cond_pol, readout_clause_of_spin=readout_clause_of_spin, term_to_clause=ttc,
        identities=tuple(all_ids) if use_red else (), feasible_simplification=bool(used_ids),
        metadata=meta, library_hash=library.hash, score_term_to_clause=score_ttc, **dep)
    t_total = time.perf_counter() - t0
    res = structural_resources(model)
    res.update({"closure_rounds": int(rounds), "gate_clauses_created": int(n_gates_created),
                "compile_seconds": float(t_total), "closure_seconds": float(t_closure),
                "preflight_seconds": float(t_pf)})
    model.metadata["resources"] = {
        "materialized": res,
        "estimated": estimated,
        "compile_host": socket.gethostname(),
        "compile_platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "compiled_at_utc": wall0.isoformat(timespec="seconds"),
    }
    return model

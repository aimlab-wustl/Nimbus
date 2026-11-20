"""Objective structure, compiled model (immutable architecture topology) and weight bindings.

Manuscript quantities:

* stored clauses (rows of H, Methods, Derivation of the transition-programmable architecture): CSR ``clause_ptr``/``clause_spins`` with role bits;
* transition neurons: groups ``r = 0..R-1`` with a spin action (Methods, Derivation of the transition-programmable architecture);
* retained connections Pi (Methods, Derivation of the transition-programmable architecture): *routes* (k, r), CSR by group, weighted
  routes (clauses with the OBJECTIVE role) first;
* switches alpha (Methods, Derivation of the transition-programmable architecture): ``route_ctrl == -1`` -> alpha = 1, else
  alpha = (1 + route_pol * T[route_ctrl]) / 2;
* encoder (Methods, Derivation of the transition-programmable architecture): v_r = sum over weighted routes J_k alpha T_k; decoder
  (Methods, Derivation of the transition-programmable architecture): sigma_k = alpha over all routes of the accepted group.

The reference helpers on :class:`CompiledModel` (``clause_values``, ``route_alpha``, ``inputs``,
``decoder_toggles``, ``apply_action``, ``energy``) evaluate these definitions directly from the
arrays with numpy. They are for validation and small problems, not the production data path.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from nimbus.core.algebra import canonical_identities, canonical_support
from nimbus.core.errors import FormatError

SCHEMA_VERSION = "nimbus.model/1"
BINDING_SCHEMA_VERSION = "nimbus.binding/1"

ROLE_OBJECTIVE = 1
ROLE_GATE = 2
ROLE_READOUT = 4
ROLE_SCORE = 8
_ROLE_ALL = ROLE_OBJECTIVE | ROLE_GATE | ROLE_READOUT | ROLE_SCORE

# spin-action kinds (identical to nimbus.transitions.descriptors.KIND_*)
KIND_MASK = 0
KIND_CONDMASK = 1
KIND_PERM = 2
KIND_READOUT = 3

__all__ = [
    "SCHEMA_VERSION", "ROLE_OBJECTIVE", "ROLE_GATE", "ROLE_READOUT", "ROLE_SCORE",
    "KIND_MASK", "KIND_CONDMASK", "KIND_PERM", "KIND_READOUT",
    "ObjectiveStructure", "CompiledModel", "WeightBinding", "bind_weights",
    "derive_dependencies", "structural_resources", "save_model", "load_model",
]


# ============================================================================================
# helpers
# ============================================================================================
def jsonable(x):
    """Recursively convert numpy scalars/arrays and tuples into JSON-serializable objects."""
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _split_path(path) -> str:
    p = os.fspath(path)
    for ext in (".npz", ".json"):
        if p.endswith(ext):
            return p[: -len(ext)]
    return p


def _hash_array(h, name: str, a: np.ndarray):
    a = np.ascontiguousarray(a)
    h.update(name.encode())
    h.update(a.dtype.str.encode())
    h.update(repr(tuple(a.shape)).encode())
    h.update(a.tobytes())


def _coerce(name: str, value, dtype, ndim: int = 1) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != ndim:
        if a.size == 0 and ndim == 1:
            a = a.reshape(0)
        else:
            raise ValueError(f"{name}: expected {ndim}-D array, got shape {a.shape}")
    if a.dtype != dtype:
        if a.size and not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_)) \
                and np.issubdtype(np.dtype(dtype), np.integer):
            raise ValueError(f"{name}: expected integer dtype {np.dtype(dtype)}, got {a.dtype}")
        b = a.astype(dtype)
        if a.size and not np.array_equal(b, a):
            raise ValueError(f"{name}: values do not fit dtype {np.dtype(dtype)}")
        a = b
    a = np.ascontiguousarray(a)
    if a is value or (isinstance(value, np.ndarray) and np.shares_memory(a, value)):
        a = a.copy()
    a.setflags(write=False)
    return a


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def _spins2d(spins) -> tuple[np.ndarray, bool]:
    s = np.asarray(spins)
    if s.ndim == 1:
        return s[None, :], True
    if s.ndim != 2:
        raise ValueError("spins must be 1-D (N,) or 2-D (states, N)")
    return s, False


# ============================================================================================
# objective structure
# ============================================================================================
@dataclass(frozen=True, eq=False)
class ObjectiveStructure:
    """Weight-free objective structure: ``n_spins`` and the term supports in structural order.

    Terms are canonicalized by parity (s_i^2 = 1): a raw index list becomes the sorted tuple of
    indices with odd multiplicity. An empty support is a constant term (it maps to no clause; its
    weight goes to the binding constant). Repeated supports are allowed; the compiler stores one
    clause per distinct support and ``bind_weights`` sums their weights.
    """

    n_spins: int
    terms: tuple = ()
    labels: tuple | None = None

    def __post_init__(self):
        n = int(self.n_spins)
        if n < 1:
            raise ValueError("ObjectiveStructure needs n_spins >= 1")
        terms = tuple(canonical_support(t) for t in self.terms)
        for t in terms:
            if t and t[-1] >= n:
                raise ValueError(f"term support {t} has a spin index >= n_spins={n}")
        labels = self.labels
        if labels is not None:
            labels = tuple(str(x) for x in labels)
            if len(labels) != len(terms):
                raise ValueError("labels and terms differ in length")
        object.__setattr__(self, "n_spins", n)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "labels", labels)

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def max_degree(self) -> int:
        return max((len(t) for t in self.terms), default=0)

    def unique_supports(self) -> list[tuple[int, ...]]:
        """Distinct nonconstant supports in order of first appearance."""
        seen, out = set(), []
        for t in self.terms:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @property
    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps({"n_spins": self.n_spins, "terms": [list(t) for t in self.terms]},
                            separators=(",", ":")).encode())
        return h.hexdigest()

    def to_dict(self) -> dict:
        return {"n_spins": self.n_spins, "terms": [list(t) for t in self.terms],
                "labels": list(self.labels) if self.labels is not None else None}

    @classmethod
    def from_dict(cls, d) -> "ObjectiveStructure":
        return cls(n_spins=d["n_spins"], terms=[tuple(t) for t in d["terms"]], labels=d.get("labels"))


# ============================================================================================
# compiled model
# ============================================================================================
_ARRAY_FIELDS = (
    ("clause_ptr", np.int64), ("clause_spins", np.int32), ("clause_role", np.uint8),
    ("route_ptr", np.int64), ("route_wend", np.int64), ("route_clause", np.int32),
    ("route_ctrl", np.int32), ("route_pol", np.int8),
    ("act_kind", np.int8), ("act_ptr", np.int64), ("act_idx", np.int32), ("act_src", np.int32),
    ("act_sgn", np.int8), ("cond_ptr", np.int64), ("cond_idx", np.int32), ("cond_pol", np.int8),
    ("readout_clause_of_spin", np.int32),
    ("clause_wroute_ptr", np.int64), ("clause_wroute", np.int64),
    ("ctrl_route_ptr", np.int64), ("ctrl_route", np.int64),
    ("ctrl_wroute_ptr", np.int64), ("ctrl_wroute", np.int64),
    ("term_to_clause", np.int32),
)
_ARRAY_NAMES = tuple(n for n, _ in _ARRAY_FIELDS)


def derive_dependencies(n_clauses: int, route_clause, route_ctrl, weighted) -> dict:
    """Dependency routes, deterministic: positions ascending within each list.

    clause_wroute: weighted route positions grouped by route_clause;
    ctrl_route: all route positions grouped by route_ctrl (ctrl >= 0);
    ctrl_wroute: weighted subset of ctrl_route.
    """
    route_clause = np.asarray(route_clause, dtype=np.int64)
    route_ctrl = np.asarray(route_ctrl, dtype=np.int64)
    weighted = np.asarray(weighted, dtype=bool)
    pos = np.arange(route_clause.size, dtype=np.int64)
    M = int(n_clauses)

    def group(keys, positions):
        order = np.argsort(keys, kind="stable")
        counts = np.bincount(keys, minlength=M) if keys.size else np.zeros(M, dtype=np.int64)
        ptr = np.zeros(M + 1, dtype=np.int64)
        np.cumsum(counts, out=ptr[1:])
        return ptr, positions[order].astype(np.int64)

    wpos = pos[weighted]
    cw_ptr, cw = group(route_clause[weighted], wpos)
    cmask = route_ctrl >= 0
    cr_ptr, cr = group(route_ctrl[cmask], pos[cmask])
    cwm = cmask & weighted
    cwr_ptr, cwr = group(route_ctrl[cwm], pos[cwm])
    return {"clause_wroute_ptr": cw_ptr, "clause_wroute": cw, "ctrl_route_ptr": cr_ptr, "ctrl_route": cr,
            "ctrl_wroute_ptr": cwr_ptr, "ctrl_wroute": cwr}


@dataclass(frozen=True, eq=False)
class CompiledModel:
    """Immutable compiled architecture. Arrays are validated and read-only.

    Extra fields beyond the base fields: ``library_hash`` (enters ``topology_hash``) and
    ``score_term_to_clause`` (slot name -> int32 map of that score structure's terms to clauses,
    -1 for constant terms), used by ``bind_weights`` for baseline score slots.
    """

    n_spins: int
    n_clauses: int
    n_groups: int
    group_label: tuple
    clause_ptr: np.ndarray
    clause_spins: np.ndarray
    clause_role: np.ndarray
    route_ptr: np.ndarray
    route_wend: np.ndarray
    route_clause: np.ndarray
    route_ctrl: np.ndarray
    route_pol: np.ndarray
    act_kind: np.ndarray
    act_ptr: np.ndarray
    act_idx: np.ndarray
    act_src: np.ndarray
    act_sgn: np.ndarray
    cond_ptr: np.ndarray
    cond_idx: np.ndarray
    cond_pol: np.ndarray
    readout_clause_of_spin: np.ndarray
    clause_wroute_ptr: np.ndarray
    clause_wroute: np.ndarray
    ctrl_route_ptr: np.ndarray
    ctrl_route: np.ndarray
    ctrl_wroute_ptr: np.ndarray
    ctrl_wroute: np.ndarray
    term_to_clause: np.ndarray
    identities: tuple = ()
    feasible_simplification: bool = False
    metadata: dict = field(default_factory=dict)
    library_hash: str = ""
    score_term_to_clause: Mapping = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    topology_hash: str = ""

    # ------------------------------------------------------------------------------------------
    def __post_init__(self):
        set_ = object.__setattr__
        if self.schema_version != SCHEMA_VERSION:
            raise FormatError(f"unsupported model schema {self.schema_version!r} (expected {SCHEMA_VERSION})")
        set_(self, "n_spins", int(self.n_spins))
        set_(self, "n_clauses", int(self.n_clauses))
        set_(self, "n_groups", int(self.n_groups))
        for name, dt in _ARRAY_FIELDS:
            set_(self, name, _coerce(name, getattr(self, name), dt))
        set_(self, "group_label", tuple(str(x) for x in self.group_label))
        ids = canonical_identities(self.identities, self.n_spins)
        set_(self, "identities", tuple((tuple(s), int(c)) for s, c in ids))
        set_(self, "feasible_simplification", bool(self.feasible_simplification))
        set_(self, "metadata", jsonable(dict(self.metadata)))
        set_(self, "library_hash", str(self.library_hash))
        stc = {str(k): _coerce(f"score_term_to_clause[{k}]", v, np.int32)
               for k, v in dict(self.score_term_to_clause).items()}
        set_(self, "score_term_to_clause", MappingProxyType(dict(sorted(stc.items()))))
        self._validate()
        h = self._compute_topology_hash()
        if self.topology_hash and self.topology_hash != h:
            raise FormatError("topology hash mismatch: the model arrays do not match the recorded hash")
        set_(self, "topology_hash", h)

    # ------------------------------------------------------------------------------------------
    def _validate(self):
        N, M, R = self.n_spins, self.n_clauses, self.n_groups

        def fail(msg):
            raise ValueError(f"CompiledModel: {msg}")

        if N < 1:
            fail("n_spins must be >= 1")
        cp, cs = self.clause_ptr, self.clause_spins
        if cp.size != M + 1 or cp[0] != 0 or cp[-1] != cs.size:
            fail("clause_ptr must have M~+1 entries from 0 to len(clause_spins)")
        if M and np.any(np.diff(cp) < 1):
            fail("empty clause support (constants are not stored clauses)")
        if cs.size and (cs.min() < 0 or cs.max() >= N):
            fail("clause_spins index out of range")
        if cs.size > 1:
            inc = np.diff(cs.astype(np.int64)) > 0
            boundary = np.zeros(cs.size - 1, dtype=bool)
            starts = cp[1:-1]
            boundary[starts[(starts > 0) & (starts < cs.size)] - 1] = True
            if np.any(~inc & ~boundary):
                fail("clause supports must be sorted with unique indices")
        role = self.clause_role
        if role.size != M or (M and (np.any(role == 0) or np.any(role & ~np.uint8(_ROLE_ALL)))):
            fail("clause_role must be a nonzero role bitmask per clause")
        rp, we = self.route_ptr, self.route_wend
        nnz = self.route_clause.size
        if rp.size != R + 1 or rp[0] != 0 or rp[-1] != nnz or np.any(np.diff(rp) < 0):
            fail("route_ptr must be a CSR pointer over groups")
        if we.size != R or np.any(we < rp[:-1]) or np.any(we > rp[1:]):
            fail("route_wend out of [route_ptr[r], route_ptr[r+1]]")
        if self.route_ctrl.size != nnz or self.route_pol.size != nnz:
            fail("route arrays differ in length")
        rc, rctrl, rpol = self.route_clause, self.route_ctrl, self.route_pol
        if nnz:
            if rc.min() < 0 or rc.max() >= M:
                fail("route_clause out of range")
            if rctrl.min() < -1 or rctrl.max() >= M:
                fail("route_ctrl out of range")
            if np.any((rpol != 1) & (rpol != -1)):
                fail("route_pol must be +1/-1")
            if np.any(rpol[rctrl == -1] != 1):
                fail("route_pol must be +1 for constant routes")
            group_of = np.repeat(np.arange(R, dtype=np.int64), np.diff(rp))
            weighted_pos = np.arange(nnz) < we[group_of]
            is_obj = (role[rc] & ROLE_OBJECTIVE) != 0
            if np.any(weighted_pos != is_obj):
                fail("weighted routes must be exactly the routes of OBJECTIVE clauses, listed first")
            # sorted, unique within each part: next position in the same part must have a larger clause
            same_part = (group_of[1:] == group_of[:-1]) & (weighted_pos[1:] == weighted_pos[:-1])
            if np.any(same_part & (np.diff(rc.astype(np.int64)) <= 0)):
                fail("routes must be sorted by clause index within the weighted/unweighted parts of a group")
        # spin actions
        ak, ap = self.act_kind, self.act_ptr
        if ak.size != R or np.any((ak < 0) | (ak > 3)):
            fail("act_kind must be one of 0..3 per group")
        na = self.act_idx.size
        if ap.size != R + 1 or ap[0] != 0 or ap[-1] != na or np.any(np.diff(ap) < 0):
            fail("act_ptr must be a CSR pointer over groups")
        if self.act_src.size != na or self.act_sgn.size != na:
            fail("act arrays differ in length")
        if na and (self.act_idx.min() < 0 or self.act_idx.max() >= N):
            fail("act_idx out of range")
        cpt, ci = self.cond_ptr, self.cond_idx
        if cpt.size != R + 1 or cpt[0] != 0 or cpt[-1] != ci.size or np.any(np.diff(cpt) < 0):
            fail("cond_ptr must be a CSR pointer over groups")
        if ci.size and (ci.min() < 0 or ci.max() >= N):
            fail("cond_idx out of range")
        if self.cond_pol.size != R or np.any((self.cond_pol != 1) & (self.cond_pol != -1)):
            fail("cond_pol must be +1/-1 per group")
        for r in range(R):
            k = int(ak[r])
            a0, a1 = int(ap[r]), int(ap[r + 1])
            idx = self.act_idx[a0:a1]
            src = self.act_src[a0:a1]
            sgn = self.act_sgn[a0:a1]
            c0, c1 = int(cpt[r]), int(cpt[r + 1])
            if k != KIND_CONDMASK and c1 > c0:
                fail(f"group {r}: condition spins on a non-CONDMASK action")
            if k in (KIND_MASK, KIND_CONDMASK):
                if a1 == a0 or np.any(src != -1) or np.any(sgn != 1) or np.unique(idx).size != idx.size:
                    fail(f"group {r}: malformed mask action")
                if k == KIND_CONDMASK and np.intersect1d(idx, ci[c0:c1]).size % 2:
                    fail(f"group {r}: |cond ∩ mask| must be even")
            elif k == KIND_PERM:
                if a1 == a0 or src.min() < 0 or src.max() >= N or np.any((sgn != 1) & (sgn != -1)):
                    fail(f"group {r}: malformed permutation action")
                if np.unique(idx).size != idx.size or not np.array_equal(np.sort(idx), np.sort(src)):
                    fail(f"group {r}: permutation is not a bijection on its moved coordinates")
            else:  # READOUT
                if a1 != a0:
                    fail(f"group {r}: READOUT actions carry no act_idx entries")
        rcs = self.readout_clause_of_spin
        if rcs.size != N:
            fail("readout_clause_of_spin must have N entries")
        if rcs.size and (rcs.min() < -1 or rcs.max() >= M):
            fail("readout_clause_of_spin out of range")
        for i in np.flatnonzero(rcs >= 0).tolist():
            k = int(rcs[i])
            if cp[k + 1] - cp[k] != 1 or cs[cp[k]] != i:
                fail(f"readout_clause_of_spin[{i}] is not the singleton clause {{{i}}}")
        if np.any(rcs < 0):
            fail("every spin must be stored as a singleton clause (spin states are read out from their singleton "
                 "clauses); missing for spins "
                 f"{np.flatnonzero(rcs < 0)[:10].tolist()}")
        # dependencies
        weighted = np.zeros(nnz, dtype=bool)
        if nnz:
            group_of = np.repeat(np.arange(R, dtype=np.int64), np.diff(rp))
            weighted = np.arange(nnz) < we[group_of]
        dep = derive_dependencies(M, rc, rctrl, weighted)
        for name, arr in dep.items():
            if not np.array_equal(getattr(self, name), arr):
                fail(f"{name} does not match the routes")
        ttc = self.term_to_clause
        if ttc.size and (ttc.min() < -1 or ttc.max() >= M):
            fail("term_to_clause out of range")
        if ttc.size and np.any((role[ttc[ttc >= 0]] & ROLE_OBJECTIVE) == 0):
            fail("term_to_clause must map to OBJECTIVE clauses")
        for name, m in self.score_term_to_clause.items():
            if name == "run":
                fail("score slot name 'run' is reserved")
            if m.size and (m.min() < -1 or m.max() >= M):
                fail(f"score_term_to_clause[{name}] out of range")
            if m.size and np.any((role[m[m >= 0]] & (ROLE_OBJECTIVE | ROLE_SCORE)) == 0):
                fail(f"score_term_to_clause[{name}] must map to OBJECTIVE or SCORE clauses")
        if len(self.group_label) != R:
            fail("group_label must have R entries")

    def _compute_topology_hash(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps({"schema": self.schema_version, "n_spins": self.n_spins, "n_clauses": self.n_clauses,
                             "n_groups": self.n_groups, "group_label": list(self.group_label),
                             "identities": [[list(s), c] for s, c in self.identities],
                             "feasible_simplification": self.feasible_simplification,
                             "library_hash": self.library_hash}, separators=(",", ":")).encode())
        for name in _ARRAY_NAMES:
            _hash_array(h, name, getattr(self, name))
        for name, arr in self.score_term_to_clause.items():
            _hash_array(h, "score:" + name, arr)
        return h.hexdigest()

    # ------------------------------------------------------------------------------------------
    # structure access
    # ------------------------------------------------------------------------------------------
    @property
    def n_routes(self) -> int:
        return int(self.route_clause.size)

    @property
    def n_terms(self) -> int:
        return int(self.term_to_clause.size)

    def clause_support(self, k: int) -> tuple[int, ...]:
        return tuple(int(i) for i in self.clause_spins[self.clause_ptr[k]:self.clause_ptr[k + 1]])

    def clause_supports(self) -> list[tuple[int, ...]]:
        cs = self.clause_spins.tolist()
        cp = self.clause_ptr.tolist()
        return [tuple(cs[cp[k]:cp[k + 1]]) for k in range(self.n_clauses)]

    def clauses_with_role(self, role: int) -> np.ndarray:
        return np.flatnonzero(self.clause_role & np.uint8(role))

    def group_routes(self, r: int) -> dict:
        a, w, b = int(self.route_ptr[r]), int(self.route_wend[r]), int(self.route_ptr[r + 1])
        return {"clause": self.route_clause[a:b], "ctrl": self.route_ctrl[a:b], "pol": self.route_pol[a:b],
                "weighted": np.arange(a, b) < w, "positions": np.arange(a, b)}

    def route_group(self) -> np.ndarray:
        """Group index of every route position."""
        return np.repeat(np.arange(self.n_groups, dtype=np.int64), np.diff(self.route_ptr))

    def weighted_route_mask(self) -> np.ndarray:
        g = self.route_group()
        return np.arange(self.n_routes) < self.route_wend[g]

    def pi_matrix(self) -> np.ndarray:
        """Dense Pi (M~ x R, uint8) for tiny models / tests."""
        pi = np.zeros((self.n_clauses, self.n_groups), dtype=np.uint8)
        pi[self.route_clause, self.route_group()] = 1
        return pi

    # ------------------------------------------------------------------------------------------
    # reference evaluations (validation / small problems)
    # ------------------------------------------------------------------------------------------
    def clause_values(self, spins) -> np.ndarray:
        """T_k(s) for all stored clauses: (M~,) or (S, M~) int8."""
        s, single = _spins2d(spins)
        if s.shape[1] != self.n_spins:
            raise ValueError("spin array length does not match the model")
        if self.n_clauses == 0:
            out = np.ones((s.shape[0], 0), dtype=np.int8)
        else:
            neg = (s < 0).astype(np.int64)
            cnt = np.add.reduceat(neg[:, self.clause_spins], self.clause_ptr[:-1], axis=1)
            out = (1 - 2 * (cnt & 1)).astype(np.int8)
        return out[0] if single else out

    def route_alpha(self, T) -> np.ndarray:
        """alpha of every route from clause signs T: (nnz,) or (S, nnz) uint8."""
        T2, single = _spins2d(T)
        ctrl = self.route_ctrl.astype(np.int64)
        alpha = np.ones((T2.shape[0], self.n_routes), dtype=np.int64)
        m = ctrl >= 0
        if np.any(m):
            alpha[:, m] = (1 + self.route_pol[m].astype(np.int64) * T2[:, ctrl[m]].astype(np.int64)) // 2
        out = alpha.astype(np.uint8)
        return out[0] if single else out

    def route_sums(self, T, binding: "WeightBinding", slot=0, weighted_only: bool = True) -> np.ndarray:
        """sum over routes of each group of J[slot,k] alpha T_k: (R,) or (S, R) float64.

        ``weighted_only=True`` is the encoder input v_r (Methods, Derivation of the transition-programmable architecture); ``False`` sums all routes,
        which is used for score/violation slots.
        """
        si = binding.slot_index(slot) if isinstance(slot, str) else int(slot)
        T2, single = _spins2d(T)
        alpha = self.route_alpha(T2).astype(np.float64)
        rc = self.route_clause.astype(np.int64)
        contrib = binding.J[si, rc][None, :] * alpha * T2[:, rc].astype(np.float64)
        g = self.route_group()
        mask = self.weighted_route_mask() if weighted_only else np.ones(self.n_routes, dtype=bool)
        v = np.zeros((T2.shape[0], self.n_groups), dtype=np.float64)
        np.add.at(v, (slice(None), g[mask]), contrib[:, mask])
        return v[0] if single else v

    def inputs(self, T, binding: "WeightBinding", slot=0) -> np.ndarray:
        """Encoder inputs v_r = sum over weighted routes J alpha T (Methods, Derivation of the transition-programmable architecture)."""
        return self.route_sums(T, binding, slot, weighted_only=True)

    def decoder_toggles(self, r: int, T) -> np.ndarray:
        """Latched sigma (Methods, Derivation of the transition-programmable architecture) for accepted group r: (M~,) or (S, M~) uint8."""
        T2, single = _spins2d(T)
        a, b = int(self.route_ptr[r]), int(self.route_ptr[r + 1])
        sigma = np.zeros((T2.shape[0], self.n_clauses), dtype=np.uint8)
        if b > a:
            ctrl = self.route_ctrl[a:b].astype(np.int64)
            pol = self.route_pol[a:b].astype(np.int64)
            al = np.ones((T2.shape[0], b - a), dtype=np.int64)
            m = ctrl >= 0
            if np.any(m):
                al[:, m] = (1 + pol[m] * T2[:, ctrl[m]].astype(np.int64)) // 2
            sigma[:, self.route_clause[a:b]] = al.astype(np.uint8)
        return sigma[0] if single else sigma

    def spin_readout(self, T) -> np.ndarray:
        """Spin readout s_i = T_{k(i)} from the stored singleton clauses (int8, same leading shape as T).

        Clause state is authoritative: the architecture backends hold no separately
        updated spin array; every spin value they report is read from its singleton clause."""
        T = np.asarray(T)
        return np.ascontiguousarray(T[..., self.readout_clause_of_spin.astype(np.int64)]).astype(np.int8)

    def apply_action(self, r: int, spins, T=None) -> np.ndarray:
        """Reference transformation mu^(r) of group r from its descriptor, applied to a copy of spins.

        Used only by independent reference paths (the oracle backend, compile-time verification,
        tests). The architecture backends never call it: they execute mu^(r) through the decoder
        toggles of the singleton clauses.

        READOUT actions need the pre-transition clause signs T (computed from spins if omitted):
        spin i flips iff the latched sigma of its singleton readout clause is 1.
        """
        s, single = _spins2d(spins)
        out = s.copy()
        kind = int(self.act_kind[r])
        a0, a1 = int(self.act_ptr[r]), int(self.act_ptr[r + 1])
        idx = self.act_idx[a0:a1].astype(np.int64)
        if kind == KIND_MASK:
            out[:, idx] *= -1
        elif kind == KIND_CONDMASK:
            c0, c1 = int(self.cond_ptr[r]), int(self.cond_ptr[r + 1])
            g = self.cond_idx[c0:c1].astype(np.int64)
            prod = 1 - 2 * (np.count_nonzero(s[:, g] < 0, axis=1) & 1) if g.size else np.ones(s.shape[0], np.int64)
            active = (int(self.cond_pol[r]) * prod) == 1
            rows = np.flatnonzero(active)
            if rows.size:
                out[np.ix_(rows, idx)] *= -1
        elif kind == KIND_PERM:
            src = self.act_src[a0:a1].astype(np.int64)
            sgn = self.act_sgn[a0:a1].astype(s.dtype)
            out[:, idx] = s[:, src] * sgn
        else:
            if T is None:
                T = self.clause_values(s)
            T2, _ = _spins2d(T)
            sigma = self.decoder_toggles(r, T2)
            flip = sigma[:, self.readout_clause_of_spin.astype(np.int64)] == 1
            out = np.where(flip, -out, out).astype(s.dtype)
        return out[0] if single else out

    def energy(self, spins, binding: "WeightBinding", slot=0):
        """Direct E = const[slot] - sum_k J[slot,k] T_k(s) (Methods, Derivation of the transition-programmable architecture); float or (S,)."""
        si = binding.slot_index(slot) if isinstance(slot, str) else int(slot)
        T = self.clause_values(spins)
        if T.ndim == 1:
            return float(binding.const[si] - T.astype(np.float64) @ binding.J[si])
        return binding.const[si] - T.astype(np.float64) @ binding.J[si]

    # ------------------------------------------------------------------------------------------
    # reports and persistence
    # ------------------------------------------------------------------------------------------
    @property
    def nbytes(self) -> int:
        return int(sum(getattr(self, n).nbytes for n in _ARRAY_NAMES)
                   + sum(a.nbytes for a in self.score_term_to_clause.values()))

    @property
    def resources(self) -> dict:
        res = self.metadata.get("resources")
        return dict(res) if res else structural_resources(self)

    def describe(self) -> dict:
        res = self.resources
        mat = res.get("materialized", res)
        return {"schema_version": self.schema_version, "n_spins": self.n_spins, "n_clauses": self.n_clauses,
                "n_groups": self.n_groups, "n_routes": self.n_routes,
                "M": mat.get("M"), "M_add": mat.get("M_add"), "M_read": mat.get("M_read"),
                "feasible_simplification": self.feasible_simplification,
                "n_identities": len(self.identities), "topology_hash": self.topology_hash,
                "library": self.metadata.get("library"), "family": self.metadata.get("family"),
                "resources": res}

    def arrays(self) -> dict:
        return {n: getattr(self, n) for n in _ARRAY_NAMES}

    def save(self, path) -> tuple[str, str]:
        """Write ``<path>.npz`` (arrays) and ``<path>.json`` (scalars, labels, metadata, hashes)."""
        base = _split_path(path)
        arrays = {n: np.asarray(getattr(self, n)) for n in _ARRAY_NAMES}
        for name, arr in self.score_term_to_clause.items():
            arrays["score__" + name] = np.asarray(arr)
        np.savez_compressed(base + ".npz", **arrays)
        meta = {"schema_version": self.schema_version, "n_spins": self.n_spins, "n_clauses": self.n_clauses,
                "n_groups": self.n_groups, "group_label": list(self.group_label),
                "identities": [[list(s), c] for s, c in self.identities],
                "feasible_simplification": self.feasible_simplification, "library_hash": self.library_hash,
                "score_slots": list(self.score_term_to_clause.keys()), "metadata": self.metadata,
                "topology_hash": self.topology_hash}
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(jsonable(meta), f, indent=1)
        return base + ".npz", base + ".json"

    @classmethod
    def load(cls, path) -> "CompiledModel":
        """Load a saved model; the topology hash is recomputed and must match the recorded one."""
        base = _split_path(path)
        try:
            with open(base + ".json", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise FormatError(f"{base}.json: cannot read model metadata ({exc})") from exc
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise FormatError(f"{base}.json: unsupported schema {meta.get('schema_version')!r}")
        try:
            with np.load(base + ".npz", allow_pickle=False) as z:
                arrays = {n: z[n] for n in _ARRAY_NAMES}
                score = {n: z["score__" + n] for n in meta.get("score_slots", [])}
        except (OSError, KeyError, ValueError) as exc:
            raise FormatError(f"{base}.npz: cannot read model arrays ({exc})") from exc
        try:
            return cls(n_spins=meta["n_spins"], n_clauses=meta["n_clauses"], n_groups=meta["n_groups"],
                       group_label=meta["group_label"], identities=[(tuple(s), c) for s, c in meta["identities"]],
                       feasible_simplification=meta["feasible_simplification"], metadata=meta.get("metadata", {}),
                       library_hash=meta.get("library_hash", ""), score_term_to_clause=score,
                       schema_version=meta["schema_version"], topology_hash=meta["topology_hash"], **arrays)
        except ValueError as exc:
            raise FormatError(f"{base}: invalid model ({exc})") from exc


def save_model(model: CompiledModel, path):
    return model.save(path)


def load_model(path) -> CompiledModel:
    return CompiledModel.load(path)


def structural_resources(model: CompiledModel) -> dict:
    """Resource counts derivable from the model arrays (Supplementary Information, Size of the compiled machine).

    Counting convention (Supplementary Algorithm 1 (compilation of a move library)): a clause that is both gate and readout counts as gate;
    M = #OBJECTIVE, M_add = #(GATE and not OBJECTIVE), M_read = #(READOUT and not OBJECTIVE and not GATE).
    """
    role = model.clause_role.astype(np.int64)
    obj = (role & ROLE_OBJECTIVE) != 0
    gate = (role & ROLE_GATE) != 0
    read = (role & ROLE_READOUT) != 0
    score = (role & ROLE_SCORE) != 0
    nnz = model.n_routes
    R = model.n_groups
    wmask = model.weighted_route_mask()
    ctrl = model.route_ctrl
    rc = model.route_clause.astype(np.int64)
    fan_in = (model.route_wend - model.route_ptr[:-1]).astype(np.int64)
    dec_fan = np.diff(model.route_ptr).astype(np.int64)
    ctrl_fan = np.diff(model.ctrl_route_ptr).astype(np.int64)
    controllers = ctrl_fan[ctrl_fan > 0]
    is_gate_only = gate & ~obj
    is_read_only = read & ~obj & ~gate
    is_score_only = score & ~obj & ~gate & ~read

    def mm(a):
        return (float(a.mean()) if a.size else 0.0), (int(a.max()) if a.size else 0)

    fi_mean, fi_max = mm(fan_in)
    df_mean, df_max = mm(dec_fan)
    cf_mean, cf_max = mm(controllers)
    return {
        "M": int(obj.sum()), "M_add": int(is_gate_only.sum()), "M_read": int(is_read_only.sum()),
        "M_score": int(is_score_only.sum()), "M_total": int(model.n_clauses), "R": int(R), "N": int(model.n_spins),
        "role_overlaps": {
            "objective+gate": int((obj & gate).sum()), "objective+readout": int((obj & read).sum()),
            "gate+readout": int((gate & read).sum()), "objective+gate+readout": int((obj & gate & read).sum()),
            "score+gate": int((score & gate).sum()), "score+readout": int((score & read).sum()),
        },
        "routes_weighted": int(wmask.sum()), "routes_decoder": int(nnz),
        "routes_gate_controlled": int((ctrl >= 0).sum()), "routes_constant": int((ctrl < 0).sum()),
        "routes_by_clause_class": {
            "objective": int(obj[rc].sum()) if nnz else 0, "gate_only": int(is_gate_only[rc].sum()) if nnz else 0,
            "readout_only": int(is_read_only[rc].sum()) if nnz else 0,
            "score_only": int(is_score_only[rc].sum()) if nnz else 0,
        },
        "readout_routes": int(read[rc].sum()) if nnz else 0,
        "readout_only_routes": int(is_read_only[rc].sum()) if nnz else 0,
        "support_storage": int(model.clause_spins.size),
        "max_clause_degree": int(np.diff(model.clause_ptr).max()) if model.n_clauses else 0,
        "fan_in_mean": fi_mean, "fan_in_max": fi_max,
        "decoder_fan_out_mean": df_mean, "decoder_fan_out_max": df_max,
        "n_controllers": int(controllers.size), "control_fan_out_mean": cf_mean, "control_fan_out_max": cf_max,
        "memory_bytes": model.nbytes,
    }


# ============================================================================================
# weight binding
# ============================================================================================
@dataclass(frozen=True, eq=False)
class WeightBinding:
    """Clause weights and constants for one observation/instance.

    ``J[slot, k]``: weight of stored clause k in slot ``slot``; slot 0 is always ``"run"`` and
    drives acceptance. ``weight_hash`` = sha256(J, const, slot names, topology_hash).
    """

    slot_names: tuple
    J: np.ndarray
    const: np.ndarray
    topology_hash: str
    weight_hash: str = ""

    def __post_init__(self):
        names = tuple(str(x) for x in self.slot_names)
        if not names or names[0] != "run":
            raise ValueError("slot 0 of a WeightBinding must be named 'run'")
        if len(set(names)) != len(names):
            raise ValueError("duplicate slot names")
        J = np.array(self.J, dtype=np.float64, copy=True)
        if J.ndim != 2 or J.shape[0] != len(names):
            raise ValueError("J must be float64[n_slots, M~]")
        const = np.array(self.const, dtype=np.float64, copy=True).reshape(-1)
        if const.size != len(names):
            raise ValueError("const must have one entry per slot")
        if not (np.all(np.isfinite(J)) and np.all(np.isfinite(const))):
            raise ValueError("weights and constants must be finite")
        object.__setattr__(self, "slot_names", names)
        object.__setattr__(self, "J", _readonly(J))
        object.__setattr__(self, "const", _readonly(const))
        h = self._compute_hash()
        if self.weight_hash and self.weight_hash != h:
            raise FormatError("weight hash mismatch: the binding does not match its recorded hash")
        object.__setattr__(self, "weight_hash", h)

    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps({"slots": list(self.slot_names), "topology_hash": self.topology_hash},
                            separators=(",", ":")).encode())
        _hash_array(h, "J", self.J)
        _hash_array(h, "const", self.const)
        return h.hexdigest()

    @property
    def n_slots(self) -> int:
        return len(self.slot_names)

    @property
    def n_clauses(self) -> int:
        return int(self.J.shape[1])

    def slot_index(self, name) -> int:
        if isinstance(name, (int, np.integer)):
            if not 0 <= int(name) < self.n_slots:
                raise KeyError(f"slot {name} out of range")
            return int(name)
        try:
            return self.slot_names.index(str(name))
        except ValueError:
            raise KeyError(f"unknown slot {name!r}; have {self.slot_names}") from None

    def validate_for(self, model: CompiledModel):
        """Check topology hash, shape and that slot 0 weights only weighted (OBJECTIVE) clauses."""
        if self.topology_hash != model.topology_hash:
            raise ValueError("binding was made for a different model topology")
        if self.J.shape[1] != model.n_clauses:
            raise ValueError("binding J has the wrong number of clauses")
        nonobj = (model.clause_role & ROLE_OBJECTIVE) == 0
        if np.any(self.J[0, nonobj] != 0.0):
            raise ValueError("slot 'run' must be zero on non-objective clauses")

    def save(self, path) -> tuple[str, str]:
        base = _split_path(path)
        np.savez_compressed(base + ".npz", J=np.asarray(self.J), const=np.asarray(self.const))
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({"schema_version": BINDING_SCHEMA_VERSION, "slot_names": list(self.slot_names),
                       "topology_hash": self.topology_hash, "weight_hash": self.weight_hash}, f, indent=1)
        return base + ".npz", base + ".json"

    @classmethod
    def load(cls, path, model: CompiledModel | None = None) -> "WeightBinding":
        base = _split_path(path)
        try:
            with open(base + ".json", encoding="utf-8") as f:
                meta = json.load(f)
            with np.load(base + ".npz", allow_pickle=False) as z:
                J, const = z["J"], z["const"]
        except (OSError, KeyError, ValueError) as exc:
            raise FormatError(f"{base}: cannot read weight binding ({exc})") from exc
        if meta.get("schema_version") != BINDING_SCHEMA_VERSION:
            raise FormatError(f"{base}.json: unsupported binding schema {meta.get('schema_version')!r}")
        b = cls(slot_names=meta["slot_names"], J=J, const=const, topology_hash=meta["topology_hash"],
                weight_hash=meta["weight_hash"])
        if model is not None:
            b.validate_for(model)
        return b


def _accumulate(n_clauses: int, mapping: np.ndarray, weights, const: float, what: str):
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != mapping.size:
        raise ValueError(f"{what}: got {w.size} term weights for {mapping.size} terms")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{what}: weights must be finite")
    J = np.zeros(n_clauses, dtype=np.float64)
    m = mapping.astype(np.int64)
    valid = m >= 0
    np.add.at(J, m[valid], w[valid])  # sequential, deterministic order (term order)
    # a constant term (empty support, T = 1) with weight J_t contributes -J_t under E = E_const - sum J T
    c = float(const) - float(np.sum(w[~valid])) if np.any(~valid) else float(const)
    return J, c


def bind_weights(model: CompiledModel, term_weights, const: float = 0.0,
                 extra_slots: Mapping | None = None) -> WeightBinding:
    """Sum term weights into clause weights through ``term_to_clause`` (never changes topology).

    ``term_weights``: J per objective term in structural order (manuscript sign:
    E = E_const - sum J T); a constant (empty-support) term has T = 1, so its weight J_t is folded
    into the slot constant as ``const - J_t``.
    ``extra_slots``: optional ``{name: (term_weights, const)}``. A slot whose name was compiled as a
    score structure uses that structure's own term order (``model.score_term_to_clause[name]``);
    any other name uses the objective structure's term order.
    """
    J0, c0 = _accumulate(model.n_clauses, model.term_to_clause, term_weights, const, "run")
    names, Js, cs = ["run"], [J0], [c0]
    for name, spec in dict(extra_slots or {}).items():
        name = str(name)
        if name in names:
            raise ValueError(f"duplicate slot {name!r}")
        if isinstance(spec, Mapping):
            w, c = spec["weights"], spec.get("const", 0.0)
        else:
            w, c = spec
        mapping = model.score_term_to_clause.get(name, model.term_to_clause)
        Jx, cx = _accumulate(model.n_clauses, mapping, w, c, name)
        names.append(name)
        Js.append(Jx)
        cs.append(cx)
    b = WeightBinding(slot_names=tuple(names), J=np.stack(Js), const=np.asarray(cs),
                      topology_hash=model.topology_hash)
    b.validate_for(model)
    return b

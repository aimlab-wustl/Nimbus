"""Compile-time coloring of transition groups for the ``colored`` process.

Two groups r, r' *interact* iff

* their route clause sets overlap (a clause that one group toggles is also read or toggled by the other:
  v_r and v_r' both read stored clause signs, and a doubly latched clause would be toggled twice), or
* a clause on a route of one group is the controller of a route of the other (a commit of one group can
  re-evaluate the other's switches).

Every route of every group counts: weighted routes (objective clauses), unweighted routes (gates) and the
READOUT routes of the spin singleton clauses. The rule is a static over-approximation of
the dynamic one (it ignores which switches are open), so it is state independent and decided once.

Groups of one color class never interact, therefore (i) the inputs v_r of the class are mutually
independent, (ii) the toggled clause sets are disjoint, so committing any subset of the class in any
order gives the same clause state T and the same input vector v (exact equality; tested), and (iii) the
energy change of the committed set is the sum of the 2 v_r.

Coloring: DSATUR (largest saturation, then largest degree, then smallest group id), deterministic. The
concatenation of the classes (class 0 first, groups ascending inside a class) is the *position order*
of the cyclic visit used by the process: attempt clock a visits position a mod R.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from nimbus.core.model import CompiledModel

__all__ = ["ColoringPlan", "interaction_graph", "dsatur", "color_groups", "check_coloring"]


@dataclass(frozen=True)
class ColoringPlan:
    """Result of :func:`color_groups` (all arrays int64)."""

    color_of_group: np.ndarray   # [R] class id of each group
    order: np.ndarray            # [R] groups in position order (class 0 first, ascending r inside a class)
    class_ptr: np.ndarray        # [C+1] class c occupies positions class_ptr[c]:class_ptr[c+1]
    class_of_pos: np.ndarray     # [R] class id of each position
    n_edges: int                 # interaction-graph edges (undirected)

    @property
    def n_colors(self) -> int:
        return int(self.class_ptr.size - 1)

    @property
    def class_sizes(self) -> np.ndarray:
        return np.diff(self.class_ptr)

    def classes(self):
        return [self.order[self.class_ptr[c]:self.class_ptr[c + 1]] for c in range(self.n_colors)]

    def summary(self) -> dict:
        sz = self.class_sizes
        R = int(self.order.size)
        return {"n_colors": self.n_colors, "n_groups": R, "n_edges": int(self.n_edges),
                "class_size_min": int(sz.min()) if sz.size else 0, "class_size_max": int(sz.max()) if sz.size else 0,
                "class_size_mean": float(sz.mean()) if sz.size else 0.0,
                "class_sizes": sz.tolist() if sz.size <= 128 else None,
                "mean_attempts_per_step": float(R / max(1, self.n_colors))}


def interaction_graph(model: CompiledModel) -> sp.csr_matrix:
    """Symmetric boolean adjacency [R, R] (no diagonal) of the interaction rule above."""
    R, M = model.n_groups, model.n_clauses
    grp = model.route_group().astype(np.int64)
    rc = model.route_clause.astype(np.int64)
    rk = model.route_ctrl.astype(np.int64)
    ones = np.ones(grp.size, dtype=np.int8)
    A = sp.csr_matrix((ones, (grp, rc)), shape=(R, M))          # group -> clauses on its routes
    m = rk >= 0
    C = sp.csr_matrix((ones[m], (grp[m], rk[m])), shape=(R, M))  # group -> controllers of its routes
    A.data[:] = 1
    C.data[:] = 1
    adj = (A @ A.T) + (A @ C.T)
    adj = adj + adj.T
    adj = adj.tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1
    return adj


def dsatur(adj: sp.csr_matrix) -> np.ndarray:
    """Deterministic DSATUR coloring of a symmetric adjacency; returns int64[R] colors 0..C-1."""
    R = adj.shape[0]
    color = np.full(R, -1, dtype=np.int64)
    if R == 0:
        return color
    deg = np.diff(adj.indptr).astype(np.int64)
    nbrs = [adj.indices[adj.indptr[i]:adj.indptr[i + 1]] for i in range(R)]
    sat = [set() for _ in range(R)]
    for _ in range(R):
        best, key = -1, None
        for i in range(R):
            if color[i] >= 0:
                continue
            k = (len(sat[i]), int(deg[i]), -i)
            if key is None or k > key:
                best, key = i, k
        c = 0
        while c in sat[best]:
            c += 1
        color[best] = c
        for j in nbrs[best]:
            sat[j].add(c)
    return color


def color_groups(model: CompiledModel) -> ColoringPlan:
    """Interaction graph, DSATUR coloring and the position order of the cyclic class visit."""
    adj = interaction_graph(model)
    color = dsatur(adj)
    R = model.n_groups
    n_col = int(color.max()) + 1 if R else 0
    order = np.lexsort((np.arange(R), color)).astype(np.int64)   # by class, then ascending r
    sizes = np.bincount(color, minlength=n_col).astype(np.int64) if R else np.zeros(0, np.int64)
    ptr = np.zeros(n_col + 1, dtype=np.int64)
    np.cumsum(sizes, out=ptr[1:])
    class_of_pos = np.repeat(np.arange(n_col, dtype=np.int64), sizes)
    return ColoringPlan(color_of_group=color, order=order, class_ptr=ptr, class_of_pos=class_of_pos,
                        n_edges=int(adj.nnz // 2))


def check_coloring(model: CompiledModel, plan: ColoringPlan) -> None:
    """Raise AssertionError if two groups of one class interact (independent re-derivation from the
    route arrays, set based)."""
    R = model.n_groups
    grp = model.route_group().astype(np.int64)
    clauses = [set() for _ in range(R)]
    ctrls = [set() for _ in range(R)]
    for g, k, c in zip(grp.tolist(), model.route_clause.astype(np.int64).tolist(),
                       model.route_ctrl.astype(np.int64).tolist()):
        clauses[g].add(k)
        if c >= 0:
            ctrls[g].add(c)
    for cls in plan.classes():
        cls = cls.tolist()
        for i, a in enumerate(cls):
            for b in cls[i + 1:]:
                assert not (clauses[a] & clauses[b]), (a, b, "route clause sets overlap")
                assert not (clauses[a] & ctrls[b]) and not (clauses[b] & ctrls[a]), (a, b, "controller overlap")

"""Structural verification of transition libraries on a feasible domain (Supplementary Algorithm 1 (compilation of a move library) "Finding and selecting grouped transitions").

``verify_library`` checks, for every group: feasibility preservation (every image of a feasible
state is feasible), the involution property (mu(mu(s)) = s) or, for a move that is not its own
inverse, that another move of the library undoes it, and self-loop fractions; and for the whole
library: connectivity of the induced state graph by BFS/union-find.

* Exhaustive mode (``states`` = the complete enumerated Omega): the checks are proofs on Omega.
* Sampled mode (``sample_states`` + ``is_feasible``): checks only the supplied states and their
  images; the report is labelled ``"sampled, not a proof"`` and makes no connectivity claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from nimbus.core.errors import UnsupportedTransition

__all__ = ["LibraryVerification", "verify_library", "state_keys"]


def state_keys(states: np.ndarray) -> np.ndarray:
    """Hashable byte keys of ±1 states (one numpy void scalar per row)."""
    s = np.ascontiguousarray(np.asarray(states) < 0)
    packed = np.packbits(s, axis=1)
    return np.ascontiguousarray(packed).view(np.dtype((np.void, packed.shape[1]))).reshape(-1)


@dataclass
class LibraryVerification:
    mode: str  # "exhaustive" | "sampled, not a proof"
    n_states: int
    n_groups: int
    feasibility_preserved: bool
    involution: bool
    connected: bool | None
    n_components: int | None
    selfloop_fraction: list = field(default_factory=list)
    infeasible_groups: list = field(default_factory=list)
    non_involutive_groups: list = field(default_factory=list)
    examples: list = field(default_factory=list)
    n_edges_changed: int = 0
    missing_reverse_groups: list = field(default_factory=list)

    @property
    def reversible(self) -> bool:
        """Every move is its own inverse or has its reverse move in the library."""
        return not self.missing_reverse_groups

    @property
    def ok(self) -> bool:
        conn_ok = self.connected is None or self.connected
        return self.feasibility_preserved and self.reversible and conn_ok

    def summary(self) -> dict:
        d = asdict(self)
        sf = np.asarray(self.selfloop_fraction, dtype=float)
        d["mean_selfloop_fraction"] = float(sf.mean()) if sf.size else 0.0
        d["selfloop_fraction"] = [float(x) for x in self.selfloop_fraction]
        d["reversible"] = self.reversible
        d["ok"] = self.ok
        return d


def verify_library(library, states=None, *, is_feasible=None, sample_states=None, check_connectivity: bool = True,
                   require: bool = False, max_examples: int = 5) -> LibraryVerification:
    """Verify a TransitionLibrary on an enumerated domain (exhaustive) or on sample states (sampled).

    Parameters
    ----------
    states: (S, N) array of ALL feasible states (exhaustive mode).
    is_feasible: callable(state 1-D) -> bool, required in sampled mode (and optional extra check).
    sample_states: (S, N) feasible sample states (sampled mode; not a proof).
    require: raise UnsupportedTransition if a check fails.
    """
    if states is not None:
        S = np.asarray(states)
        mode = "exhaustive"
    elif sample_states is not None:
        if is_feasible is None:
            raise ValueError("sampled verification needs is_feasible")
        S = np.asarray(sample_states)
        mode = "sampled, not a proof"
    else:
        raise ValueError("give the enumerated feasible states or sample_states + is_feasible")
    if S.ndim != 2 or S.shape[1] != library.n_spins:
        raise ValueError("states must be (S, N) with N = library.n_spins")
    n = S.shape[0]
    keys = state_keys(S)
    index = {k.tobytes(): i for i, k in enumerate(keys)}
    if mode == "exhaustive" and len(index) != n:
        raise ValueError("duplicate states in the enumerated domain")
    infeasible, non_inv, examples, selfloop = [], [], [], []
    src_e, dst_e = [], []
    for r, g in enumerate(library.groups):
        mu = g.apply(S)
        same = np.all(mu == S, axis=1)
        selfloop.append(float(same.mean()) if n else 0.0)
        mkeys = state_keys(mu)
        if mode == "exhaustive":
            dest = np.fromiter((index.get(k.tobytes(), -1) for k in mkeys), dtype=np.int64, count=n)
            bad = dest < 0
            if is_feasible is not None:
                bad |= ~np.fromiter((bool(is_feasible(x)) for x in mu), dtype=bool, count=n)
        else:
            bad = ~np.fromiter((bool(is_feasible(x)) for x in mu), dtype=bool, count=n)
            dest = None
        if np.any(bad):
            infeasible.append(r)
            if len(examples) < max_examples:
                i = int(np.flatnonzero(bad)[0])
                examples.append({"group": r, "label": library.labels[r], "issue": "leaves the feasible set",
                                 "state": S[i].tolist(), "image": mu[i].tolist()})
        back = g.apply(mu[~bad]) if np.any(~bad) else mu[:0]
        if not np.array_equal(back, S[~bad]):
            non_inv.append(r)
            if len(examples) < max_examples:
                i = int(np.flatnonzero(np.any(back != S[~bad], axis=1))[0])
                examples.append({"group": r, "label": library.labels[r], "issue": "not an involution",
                                 "state": S[~bad][i].tolist()})
        if dest is not None:
            ok = (~bad) & (~same)
            src_e.append(np.flatnonzero(ok))
            dst_e.append(dest[ok])
    # a move that is not its own inverse needs a move in the library that undoes it on these states
    missing = []
    for r in non_inv:
        g = library.groups[r]
        mu = g.apply(S)
        if not any(q != r and np.array_equal(library.groups[q].apply(mu), S) for q in non_inv):
            missing.append(r)
            if len(examples) < max_examples:
                examples.append({"group": r, "label": library.labels[r],
                                 "issue": "not its own inverse and no move of the library reverses it"})
    connected = None
    ncomp = None
    n_changed = int(sum(x.size for x in src_e))
    if mode == "exhaustive" and check_connectivity and n:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components

        if src_e:
            a = np.concatenate(src_e)
            b = np.concatenate(dst_e)
        else:
            a = b = np.zeros(0, dtype=np.int64)
        graph = sp.coo_matrix((np.ones(a.size, dtype=np.int8), (a, b)), shape=(n, n)).tocsr()
        ncomp, _labels = connected_components(graph, directed=True, connection="weak")
        connected = bool(ncomp == 1)
    rep = LibraryVerification(mode=mode, n_states=n, n_groups=library.n_groups, feasibility_preserved=not infeasible,
                              involution=not non_inv, connected=connected, n_components=ncomp,
                              selfloop_fraction=selfloop, infeasible_groups=infeasible,
                              non_involutive_groups=non_inv, examples=examples, n_edges_changed=n_changed,
                              missing_reverse_groups=missing)
    if require and not rep.ok:
        raise UnsupportedTransition(f"library {library.name!r} failed verification: {rep.summary()}")
    return rep

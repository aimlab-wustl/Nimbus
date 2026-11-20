"""Finding moves by grouping spin transitions (finite-table route).

For a problem small enough to enumerate its feasible states, the moves can be found from the
feasible states alone, without knowing the structure of the problem:

1. The spin transitions are the pairs of feasible states that differ in the fewest spins
   (or in at most ``max_distance`` spins).
2. Transitions that one spin transformation reproduces are collected into a move. The
   transformation must map every feasible state to a feasible state (or to itself, a
   self-loop), be its own inverse and produce only transitions of step 1.
3. Moves whose transitions connect all feasible states are selected as the backbone; the
   remaining moves are shortcut candidates.

**Which transformations are accepted.** The compiler stores every spin as a clause, so a move can
be wired only if, for every spin, *whether the move flips it* is set by one signed spin product
of the current state. On the feasible states this means that the flip pattern is an affine
function over GF(2): with ``y = (1 - s)/2``, the flip mask is ``d(y) = A y + b``. Every clause
change is then one spin product as well, so every move found here compiles with one clause per
switch, whatever the objective.

**How the moves are found.** The transitions first, then four searches, from the cheapest:

* *Transitions*: a few states give an upper bound on the smallest distance, and every pair within
  it is found by the cheapest exact method: hashing the states with every flip pattern of few
  spins, multi-index hashing (states within distance d agree on one of d + 1 spin blocks, so only
  states sharing a block are compared), or bit-parallel pairwise distances.
* *Flip moves* (a fixed set of spins flipped wherever a condition holds, including pair
  exchanges): for each flip pattern, the condition is an affine function that vanishes outside
  the states having that transition and is invariant under it. These are linear equations, so
  the maximal flip moves follow from one null space per pattern, solved on a sample of the
  equations and verified on all of them.
* *Permutation moves* (exchanges of spins, possibly with sign reversals, such as exchanges of
  rows or columns of a one-hot matrix): a constraint search over involutive spin permutations.
  Every feasible state keeps a short list of possible destinations (itself and its transitions);
  each spin assignment filters all lists at once and forced destinations are propagated, so
  inconsistent branches end after few steps. To list every permutation move, the search is
  seeded by the first moved spin and its image, about N^2 seeds, each move found once.
* *Compositions*: affine moves are closed under composition, so a transition that no move covers
  is tried as the composition of two commuting moves already found (one pass over the states).
* *General affine moves*, only for transitions still uncovered: a search over the destinations of
  single states, in which the affine form determines every state whose coordinates lie in the
  span of the chosen ones (bounded by ``max_nodes``).

States and flip masks are bit-packed; feasibility tests use a hash of the packed states.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from nimbus.compiler.table_fit import AffineFitter
from nimbus.core.errors import UnsupportedTransition
from nimbus.core.gf2 import bits_to_indices, nullspace
from nimbus.transitions.descriptors import (
    AffineMove,
    ConditionalMask,
    FixedMask,
    SignedPermutation,
    SpinAction,
    TableTransition,
)
from nimbus.transitions.library import TransitionLibrary

__all__ = ["candidate_transitions", "group_transitions", "FoundMove", "MoveGrouping"]

_ONE = np.uint64(1)
_MIX = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93)


# ============================================================================================
# bit-packed helpers
# ============================================================================================
def _pack(bits: np.ndarray) -> np.ndarray:
    """(m, n) 0/1 array -> (m, ceil(n/64)) uint64 words; bit j of word w is column 64 w + j."""
    bits = np.asarray(bits, dtype=np.uint8)
    m, n = bits.shape
    nw = max(1, -(-n // 64))
    padded = np.zeros((m, nw * 64), dtype=np.uint8)
    padded[:, :n] = bits
    return np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little")).view("<u8").astype(np.uint64)


def _unpack(words: np.ndarray, n: int) -> np.ndarray:
    b = np.unpackbits(np.ascontiguousarray(words.astype("<u8")).view(np.uint8), axis=1, bitorder="little")
    return b[:, :n]


def _bit(words: np.ndarray, p: int) -> np.ndarray:
    return ((words[:, p >> 6] >> np.uint64(p & 63)) & _ONE).astype(bool)


def _to_int(row: np.ndarray) -> int:
    v = 0
    for k, w in enumerate(row.tolist()):
        v |= int(w) << (64 * k)
    return v


def _to_words(v: int, nw: int) -> np.ndarray:
    return np.array([(v >> (64 * k)) & 0xFFFFFFFFFFFFFFFF for k in range(nw)], dtype=np.uint64)


if hasattr(np, "bitwise_count"):
    def _popcount_rows(words: np.ndarray) -> np.ndarray:
        return np.bitwise_count(words).sum(axis=-1, dtype=np.int64)
else:  # numpy < 2
    _LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount_rows(words: np.ndarray) -> np.ndarray:
        b = np.ascontiguousarray(words.astype("<u8")).view(np.uint8)
        return _LUT[b].reshape(*words.shape[:-1], -1).sum(axis=-1, dtype=np.int64)


# ============================================================================================
# the feasible domain
# ============================================================================================
class _Domain:
    """Enumerated feasible states: packed rows, a hash index and affine coordinates."""

    def __init__(self, states):
        S = np.asarray(states)
        if S.ndim != 2 or S.shape[0] == 0:
            raise ValueError("states must be a non-empty (states, N) array")
        if np.any((S != 1) & (S != -1)):
            raise ValueError("spins must be -1/+1")
        self.S = S.astype(np.int8)
        self.m, self.N = self.S.shape
        self.Yw = _pack(self.S < 0)
        self.nw = self.Yw.shape[1]
        self._build_index()
        # affine coordinates: independent columns of [y, 1] (forward elimination, column order)
        aug = np.hstack([(self.S < 0), np.ones((self.m, 1), dtype=bool)])
        A = _pack(aug)
        free = np.ones(self.m, dtype=bool)
        piv = []
        for c in range(self.N + 1):
            idx = np.flatnonzero(_bit(A, c) & free)
            if idx.size == 0:
                continue
            r = idx[0]
            free[r] = False
            piv.append(c)
            if idx.size > 1:
                A[idx[1:]] ^= A[r]
        self.pivot_columns = piv
        self.rho = len(piv)
        self.Xw = _pack(aug[:, piv])
        self.nx = self.Xw.shape[1]

    def _keys(self, words: np.ndarray) -> np.ndarray:
        if words.shape[1] == 1:
            return words[:, 0].copy()
        h = np.zeros(words.shape[0], dtype=np.uint64)
        with np.errstate(over="ignore"):
            for k in range(words.shape[1]):
                h ^= words[:, k] * np.uint64(_MIX[k % 4] | 1)
                h = (h << np.uint64(31)) | (h >> np.uint64(33))
                h *= np.uint64(_MIX[(k + 1) % 4] | 1)
        return h

    def _build_index(self):
        keys = self._keys(self.Yw)
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        dup = np.flatnonzero(sk[1:] == sk[:-1])
        if dup.size:
            a, b = order[dup[0]], order[dup[0] + 1]
            if np.array_equal(self.Yw[a], self.Yw[b]):
                raise ValueError(f"duplicate feasible states {int(a)} and {int(b)}")
            raise RuntimeError("hash collision between two feasible states")  # pragma: no cover
        self._order, self._sorted = order, sk

    def lookup(self, words: np.ndarray) -> np.ndarray:
        """Index of each packed row in the domain, or -1."""
        if words.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        q = self._keys(words)
        pos = np.minimum(np.searchsorted(self._sorted, q), self.m - 1)
        idx = self._order[pos]
        ok = (self._sorted[pos] == q) & np.all(self.Yw[idx] == words, axis=1)
        return np.where(ok, idx, -1).astype(np.int64)


# ============================================================================================
# candidate transitions
# ============================================================================================
def _pairs_by_masks(dom: _Domain, k: int, budget: int):
    """Unordered pairs at distance exactly k by hashing flipped rows; None if over budget."""
    n_masks = math.comb(dom.N, k)
    if n_masks * dom.m > budget:
        return None
    out_a, out_b = [], []
    chunk = max(1, (1 << 22) // max(1, dom.m))
    combos = itertools.combinations(range(dom.N), k)
    while True:
        block = list(itertools.islice(combos, chunk))
        if not block:
            break
        bits = np.zeros((len(block), dom.N), dtype=np.uint8)
        for r, c in enumerate(block):
            bits[r, list(c)] = 1
        M = _pack(bits)                                            # (B, nw)
        Q = (dom.Yw[None, :, :] ^ M[:, None, :]).reshape(-1, dom.nw)
        j = dom.lookup(Q)
        i = np.tile(np.arange(dom.m, dtype=np.int64), len(block))
        keep = j > i
        out_a.append(i[keep])
        out_b.append(j[keep])
    a = np.concatenate(out_a) if out_a else np.zeros(0, dtype=np.int64)
    b = np.concatenate(out_b) if out_b else np.zeros(0, dtype=np.int64)
    return a, b


def _pairs_pairwise(dom: _Domain, max_distance):
    """All unordered pairs with distance <= max_distance (or at the minimum distance if None)."""
    m = dom.m
    chunk = max(1, (1 << 24) // max(1, m * dom.nw))
    best = None
    out = []
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        D = _popcount_rows(dom.Yw[i0:i1, None, :] ^ dom.Yw[None, :, :])   # (c, m)
        ii, jj = np.nonzero(np.arange(m)[None, :] > np.arange(i0, i1)[:, None])
        d = D[ii, jj]
        ii = ii + i0
        if max_distance is None:
            pos = d > 0
            if not np.any(pos):
                continue
            dmin = int(d[pos].min())
            if best is None or dmin < best:
                best, out = dmin, []
            if dmin == best:
                sel = d == best
                out.append((ii[sel], jj[sel], d[sel]))
        else:
            sel = (d > 0) & (d <= max_distance)
            out.append((ii[sel], jj[sel], d[sel]))
    if not out:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    return tuple(np.concatenate([o[t] for o in out]).astype(np.int64) for t in range(3))


def _blocks(N: int, d: int):
    """d + 1 contiguous spin blocks (pigeonhole: states at distance <= d agree on one block)."""
    return [b for b in np.array_split(np.arange(N), d + 1) if b.size]


def _mih_cost(dom: _Domain, d: int):
    """Number of state pairs that multi-index hashing compares, or None if a block is too wide."""
    blocks = _blocks(dom.N, d)
    if len(blocks) < d + 1 or max(b.size for b in blocks) > 64:
        return None
    Y = (dom.S < 0)
    cost = 0
    for b in blocks:
        _, counts = np.unique(_pack(Y[:, b])[:, 0], return_counts=True)
        cost += int((counts * (counts - 1) // 2).sum())
    return cost


def _pairs_mih(dom: _Domain, d: int):
    """All unordered pairs with 0 < distance <= d, comparing only states that agree on one block."""
    Y = (dom.S < 0)
    keys = [_pack(Y[:, b])[:, 0] for b in _blocks(dom.N, d)]
    out_a, out_b, out_d = [], [], []
    tri = {}
    for bi, key in enumerate(keys):
        order = np.argsort(key, kind="stable")
        ks = key[order]
        cuts = np.flatnonzero(ks[1:] != ks[:-1]) + 1
        for grp in np.split(order, cuts):
            n = grp.size
            if n < 2:
                continue
            if n not in tri:
                tri[n] = np.triu_indices(n, 1)
            i, j = tri[n]
            a, c = grp[i], grp[j]
            first = np.ones(a.size, dtype=bool)        # count each pair in the first block it agrees on
            for prev in keys[:bi]:
                first &= prev[a] != prev[c]
            a, c = a[first], c[first]
            dist = _popcount_rows(dom.Yw[a] ^ dom.Yw[c])
            keep = (dist > 0) & (dist <= d)
            out_a.append(np.minimum(a[keep], c[keep]))
            out_b.append(np.maximum(a[keep], c[keep]))
            out_d.append(dist[keep])
    if not out_a:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    return np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_d)


def _distance_bound(dom: _Domain, n_probe: int = 64) -> int:
    """An upper bound on the smallest distance between two states: distances from a few states."""
    probe = np.unique(np.linspace(0, dom.m - 1, min(n_probe, dom.m)).astype(np.int64))
    D = _popcount_rows(dom.Yw[probe][:, None, :] ^ dom.Yw[None, :, :])
    D = D[D > 0]
    return int(D.min()) if D.size else dom.N


def _candidate_pairs(dom: _Domain, max_distance):
    """Pairs within max_distance (or at the smallest distance), by the cheapest exact method:
    hashing flipped states, multi-index hashing on spin blocks, or bit-parallel pairwise distances."""
    if dom.m < 2:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    d = min(int(max_distance), dom.N) if max_distance is not None else _distance_bound(dom)
    costs = {"pairwise": dom.m * (dom.m - 1) // 2 * dom.nw,
             "masks": dom.m * sum(math.comb(dom.N, k) for k in range(1, d + 1))}
    mih = _mih_cost(dom, d)
    if mih is not None:
        costs["blocks"] = mih + dom.m * (d + 1)
    method = min(costs, key=costs.get)
    if method == "masks":
        A, Bs, Ds = [], [], []
        for k in range(1, d + 1):
            a, b = _pairs_by_masks(dom, k, budget=1 << 62)
            if a.size:
                A.append(a)
                Bs.append(b)
                Ds.append(np.full(a.size, k, dtype=np.int64))
                if max_distance is None:
                    break
        if not A:
            z = np.zeros(0, dtype=np.int64)
            return z, z, z
        return np.concatenate(A), np.concatenate(Bs), np.concatenate(Ds)
    if method == "blocks":
        a, b, dist = _pairs_mih(dom, d)
    else:
        a, b, dist = _pairs_pairwise(dom, d)
    if max_distance is None and dist.size:
        keep = dist == dist.min()                       # every pair within the bound was found
        a, b, dist = a[keep], b[keep], dist[keep]
    return a, b, dist


def candidate_transitions(states, max_distance: int | None = None):
    """Spin transitions between feasible states: ``(pairs, distances)``.

    ``pairs`` is an (E, 2) array of state indices ``a < b``, sorted. With ``max_distance=None`` the
    transitions are the pairs that differ in the fewest spins; otherwise every pair that differs
    in at most ``max_distance`` spins.
    """
    dom = states if isinstance(states, _Domain) else _Domain(states)
    a, b, d = _candidate_pairs(dom, max_distance)
    order = np.lexsort((b, a))
    return np.stack([a[order], b[order]], axis=1), d[order]




# ============================================================================================
# flip moves: one null space per flip pattern
# ============================================================================================
def _row_basis(W: np.ndarray, nbits: int) -> list[int]:
    """Basis (Python ints) of the GF(2) row space of packed rows W."""
    if W.shape[0] == 0:
        return []
    A = W.copy()
    free = np.ones(A.shape[0], dtype=bool)
    out = []
    for c in range(nbits):
        idx = np.flatnonzero(_bit(A, c) & free)
        if idx.size == 0:
            continue
        r = idx[0]
        free[r] = False
        out.append(_to_int(A[r]))
        if idx.size > 1:
            A[idx[1:]] ^= A[r]
    return out


def _null_space(eq: np.ndarray, nbits: int) -> list[int]:
    """Null space of the packed equations: solved on a sample, verified on all equations; any
    violated equation joins the sample and the solve repeats (usually one or two rounds)."""
    if eq.shape[0] == 0:
        return [1 << j for j in range(nbits)]
    take = np.unique(np.linspace(0, eq.shape[0] - 1, min(eq.shape[0], 4 * nbits)).astype(np.int64))
    while True:
        basis = _row_basis(eq[take], nbits)
        if len(basis) == nbits:
            return []                                  # full rank: only the zero solution
        K = nullspace(basis, nbits) if basis else [1 << j for j in range(nbits)]
        bad = np.zeros(eq.shape[0], dtype=bool)
        for w in K:
            bad |= _parity(eq, w)
        if not bad.any():
            return K
        take = np.union1d(take, np.flatnonzero(bad)[:4 * nbits])


def _parity(Xw: np.ndarray, w: int) -> np.ndarray:
    ww = _to_words(w, Xw.shape[1])
    return (_popcount_rows(Xw & ww[None, :]) & 1).astype(bool)


def _flip_moves(dom: _Domain, pairs: np.ndarray, max_dim: int = 8) -> list[np.ndarray]:
    """Maximal moves that flip one fixed set of spins wherever an affine condition holds."""
    if len(pairs) == 0:
        return []
    masks = dom.Yw[pairs[:, 0]] ^ dom.Yw[pairs[:, 1]]
    _, inv = np.unique(masks, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    cuts = np.flatnonzero(np.diff(inv[order])) + 1
    out = []
    for grp in np.split(order, cuts):
        P = pairs[grp]
        inA = np.zeros(dom.m, dtype=bool)
        inA[P[:, 0]] = True
        inA[P[:, 1]] = True
        # the condition a(t) = x(t) w vanishes off the pattern's states and is invariant under it
        eq = np.concatenate([dom.Xw[~inA], dom.Xw[P[:, 0]] ^ dom.Xw[P[:, 1]]])
        K = _null_space(eq, dom.rho)
        if not K:
            continue
        sup = [_parity(dom.Xw, w) for w in K]
        if len(K) <= max_dim:
            cand = {}
            for r in range(1, 1 << len(K)):
                z = np.zeros(dom.m, dtype=bool)
                for j in range(len(K)):
                    if (r >> j) & 1:
                        z ^= sup[j]
                cand[z.tobytes()] = z
            zs = list(cand.values())
            zs = [z for z in zs if not any(o.sum() > z.sum() and np.all(o[z]) for o in zs)]
        else:
            zs = sup
        for z in zs:
            if not z.any():
                continue
            dest = np.arange(dom.m, dtype=np.int64)
            keep = z[P[:, 0]]
            dest[P[keep, 0]] = P[keep, 1]
            dest[P[keep, 1]] = P[keep, 0]
            out.append(dest)
    return out


# ============================================================================================
# permutation moves: constraint search over involutive (signed) spin permutations
# ============================================================================================
class _Budget(Exception):
    pass


class _PermSearch:
    def __init__(self, dom: _Domain, pairs: np.ndarray, pair_index, max_nodes: int, signed: bool = True):
        m = dom.m
        self.S = dom.S
        self.N = dom.N
        self.signed = signed
        self.max_nodes = int(max_nodes)
        both = np.concatenate([pairs, pairs[:, ::-1]]) if len(pairs) else np.zeros((0, 2), dtype=np.int64)
        deg = np.bincount(both[:, 0], minlength=m) if both.size else np.zeros(m, dtype=np.int64)
        K = 1 + int(deg.max()) if m else 1
        order = np.lexsort((both[:, 1], both[:, 0])) if both.size else np.zeros(0, dtype=np.int64)
        both = both[order]
        Cand = np.full((m, K), -1, dtype=np.int64)
        Cand[:, 0] = np.arange(m)
        if both.size:
            start = np.searchsorted(both[:, 0], np.arange(m))
            col = 1 + np.arange(both.shape[0]) - start[both[:, 0]]
            Cand[both[:, 0], col] = both[:, 1]
        self.Cand = Cand
        self.valid0 = Cand >= 0
        self.Cc = np.where(Cand >= 0, Cand, 0)
        PI = np.full((m, K), -1, dtype=np.int64)
        nb = Cand[:, 1:] >= 0
        if nb.any():
            rows = np.nonzero(nb)[0]
            PI[:, 1:][nb] = pair_index(rows, Cand[:, 1:][nb])
        self.PI = PI
        self.rowid = np.arange(m)[:, None]

    def _propagate(self, V: np.ndarray):
        """Arc consistency of the destination lists; None if some state has no destination left."""
        Cand, Cc = self.Cand, self.Cc
        while True:
            cnt = V.sum(axis=1)
            if np.any(cnt == 0):
                return None
            rows = np.flatnonzero(cnt == 1)
            if rows.size == 0:
                return V
            us = Cand[rows, np.argmax(V[rows], axis=1)]
            if np.bincount(us, minlength=V.shape[0]).max() > 1:
                return None                              # two states forced to one destination
            owner = np.full(V.shape[0], -1, dtype=np.int64)
            owner[us] = rows
            oc = owner[Cc]
            W = V & ~((oc >= 0) & (oc != self.rowid))
            mv = us != rows
            if mv.any():                                  # t -> u forces u -> t
                W[us[mv]] &= Cand[us[mv]] == rows[mv][:, None]
            if np.array_equal(W, V):
                return V
            V = W

    def _assign(self, V, i, j, e):
        S, Cc = self.S, self.Cc
        W = V & ((S[:, j][Cc] * e) == S[:, i][:, None])
        if j != i:
            W &= (S[:, i][Cc] * e) == S[:, j][:, None]
        return self._propagate(W)

    def _targets(self, V, g, i):
        S = self.S
        cnt = V.sum(axis=1)
        rows = np.flatnonzero(cnt == 1)
        us = self.Cand[rows, np.argmax(V[rows], axis=1)]
        mv = us != rows                                    # moved states carry the information
        rows, us = (rows[mv], us[mv]) if mv.any() else (rows[:1], us[:1])
        if rows.size > 256:
            rows, us = rows[:256], us[:256]                # a necessary condition, checked in full by _assign
        Su, St = S[us].astype(np.int64), S[rows].astype(np.int64)
        free = g == -2
        out = []
        for e in ((1, -1) if self.signed else (1,)):
            ok = np.all(Su * e == St[:, i][:, None], axis=0) & np.all(Su[:, i][:, None] * e == St, axis=0)
            ok &= free
            js = np.flatnonzero(ok)
            if e == 1 and i in js:
                out.insert(0, (i, 1))                      # leave the spin in place first
                js = js[js != i]
            out.extend((int(j), e) for j in js)
        return out

    def search(self, a: int, b: int, *, forbid_below: int = 0, collect_all: bool = False):
        V = self.valid0.copy()
        V[a] = self.Cand[a] == b
        V[b] = self.Cand[b] == a
        if forbid_below > 0:
            V &= ~((self.PI >= 0) & (self.PI < forbid_below))
        V = self._propagate(V)
        if V is None:
            return [], False
        changed = np.flatnonzero(self.S[a] != self.S[b])
        rest = np.flatnonzero(self.S[a] == self.S[b])
        self._order = np.concatenate([changed, rest])
        self._nodes = 0
        found: list[np.ndarray] = []
        g = np.full(self.N, -2, dtype=np.int64)
        try:
            self._dfs(V, g, 0, collect_all, found)
            truncated = False
        except _Budget:
            truncated = True
        return found, truncated

    def enumerate_all(self):
        """Every permutation move, each found once: seeded by its first moved spin i and that spin's
        image j (sign e), with the spins before i left in place. About N^2 seeds instead of one per
        transition."""
        S, Cc, N, m = self.S, self.Cc, self.N, self.S.shape[0]
        found: list[np.ndarray] = []
        truncated = False
        prefix = self.valid0.copy()
        ident = np.arange(m)
        for i in range(N):
            for j in range(i, N):
                for e in ((1, -1) if self.signed else (1,)):
                    if j == i and e == 1:
                        continue
                    W = self._assign(prefix, i, j, e)
                    if W is None:
                        continue
                    g = np.full(N, -2, dtype=np.int64)
                    g[:i] = np.arange(i)
                    g[i], g[j] = j, i
                    self._order = np.arange(N)
                    self._nodes = 0
                    out: list[np.ndarray] = []
                    try:
                        self._dfs(W, g, i + 1, True, out)
                    except _Budget:
                        truncated = True
                    found.extend(d for d in out if not np.array_equal(d, ident))
            prefix = prefix & (S[:, i][Cc] == S[:, i][:, None])     # spin i stays from now on
            prefix = self._propagate(prefix)
            if prefix is None:
                break
        return found, truncated

    def _dfs(self, V, g, pos, collect_all, found) -> bool:
        if np.all(V.sum(axis=1) == 1):                    # every destination is fixed
            found.append(self.Cand[np.arange(V.shape[0]), np.argmax(V, axis=1)])
            return True
        order = self._order
        while pos < order.size and g[order[pos]] != -2:
            pos += 1
        if pos == order.size:
            return False
        self._nodes += 1
        if self._nodes > self.max_nodes:
            raise _Budget
        i = int(order[pos])
        ok_any = False
        for j, e in self._targets(V, g, i):
            W = self._assign(V, i, j, e)
            if W is None:
                continue
            g2 = g.copy()
            g2[i] = j
            g2[j] = i
            if self._dfs(W, g2, pos + 1, collect_all, found):
                ok_any = True
                if not collect_all:
                    return True
        return ok_any


# ============================================================================================
# general affine moves (fallback): search over single-state destinations
# ============================================================================================
class _Infeasible(Exception):
    pass


class _State:
    """Linear equations x W = d in echelon form, with every state's reduced coordinates."""

    def __init__(self, dom: _Domain):
        self.dom = dom
        self.rows: dict[int, tuple[int, int]] = {}
        self.Xres = dom.Xw.copy()
        self.Dacc = np.zeros_like(dom.Yw)
        self.dest = np.full(dom.m, -1, dtype=np.int64)
        self.pend_t = np.zeros(0, dtype=np.int64)
        self.pend_u = np.zeros(0, dtype=np.int64)

    def copy(self) -> "_State":
        c = _State.__new__(_State)
        c.dom = self.dom
        c.rows = dict(self.rows)
        c.Xres, c.Dacc, c.dest = self.Xres.copy(), self.Dacc.copy(), self.dest.copy()
        c.pend_t, c.pend_u = self.pend_t.copy(), self.pend_u.copy()
        return c

    def insert(self, x: int, d: int) -> None:
        for p in sorted(self.rows, reverse=True):
            if (x >> p) & 1:
                bx, bd = self.rows[p]
                x ^= bx
                d ^= bd
        if x == 0:
            if d:
                raise _Infeasible
            return
        p = x.bit_length() - 1
        self.rows[p] = (x, d)
        hit = _bit(self.Xres, p)
        if hit.any():
            self.Xres[hit] ^= _to_words(x, self.dom.nx)
            self.Dacc[hit] ^= _to_words(d, self.dom.nw)

    def require(self, t: int, u: int) -> None:
        """Add the transition t -> u (u == t: self-loop)."""
        dom = self.dom
        self.insert(_to_int(dom.Xw[t]), _to_int(dom.Yw[t] ^ dom.Yw[u]))


class _AffineSearch:
    def __init__(self, dom: _Domain, pairs: np.ndarray, pair_index, max_nodes: int):
        self.dom = dom
        self.pair_index = pair_index
        self.max_nodes = int(max_nodes)
        m = dom.m
        both = np.concatenate([pairs, pairs[:, ::-1]]) if len(pairs) else np.zeros((0, 2), dtype=np.int64)
        order = np.lexsort((both[:, 1], both[:, 0])) if both.size else np.zeros(0, dtype=np.int64)
        both = both[order]
        self.indptr = np.searchsorted(both[:, 0], np.arange(m + 1)) if both.size else np.zeros(m + 1, dtype=np.int64)
        self.nbr = both[:, 1] if both.size else np.zeros(0, dtype=np.int64)

    def propagate(self, st: _State) -> None:
        dom = self.dom
        while True:
            det = ~np.any(st.Xres, axis=1)
            if st.pend_u.size:
                du = det[st.pend_u]
                if du.any():
                    if np.any(st.Dacc[st.pend_u[du]] != st.Dacc[st.pend_t[du]]):
                        raise _Infeasible
                    st.pend_t, st.pend_u = st.pend_t[~du], st.pend_u[~du]
            new = np.flatnonzero(det & (st.dest < 0))
            if new.size:
                u = dom.lookup(dom.Yw[new] ^ st.Dacc[new])
                if np.any(u < 0):
                    raise _Infeasible                      # leaves the feasible set
                moved = u != new
                if np.any(self.pair_index(new[moved], u[moved]) < 0):
                    raise _Infeasible                      # a transition that is not a candidate
                st.dest[new] = u
                du = det[u]
                if np.any(st.Dacc[u[du]] != st.Dacc[new[du]]):
                    raise _Infeasible                      # not its own inverse
                if (~du).any():
                    st.pend_t = np.concatenate([st.pend_t, new[~du]])
                    st.pend_u = np.concatenate([st.pend_u, u[~du]])
            if st.pend_u.size:
                t, u = int(st.pend_t[0]), int(st.pend_u[0])
                st.insert(_to_int(dom.Xw[u]), _to_int(st.Dacc[t]))
                continue
            if not new.size:
                return

    def search(self, a: int, b: int):
        st = _State(self.dom)
        self._seed_mask = self.dom.Yw[a] ^ self.dom.Yw[b]
        try:
            st.require(a, b)
            st.require(b, a)
            self.propagate(st)
        except _Infeasible:
            return [], False
        self._nodes = 0
        found: list[np.ndarray] = []
        try:
            self._dfs(st, found)
            return found, False
        except _Budget:
            return found, True

    def _dfs(self, st: _State, found: list) -> bool:
        undet = np.flatnonzero(st.dest < 0)
        if undet.size == 0:
            found.append(st.dest.copy())
            return True
        self._nodes += 1
        if self._nodes > self.max_nodes:
            raise _Budget
        c = int(undet[0])
        nb = self.nbr[self.indptr[c]:self.indptr[c + 1]]
        nb = nb[st.dest[nb] < 0]
        if nb.size > 1:                                    # the seed's own flip pattern first
            same = np.all((self.dom.Yw[nb] ^ self.dom.Yw[c]) == self._seed_mask, axis=1)
            nb = np.concatenate([nb[same], nb[~same]])
        for v in nb.tolist():
            child = st.copy()
            try:
                child.require(c, v)
                child.require(v, c)
                self.propagate(child)
            except _Infeasible:
                continue
            if self._dfs(child, found):
                return True
        child = st.copy()                                  # self-loop when no transition fits
        try:
            child.require(c, c)
            self.propagate(child)
        except _Infeasible:
            return False
        return self._dfs(child, found)


# ============================================================================================
# descriptors, labels, filters and the backbone
# ============================================================================================
def _signed_permutation(S: np.ndarray, Z: np.ndarray):
    N = S.shape[1]
    cols = {}
    for j in range(N):
        cols.setdefault(S[:, j].tobytes(), []).append((j, 1))
        cols.setdefault((-S[:, j]).tobytes(), []).append((j, -1))
    cand = []
    for i in range(N):
        c = cols.get(Z[:, i].tobytes())
        if not c:
            return None
        cand.append(c)
    src = [None] * N
    used = set()
    for i in range(N):                                  # spins left in place first
        if (i, 1) in cand[i]:
            src[i] = (i, 1)
            used.add(i)
    for i in range(N):                                  # then mutual exchanges
        if src[i] is not None:
            continue
        for j, e in cand[i]:
            if j in used or j == i or src[j] is not None:
                continue
            back = [ee for jj, ee in cand[j] if jj == i]
            if e in back:
                src[i], src[j] = (j, e), (i, e)
                used.update((i, j))
                break
    for i in range(N):
        if src[i] is None:
            for j, e in cand[i]:
                if j not in used:
                    src[i] = (j, e)
                    used.add(j)
                    break
        if src[i] is None:
            return None
    try:
        return SignedPermutation([(i, j, e) for i, (j, e) in enumerate(src)])
    except (ValueError, UnsupportedTransition):
        return None


def _structured(S: np.ndarray, dest: np.ndarray, fitter: AffineFitter) -> SpinAction | None:
    Z = S[dest]
    flip = S != Z
    moved = flip.any(axis=1)
    makers = []
    if moved.all() and np.all(flip == flip[0]):
        makers.append(lambda: FixedMask(np.flatnonzero(flip[0]).tolist()))
    makers.append(lambda: _signed_permutation(S, Z))
    first = flip[moved][0] if moved.any() else None
    if first is not None and np.all(flip[moved] == first):
        def condmask():
            g, c = fitter.fit(moved.astype(np.uint8))
            return ConditionalMask(np.flatnonzero(first).tolist(), bits_to_indices(g), -1 if c == 0 else 1)
        makers.append(condmask)

    def affine():
        # spin i flips where <g, y> + c = 1, so new s_i = s_i * (-1)^c * prod_{j in g} s_j
        items = []
        for i in np.flatnonzero(flip.any(axis=0)):
            g, c = fitter.fit(flip[:, i].astype(np.uint8))
            items.append((int(i), sorted(set(bits_to_indices(g)) ^ {int(i)}), -1 if c else 1))
        return AffineMove(items)
    makers.append(affine)
    for make in makers:
        try:
            g = make()
        except (ValueError, UnsupportedTransition):
            continue
        if g is not None and g.is_involution and np.array_equal(g.apply(S), Z):
            return g
    return None


def _is_affine(S: np.ndarray, dest: np.ndarray, fitter: AffineFitter) -> bool:
    flip = (S != S[dest]).astype(np.uint8)
    try:
        for i in np.flatnonzero(flip.any(axis=0)):
            fitter.fit(flip[:, i])
    except UnsupportedTransition:
        return False
    return True


def _label(g: SpinAction, n_pairs: int) -> str:
    if isinstance(g, SignedPermutation):
        mp = g.mapping
        parts, seen = [], set()
        for d, (s, e) in sorted(mp.items()):
            if d in seen:
                continue
            seen.update((d, s))
            if s == d:
                parts.append(f"-s{d}")
            elif mp.get(s, (None,))[0] == d and e == 1 and mp[s][1] == 1:
                parts.append(f"s{d}<->s{s}")
            else:
                parts.append(f"s{d}<-{'-' if e < 0 else ''}s{s}")
        return "exchange " + ", ".join(parts)
    if isinstance(g, TableTransition):
        return f"table[{n_pairs} transitions]"
    if isinstance(g, AffineMove):
        parts = []
        for d, (src, e) in sorted(g.mapping.items()):
            prod = "*".join(f"s{j}" for j in src)
            parts.append(f"s{d}<-{'-' if e < 0 else ''}{prod}")
        return "affine " + ", ".join(parts) if len(parts) <= 6 else f"affine[{len(parts)} spins]"
    return g.default_label()


def _compose(dests: list, pairs: np.ndarray, covered: np.ndarray, pair_index, m: int, max_per_transition: int = 64):
    """Cover transitions with compositions of two commuting moves already found.

    Affine moves are closed under composition, and two commuting moves that are their own inverse
    compose to a move that is its own inverse, so only the candidate transitions need checking."""
    if not dests:
        return []
    ar = np.arange(m)
    ts, rs = [], []
    for r, d in enumerate(dests):
        t = np.flatnonzero(d != ar)
        ts.append(t)
        rs.append(np.full(t.size, r, dtype=np.int64))
    t_all, r_all = np.concatenate(ts), np.concatenate(rs)
    order = np.argsort(t_all, kind="stable")
    t_all, r_all = t_all[order], r_all[order]
    start = np.searchsorted(t_all, np.arange(m + 1))
    out, seen = [], set()
    for e in np.flatnonzero(~covered).tolist():
        if covered[e]:
            continue
        a, b = int(pairs[e, 0]), int(pairs[e, 1])
        tried = 0
        for r1 in r_all[start[a]:start[a + 1]].tolist():
            c = int(dests[r1][a])
            for r2 in r_all[start[c]:start[c + 1]].tolist():
                if r2 == r1 or int(dests[r2][c]) != b:
                    continue
                tried += 1
                d1, d2 = dests[r1], dests[r2]
                comp = d2[d1]
                if not np.array_equal(comp, d1[d2]):
                    continue                                     # the two moves do not commute
                t = np.flatnonzero(comp > ar)
                k = pair_index(t, comp[t])
                if np.any(k < 0):
                    continue                                     # produces a transition outside the candidates
                key = comp.tobytes()
                if key not in seen:
                    seen.add(key)
                    out.append(comp)
                    covered[k] = True
                break
            if covered[e] or tried >= max_per_transition:
                break
    return out


def _pair_sets(dests: list, pair_index, m: int) -> list[np.ndarray]:
    out = []
    for d in dests:
        t = np.flatnonzero(d > np.arange(m))
        out.append(np.sort(pair_index(t, d[t])))
    return out


def _maximal(dests: list, sets: list) -> list[int]:
    """Indices of moves whose transitions are not all inside another, larger move."""
    sizes = np.array([x.size for x in sets])
    keep = []
    for r, x in enumerate(sets):
        dominated = False
        if x.size:
            for q in np.flatnonzero(sizes > x.size).tolist():   # only larger moves can contain it
                y = sets[q]
                pos = np.searchsorted(y, x[0])
                if pos < y.size and y[pos] == x[0] and np.all(np.isin(x, y, assume_unique=True)):
                    dominated = True
                    break
        if not dominated:
            keep.append(r)
    return keep


def _components(m: int, a: np.ndarray, b: np.ndarray) -> tuple[int, np.ndarray]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    g = coo_matrix((np.ones(a.size, dtype=np.int8), (a, b)), shape=(m, m))
    return connected_components(g, directed=False)


def _select_backbone(m: int, moves: list) -> tuple[list[int], bool]:
    """Greedy choice (lazy, exact gains) of moves until their transitions connect all states."""
    n_comp, comp = m, np.arange(m)
    edges = []
    for mv in moves:
        t = np.flatnonzero(mv.dest > np.arange(m))
        edges.append((t, mv.dest[t]))

    def gain(r):
        t, u = edges[r]
        ca, cb = comp[t], comp[u]
        k = ca != cb
        if not k.any():
            return 0
        labels, inv = np.unique(np.concatenate([ca[k], cb[k]]), return_inverse=True)
        n, _ = _components(labels.size, inv[:k.sum()], inv[k.sum():])
        return labels.size - n

    heap = [(-int(edges[r][0].size), r) for r in range(len(moves))]   # upper bounds, refined when popped
    heapq.heapify(heap)
    chosen = []
    all_t = [np.zeros(0, dtype=np.int64)]
    all_u = [np.zeros(0, dtype=np.int64)]
    while n_comp > 1 and heap:
        _, r = heapq.heappop(heap)
        g = gain(r)
        if g == 0:
            continue
        if heap and g < -heap[0][0]:
            heapq.heappush(heap, (-g, r))
            continue
        chosen.append(r)
        all_t.append(edges[r][0])
        all_u.append(edges[r][1])
        n_comp, comp = _components(m, np.concatenate(all_t), np.concatenate(all_u))
    return sorted(chosen), n_comp == 1


# ============================================================================================
# public API
# ============================================================================================
@dataclass
class FoundMove:
    """One move found by grouping.

    ``dest[t]`` is the destination of feasible state ``t`` (``t`` itself for a self-loop);
    ``descriptor`` is a ``FixedMask``, ``ConditionalMask``, ``SignedPermutation`` or ``AffineMove``
    when the transformation has that form on the feasible states, and a ``TableTransition`` otherwise;
    ``found_by`` is ``"flip"``, ``"permutation"``, ``"composition"`` or ``"affine"``.
    """

    dest: np.ndarray
    n_transitions: int
    descriptor: SpinAction
    label: str
    found_by: str


@dataclass
class MoveGrouping:
    """Result of :func:`group_transitions`."""

    states: np.ndarray
    transitions: np.ndarray                 # (E, 2) candidate transitions a < b
    distances: np.ndarray                   # spins changed by each transition
    moves: list[FoundMove]
    backbone: list[int]                     # indices into moves; their transitions connect all states
    connected: bool                         # whether the backbone connects the feasible states
    ungrouped: np.ndarray                   # candidate transitions that no found move contains
    identities: list                        # parity identities of the feasible states (for the compiler)
    truncated: bool = False                 # a search hit max_nodes (results are valid, maybe incomplete)
    stats: dict = field(default_factory=dict)

    @property
    def shortcuts(self) -> list[int]:
        """Moves outside the backbone."""
        b = set(self.backbone)
        return [r for r in range(len(self.moves)) if r not in b]

    def library(self, which="backbone", *, name: str | None = None) -> TransitionLibrary:
        """A TransitionLibrary of ``"backbone"``, ``"all"``, ``"shortcuts"`` or a list of move indices."""
        if isinstance(which, str):
            idx = {"backbone": self.backbone, "all": list(range(len(self.moves))),
                   "shortcuts": self.shortcuts}.get(which)
            if idx is None:
                raise ValueError(f"unknown selection {which!r}")
            default_name = f"grouped_{which}"
        else:
            idx = [int(r) for r in which]
            default_name = "grouped"
        groups = [self.moves[r].descriptor for r in idx]
        labels = [self.moves[r].label for r in idx]
        return TransitionLibrary(self.states.shape[1], groups, labels, name=name or default_name,
                                 template="grouped", params={"moves": idx})

    def summary(self) -> str:
        by_d = ", ".join(f"{k} spins: {v}" for k, v in sorted(self.stats.get("by_distance", {}).items()))
        lines = [f"{self.states.shape[0]} feasible states, {len(self.transitions)} transitions ({by_d})",
                 f"{len(self.moves)} moves; backbone {self.backbone} "
                 f"({'connects all states' if self.connected else 'does NOT connect all states'})"]
        for r, mv in enumerate(self.moves):
            tag = "backbone" if r in self.backbone else "shortcut"
            lines.append(f"  move {r} ({tag}): {mv.label}, {mv.n_transitions} transitions")
        if len(self.ungrouped):
            lines.append(f"{len(self.ungrouped)} transitions are in no move")
        if self.truncated:
            lines.append("a search hit max_nodes (the moves are valid, the list may be incomplete)")
        return "\n".join(lines)


def group_transitions(states, *, max_distance: int | None = None, mode: str = "cover",
                      structured: bool = True, max_nodes: int = 2_000, max_moves: int | None = None
                      ) -> MoveGrouping:
    """Find moves by grouping the spin transitions between enumerated feasible states.

    Parameters
    ----------
    states: (m, N) array of all feasible states (spins -1/+1, no duplicates).
    max_distance: transitions are the pairs that differ in the fewest spins (``None``) or in at
        most this many spins.
    mode: ``"cover"`` searches until every transition lies in a move (fast); ``"all"`` lists every
        maximal move that the searches find from every transition.
    structured: return a ``FixedMask``, ``ConditionalMask``, ``SignedPermutation`` or ``AffineMove``
        descriptor when the transformation has that form on the feasible states (else a
        ``TableTransition``, limited to 65,536 feasible states at compilation).
    max_nodes: choice points per searched transition before a search gives up on it.
    max_moves: stop after this many moves.

    Every returned move maps feasible states to feasible states, is its own inverse, produces only
    candidate transitions and is affine on the feasible states, so it compiles with one clause per
    switch for any objective. Pass ``result.identities`` to ``compile_model`` so that the compiler
    can reuse stored clauses as switches.
    """
    if mode not in ("cover", "all"):
        raise ValueError("mode must be 'cover' or 'all'")
    dom = _Domain(states)
    pairs, dist = candidate_transitions(dom, max_distance)
    E = len(pairs)
    m = dom.m
    pair_key = pairs[:, 0] * m + pairs[:, 1] if E else np.zeros(0, dtype=np.int64)

    def pair_index(a, b):
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = np.asarray(lo * m + hi, dtype=np.int64)
        if E == 0:
            return np.full(key.shape, -1, dtype=np.int64)
        pos = np.minimum(np.searchsorted(pair_key, key), E - 1)
        return np.where(pair_key[pos] == key, pos, -1)

    fitter = AffineFitter(dom.S)
    covered = np.zeros(E, dtype=bool)
    dests: list[np.ndarray] = []
    kinds: list[str] = []
    seen: set = set()
    truncated = False

    def add(d, kind):
        key = d.tobytes()
        if key in seen or (max_moves is not None and len(dests) >= max_moves):
            return
        seen.add(key)
        dests.append(d)
        kinds.append(kind)
        t = np.flatnonzero(d > np.arange(m))
        covered[pair_index(t, d[t])] = True

    for d in _flip_moves(dom, pairs):
        add(d, "flip")
    perm = _PermSearch(dom, pairs, pair_index, max_nodes)
    if mode == "all":
        found, trunc = perm.enumerate_all()
        truncated |= trunc
        for d in found:
            if _is_affine(dom.S, d, fitter):
                add(d, "permutation")
    else:
        for e in range(E):
            if covered[e]:
                continue
            found, trunc = perm.search(int(pairs[e, 0]), int(pairs[e, 1]))
            truncated |= trunc
            for d in found:
                if _is_affine(dom.S, d, fitter):
                    add(d, "permutation")
    for d in _compose(dests, pairs, covered, pair_index, m):
        add(d, "composition")
    aff = _AffineSearch(dom, pairs, pair_index, max_nodes)
    for e in range(E):
        if covered[e]:
            continue
        found, trunc = aff.search(int(pairs[e, 0]), int(pairs[e, 1]))
        truncated |= trunc
        for d in found:
            add(d, "affine")

    # keep maximal moves; drop generic moves that only recombine transitions of simple moves
    sets = _pair_sets(dests, pair_index, m)
    keep = _maximal(dests, sets)
    simple = {r: _structured(dom.S, dests[r], fitter) for r in keep}
    held = np.zeros(E, dtype=bool)
    for r in keep:
        if simple[r] is not None:
            held[sets[r]] = True
    keep = [r for r in keep if simple[r] is not None or not np.all(held[sets[r]])]

    moves = []
    proto = None                                   # tables share one domain object
    for r in keep:
        d = dests[r]
        g = simple[r] if structured else None
        if g is None:
            g = TableTransition.from_destinations(dom.S, d, like=proto)
            proto = proto or g
        n_pairs = int(sets[r].size)
        moves.append(FoundMove(dest=d, n_transitions=n_pairs, descriptor=g, label=_label(g, n_pairs),
                               found_by=kinds[r]))
    backbone, connected = _select_backbone(m, moves) if moves else ([], m == 1)
    by_d = {int(k): int(v) for k, v in zip(*np.unique(dist, return_counts=True))} if E else {}
    return MoveGrouping(states=dom.S, transitions=pairs, distances=dist, moves=moves, backbone=backbone,
                        connected=connected, ungrouped=np.flatnonzero(~covered), identities=fitter.identities(),
                        truncated=truncated,
                        stats={"by_distance": by_d, "rho": dom.rho, "n_states": m, "n_spins": dom.N})

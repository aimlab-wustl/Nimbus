"""Builders for the transition templates used by the target constraint families.

Every builder returns a :class:`TransitionLibrary` (or a single descriptor) whose ``template`` and
``params`` record how it was built. Randomized constructions (random spanning trees, random
addition controls) are deterministic functions of a uint32 ``seed`` through the DATA stream of
``nimbus.core.rng`` (never INIT or SOLVER): counter = (block, 0, tag, 0) with a template-specific
``tag`` in counter word c2 (constants ``TAG_*`` below).

Index conventions
-----------------
* One-hot P x P assignment: spin index ``a * P + p`` for item (row) ``a`` at position
  (column) ``p``; ``+1`` = item a occupies position p.
* Templates of the additional constraint families: chain precedence toggles position i iff
  s_{i-1} s_{i+1} = -1 with boundary constants s_0 = +1, s_{m+1} = -1 folded in; OR blocks toggle
  spin i iff literal j is satisfied (both orientations of every tree edge); AMO blocks toggle i iff
  all other members are -1, i.e. condition = others with polarity (-1)^(m-1).
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np

from nimbus.core import gf2
from nimbus.core.rng import DATA, group_index, stream_integers, stream_words
from nimbus.transitions.descriptors import ConditionalMask, FixedMask, SignedPermutation
from nimbus.transitions.library import TransitionLibrary

__all__ = [
    "TAG_RANDOM_TREE", "TAG_RANDOM_ADDITION", "TAG_RANDOM_MASKS",
    "single_flips", "fixed_masks", "nullspace_basis", "random_fixed_masks",
    "tree_edges", "pair_exchange", "pair_exchanges", "simultaneous_exchanges",
    "block_exchange", "block_exchanges", "onehot_index", "onehot_rows", "onehot_columns", "row_exchange",
    "row_exchanges", "row_cycle", "row_cycles", "column_permutation", "segment_reversal", "segment_reversals",
    "random_segment_reversals", "random_addition", "chain_precedence", "or_block", "amo_block",
]

TAG_RANDOM_TREE = 0x54524545  # 'TREE'
TAG_RANDOM_ADDITION = 0x41444421  # 'ADD!'
TAG_RANDOM_MASKS = 0x4D41534B  # 'MASK'


# ============================================================================================
# deterministic randomness (DATA stream)
# ============================================================================================
def _sample_without_replacement(n: int, count: int, seed: int, tag: int) -> list[int]:
    """Partial Fisher-Yates on [0, n) with draws j = i + floor(w * (n - i) / 2^64) from DATA words."""
    if not 0 <= count <= n:
        raise ValueError(f"cannot choose {count} of {n}")
    if count == 0:
        return []
    w = stream_words(seed, DATA, 2 * count, c2=tag).astype(np.uint64)
    perm = list(range(n))
    for i in range(count):
        j = i + group_index(int(w[2 * i]), int(w[2 * i + 1]), n - i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm[:count]


def _random_tree(n: int, seed: int) -> list[tuple[int, int]]:
    """Uniform random labelled tree on n nodes via a DATA-stream Pruefer sequence."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    code = stream_integers(seed, DATA, n - 2, n, c2=TAG_RANDOM_TREE).tolist()
    degree = [1] * n
    for x in code:
        degree[x] += 1
    edges = []
    import heapq

    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)
    for x in code:
        leaf = heapq.heappop(leaves)
        edges.append((min(leaf, x), max(leaf, x)))
        degree[x] -= 1
        if degree[x] == 1:
            heapq.heappush(leaves, x)
    u, v = heapq.heappop(leaves), heapq.heappop(leaves)
    edges.append((min(u, v), max(u, v)))
    return sorted(edges)


def tree_edges(n: int, pattern: str = "path", seed: int | None = None) -> list[tuple[int, int]]:
    """Edges (a < b) over item indices 0..n-1: 'complete', 'path', 'star' or 'random_tree' (seeded)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if pattern == "complete":
        return list(itertools.combinations(range(n), 2))
    if pattern == "path":
        return [(i, i + 1) for i in range(n - 1)]
    if pattern == "star":
        return [(0, i) for i in range(1, n)]
    if pattern == "random_tree":
        if seed is None:
            raise ValueError("pattern 'random_tree' needs a seed")
        return _random_tree(n, int(seed))
    raise ValueError(f"unknown pattern {pattern!r} (complete, path, star, random_tree)")


def _edges(n: int, pattern, seed, edges):
    if edges is not None:
        out = [(int(a), int(b)) for a, b in edges]
        for a, b in out:
            if not (0 <= a < n and 0 <= b < n) or a == b:
                raise ValueError(f"invalid edge {(a, b)} for {n} items")
        return out, "edges"
    return tree_edges(n, pattern, seed), pattern


# ============================================================================================
# fixed masks
# ============================================================================================
def single_flips(n_spins: int, spins: Sequence[int] | None = None, name: str = "single_flips") -> TransitionLibrary:
    """One FixedMask({i}) per spin (N-HOIM single-spin transitions)."""
    idx = list(range(n_spins)) if spins is None else [int(i) for i in spins]
    return TransitionLibrary(n_spins, [FixedMask([i]) for i in idx], [f"flip({i})" for i in idx], name=name,
                             template="single_flips", params={"spins": idx})


def _mask_rows(masks, n_spins):
    import scipy.sparse as sp

    if sp.issparse(masks) or isinstance(masks, np.ndarray):
        rows, ncols = gf2.as_rows(masks)
        n = int(n_spins if n_spins is not None else ncols)
        return [gf2.bits_to_indices(x) for x in rows], n
    lists = [sorted({int(i) for i in m}) for m in masks]
    n = int(n_spins) if n_spins is not None else (max((m[-1] for m in lists if m), default=-1) + 1)
    return lists, n


def fixed_masks(masks, n_spins: int | None = None, name: str = "fixed_masks", template: str = "fixed_masks",
                params: dict | None = None, labels: Sequence[str] | None = None) -> TransitionLibrary:
    """FixedMask groups from a list of index lists, a 0/1 matrix (rows = masks) or a sparse matrix."""
    lists, n = _mask_rows(masks, n_spins)
    if any(not m for m in lists):
        raise ValueError("fixed_masks: empty mask")
    groups = [FixedMask(m) for m in lists]
    p = dict(params or {})
    p.setdefault("n_masks", len(groups))
    p.setdefault("weights", [len(m) for m in lists])
    return TransitionLibrary(n, groups, labels or [f"mask#{j}" for j in range(len(groups))], name=name,
                             template=template, params=p)


def nullspace_basis(check_matrix, n_spins: int | None = None, name: str = "nullspace_basis") -> TransitionLibrary:
    """Systematic GF(2) nullspace basis of a parity-check matrix C as fixed masks (C d = 0).

    Flipping spins in d preserves C y = b (y = (1 - s)/2), and the N - rank(C) basis masks connect
    the affine solution set (Supplementary Information, Surface-code decoding: objective, moves and reference decoders). Sparse checks need not give sparse masks.
    """
    rows, ncols = gf2.as_rows(check_matrix, n_spins)
    basis = gf2.nullspace(rows, ncols)
    if not basis:
        raise ValueError("the parity-check matrix has full column rank: no feasible move")
    lists = [gf2.bits_to_indices(x) for x in basis]
    return fixed_masks(lists, ncols, name=name, template="nullspace_basis",
                       params={"rank": ncols - len(basis), "K": len(basis), "systematic": True})


def random_fixed_masks(n_spins: int, count: int, weight: int, seed: int, name: str = "random_masks") -> TransitionLibrary:
    """``count`` distinct random masks of fixed Hamming weight (DATA stream, seeded), e.g. switch controls."""
    if not 1 <= weight <= n_spins:
        raise ValueError("weight must be in [1, n_spins]")
    out, seen = [], set()
    draw = 0
    while len(out) < count:
        tag = (TAG_RANDOM_MASKS + draw) & 0xFFFFFFFF
        m = tuple(sorted(_sample_without_replacement(n_spins, weight, seed, tag)))
        draw += 1
        if m in seen:
            if draw > 100 * (count + 1):
                raise ValueError("could not draw enough distinct masks")
            continue
        seen.add(m)
        out.append(list(m))
    return fixed_masks(out, n_spins, name=name, template="random_fixed_masks",
                       params={"count": count, "weight": weight, "seed": int(seed)})


# ============================================================================================
# exchanges and permutations
# ============================================================================================
def pair_exchange(i: int, j: int) -> SignedPermutation:
    """Exchange spins i and j (exact-cardinality pair exchange, Supplementary Information, Number of auxiliary clauses added by compilation (exact cardinality: pair exchanges))."""
    i, j = int(i), int(j)
    if i == j:
        raise ValueError("pair exchange needs two distinct spins")
    return SignedPermutation({i: (j, 1), j: (i, 1)})


def pair_exchanges(positions: Sequence[int], pattern: str = "complete", seed: int | None = None,
                   edges=None, n_spins: int | None = None, name: str | None = None) -> TransitionLibrary:
    """Pair-exchange library over ``positions``: complete, path, star, random_tree (seeded) or explicit edges
    (edges index into ``positions``)."""
    pos = [int(i) for i in positions]
    n = int(n_spins) if n_spins is not None else max(pos) + 1
    E, pat = _edges(len(pos), pattern, seed, edges)
    groups = [pair_exchange(pos[a], pos[b]) for a, b in E]
    labels = [f"swap({pos[a]},{pos[b]})" for a, b in E]
    return TransitionLibrary(n, groups, labels, name=name or f"pair_exchange_{pat}",
                             template=f"{pat}_pair_exchange",
                             params={"positions": pos, "pattern": pat, "seed": seed, "edges": E})


def simultaneous_exchanges(pairs: Iterable[tuple[int, int]]) -> SignedPermutation:
    """One group exchanging several disjoint pairs at once (e.g. the toy shortcut (1,4)+(2,5)).

    Compiled from the net before/after product (Methods, An example toy problem), not as an AND of the
    separate exchange conditions.
    """
    mp = {}
    for i, j in pairs:
        i, j = int(i), int(j)
        if i == j or i in mp or j in mp:
            raise ValueError("simultaneous exchanges must be disjoint pairs of distinct spins")
        mp[i] = (j, 1)
        mp[j] = (i, 1)
    return SignedPermutation(mp)


def block_exchange(block_a: Sequence[int], block_b: Sequence[int]) -> SignedPermutation:
    """Exchange two equal-size blocks position-wise (Supplementary Information, Number of auxiliary clauses added by compilation (row exchanges))."""
    a = [int(i) for i in block_a]
    b = [int(i) for i in block_b]
    if len(a) != len(b):
        raise ValueError("blocks must have equal size")
    if set(a) & set(b):
        raise ValueError("blocks must be disjoint")
    mp = {}
    for u, v in zip(a, b):
        mp[u] = (v, 1)
        mp[v] = (u, 1)
    return SignedPermutation(mp)


def block_exchanges(blocks: Sequence[Sequence[int]], pattern: str = "complete", seed: int | None = None,
                    edges=None, n_spins: int | None = None, name: str | None = None,
                    template: str | None = None, extra_params: dict | None = None) -> TransitionLibrary:
    """Exchanges of equal-size blocks along a pattern over block indices; params record ``blocks``."""
    blks = [[int(i) for i in b] for b in blocks]
    n = int(n_spins) if n_spins is not None else max(max(b) for b in blks) + 1
    E, pat = _edges(len(blks), pattern, seed, edges)
    groups = [block_exchange(blks[a], blks[b]) for a, b in E]
    labels = [f"block_swap({a},{b})" for a, b in E]
    params = {"blocks": blks, "pattern": pat, "seed": seed, "edges": E}
    params.update(extra_params or {})
    return TransitionLibrary(n, groups, labels, name=name or f"block_exchange_{pat}",
                             template=template or f"{pat}_block_exchange", params=params)


def onehot_index(a: int, p: int, P: int) -> int:
    """Spin index of city/row a at position/column p in a P x P one-hot matrix."""
    return int(a) * int(P) + int(p)


def onehot_rows(P: int, n_rows: int | None = None) -> list[list[int]]:
    n_rows = P if n_rows is None else int(n_rows)
    return [[onehot_index(a, p, P) for p in range(P)] for a in range(n_rows)]


def onehot_columns(P: int, n_rows: int | None = None) -> list[list[int]]:
    n_rows = P if n_rows is None else int(n_rows)
    return [[onehot_index(a, p, P) for a in range(n_rows)] for p in range(P)]


def row_exchange(a: int, b: int, P: int) -> SignedPermutation:
    """Exchange rows a and b of the one-hot matrix (row exchange)."""
    return block_exchange(onehot_rows(P)[a], onehot_rows(P)[b])


def row_exchanges(P: int, pattern: str = "complete", seed: int | None = None, edges=None,
                  name: str | None = None) -> TransitionLibrary:
    """Row-exchange library on a P x P one-hot assignment: 'complete' (P(P-1)/2 groups) or a connected
    tree ('path', 'star', 'random_tree' seeded; P - 1 groups), or explicit row-pair edges."""
    E, pat = _edges(P, pattern, seed, edges)
    tmpl = "complete_row_exchange" if pat == "complete" else ("custom_row_exchange" if pat == "edges"
                                                               else "tree_row_exchange")
    rows = onehot_rows(P)
    groups = [block_exchange(rows[a], rows[b]) for a, b in E]
    labels = [f"row_swap({a},{b})" for a, b in E]
    return TransitionLibrary(P * P, groups, labels, name=name or f"{tmpl}_P{P}", template=tmpl,
                             params={"P": P, "blocks": rows, "pattern": pat, "seed": seed, "edges": E,
                                     "one_hot": True})


def row_cycle(a: int, b: int, c: int, P: int) -> SignedPermutation:
    """Cyclic exchange of rows a, b and c of the one-hot matrix: row a receives the content of row b,
    row b that of row c and row c that of row a. Not its own inverse: its reverse is ``row_cycle(a, c, b, P)``."""
    if len({a, b, c}) != 3:
        raise ValueError("a row cycle needs three distinct rows")
    rows = onehot_rows(P)
    mp = []
    for dst, src in ((a, b), (b, c), (c, a)):
        mp += [(rows[dst][p], rows[src][p], 1) for p in range(P)]
    return SignedPermutation(mp)


def row_cycles(P: int, cycles=None, name: str | None = None) -> TransitionLibrary:
    """Cyclic exchanges of three rows on a P x P one-hot assignment, each with its reverse.

    ``cycles``: triples (a, b, c), each giving ``row_cycle(a, b, c)`` and its reverse
    ``row_cycle(a, c, b)``; default all ``2 * C(P, 3)`` cycles. The two directions are separate moves
    with the same connections but different switches, inputs and destinations.
    """
    trip = [tuple(int(x) for x in t) for t in (cycles if cycles is not None else itertools.combinations(range(P), 3))]
    groups, labels = [], []
    for a, b, c in trip:
        groups += [row_cycle(a, b, c, P), row_cycle(a, c, b, P)]
        labels += [f"row_cycle({a},{b},{c})", f"row_cycle({a},{c},{b})"]
    return TransitionLibrary(P * P, groups, labels, name=name or f"row_cycles_P{P}", template="row_cycles",
                             params={"P": P, "cycles": trip, "one_hot": True})


def column_permutation(P: int, perm: Sequence[int], n_rows: int | None = None) -> SignedPermutation:
    """Column permutation of the one-hot matrix: new s[a, p] = old s[a, perm[p]] for every row a."""
    perm = [int(x) for x in perm]
    if sorted(perm) != list(range(P)):
        raise ValueError("perm must be a permutation of 0..P-1")
    n_rows = P if n_rows is None else int(n_rows)
    mp = []
    for a in range(n_rows):
        for p in range(P):
            if perm[p] != p:
                mp.append((onehot_index(a, p, P), onehot_index(a, perm[p], P), 1))
    return SignedPermutation(mp)


def segment_reversal(P: int, p: int, q: int, n_rows: int | None = None) -> SignedPermutation:
    """Reverse tour positions p..q (inclusive, 0 <= p < q < P): new position p+t holds the city of q-t.

    Reversing 2 or 3 positions equals one column swap; longer reversals are distinct shortcuts
    relative to the pair-exchange backbone.
    """
    if not 0 <= p < q < P:
        raise ValueError("need 0 <= p < q < P")
    perm = list(range(P))
    for t in range(q - p + 1):
        perm[p + t] = q - t
    return column_permutation(P, perm, n_rows)


def segment_reversals(P: int, segments: Iterable[tuple[int, int]] | None = None, min_len: int = 4,
                      max_len: int | None = None, name: str | None = None) -> TransitionLibrary:
    """Segment-reversal library: explicit (p, q) list, or all segments with min_len <= q-p+1 <= max_len."""
    if segments is None:
        max_len = P if max_len is None else int(max_len)
        segments = [(p, p + L - 1) for L in range(int(min_len), max_len + 1) for p in range(0, P - L + 1)]
    segs = [(int(p), int(q)) for p, q in segments]
    groups = [segment_reversal(P, p, q) for p, q in segs]
    return TransitionLibrary(P * P, groups, [f"rev({p},{q})" for p, q in segs], name=name or f"segment_reversal_P{P}",
                             template="segment_reversal", params={"P": P, "segments": segs})


def random_addition(candidates: TransitionLibrary, count: int, seed: int, exclude: Iterable[int] = (),
                    name: str | None = None) -> TransitionLibrary:
    """Matched random-addition control: ``count`` distinct groups drawn uniformly (seeded, DATA stream)
    from ``candidates`` excluding the given candidate indices."""
    excl = set(int(i) for i in exclude)
    pool = [i for i in range(candidates.n_groups) if i not in excl]
    pick = sorted(pool[j] for j in _sample_without_replacement(len(pool), int(count), int(seed), TAG_RANDOM_ADDITION))
    return candidates.subset(pick, name=name or f"random_addition({candidates.name})", template="random_addition",
                             params={"candidates": candidates.hash, "candidate_template": candidates.template,
                                     "count": int(count), "seed": int(seed), "indices": pick,
                                     "excluded": sorted(excl)})


def random_segment_reversals(P: int, count: int, seed: int, min_len: int = 4, max_len: int | None = None,
                             exclude: Iterable[tuple[int, int]] = ()) -> TransitionLibrary:
    """Random-addition control of segment reversals (lengths in [min_len, max_len])."""
    cand = segment_reversals(P, min_len=min_len, max_len=max_len)
    ex = set((int(p), int(q)) for p, q in exclude)
    idx = [i for i, seg in enumerate(cand.params["segments"]) if tuple(seg) in ex]
    return random_addition(cand, count, seed, exclude=idx, name=f"random_segment_reversals_P{P}")


# ============================================================================================
# block-constraint templates (ConditionalMask groups)
# ============================================================================================
def chain_precedence(chain: Sequence[int], n_spins: int | None = None, name: str | None = None) -> TransitionLibrary:
    """Ordering/precedence on one chain (spins in order; feasible = +1 prefix then -1 suffix).

    Group for position i toggles s_i iff s_{i-1} s_{i+1} = -1, with s_0 = +1 and s_{m+1} = -1 folded:
    interior cond {i-1, i+1} pol -1; first cond {i+1} pol -1; last cond {i-1} pol +1; m = 1: always.
    """
    c = [int(i) for i in chain]
    if len(set(c)) != len(c) or not c:
        raise ValueError("chain must be a non-empty list of distinct spins")
    m = len(c)
    n = int(n_spins) if n_spins is not None else max(c) + 1
    groups, labels = [], []
    for t in range(m):
        # active iff left * right == -1, left = s_{t-1} (or +1), right = s_{t+1} (or -1).
        # left * right = const * prod(cond) with const = -1 iff the right boundary is folded,
        # so active iff (-const) * prod(cond) == +1: pol = -const.
        cond = ([c[t - 1]] if t > 0 else []) + ([c[t + 1]] if t < m - 1 else [])
        pol = -1 if t < m - 1 else 1
        groups.append(ConditionalMask([c[t]], cond, pol))
        labels.append(f"chain[{t}]@{c[t]}")
    return TransitionLibrary(n, groups, labels, name=name or "chain_precedence", template="chain_precedence",
                             params={"chain": c})


def or_block(literals: Sequence[tuple[int, int]], tree: str = "path", seed: int | None = None, edges=None,
             n_spins: int | None = None, name: str | None = None) -> TransitionLibrary:
    """Disjoint hard OR requirement over signed literals [(spin, sign)] (satisfied iff sign * s == +1).

    For each tree edge (a, b) both orientations: toggle spin of a iff literal b satisfied (cond {spin_b},
    pol sign_b), then toggle spin of b iff literal a satisfied. m = 1 gives no groups (variable fixed).
    """
    lits = [(int(i), int(sg)) for i, sg in literals]
    if any(sg not in (1, -1) for _, sg in lits):
        raise ValueError("literal signs must be +1/-1")
    spins = [i for i, _ in lits]
    if len(set(spins)) != len(spins) or not spins:
        raise ValueError("an OR block needs distinct spins")
    n = int(n_spins) if n_spins is not None else max(spins) + 1
    m = len(lits)
    E, pat = _edges(m, tree, seed, edges) if m > 1 else ([], tree)
    groups, labels = [], []
    for a, b in E:
        for i, j in ((a, b), (b, a)):
            groups.append(ConditionalMask([lits[i][0]], [lits[j][0]], lits[j][1]))
            labels.append(f"or:toggle({lits[i][0]})|lit({lits[j][0]},{lits[j][1]:+d})")
    return TransitionLibrary(n, groups, labels, name=name or "or_block", template="or_block",
                             params={"literals": lits, "tree": pat, "edges": E, "seed": seed})


def amo_block(spins: Sequence[int], n_spins: int | None = None, name: str | None = None) -> TransitionLibrary:
    """Unit-capacity packing (at most one +1 in the block): toggle i iff all others are -1, i.e.
    cond = others, pol = (-1)^(m-1) (alpha = (1 + (-1)^(m-1) prod_{j != i} s_j)/2)."""
    sp_ = [int(i) for i in spins]
    if len(set(sp_)) != len(sp_) or not sp_:
        raise ValueError("an AMO block needs distinct spins")
    m = len(sp_)
    n = int(n_spins) if n_spins is not None else max(sp_) + 1
    pol = -1 if (m - 1) % 2 else 1
    groups = [ConditionalMask([i], [j for j in sp_ if j != i], pol) for i in sp_]
    labels = [f"amo:toggle({i})" for i in sp_]
    return TransitionLibrary(n, groups, labels, name=name or "amo_block", template="amo_block",
                             params={"spins": sp_})


def contiguity_block(spins: Sequence[int], cyclic: bool = False, n_spins: int | None = None,
                     name: str | None = None) -> TransitionLibrary:
    """Contiguity (consecutive ones): the selected (+1) spins of the block form one nonempty contiguous
    interval of the path order given by `spins` (cyclic=False), or one nonempty proper arc of the cycle
    (cyclic=True, m >= 3; the all-selected state is excluded).

    Group for position t toggles s_t iff exactly one neighbour is selected, i.e. iff s_{t-1} s_{t+1} = -1,
    with fixed boundary values s_0 = s_{m+1} = -1 on a path (folded into the condition polarity). The
    condition excludes s_t, so every group is an involution; alpha = (1 - s_{t-1} s_{t+1})/2 for clauses
    containing t. Verified exhaustively by scripts/verify/check_contiguity_family.py.
    A one-spin path block has no groups (the variable is fixed to +1).
    """
    c = [int(i) for i in spins]
    if len(set(c)) != len(c) or not c:
        raise ValueError("a contiguity block needs a non-empty list of distinct spins")
    m = len(c)
    if cyclic and m < 3:
        raise ValueError("a cyclic contiguity block needs m >= 3")
    n = int(n_spins) if n_spins is not None else max(c) + 1
    groups, labels = [], []
    for t in range(m):
        if cyclic:
            cond, const = [c[(t - 1) % m], c[(t + 1) % m]], 1
        else:
            cond, const = [], 1
            if t > 0:
                cond.append(c[t - 1])
            else:
                const = -const  # left boundary s_0 = -1
            if t < m - 1:
                cond.append(c[t + 1])
            else:
                const = -const  # right boundary s_{m+1} = -1
        if not cond and -const != 1:
            continue  # never active (m = 1 path): no transition neuron needed
        groups.append(ConditionalMask([c[t]], cond, -const))
        labels.append(f"contig[{t}]@{c[t]}")
    return TransitionLibrary(n, groups, labels, name=name or ("contiguity_cycle" if cyclic else "contiguity_path"),
                             template="contiguity_block", params={"spins": c, "cyclic": bool(cyclic)})

"""Serialized third-order matching tensor and the triangle-affinity generators.

Tensor protocol. A matching instance of size ``n`` is a sparse symmetric third-order affinity tensor
over candidate pairs ``(i, a)`` (source point ``i`` matched to target point ``a``)::

    F(X) = sum_m v_m X[i1_m, a1_m] X[i2_m, a2_m] X[i3_m, a3_m],      X a permutation matrix.

File format (``.npz``): ``n`` (int), ``triples`` int32[m, 3, 2] (rows ``(i, a)`` of the three pairs),
``values`` float64[m]. Nothing else is read by any solver, so every solver sees the identical tensor.
Ground truth, when known (synthetic, WILLOW), lives in a sidecar ``<name>.truth.npz`` that only the
accuracy evaluation reads; it is never used to build a tensor.

Only entries with distinct source indices and distinct target indices are meaningful (a product
containing two pairs of the same source or target vanishes on every permutation); the builders never
produce others and :func:`canonical_tensor` drops them.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["MatchingTensor", "save_tensor", "load_tensor", "canonical_tensor", "evaluate_F", "triangle_features",
           "triangle_affinity", "similarity_deformation_pair", "point_pair_tensor", "perm_to_matrix"]


@dataclass(frozen=True)
class MatchingTensor:
    n: int
    triples: np.ndarray  # int32[m, 3, 2]
    values: np.ndarray  # float64[m]

    @property
    def nnz(self) -> int:
        return int(self.values.shape[0])


def canonical_tensor(n: int, triples, values) -> MatchingTensor:
    """Sort each entry's three pairs, merge equal entries (sum of values), drop degenerate entries."""
    t = np.asarray(triples, dtype=np.int64).reshape(-1, 3, 2)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if t.shape[0] != v.shape[0]:
        raise ValueError("triples and values differ in length")
    if t.size and (t.min() < 0 or t.max() >= n):
        raise ValueError("tensor index out of range")
    code = t[:, :, 0] * n + t[:, :, 1]  # pair code i*n+a
    code.sort(axis=1)
    i = code // n
    a = code % n
    ok = ((i[:, 0] != i[:, 1]) & (i[:, 1] != i[:, 2]) & (i[:, 0] != i[:, 2]) &
          (a[:, 0] != a[:, 1]) & (a[:, 1] != a[:, 2]) & (a[:, 0] != a[:, 2]))
    code, v = code[ok], v[ok]
    key = (code[:, 0] * (n * n) + code[:, 1]) * (n * n) + code[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    vv = np.bincount(inv, weights=v, minlength=uk.size)
    c2 = uk % (n * n)
    c1 = (uk // (n * n)) % (n * n)
    c0 = uk // (n * n * n * n)
    cc = np.stack([c0, c1, c2], axis=1)
    out = np.stack([cc // n, cc % n], axis=2).astype(np.int32)
    keep = vv != 0.0
    return MatchingTensor(int(n), out[keep], vv[keep])


def save_tensor(path, tensor: MatchingTensor, **meta) -> None:
    """Write the tensor npz (plus optional scalar/array metadata keys prefixed ``meta_``)."""
    extra = {f"meta_{k}": np.asarray(v) for k, v in meta.items()}
    np.savez_compressed(path, n=np.int64(tensor.n), triples=tensor.triples.astype(np.int32),
                        values=tensor.values.astype(np.float64), **extra)


def load_tensor(path) -> MatchingTensor:
    with np.load(Path(path)) as z:
        return MatchingTensor(int(z["n"]), np.ascontiguousarray(z["triples"], dtype=np.int32),
                              np.ascontiguousarray(z["values"], dtype=np.float64))


def perm_to_matrix(perm) -> np.ndarray:
    """perm[i] = a  ->  X[i, a] = 1."""
    perm = np.asarray(perm)
    X = np.zeros((perm.size, perm.size), dtype=np.int8)
    X[np.arange(perm.size), perm] = 1
    return X


def evaluate_F(tensor: MatchingTensor, perms) -> np.ndarray:
    """F for permutations ``perms[b, i] = a`` (or one 1-D permutation), directly from the tensor."""
    P = np.atleast_2d(np.asarray(perms))
    t = tensor.triples
    hit = np.ones((P.shape[0], tensor.nnz), dtype=bool)
    for q in range(3):
        hit &= P[:, t[:, q, 0]] == t[:, q, 1][None, :]
    out = hit.astype(np.float64) @ tensor.values
    return out if np.ndim(perms) == 2 else float(out[0])


# ---------------------------------------------------------------------------------------------
# triangle affinity (Duchenne et al. / Nguyen et al. style; angle-sine features)
# ---------------------------------------------------------------------------------------------
def triangle_features(P: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Sines of the three interior angles (at vertices 0, 1, 2) of each ordered triangle."""
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]

    def ang(u, v, w):
        x, y = v - u, w - u
        cs = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
        return np.arccos(np.clip(cs, -1, 1))

    return np.stack([np.sin(ang(a, b, c)), np.sin(ang(b, c, a)), np.sin(ang(c, a, b))], 1)


def triangle_affinity(P1: np.ndarray, P2: np.ndarray, T1: int, K: int, rng: np.random.Generator) -> MatchingTensor:
    """Sparse triangle affinity between point sets ``P1`` (source) and ``P2`` (target), both n x 2.

    ``T1`` source triangles are sampled (all C(n,3) if fewer); each is linked to its ``K`` nearest
    *ordered* target triangles in angle-sine space; the entry value is ``exp(-d^2 / gamma)`` with
    ``gamma = mean(d)^2``. Uses coordinates only, never any correspondence.
    """
    from scipy.spatial import cKDTree

    n = P1.shape[0]
    all1 = np.array(list(itertools.combinations(range(n), 3)))
    t1 = all1[rng.choice(len(all1), size=min(int(T1), len(all1)), replace=False)]
    t2 = np.array(list(itertools.permutations(range(n), 3)))
    f1, f2 = triangle_features(P1, t1), triangle_features(P2, t2)
    d, idx = cKDTree(f2).query(f1, k=min(int(K), len(t2)))
    d = d.reshape(len(t1), -1)
    idx = idx.reshape(len(t1), -1)
    gamma = float(np.mean(d)) ** 2 + 1e-12
    val = np.exp(-d * d / gamma)
    rows = np.repeat(np.arange(len(t1)), idx.shape[1])
    src = t1[rows]  # (m, 3) source indices
    tgt = t2[idx.reshape(-1)]  # (m, 3) target indices
    triples = np.stack([src, tgt], axis=2)
    return canonical_tensor(n, triples, val.reshape(-1))


def similarity_deformation_pair(n: int, sigma: float, rng: np.random.Generator, n_outliers: int = 0):
    """Source points, transformed+noisy+shuffled target points, and the ground-truth permutation.

    P2 = scale * R(theta) P1 + N(0, sigma^2), then ``n_outliers`` target points are replaced by
    uniform random points (their true source is "none": excluded from accuracy), then the target ids
    are shuffled. ``gt[i] = a`` gives the target index of source ``i``. Returns ``(P1, P2, gt, valid)``
    with ``valid[i]`` False for sources whose target was replaced by an outlier.
    """
    P1 = rng.random((n, 2))
    th = rng.uniform(0, np.pi / 4)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P2 = (P1 @ R.T) * rng.uniform(0.8, 1.2) + rng.normal(0, sigma, (n, 2))
    valid = np.ones(n, dtype=bool)
    if n_outliers:
        out = rng.choice(n, size=int(n_outliers), replace=False)
        valid[out] = False
        lo, hi = P2.min(0), P2.max(0)
        P2[out] = lo + rng.random((len(out), 2)) * (hi - lo)
    shuffle = rng.permutation(n)  # P2s[k] = P2[shuffle[k]]; source i's target id is argsort position
    P2s = P2[shuffle]
    gt = np.empty(n, dtype=np.int64)
    gt[shuffle] = np.arange(n)
    return P1, P2s, gt, valid


def point_pair_tensor(P1, P2, *, T1: int | None = None, K: int | None = None, seed: int = 0) -> MatchingTensor:
    """Protocol defaults: ``T1 = n^2`` sampled source triangles, ``K = 3n`` nearest target triangles."""
    n = np.asarray(P1).shape[0]
    T1 = n * n if T1 is None else int(T1)
    K = 3 * n if K is None else int(K)
    return triangle_affinity(np.asarray(P1, dtype=float), np.asarray(P2, dtype=float), T1, K,
                             np.random.default_rng(seed))

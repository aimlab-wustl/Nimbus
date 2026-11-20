"""GF(2) linear algebra on Python-int row bitsets.

Convention: a matrix row is a non-negative Python ``int``; bit ``j`` holds column ``j``.
Python ints give exact arbitrary-width XOR and popcount (``int.bit_count``), so a 1000 x 2000
elimination runs in well under a second without packing tricks.

The central object is :class:`GF2Basis`, an incrementally maintained *fully reduced* row-echelon
basis: every basis row has a distinct pivot (its lowest set column bit) and no basis row contains
another row's pivot. Reducing a vector is then a single pass over its pivot bits.

Rows may carry *payload* bits at positions ``>= n_cols`` (never chosen as pivots). They ride along
with the elimination and are used for right-hand sides (``solve``), parity-identity signs
(``nimbus.core.algebra.IdentityReducer``) or combination tracking.

Helpers convert from numpy ``uint8`` matrices and scipy.sparse matrices.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "GF2Basis",
    "bits_to_indices",
    "indices_to_bits",
    "iter_bits",
    "lowbit",
    "rows_from_dense",
    "rows_to_dense",
    "rows_from_sparse",
    "as_rows",
    "rank",
    "rref",
    "nullspace",
    "solve",
    "in_row_space",
    "matvec",
]


# --------------------------------------------------------------------------------------------
# bit helpers
# --------------------------------------------------------------------------------------------
def lowbit(x: int) -> int:
    """Index of the lowest set bit of x (x > 0)."""
    return (x & -x).bit_length() - 1


def iter_bits(x: int):
    """Yield set-bit indices of x in increasing order."""
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


def bits_to_indices(x: int) -> list[int]:
    """Sorted list of set-bit indices of x."""
    if x < 0:
        raise ValueError("bitsets must be non-negative")
    if x.bit_count() <= 48:
        return list(iter_bits(x))
    nbytes = (x.bit_length() + 7) // 8
    arr = np.frombuffer(x.to_bytes(nbytes, "little"), dtype=np.uint8)
    return np.flatnonzero(np.unpackbits(arr, bitorder="little")).tolist()


def indices_to_bits(indices: Iterable[int]) -> int:
    """Bitset with the given column indices set (repeated indices cancel in pairs, i.e. XOR)."""
    idx = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices, dtype=np.int64)
    if idx.size == 0:
        return 0
    if idx.min() < 0:
        raise ValueError("negative column index")
    if idx.size <= 48:
        x = 0
        for i in idx.tolist():
            x ^= 1 << i
        return x
    n = int(idx.max()) + 1
    v = np.zeros(n, dtype=np.uint8)
    np.add.at(v, idx, 1)
    v &= 1
    return int.from_bytes(np.packbits(v, bitorder="little").tobytes(), "little")


# --------------------------------------------------------------------------------------------
# conversions
# --------------------------------------------------------------------------------------------
def rows_from_dense(a) -> list[int]:
    """Rows of a dense 0/1 matrix (any integer/bool dtype; values taken mod 2) as bitsets."""
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    if a.shape[1] == 0:
        return [0] * a.shape[0]
    bits = (a.astype(np.int64) & 1).astype(np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


def rows_to_dense(rows: Sequence[int], n_cols: int) -> np.ndarray:
    """uint8 matrix (len(rows) x n_cols) from bitset rows (bits >= n_cols must be zero)."""
    n_cols = int(n_cols)
    nbytes = (n_cols + 7) // 8
    out = np.zeros((len(rows), n_cols), dtype=np.uint8)
    if n_cols == 0:
        return out
    for i, x in enumerate(rows):
        if x >> n_cols:
            raise ValueError(f"row {i} has bits beyond n_cols={n_cols}")
        arr = np.frombuffer(int(x).to_bytes(nbytes, "little"), dtype=np.uint8)
        out[i] = np.unpackbits(arr, bitorder="little")[:n_cols]
    return out


def rows_from_sparse(a) -> list[int]:
    """Rows of a scipy.sparse matrix (any format; nonzero entries taken mod 2 after summing)."""
    import scipy.sparse as sp

    m = sp.csr_matrix(a)
    m.sum_duplicates()
    out = []
    indptr, indices, data = m.indptr, m.indices, m.data
    for r in range(m.shape[0]):
        sl = slice(indptr[r], indptr[r + 1])
        cols = indices[sl][(np.asarray(data[sl]).astype(np.int64) & 1) == 1]
        out.append(indices_to_bits(cols))
    return out


def as_rows(a, n_cols: int | None = None) -> tuple[list[int], int]:
    """Normalize a matrix to (bitset rows, n_cols).

    Accepts a list of ints (bitsets), a dense array, or a scipy.sparse matrix.
    """
    try:
        import scipy.sparse as sp

        if sp.issparse(a):
            return rows_from_sparse(a), int(a.shape[1]) if n_cols is None else int(n_cols)
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        pass
    if isinstance(a, np.ndarray):
        if a.ndim != 2:
            raise ValueError("expected a 2-D matrix")
        return rows_from_dense(a), int(a.shape[1]) if n_cols is None else int(n_cols)
    rows = [int(x) for x in a]
    if any(x < 0 for x in rows):
        raise ValueError("bitset rows must be non-negative")
    if n_cols is None:
        n_cols = max((x.bit_length() for x in rows), default=0)
    return rows, int(n_cols)


# --------------------------------------------------------------------------------------------
# incremental basis
# --------------------------------------------------------------------------------------------
class GF2Basis:
    """Fully reduced row-echelon basis over GF(2) with optional payload bits.

    ``n_cols`` columns take part in elimination; bits at ``>= n_cols`` are payload. The pivot of a
    basis row is its lowest set column bit. Invariant: no basis row contains another pivot bit.
    """

    __slots__ = ("n_cols", "colmask", "pivmask", "_row", "pivots")

    def __init__(self, n_cols: int):
        self.n_cols = int(n_cols)
        self.colmask = (1 << self.n_cols) - 1
        self.pivmask = 0
        self._row: dict[int, int] = {}  # pivot column -> row (incl. payload)
        self.pivots: list[int] = []  # insertion order

    @property
    def rank(self) -> int:
        return len(self.pivots)

    def reduce(self, x: int) -> int:
        """x minus its projection on the basis (payload bits included). Result has no pivot bits."""
        t = x & self.pivmask
        row = self._row
        while t:
            low = t & -t
            x ^= row[low.bit_length() - 1]
            t ^= low
        return x

    def insert(self, x: int) -> int:
        """Insert x; returns the reduced residual (0 or payload-only if x was dependent)."""
        x = self.reduce(x)
        col = x & self.colmask
        if not col:
            return x
        q = (col & -col).bit_length() - 1
        qbit = 1 << q
        row = self._row
        for p in self.pivots:
            if row[p] & qbit:
                row[p] ^= x
        row[q] = x
        self.pivots.append(q)
        self.pivmask |= qbit
        return x

    def contains(self, x: int) -> bool:
        """True iff the column part of x lies in the row space."""
        return not (self.reduce(x & self.colmask) & self.colmask)

    def rows(self) -> list[int]:
        """Basis rows sorted by pivot (payload bits included)."""
        return [self._row[p] for p in sorted(self.pivots)]

    def row_of_pivot(self, p: int) -> int:
        return self._row[p]

    def sorted_pivots(self) -> list[int]:
        return sorted(self.pivots)


# --------------------------------------------------------------------------------------------
# matrix-level operations
# --------------------------------------------------------------------------------------------
def rref(a, n_cols: int | None = None) -> tuple[list[int], list[int]]:
    """Reduced row-echelon form. Returns (rows sorted by pivot, pivot columns ascending).

    Zero / dependent rows are dropped, so ``len(rows) == rank``.
    """
    rows, n_cols = as_rows(a, n_cols)
    basis = GF2Basis(n_cols)
    for x in rows:
        if x >> n_cols:
            raise ValueError("row has bits beyond n_cols")
        basis.insert(x)
    piv = basis.sorted_pivots()
    return [basis.row_of_pivot(p) for p in piv], piv


def rank(a, n_cols: int | None = None) -> int:
    rows, n_cols = as_rows(a, n_cols)
    basis = GF2Basis(n_cols)
    for x in rows:
        basis.insert(x)
    return basis.rank


def nullspace(a, n_cols: int | None = None) -> list[int]:
    """Systematic basis of {x : A x = 0}: one vector per free (non-pivot) column f, with x_f = 1,
    the other free coordinates 0 and the pivot coordinates determined. Returned in increasing f."""
    rows, n_cols = as_rows(a, n_cols)
    rr, piv = rref(rows, n_cols)
    pivset = set(piv)
    free = [j for j in range(n_cols) if j not in pivset]
    if not free:
        return []
    if not piv:
        return [1 << f for f in free]
    dense = rows_to_dense(rr, n_cols)  # rank x n_cols
    ns = np.zeros((len(free), n_cols), dtype=np.uint8)
    ns[np.arange(len(free)), free] = 1
    ns[:, piv] = dense[:, free].T
    return rows_from_dense(ns)


def solve(a, b, n_cols: int | None = None):
    """One solution x (bitset over columns, free variables 0) of A x = b over GF(2), or None if the
    system is inconsistent. ``b`` is a sequence of bits (one per row) or an int bitset over rows."""
    rows, n_cols = as_rows(a, n_cols)
    if isinstance(b, (int, np.integer)):
        bvec = [(int(b) >> i) & 1 for i in range(len(rows))]
    else:
        bvec = [int(v) & 1 for v in b]
    if len(bvec) != len(rows):
        raise ValueError("rhs length does not match the number of rows")
    rhs_bit = 1 << n_cols
    basis = GF2Basis(n_cols)
    for x, bi in zip(rows, bvec):
        if x >> n_cols:
            raise ValueError("row has bits beyond n_cols")
        res = basis.insert(x | (rhs_bit if bi else 0))
        if res == rhs_bit:  # 0 = 1
            return None
    sol = 0
    for p in basis.pivots:
        if basis.row_of_pivot(p) & rhs_bit:
            sol |= 1 << p
    return sol


def in_row_space(a, v: int, n_cols: int | None = None) -> bool:
    """Is bitset v in the GF(2) row space of A?"""
    rows, n_cols = as_rows(a, n_cols)
    basis = GF2Basis(max(n_cols, int(v).bit_length()))
    for x in rows:
        basis.insert(x)
    return basis.contains(int(v))


def matvec(a, x: int, n_cols: int | None = None) -> list[int]:
    """A x over GF(2) as a list of bits."""
    rows, _ = as_rows(a, n_cols)
    return [(r & x).bit_count() & 1 for r in rows]

"""MacKay ``alist`` parity-check matrices, including the AFF3CT variant.

Layout (D. J. C. MacKay, http://www.inference.org.uk/mackay/codes/alist.html)::

    N M                 number of variable nodes (columns of H) and of checks (rows of H)
    dmax_v dmax_c       largest column degree and largest row degree
    d_v[1..N]           column (variable-node) degrees
    d_c[1..M]           row (check-node) degrees
    N lines             for each column, the 1-based indices of its checks ("VN to CN")
    M lines             for each row, the 1-based indices of its variables ("CN to VN")

Conventions handled here:

- Indices are one-based in the file and zero-based in :class:`ParityCheckMatrix`.
- Zero padding: MacKay's files pad every list with ``0`` up to ``dmax``; AFF3CT reads and writes
  both padded and unpadded lists. A ``0`` is accepted only as *trailing* padding and a list may not
  be longer than its ``dmax``. A degree-0 node must therefore be written as padding zeros.
- AFF3CT's configuration files carry ``#`` comment lines (e.g. ``# VN to CN (first CN id is 1)``);
  everything from ``#`` to the end of a line is a comment. Blank lines are ignored.
- The header numbers may wrap over several lines; every adjacency list must be on its own line.
- The CN-to-VN block is optional. When present it must describe exactly the same matrix as the
  VN-to-CN block; disagreement is an error, never silently resolved.
- Declared degrees must match the lists, ``dmax`` must equal the largest declared degree, and the
  two degree sums (edge counts) must agree. Duplicate entries, out-of-range indices, truncated
  files and trailing content are rejected with ``FormatError("<file>:<line>: ...")``.

The returned :class:`ParityCheckMatrix` keeps both CSR descriptions, degree distributions, the
GF(2) rank (computed on demand) and the sha256 of the raw file bytes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from nimbus.problems.formats import SourceText, format_error, read_source, source_from_text

__all__ = [
    "ParityCheckMatrix",
    "read_alist",
    "parse_alist",
    "write_alist",
    "format_alist",
    "gf2_rank",
    "gf2_rank_dense",
    "pack_rows",
    "GF2RowSpace",
]


# ---------------------------------------------------------------------------------------------
# GF(2) helpers (bit-packed rows, uint64 words)
# ---------------------------------------------------------------------------------------------
def pack_rows(rows: Sequence[Iterable[int]], n_cols: int) -> np.ndarray:
    """Pack 0-based column-index lists into a uint64 bit matrix of shape (len(rows), ceil(n/64))."""
    n_words = max(1, (n_cols + 63) // 64)
    packed = np.zeros((len(rows), n_words), dtype=np.uint64)
    for r, cols in enumerate(rows):
        c = np.asarray(list(cols), dtype=np.int64)
        if c.size == 0:
            continue
        if c.min() < 0 or c.max() >= n_cols:
            raise ValueError(f"row {r}: column index out of range 0..{n_cols - 1}")
        bits = np.left_shift(np.uint64(1), (c & 63).astype(np.uint64))
        # duplicates cancel over GF(2): use XOR accumulation
        np.bitwise_xor.at(packed[r], c >> 6, bits)
    return packed


def _rank_packed(packed: np.ndarray, n_cols: int) -> int:
    a = packed.copy()
    m = a.shape[0]
    rank = 0
    for col in range(n_cols):
        if rank == m:
            break
        w = col >> 6
        mask = np.uint64(1) << np.uint64(col & 63)
        hits = np.flatnonzero(a[rank:, w] & mask)
        if hits.size == 0:
            continue
        p = rank + int(hits[0])
        if p != rank:
            a[[rank, p]] = a[[p, rank]]
        below = rank + 1 + np.flatnonzero(a[rank + 1:, w] & mask)
        if below.size:
            # rows at or below `rank` are zero in all columns < col, so only words >= w change
            a[below, w:] ^= a[rank, w:]
        rank += 1
    return rank


class GF2RowSpace:
    """Reduced row-echelon basis of the GF(2) row space of a 0/1 matrix (bit-packed).

    ``reduce(cols)`` returns the unique normal form of the vector with ones at ``cols`` modulo the
    row space, so two supports A, B satisfy ``A xor B in rowspace`` iff their normal forms are equal,
    and ``A in rowspace`` iff the normal form is zero (``contains``).
    """

    def __init__(self, rows: Sequence[Iterable[int]], n_cols: int):
        self.n_cols = int(n_cols)
        a = pack_rows(rows, n_cols) if len(rows) else np.zeros((0, max(1, (n_cols + 63) // 64)), dtype=np.uint64)
        m = a.shape[0]
        pivots = []
        rank = 0
        for col in range(n_cols):
            if rank == m:
                break
            w = col >> 6
            mask = np.uint64(1) << np.uint64(col & 63)
            hits = np.flatnonzero(a[rank:, w] & mask)
            if hits.size == 0:
                continue
            p = rank + int(hits[0])
            if p != rank:
                a[[rank, p]] = a[[p, rank]]
            others = np.flatnonzero(a[:, w] & mask)
            others = others[others != rank]
            if others.size:
                a[others] ^= a[rank]
            pivots.append(col)
            rank += 1
        self.basis = a[:rank].copy()
        self.pivots = np.array(pivots, dtype=np.int64)

    @property
    def rank(self) -> int:
        return int(self.pivots.size)

    def _vec(self, cols: Iterable[int]) -> np.ndarray:
        return pack_rows([list(cols)], self.n_cols)[0]

    def reduce(self, cols: Iterable[int]) -> bytes:
        v = self._vec(cols)
        for r, col in enumerate(self.pivots):
            if (int(v[col >> 6]) >> (int(col) & 63)) & 1:
                v ^= self.basis[r]
        return v.tobytes()

    def contains(self, cols: Iterable[int]) -> bool:
        return not any(self.reduce(cols))


def gf2_rank(rows: Sequence[Iterable[int]], n_cols: int) -> int:
    """GF(2) rank of the matrix whose row r has ones at the 0-based columns ``rows[r]``."""
    if len(rows) == 0 or n_cols == 0:
        return 0
    return _rank_packed(pack_rows(rows, n_cols), n_cols)


def gf2_rank_dense(matrix) -> int:
    """GF(2) rank of a dense 0/1 matrix."""
    h = np.asarray(matrix)
    if h.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    rows = [np.flatnonzero(h[r] & 1) for r in range(h.shape[0])]
    return gf2_rank(rows, h.shape[1])


# ---------------------------------------------------------------------------------------------
# Parity-check matrix container
# ---------------------------------------------------------------------------------------------
def _csr_from_lists(lists: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    for i, lst in enumerate(lists):
        ptr[i + 1] = ptr[i] + len(lst)
    idx = np.empty(int(ptr[-1]), dtype=np.int32)
    for i, lst in enumerate(lists):
        idx[ptr[i]:ptr[i + 1]] = np.sort(np.asarray(lst, dtype=np.int32))
    return ptr, idx


def _transpose_lists(lists: Sequence[Sequence[int]], n_other: int) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(n_other)]
    for i, lst in enumerate(lists):
        for j in lst:
            out[j].append(i)
    return out


@dataclass(frozen=True, eq=False)
class ParityCheckMatrix:
    """Sparse binary parity-check matrix H (M checks x N variables), zero-based CSR both ways.

    ``row_ptr/row_idx``: variables of each check (sorted); ``col_ptr/col_idx``: checks of each
    variable (sorted). ``sha256`` is the checksum of the source file bytes (None if built in memory).
    """

    n: int
    m: int
    row_ptr: np.ndarray
    row_idx: np.ndarray
    col_ptr: np.ndarray
    col_idx: np.ndarray
    source: str | None = None
    sha256: str | None = None
    has_row_lists: bool = True
    zero_padded: bool = False
    comments: tuple = ()
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ---- construction --------------------------------------------------------------------
    @classmethod
    def from_rows(cls, n: int, rows: Sequence[Iterable[int]], **kw) -> "ParityCheckMatrix":
        """Build from 0-based column lists of each check (duplicates rejected)."""
        rows_l = [list(map(int, r)) for r in rows]
        for i, r in enumerate(rows_l):
            if len(set(r)) != len(r):
                raise ValueError(f"check {i}: duplicate variable index")
            if any(j < 0 or j >= n for j in r):
                raise ValueError(f"check {i}: variable index out of range 0..{n - 1}")
        cols_l = _transpose_lists(rows_l, n)
        row_ptr, row_idx = _csr_from_lists(rows_l)
        col_ptr, col_idx = _csr_from_lists(cols_l)
        return cls(n=int(n), m=len(rows_l), row_ptr=row_ptr, row_idx=row_idx, col_ptr=col_ptr,
                   col_idx=col_idx, **kw)

    @classmethod
    def from_dense(cls, h, **kw) -> "ParityCheckMatrix":
        a = np.asarray(h)
        if a.ndim != 2 or not np.isin(a, (0, 1)).all():
            raise ValueError("expected a 2-D 0/1 matrix")
        return cls.from_rows(a.shape[1], [np.flatnonzero(a[i]).tolist() for i in range(a.shape[0])], **kw)

    # ---- views ---------------------------------------------------------------------------
    @property
    def nnz(self) -> int:
        return int(self.row_ptr[-1])

    def row(self, i: int) -> np.ndarray:
        return self.row_idx[self.row_ptr[i]:self.row_ptr[i + 1]]

    def col(self, j: int) -> np.ndarray:
        return self.col_idx[self.col_ptr[j]:self.col_ptr[j + 1]]

    @property
    def rows(self) -> list[tuple[int, ...]]:
        """0-based variable supports of the checks."""
        return [tuple(int(x) for x in self.row(i)) for i in range(self.m)]

    @property
    def cols(self) -> list[tuple[int, ...]]:
        return [tuple(int(x) for x in self.col(j)) for j in range(self.n)]

    @property
    def var_degrees(self) -> np.ndarray:
        return np.diff(self.col_ptr)

    @property
    def check_degrees(self) -> np.ndarray:
        return np.diff(self.row_ptr)

    def degree_distribution(self) -> dict:
        """{'variable': {degree: count}, 'check': {degree: count}} (node-perspective counts)."""
        def hist(d):
            vals, counts = np.unique(d, return_counts=True)
            return {int(v): int(c) for v, c in zip(vals, counts)}
        return {"variable": hist(self.var_degrees), "check": hist(self.check_degrees)}

    def to_dense(self, dtype=np.uint8) -> np.ndarray:
        h = np.zeros((self.m, self.n), dtype=dtype)
        for i in range(self.m):
            h[i, self.row(i)] = 1
        return h

    def to_csr(self):
        """scipy.sparse.csr_matrix (uint8) of H."""
        from scipy.sparse import csr_matrix

        data = np.ones(self.nnz, dtype=np.uint8)
        return csr_matrix((data, self.row_idx, self.row_ptr), shape=(self.m, self.n))

    def syndrome(self, bits) -> np.ndarray:
        """H y mod 2 for bits y of shape (N,) or (F, N); returns uint8 (M,) or (F, M)."""
        y = np.asarray(bits)
        if y.shape[-1] != self.n:
            raise ValueError(f"bits have length {y.shape[-1]}, expected N={self.n}")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("bits must be 0/1")
        h = self.to_csr().astype(np.int64)
        s = (h @ y.astype(np.int64).T) % 2
        return np.asarray(s.T, dtype=np.uint8)

    def rank(self) -> int:
        """GF(2) rank (cached)."""
        if "rank" not in self._cache:
            self._cache["rank"] = gf2_rank(self.rows, self.n)
        return self._cache["rank"]

    def syndrome_consistent(self, syndrome) -> bool:
        """True iff H y = b has a solution over GF(2) (rank H == rank [H | b])."""
        b = np.asarray(syndrome).reshape(-1)
        if b.size != self.m or not np.isin(b, (0, 1)).all():
            raise ValueError(f"syndrome must be a 0/1 vector of length M={self.m}")
        aug_rows = [tuple(r) + ((self.n,) if b[i] else ()) for i, r in enumerate(self.rows)]
        return gf2_rank(aug_rows, self.n + 1) == self.rank()

    def summary(self, with_rank: bool = True) -> dict:
        out = {
            "n": self.n,
            "m": self.m,
            "nnz": self.nnz,
            "var_degree_max": int(self.var_degrees.max(initial=0)),
            "check_degree_max": int(self.check_degrees.max(initial=0)),
            "degree_distribution": self.degree_distribution(),
            "sha256": self.sha256,
            "source": self.source,
        }
        if with_rank:
            r = self.rank()
            out["rank"] = r
            out["k"] = self.n - r
            out["redundant_rows"] = self.m - r
        return out

    def same_matrix(self, other: "ParityCheckMatrix") -> bool:
        return (self.n == other.n and self.m == other.m and np.array_equal(self.row_ptr, other.row_ptr)
                and np.array_equal(self.row_idx, other.row_idx))

    def __eq__(self, other):  # matrix equality, provenance ignored
        return isinstance(other, ParityCheckMatrix) and self.same_matrix(other)

    def __hash__(self):
        return hash((self.n, self.m, self.row_idx.tobytes(), self.row_ptr.tobytes()))


# ---------------------------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------------------------
def _tokenize(src: SourceText) -> tuple[list[tuple[int, list[str]]], list[str]]:
    lines: list[tuple[int, list[str]]] = []
    comments: list[str] = []
    for lineno, raw in enumerate(src.text.splitlines(), start=1):
        body, hash_, comment = raw.partition("#")
        if hash_:
            comments.append(comment.strip())
        toks = body.split()
        if toks:
            lines.append((lineno, toks))
    return lines, comments


def _to_int(name: str, lineno: int, tok: str, what: str) -> int:
    try:
        return int(tok)
    except ValueError:
        raise format_error(name, lineno, f"expected an integer ({what}), found {tok!r}") from None


def parse_alist(src: SourceText | str, name: str | None = None) -> ParityCheckMatrix:
    """Parse alist text (a :class:`SourceText` or a plain string)."""
    if isinstance(src, str):
        src = source_from_text(src, name or "<string>")
    fname = src.name
    lines, comments = _tokenize(src)
    if not lines:
        raise format_error(fname, None, "empty alist file")

    # ---- header as a token stream (may wrap lines) but must end on a line boundary ------------
    li = 0  # index into `lines`
    ti = 0  # token index within lines[li]

    def take(count: int, what: str) -> list[tuple[int, int]]:
        nonlocal li, ti
        out = []
        while len(out) < count:
            if li >= len(lines):
                last = lines[-1][0] if lines else None
                raise format_error(fname, last, f"truncated file: expected {count} values for {what}, found {len(out)}")
            lineno, toks = lines[li]
            while ti < len(toks) and len(out) < count:
                out.append((_to_int(fname, lineno, toks[ti], what), lineno))
                ti += 1
            if ti == len(toks):
                li, ti = li + 1, 0
        return out

    (n, ln_n), (m, _) = take(2, "N M")
    if n <= 0 or m <= 0:
        raise format_error(fname, ln_n, f"N and M must be positive, found N={n} M={m}")
    (dmax_v, ln_d), (dmax_c, _) = take(2, "dmax_v dmax_c")
    if dmax_v < 0 or dmax_c < 0:
        raise format_error(fname, ln_d, "maximum degrees must be non-negative")
    dv = take(n, "variable-node degrees")
    dc = take(m, "check-node degrees")
    if ti != 0:
        lineno = lines[li][0]
        raise format_error(fname, lineno, "unexpected extra values after the check-node degree list "
                                          "(each adjacency list must start on its own line)")
    deg_v = [d for d, _ in dv]
    deg_c = [d for d, _ in dc]
    for j, (d, lineno) in enumerate(dv):
        if d < 0 or d > dmax_v:
            raise format_error(fname, lineno, f"variable node {j + 1}: degree {d} outside 0..dmax_v={dmax_v}")
    for i, (d, lineno) in enumerate(dc):
        if d < 0 or d > dmax_c:
            raise format_error(fname, lineno, f"check node {i + 1}: degree {d} outside 0..dmax_c={dmax_c}")
    if max(deg_v) != dmax_v:
        raise format_error(fname, ln_d, f"dmax_v={dmax_v} but the largest declared variable degree is {max(deg_v)}")
    if max(deg_c) != dmax_c:
        raise format_error(fname, ln_d, f"dmax_c={dmax_c} but the largest declared check degree is {max(deg_c)}")
    if sum(deg_v) != sum(deg_c):
        raise format_error(fname, ln_d, f"degree sums disagree: variable degrees sum to {sum(deg_v)}, "
                                        f"check degrees sum to {sum(deg_c)}")

    padded = False

    def read_lists(count: int, degrees: list[int], dmax: int, upper: int, node: str,
                   other: str) -> tuple[list[list[int]], list[int]]:
        nonlocal li, padded
        out = []
        linenos = []
        for k in range(count):
            if li >= len(lines):
                last = lines[-1][0]
                raise format_error(fname, last, f"truncated file: expected {count} {node} lists, found {k}")
            lineno, toks = lines[li]
            li += 1
            linenos.append(lineno)
            vals = [_to_int(fname, lineno, t, f"{other} index") for t in toks]
            if len(vals) > max(dmax, degrees[k]):
                raise format_error(fname, lineno, f"{node} {k + 1}: {len(vals)} entries exceed dmax={dmax}")
            nz = 0
            while nz < len(vals) and vals[nz] != 0:
                nz += 1
            if any(v != 0 for v in vals[nz:]):
                raise format_error(fname, lineno, f"{node} {k + 1}: zero inside the list (zeros are only "
                                                  "allowed as trailing padding)")
            if nz < len(vals):
                padded = True
            entries = vals[:nz]
            if len(entries) != degrees[k]:
                raise format_error(fname, lineno, f"{node} {k + 1}: lists {len(entries)} {other} indices but its "
                                                  f"declared degree is {degrees[k]}")
            bad = [v for v in entries if v < 1 or v > upper]
            if bad:
                raise format_error(fname, lineno, f"{node} {k + 1}: {other} index {bad[0]} out of range 1..{upper}")
            if len(set(entries)) != len(entries):
                dup = next(v for v in entries if entries.count(v) > 1)
                raise format_error(fname, lineno, f"{node} {k + 1}: duplicate {other} index {dup}")
            out.append([v - 1 for v in entries])
        return out, linenos

    col_lists, _ = read_lists(n, deg_v, dmax_v, m, "variable node", "check")
    rows_from_cols = _transpose_lists(col_lists, m)
    has_row_lists = li < len(lines)
    if has_row_lists:
        row_lists, row_lines = read_lists(m, deg_c, dmax_c, n, "check node", "variable")
        if li < len(lines):
            raise format_error(fname, lines[li][0], "unexpected content after the check-node lists")
        for i in range(m):
            a = set(row_lists[i])
            b = set(rows_from_cols[i])
            if a != b:
                only_rows = sorted(x + 1 for x in a - b)[:5]
                only_cols = sorted(x + 1 for x in b - a)[:5]
                raise format_error(
                    fname, row_lines[i],
                    f"the two adjacency descriptions disagree at check {i + 1}: CN-to-VN list has extra "
                    f"variables {only_rows}, VN-to-CN lists imply extra variables {only_cols}")
    else:
        for i in range(m):
            if len(rows_from_cols[i]) != deg_c[i]:
                raise format_error(fname, dc[i][1], f"check node {i + 1}: declared degree {deg_c[i]} but the "
                                                    f"VN-to-CN lists give {len(rows_from_cols[i])}")

    row_ptr, row_idx = _csr_from_lists(rows_from_cols)
    col_ptr, col_idx = _csr_from_lists(col_lists)
    return ParityCheckMatrix(n=n, m=m, row_ptr=row_ptr, row_idx=row_idx, col_ptr=col_ptr, col_idx=col_idx,
                             source=str(src.path) if src.path else src.name, sha256=src.sha256,
                             has_row_lists=has_row_lists, zero_padded=padded, comments=tuple(comments))


def read_alist(path: str | os.PathLike) -> ParityCheckMatrix:
    """Read an alist file (``.gz``/``.xz`` decompressed transparently)."""
    return parse_alist(read_source(path))


# ---------------------------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------------------------
def format_alist(h: ParityCheckMatrix, *, zero_padding: bool = True, comments: Sequence[str] = ()) -> str:
    """alist text for H. ``zero_padding=True`` follows MacKay's files; comments use AFF3CT's '#'."""
    dv = h.var_degrees
    dc = h.check_degrees
    dmax_v = int(dv.max(initial=0))
    dmax_c = int(dc.max(initial=0))
    if not zero_padding and (dv.min(initial=1) == 0 or dc.min(initial=1) == 0):
        raise ValueError("a degree-0 node can only be written with zero padding")
    out = [f"# {c}" for c in comments]
    out.append(f"{h.n} {h.m}")
    out.append(f"{dmax_v} {dmax_c}")
    out.append(" ".join(str(int(d)) for d in dv))
    out.append(" ".join(str(int(d)) for d in dc))

    def fmt(lst, dmax):
        vals = [str(int(x) + 1) for x in lst]
        if zero_padding:
            vals += ["0"] * (dmax - len(vals))
        return " ".join(vals)

    out.extend(fmt(h.col(j), dmax_v) for j in range(h.n))
    out.extend(fmt(h.row(i), dmax_c) for i in range(h.m))
    return "\n".join(out) + "\n"


def write_alist(path: str | os.PathLike, h: ParityCheckMatrix, *, zero_padding: bool = True,
                comments: Sequence[str] = ()) -> Path:
    p = Path(path)
    p.write_text(format_alist(h, zero_padding=zero_padding, comments=comments), encoding="ascii", newline="\n")
    return p

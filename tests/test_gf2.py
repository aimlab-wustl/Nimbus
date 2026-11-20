"""GF(2) linear algebra on Python-int bitsets, checked against brute force."""

import time

import numpy as np
import pytest
import scipy.sparse as sp

from nimbus.core import gf2


def _span(rows):
    out = {0}
    for r in rows:
        out |= {x ^ r for x in out}
    return out


def _random_matrix(rng, r, c, density=0.4):
    return (rng.random((r, c)) < density).astype(np.uint8)


@pytest.mark.parametrize("seed", range(40))
def test_rank_rref_nullspace_solve_vs_bruteforce(seed):
    rng = np.random.default_rng(seed)
    r, c = int(rng.integers(1, 7)), int(rng.integers(1, 9))
    A = _random_matrix(rng, r, c, density=float(rng.uniform(0.2, 0.7)))
    rows = gf2.rows_from_dense(A)
    span = _span(rows)
    rk = gf2.rank(rows, c)
    assert 2 ** rk == len(span)
    # rref: same row space, distinct pivots = lowest bits, fully reduced
    rr, piv = gf2.rref(rows, c)
    assert len(rr) == rk and piv == sorted(piv)
    assert _span(rr) == span
    for x, p in zip(rr, piv):
        assert gf2.lowbit(x) == p
        for q in piv:
            if q != p:
                assert not (x >> q) & 1
    # nullspace: exactly the kernel
    ns = gf2.nullspace(rows, c)
    assert len(ns) == c - rk
    kernel = {x for x in range(2 ** c) if all((row & x).bit_count() % 2 == 0 for row in rows)}
    assert _span(ns) == kernel
    pivset = set(piv)
    free = [j for j in range(c) if j not in pivset]
    for v, f in zip(ns, free):  # systematic: identity on the free columns
        assert [(v >> g) & 1 for g in free] == [int(g == f) for g in free]
    # solve: consistency and correctness for every rhs
    for b in range(2 ** r):
        bvec = [(b >> i) & 1 for i in range(r)]
        x = gf2.solve(rows, bvec, c)
        exists = any(gf2.matvec(rows, y) == bvec for y in range(2 ** c))
        assert (x is not None) == exists
        if x is not None:
            assert gf2.matvec(rows, x) == bvec
    # row-space membership
    for v in range(2 ** c):
        assert gf2.in_row_space(rows, v, c) == (v in span)


def test_conversions_roundtrip():
    rng = np.random.default_rng(1)
    A = _random_matrix(rng, 7, 133)
    rows = gf2.rows_from_dense(A)
    assert np.array_equal(gf2.rows_to_dense(rows, 133), A)
    assert gf2.rows_from_sparse(sp.csr_matrix(A)) == rows
    assert gf2.rows_from_sparse(sp.coo_matrix(A)) == rows
    r2, n2 = gf2.as_rows(sp.csc_matrix(A))
    assert r2 == rows and n2 == 133
    for x in rows:
        assert gf2.indices_to_bits(gf2.bits_to_indices(x)) == x
    big = gf2.indices_to_bits(range(0, 400, 3))
    assert gf2.bits_to_indices(big) == list(range(0, 400, 3))
    assert gf2.indices_to_bits([5, 5, 7]) == 1 << 7  # repeated indices cancel
    # values taken mod 2 (e.g. 2 -> 0, 3 -> 1)
    assert gf2.rows_from_dense(np.array([[2, 3, 1]])) == [0b110]


def test_large_matrix_is_fast():
    """~1000 x 2000 matrices must take seconds at most (eliminations of code size)."""
    rng = np.random.default_rng(7)
    n, m = 2000, 1000
    H = sp.random(m, n, density=6.0 / n, random_state=3, format="csr")
    H.data[:] = 1
    t0 = time.perf_counter()
    rows = gf2.rows_from_sparse(H)
    rk = gf2.rank(rows, n)
    ns = gf2.nullspace(rows, n)
    t_sparse = time.perf_counter() - t0
    assert len(ns) == n - rk
    for v in ns[:50]:
        assert all((r & v).bit_count() % 2 == 0 for r in rows)
    D = _random_matrix(rng, m, n, density=0.5)
    t0 = time.perf_counter()
    drows = gf2.rows_from_dense(D)
    rr, piv = gf2.rref(drows, n)
    x = gf2.solve(drows, [1] * m, n)
    t_dense = time.perf_counter() - t0
    assert len(rr) == len(piv) <= m
    if x is not None:
        assert gf2.matvec(drows, x) == [1] * m
    assert t_sparse < 10.0 and t_dense < 10.0, (t_sparse, t_dense)

"""alist reader/writer: original fixtures, both layouts, malformed input, GF(2) helpers."""

import gzip
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from nimbus.core.errors import FormatError
from nimbus.problems.formats.alist import (
    GF2RowSpace,
    ParityCheckMatrix,
    format_alist,
    gf2_rank,
    gf2_rank_dense,
    parse_alist,
    read_alist,
    write_alist,
)

pytestmark = pytest.mark.quick

FIX = Path(__file__).parent / "fixtures"
TOY = FIX / "alist" / "toy_10_4.alist"
ROWS = [(0, 1, 2, 4), (2, 3, 5, 6), (4, 5, 7, 8, 9), (0, 1, 3, 4, 5, 6)]  # row 4 = row 1 xor row 2


def test_toy_padded_fields():
    h = read_alist(TOY)
    assert (h.n, h.m, h.nnz) == (10, 4, 19)
    assert h.rows == ROWS
    assert h.var_degrees.tolist() == [2, 2, 2, 2, 3, 3, 2, 1, 1, 1]
    assert h.check_degrees.tolist() == [4, 4, 5, 6]
    assert h.zero_padded and h.has_row_lists
    assert h.sha256 == hashlib.sha256(TOY.read_bytes()).hexdigest()
    assert h.degree_distribution() == {"variable": {1: 3, 2: 5, 3: 2}, "check": {4: 2, 5: 1, 6: 1}}
    s = h.summary()
    assert (s["rank"], s["k"], s["redundant_rows"]) == (3, 7, 1)
    # both CSR descriptions agree
    assert h.cols[4] == (0, 2, 3) and h.cols[7] == (2,)
    dense = h.to_dense()
    assert dense.sum() == 19 and np.array_equal(dense, h.to_csr().toarray())


def test_aff3ct_and_vn_only_variants_describe_the_same_matrix():
    h = read_alist(TOY)
    aff = read_alist(FIX / "alist" / "toy_10_4_aff3ct.alist")
    vn = read_alist(FIX / "alist" / "toy_10_4_vn_only.alist")
    assert aff == h and vn == h
    assert not aff.zero_padded and aff.has_row_lists
    assert aff.comments[0].startswith("nimbus original fixture")
    assert vn.has_row_lists is False


def test_codewords_by_enumeration_and_redundant_row():
    h = read_alist(TOY)
    dense = h.to_dense().astype(int)
    words = np.array(list(itertools.product([0, 1], repeat=10)), dtype=np.uint8)
    synd = h.syndrome(words)
    assert np.array_equal(synd, (words.astype(int) @ dense.T) % 2)  # independent dense computation
    n_code = int((~synd.any(axis=1)).sum())
    assert n_code == 2 ** (h.n - h.rank()) == 128
    rs = GF2RowSpace(h.rows[:3], h.n)
    assert rs.rank == 3 and rs.contains(h.rows[3])  # redundant row lies in the span of the others
    assert not rs.contains([0])
    # syndrome consistency: the redundant row forces b4 = b1 xor b2
    assert h.syndrome_consistent([1, 0, 1, 1])
    assert not h.syndrome_consistent([1, 0, 1, 0])


@pytest.mark.parametrize("zero_padding", [True, False])
def test_write_read_roundtrip(tmp_path, zero_padding):
    h = read_alist(TOY)
    p = write_alist(tmp_path / "x.alist", h, zero_padding=zero_padding, comments=["round trip"])
    back = read_alist(p)
    assert back == h and back.zero_padded == zero_padding and back.comments == ("round trip",)
    assert format_alist(h) == TOY.read_text()  # the fixture is the canonical padded writer output


def test_degree_zero_column_requires_padding(tmp_path):
    h = ParityCheckMatrix.from_rows(4, [[0, 1], [1, 2]])  # column 3 has degree 0
    with pytest.raises(ValueError):
        format_alist(h, zero_padding=False)
    back = read_alist(write_alist(tmp_path / "d0.alist", h))
    assert back == h and back.var_degrees.tolist() == [1, 2, 1, 0]


def test_gzip_transparent_and_checksum_of_stored_bytes(tmp_path):
    gz = tmp_path / "toy.alist.gz"
    gz.write_bytes(gzip.compress(TOY.read_bytes()))
    h = read_alist(gz)
    assert h == read_alist(TOY)
    assert h.sha256 == hashlib.sha256(gz.read_bytes()).hexdigest()


def test_header_may_wrap_lines():
    text = TOY.read_text().replace("10 4\n3 6\n", "10\n4 3\n6\n", 1)
    assert parse_alist(text) == read_alist(TOY)


@pytest.mark.parametrize("name, line, fragment", [
    ("alist_truncated.alist", 9, "truncated file: expected 10 variable node lists, found 5"),
    ("alist_duplicate.alist", 5, "duplicate check index 1"),
    ("alist_degree_mismatch.alist", 12, "lists 1 check indices but its declared degree is 2"),
    ("alist_degree_sum.alist", 2, "degree sums disagree"),
    ("alist_out_of_range.alist", 14, "check index 5 out of range 1..4"),
    ("alist_disagree.alist", 17, "the two adjacency descriptions disagree at check 3"),
    ("alist_zero_inside.alist", 5, "zero inside the list"),
    ("alist_dmax.alist", 3, "degree 3 outside 0..dmax_v=2"),
    ("alist_trailing.alist", 19, "unexpected content after the check-node lists"),
    ("alist_not_integer.alist", 6, "expected an integer (check index), found 'x'"),
])
def test_malformed_alists(name, line, fragment):
    path = FIX / "malformed" / name
    with pytest.raises(FormatError) as exc:
        read_alist(path)
    msg = str(exc.value)
    assert msg.startswith(f"{path}:{line}: "), msg
    assert fragment in msg


def test_error_messages_name_string_sources():
    with pytest.raises(FormatError, match=r"^mycode: empty alist file"):
        parse_alist("# only a comment\n", name="mycode")
    with pytest.raises(FormatError, match=r"^mycode:1: N and M must be positive"):
        parse_alist("0 3\n", name="mycode")


def test_gf2_rank_matches_span_enumeration():
    rng = np.random.default_rng(7)
    for _ in range(40):
        m, n = rng.integers(1, 7), rng.integers(1, 12)
        a = rng.integers(0, 2, size=(m, n))
        # independent rank: |row space| = 2^rank, enumerated over all subsets of rows
        span = {tuple(np.bitwise_xor.reduce(a[list(c)], axis=0)) if c else (0,) * n
                for r in range(m + 1) for c in itertools.combinations(range(m), r)}
        assert 2 ** gf2_rank_dense(a) == len(span)
        rows = [np.flatnonzero(a[i]).tolist() for i in range(m)]
        assert gf2_rank(rows, n) == gf2_rank_dense(a)
        rs = GF2RowSpace(rows, n)
        for v in list(span)[:5]:
            assert rs.contains(np.flatnonzero(v).tolist())


def test_gf2_rank_wide_rows_cross_word_boundaries():
    n = 200
    rows = [[0, 63, 64, 199], [63, 64], [0, 199], [1, 128]]  # row 0 = row 1 xor row 2
    assert gf2_rank(rows, n) == 3
    rs = GF2RowSpace(rows, n)
    assert rs.reduce([0, 199]) == rs.reduce([63, 64])

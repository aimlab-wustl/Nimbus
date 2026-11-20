"""GF(2) coset helpers: nullspace basis, particular solution, encoding of coset members."""

from __future__ import annotations

import numpy as np

from helpers_core import codewords, random_parity_code
from nimbus.problems import channels
from nimbus.problems.formats.alist import ParityCheckMatrix


def _code_from_dense(c_mat):
    n = c_mat.shape[1]
    return ParityCheckMatrix.from_rows(n, [np.flatnonzero(c_mat[i]).tolist() for i in range(c_mat.shape[0])])


def test_generator_masks_span_the_nullspace_and_encode_all_codewords():
    n = 9
    c_mat = random_parity_code(n, 3, seed=4, row_weight=4)
    code = _code_from_dense(c_mat)
    masks = channels.generator_masks(code)
    k = code.n - code.rank()
    assert len(masks) == k
    brute = codewords(c_mat)
    assert brute.shape[0] == 1 << k
    info = ((np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1).astype(np.uint8)
    y = channels.encode(masks, n, info)
    enc_spins = (1 - 2 * y.astype(np.int16)).astype(np.int8)
    assert np.unique(enc_spins, axis=0).shape[0] == 1 << k
    for s in enc_spins:
        assert np.any(np.all(brute == s[None, :], axis=1))


def test_particular_solution_and_encode_stay_in_nonzero_syndrome_coset():
    n = 9
    c_mat = random_parity_code(n, 3, seed=4, row_weight=4)
    code = _code_from_dense(c_mat)
    syndrome = np.array([1, 0, 1], dtype=np.uint8)
    part = channels.particular_solution(code, syndrome)
    masks = channels.generator_masks(code)
    k = len(masks)
    info = ((np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1).astype(np.uint8)
    y = channels.encode(masks, n, info, part)
    got = code.syndrome(y)
    assert np.all(got == syndrome[None, :])
    assert np.unique(y, axis=0).shape[0] == 1 << k

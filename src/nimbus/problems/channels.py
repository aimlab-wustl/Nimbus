"""GF(2) coset helpers for parity-constrained problems (used by ``surface_code``).

A parity constraint ``H y = b`` (mod 2) has the feasible set ``y0 + span(G)``: ``particular_solution``
returns one ``y0``, ``generator_masks`` the systematic nullspace basis ``G`` (the same basis the
``nullspace_basis`` transition library uses, so feasible starts and the compiled architecture's moves
agree by construction), and ``encode`` maps coefficient bits to coset members.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nimbus.core import gf2
from nimbus.problems.formats.alist import ParityCheckMatrix

__all__ = [
    "generator_masks",
    "particular_solution",
    "encode",
]


def _bits_to_array(x: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.uint8)
    idx = gf2.bits_to_indices(x)
    if idx:
        out[idx] = 1
    return out


def generator_masks(code: ParityCheckMatrix) -> list[tuple]:
    """Systematic GF(2) nullspace basis of H, as 0-based spin-index tuples (``K = N - rank(H)``).

    Flipping the spins of one basis mask preserves ``C y = b`` for any target syndrome ``b``
    (Supplementary Information, Surface-code decoding: objective, moves and reference decoders); this is the same basis the ``nullspace_basis`` transition library builds,
    so the encoder below and the compiled architecture's feasible moves always agree.
    """
    rows = [gf2.indices_to_bits(r) for r in code.rows]
    basis = gf2.nullspace(rows, code.n)
    if not basis:
        raise ValueError("the parity-check matrix has full column rank: no feasible move (K = 0)")
    return [tuple(gf2.bits_to_indices(x)) for x in basis]


def particular_solution(code: ParityCheckMatrix, syndrome: Sequence[int] | None = None) -> np.ndarray:
    """One bit vector y (uint8[N]) with ``H y = syndrome`` (mod 2); the all-zero syndrome gives the
    all-zero vector (always a solution)."""
    rows = [gf2.indices_to_bits(r) for r in code.rows]
    if syndrome is None:
        b = 0
    else:
        s = np.asarray(syndrome, dtype=np.int64) & 1
        if s.size != code.m:
            raise ValueError(f"syndrome must have length {code.m}, got {s.size}")
        b = s.tolist()
    sol = gf2.solve(rows, b, code.n)
    if sol is None:
        raise ValueError("syndrome has no solution under this parity-check matrix (inconsistent with its rank)")
    return _bits_to_array(sol, code.n)


def _mask_matrix(masks: Sequence[tuple], n: int) -> np.ndarray:
    mat = np.zeros((len(masks), n), dtype=np.uint8)
    for j, m in enumerate(masks):
        mat[j, list(m)] = 1
    return mat


def encode(masks: Sequence[tuple], n: int, info_bits: np.ndarray, particular: np.ndarray | None = None) -> np.ndarray:
    """Codeword bits y = particular XOR (info_bits . mask_matrix) mod 2. ``info_bits``: (K,) or (F, K)."""
    mat = _mask_matrix(masks, n)
    info = np.asarray(info_bits, dtype=np.uint8)
    single = info.ndim == 1
    if single:
        info = info[None, :]
    if particular is None:
        particular = np.zeros(n, dtype=np.uint8)
    y = (particular[None, :] + info.astype(np.int64) @ mat.astype(np.int64)) % 2
    y = y.astype(np.uint8)
    return y[0] if single else y

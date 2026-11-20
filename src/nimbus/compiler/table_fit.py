"""Finite-table route: fit switches over an enumerated feasible domain (table
"Complete finite transition table"; Supplementary Algorithm 1 (compilation of a move library)).

For a clause support A and a group r given as an explicit table on Omega, the clause-change bit
is b(s) = |A ∩ changed_r(s)| mod 2 (Supplementary Algorithm 1 (compilation of a move library)). The architecture needs b to be
a constant or one signed spin product. With y = (1 - s)/2 in GF(2), that means

    b(s) = <g, y(s)> xor c     for all s in Omega,

i.e. alpha = (1 - T_g)/2 if c = 0 and alpha = (1 + T_g)/2 if c = 1 (T_g = prod_{i in g} s_i).
:class:`AffineFitter` solves this GF(2) system on a basis of the domain rows [y(s), 1] and then
verifies the solution on every domain state (exhaustive check). If no (g, c) fits, the switch
cannot be implemented by one stored product and ``UnsupportedTransition`` is raised.

The nullspace of the same row matrix gives every parity identity of Omega:
<g, y(s)> = c on all s  <=>  prod_{i in g} s_i = (-1)^c on Omega. ``domain_identities`` returns a
basis of them; the compiler adds them to the identity reducer for gate reuse.
"""

from __future__ import annotations

import numpy as np

from nimbus.core.errors import UnsupportedTransition
from nimbus.core.gf2 import GF2Basis, bits_to_indices, nullspace, rows_from_dense

__all__ = ["AffineFitter", "domain_identities"]


def _domain_rows(states: np.ndarray) -> tuple[np.ndarray, list[int]]:
    s = np.asarray(states)
    if s.ndim != 2 or s.shape[0] == 0:
        raise ValueError("the domain must be a non-empty 2-D array (states x N)")
    if np.any((s != 1) & (s != -1)):
        raise ValueError("domain spins must be -1/+1")
    y = (s < 0).astype(np.uint8)
    aug = np.hstack([y, np.ones((s.shape[0], 1), dtype=np.uint8)])
    return y, rows_from_dense(aug)


class AffineFitter:
    """Solve b(s) = <g, y(s)> xor c over an enumerated domain (states x N, spins -1/+1)."""

    def __init__(self, states):
        s = np.asarray(states)
        self.n_spins = int(s.shape[1])
        self.n_states = int(s.shape[0])
        self.Y, rows = _domain_rows(s)
        n_cols = self.n_spins + 1
        self._n_cols = n_cols
        basis = GF2Basis(n_cols)
        sel: list[int] = []
        for idx, row in enumerate(rows):
            if len(sel) == n_cols:
                break
            res = basis.insert(row | (1 << (n_cols + len(sel))))
            if res & ((1 << n_cols) - 1):
                sel.append(idx)
        self.basis_states = np.asarray(sel, dtype=np.int64)
        self.rank = len(sel)
        self._basis_rows = [rows[i] for i in sel]
        # per pivot: (pivot column, tracking bits over selected rows)
        self._pivots = [(p, basis.row_of_pivot(p) >> n_cols) for p in basis.sorted_pivots()]

    def identities(self) -> list[tuple[tuple[int, ...], int]]:
        """Basis of all parity identities of the domain: [(support, sign)]."""
        out = []
        for v in nullspace(self._basis_rows, self._n_cols):
            g = v & ((1 << self.n_spins) - 1)
            c = (v >> self.n_spins) & 1
            out.append((tuple(bits_to_indices(g)), -1 if c else 1))
        return out

    def fit(self, b, *, context: str = "") -> tuple[int, int]:
        """(g_bits, c) with b(s) = <g, y(s)> xor c on every domain state; raises UnsupportedTransition."""
        b = np.asarray(b).astype(np.int64).reshape(-1) & 1
        if b.size != self.n_states:
            raise ValueError("b must have one bit per domain state")
        bsel = 0
        for j, i in enumerate(self.basis_states.tolist()):
            if b[i]:
                bsel |= 1 << j
        x = 0
        for p, track in self._pivots:
            if (track & bsel).bit_count() & 1:
                x |= 1 << p
        g = x & ((1 << self.n_spins) - 1)
        c = (x >> self.n_spins) & 1
        gi = bits_to_indices(g)
        pred = (self.Y[:, gi].sum(axis=1, dtype=np.int64) + c) & 1 if gi else np.full(self.n_states, c)
        if not np.array_equal(pred, b):
            bad = int(np.flatnonzero(pred != b)[0])
            raise UnsupportedTransition(
                "finite-table route: the clause-change pattern is not a constant or a single signed spin "
                f"product over the enumerated domain (first mismatch at domain state {bad}){context}")
        return g, c


def domain_identities(states) -> list[tuple[tuple[int, ...], int]]:
    """All parity identities prod_{i in S} s_i = sign valid on the enumerated states (a basis)."""
    return AffineFitter(states).identities()

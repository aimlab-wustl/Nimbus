"""Spin-product algebra: support canonicalization, bitsets, product evaluation and the
feasible-domain identity reducer used for gate reuse.

A *support* is a sorted tuple of distinct spin indices; it denotes the spin product
``T_A(s) = prod_{i in A} s_i`` (Methods, Derivation of the transition-programmable architecture). Because ``s_i^2 = 1``, a raw index
list is canonicalized by keeping the indices of odd multiplicity. The empty support is the
constant 1.

Feasible-domain parity identities ``prod_{i in S} s_i = c`` (valid on Omega only) let the compiler
recognize that a requested switch product equals a stored clause sign up to a polarity
(Supplementary Algorithm 1 (compilation of a move library): "reuse an equivalent sign with the correct polarity").
:class:`IdentityReducer` keeps the identities as a fully reduced GF(2) row-echelon basis with the
sign carried as a payload bit, so ``reduce(A)`` returns a canonical coset key: two products are
equivalent on Omega iff their keys' bit parts are equal, and then ``T_A = sign_A * sign_B * T_B``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from nimbus.core.errors import ConfigError
from nimbus.core.gf2 import GF2Basis, bits_to_indices

__all__ = [
    "canonical_support",
    "support_to_bits",
    "bits_to_support",
    "product",
    "products",
    "IdentityReducer",
    "canonical_identities",
]


def canonical_support(indices: Iterable[int]) -> tuple[int, ...]:
    """Sorted tuple of the indices with odd multiplicity (s_i^2 = 1)."""
    odd: set[int] = set()
    for i in indices:
        i = int(i)
        if i < 0:
            raise ValueError(f"negative spin index {i}")
        if i in odd:
            odd.remove(i)
        else:
            odd.add(i)
    return tuple(sorted(odd))


def support_to_bits(support: Iterable[int]) -> int:
    """Bitset of a support (repeated indices cancel)."""
    x = 0
    for i in support:
        x ^= 1 << int(i)
    return x


def bits_to_support(bits: int) -> tuple[int, ...]:
    return tuple(bits_to_indices(int(bits)))


def product(spins, support: Sequence[int]):
    """T_A for one state (1-D spins -> int) or many states (2-D states x N -> int8 array).

    Computed as (-1)^(number of negative spins in A), exact for any support size.
    """
    s = np.asarray(spins)
    idx = np.asarray(tuple(support), dtype=np.int64)
    if s.ndim == 1:
        if idx.size == 0:
            return 1
        return -1 if int(np.count_nonzero(s[idx] < 0)) & 1 else 1
    if s.ndim != 2:
        raise ValueError("spins must be 1-D (N,) or 2-D (states, N)")
    if idx.size == 0:
        return np.ones(s.shape[0], dtype=np.int8)
    neg = np.count_nonzero(s[:, idx] < 0, axis=1)
    return (1 - 2 * (neg & 1)).astype(np.int8)


def products(spins, supports: Sequence[Sequence[int]]) -> np.ndarray:
    """Matrix of products: (K,) for 1-D spins or (S, K) for 2-D spins, int8."""
    s = np.asarray(spins)
    single = s.ndim == 1
    if single:
        s = s[None, :]
    out = np.empty((s.shape[0], len(supports)), dtype=np.int8)
    neg = (s < 0)
    for k, sup in enumerate(supports):
        idx = np.asarray(tuple(sup), dtype=np.int64)
        if idx.size == 0:
            out[:, k] = 1
        else:
            out[:, k] = 1 - 2 * (np.count_nonzero(neg[:, idx], axis=1) & 1)
    return out[0] if single else out


def canonical_identities(identities, n_spins: int | None = None) -> list[tuple[tuple[int, ...], int]]:
    """Normalize [(support, sign)] -> [(canonical support, +1/-1)], dropping trivially true empty
    identities, removing exact duplicates and rejecting contradictory duplicates or ``() = -1``."""
    out: list[tuple[tuple[int, ...], int]] = []
    seen: dict[tuple[int, ...], int] = {}
    for item in identities:
        try:
            sup, sign = item
        except (TypeError, ValueError):
            raise ConfigError(f"identity must be (support, sign), got {item!r}") from None
        sign = int(sign)
        if sign not in (1, -1):
            raise ConfigError(f"identity sign must be +1 or -1, got {sign}")
        sup = canonical_support(sup)
        if n_spins is not None and sup and sup[-1] >= n_spins:
            raise ConfigError(f"identity support {sup} has index >= n_spins={n_spins}")
        if not sup:
            if sign == -1:
                raise ConfigError("inconsistent parity identities: the empty product is +1, not -1")
            continue
        if sup in seen:
            if seen[sup] != sign:
                raise ConfigError(f"inconsistent parity identities: product {sup} declared both +1 and -1")
            continue
        seen[sup] = sign
        out.append((sup, sign))
    return out


class IdentityReducer:
    """Canonical coset keys of spin products modulo declared feasible-domain parity identities.

    ``identities``: sequence of (support, sign) meaning prod_{i in support} s_i = sign on Omega.
    The identities are stored as a fully reduced GF(2) basis (pivot = lowest bit) whose rows carry
    the sign as a payload bit at position ``n_spins`` (1 means -1).

    * ``reduce(A) -> (key_bits, sign)`` with T_A = sign * T_{key} on Omega. ``key_bits`` has no
      pivot bits, so it is canonical for the coset A + span(identities).
    * ``is_constant(A)`` -> the constant sign if A is in the span, else None.
    * ``best_representative(A) -> (support, sign)``: a low-weight support equivalent to A with
      T_A = sign * T_rep. Greedy: repeatedly XOR with the *declared* identity generator that gives
      the smallest resulting weight, as long as the weight strictly decreases (ties: first
      generator in declaration order). Example (3x3 one-hot, column identity {0,3,6} = +1):
      s_0 s_3 -> s_6. This is a heuristic (minimum-weight coset decoding is hard in general); the
      result is always equivalent to A, only its weight is not guaranteed minimal.

    Contradictory identities (some combination implies +1 = -1) raise ConfigError.
    """

    def __init__(self, n_spins: int, identities=()):
        self.n_spins = int(n_spins)
        self.generators = canonical_identities(identities, self.n_spins)
        self._sign_bit = 1 << self.n_spins
        self._colmask = self._sign_bit - 1
        self._basis = GF2Basis(self.n_spins)
        self._gen_bits: list[tuple[int, int, int]] = []  # (bits, weight, sign)
        for sup, sign in self.generators:
            bits = support_to_bits(sup)
            res = self._basis.insert(bits | (self._sign_bit if sign < 0 else 0))
            if res == self._sign_bit:
                raise ConfigError(
                    f"inconsistent parity identities: identity {sup} = {sign:+d} contradicts the others")
            self._gen_bits.append((bits, bits.bit_count(), sign))

    # -- properties -------------------------------------------------------------------------
    @property
    def rank(self) -> int:
        return self._basis.rank

    @property
    def active(self) -> bool:
        return self._basis.rank > 0

    def __len__(self) -> int:
        return len(self.generators)

    # -- core operations ---------------------------------------------------------------------
    def reduce_bits(self, bits: int) -> tuple[int, int]:
        """(key_bits, sign) for a product given as a bitset."""
        if not self._basis.pivmask:
            return bits, 1
        x = self._basis.reduce(bits)
        return x & self._colmask, (-1 if x & self._sign_bit else 1)

    def reduce(self, support) -> tuple[int, int]:
        """(key_bits, sign) with T_support = sign * T_key on Omega."""
        if isinstance(support, (int, np.integer)):
            return self.reduce_bits(int(support))
        return self.reduce_bits(support_to_bits(support))

    key = reduce

    def is_constant(self, support):
        """Sign c if T_support == c on Omega by the identities, else None."""
        kb, sg = self.reduce(support)
        return sg if kb == 0 else None

    def relation(self, a, b):
        """c such that T_a = c * T_b on Omega, or None if the identities do not relate them."""
        ka, sa = self.reduce(a)
        kb, sb = self.reduce(b)
        return sa * sb if ka == kb else None

    def best_representative_bits(self, bits: int) -> tuple[int, int]:
        """(rep_bits, sign) with T_bits = sign * T_rep; greedy weight reduction (see class doc)."""
        cur = int(bits)
        sign = 1
        w = cur.bit_count()
        gens = self._gen_bits
        while gens:
            best_w, best = w, None
            for g, _gw, gs in gens:
                nw = (cur ^ g).bit_count()
                if nw < best_w:
                    best_w, best = nw, (g, gs)
            if best is None:
                break
            cur ^= best[0]
            sign *= best[1]
            w = best_w
        return cur, sign

    def best_representative(self, support) -> tuple[tuple[int, ...], int]:
        if isinstance(support, (int, np.integer)):
            rb, sg = self.best_representative_bits(int(support))
        else:
            rb, sg = self.best_representative_bits(support_to_bits(support))
        return bits_to_support(rb), sg

    def holds_on(self, states) -> bool:
        """Check every declared identity on the given states (2-D, states x N)."""
        s = np.asarray(states)
        for sup, sign in self.generators:
            if np.any(product(s, sup) != sign):
                return False
        return True

    def describe(self) -> dict:
        return {"n_identities": len(self.generators), "rank": self.rank}

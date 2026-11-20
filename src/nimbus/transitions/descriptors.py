"""Spin-action descriptors of grouped transitions mu^(r) (Methods, Derivation of the transition-programmable architecture).

A descriptor is the *physical* transformation of the spin assignment for one transition group.
It is compiler input only: the compiler derives the switches alpha and retained connections Pi
from it (Supplementary Algorithm 1 (compilation of a move library)), including those of the spin singleton clauses, through
which the decoder executes the transformation. The architecture backends never apply a descriptor
at runtime; the independent oracle and verification use ``apply``.

Kinds:

* ``FixedMask(mask)``: flip the spins in ``mask`` (kind MASK = 0).
* ``ConditionalMask(mask, cond, pol)``: flip ``mask`` iff ``pol * prod_{i in cond} s_i == +1``
  (empty cond: iff ``pol == +1``); requires ``|cond ∩ mask|`` even so the condition is invariant
  under the flip and the action is an involution (kind CONDMASK = 1).
* ``SignedPermutation(mapping)``: ``new s_i = eps_i * old s_{pi(i)}`` for the listed destinations,
  all reads from the pre-transition state (kind PERM = 2).
* ``AffineMove(mapping)``: ``new s_i = eps_i * prod_{j in sources_i} old s_j`` for the listed
  destinations: every new spin is a signed product of old spins (an affine map over GF(2)). It
  contains permutations, reversals and state-dependent flips; compiled like a permutation and
  executed by the decoder (kind READOUT = 3).
* ``TableTransition(table, domain)``: explicit finite map on an enumerated domain Omega; compiled by
  the finite-table route into READOUT actions (kind READOUT = 3).

A move need not be its own inverse: ``inverse()`` returns the move that undoes it, which a library
then holds as a separate move (``TransitionLibrary.with_reverses``).

All ``apply`` methods are pure (operate on a copy) and accept 1-D ``(N,)`` or 2-D ``(states, N)``
int8 spin arrays.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

import numpy as np

from nimbus.core.errors import InfeasibleState, UnsupportedTransition

KIND_MASK = 0
KIND_CONDMASK = 1
KIND_PERM = 2
KIND_READOUT = 3
KIND_NAMES = {KIND_MASK: "MASK", KIND_CONDMASK: "CONDMASK", KIND_PERM: "PERM", KIND_READOUT: "READOUT"}

__all__ = [
    "KIND_MASK", "KIND_CONDMASK", "KIND_PERM", "KIND_READOUT", "KIND_NAMES",
    "SpinAction", "FixedMask", "ConditionalMask", "SignedPermutation", "AffineMove", "TableTransition",
    "descriptor_from_dict",
]


def _sorted_unique(indices, what: str, allow_empty: bool = False) -> tuple[int, ...]:
    vals = [int(i) for i in indices]
    if any(i < 0 for i in vals):
        raise ValueError(f"{what}: negative spin index")
    if len(set(vals)) != len(vals):
        raise ValueError(f"{what}: repeated spin index in {vals}")
    if not vals and not allow_empty:
        raise ValueError(f"{what}: must not be empty")
    return tuple(sorted(vals))


def _as_spins(spins) -> tuple[np.ndarray, bool]:
    s = np.asarray(spins)
    if s.ndim not in (1, 2):
        raise ValueError("spins must be 1-D (N,) or 2-D (states, N)")
    return s, s.ndim == 1


def _stable_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SpinAction:
    """Base class of spin-action descriptors."""

    kind: int = -1
    type_name: str = "abstract"

    def apply(self, spins) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def is_selfloop(self, spins):
        """True where the action leaves the state unchanged (bool or bool array)."""
        s, single = _as_spins(spins)
        out = np.all(self.apply(s) == s, axis=-1)
        return bool(out) if single else out

    def touched(self) -> tuple[int, ...]:  # pragma: no cover - abstract
        """Spins that may change under the action (sorted)."""
        raise NotImplementedError

    def max_index(self) -> int:
        """Largest spin index referenced (or -1)."""
        t = self.touched()
        return t[-1] if t else -1

    @property
    def is_involution(self) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def inverse(self) -> "SpinAction":
        """The move that undoes this one (the move itself for an involution)."""
        if self.is_involution:
            return self
        raise NotImplementedError(f"{type(self).__name__}: no inverse")

    def check_involution(self, states) -> bool:
        """Numerical check mu(mu(s)) == s on the given states."""
        s, _ = _as_spins(states)
        return bool(np.all(self.apply(self.apply(s)) == s))

    def to_dict(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())

    def default_label(self) -> str:
        return self.type_name

    def __eq__(self, other):
        return isinstance(other, SpinAction) and self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash(self.stable_hash())

    def __repr__(self):
        return f"{type(self).__name__}({self.default_label()})"


class FixedMask(SpinAction):
    """Flip a fixed set of spins (Supplementary Information, Number of auxiliary clauses added by compilation (parity constraints: fixed reversals): alpha constant)."""

    kind = KIND_MASK
    type_name = "mask"

    def __init__(self, mask: Iterable[int]):
        self.mask = _sorted_unique(mask, "FixedMask")
        self._idx = np.asarray(self.mask, dtype=np.int64)

    def apply(self, spins):
        s, _ = _as_spins(spins)
        out = s.copy()
        out[..., self._idx] *= -1
        return out

    def is_selfloop(self, spins):
        s, single = _as_spins(spins)
        return False if single else np.zeros(s.shape[0], dtype=bool)

    def touched(self):
        return self.mask

    @property
    def is_involution(self):
        return True

    def to_dict(self):
        return {"type": self.type_name, "mask": list(self.mask)}

    def default_label(self):
        if len(self.mask) <= 4:
            return "flip(" + ",".join(map(str, self.mask)) + ")"
        return f"mask[w={len(self.mask)}]"


class ConditionalMask(SpinAction):
    """Flip ``mask`` iff ``pol * prod_{cond} s == +1`` (evaluated on the pre-transition spins)."""

    kind = KIND_CONDMASK
    type_name = "condmask"

    def __init__(self, mask: Iterable[int], cond: Iterable[int], pol: int):
        self.mask = _sorted_unique(mask, "ConditionalMask.mask")
        self.cond = _sorted_unique(cond, "ConditionalMask.cond", allow_empty=True)
        pol = int(pol)
        if pol not in (1, -1):
            raise ValueError("ConditionalMask polarity must be +1 or -1")
        self.pol = pol
        if len(set(self.mask) & set(self.cond)) % 2:
            raise UnsupportedTransition(
                f"ConditionalMask needs |cond ∩ mask| even (condition invariant under the flip); "
                f"mask={self.mask}, cond={self.cond}")
        self._m = np.asarray(self.mask, dtype=np.int64)
        self._c = np.asarray(self.cond, dtype=np.int64)

    def active(self, spins):
        """Bool (or bool array): whether the flip is applied in this state."""
        s, single = _as_spins(spins)
        if self._c.size == 0:
            act = np.full(1 if single else s.shape[0], self.pol == 1)
        else:
            ss = s[None, :] if single else s
            neg = np.count_nonzero(ss[:, self._c] < 0, axis=1) & 1
            prod = 1 - 2 * neg
            act = (self.pol * prod) == 1
        return bool(act[0]) if single else act

    def apply(self, spins):
        s, single = _as_spins(spins)
        out = s.copy()
        act = self.active(s)
        if single:
            if act:
                out[self._m] *= -1
        else:
            rows = np.flatnonzero(act)
            if rows.size:
                out[np.ix_(rows, self._m)] *= -1
        return out

    def is_selfloop(self, spins):
        act = self.active(spins)
        return (not act) if isinstance(act, bool) else ~act

    def touched(self):
        return self.mask

    def max_index(self):
        return max(self.mask[-1], self.cond[-1] if self.cond else -1)

    @property
    def is_involution(self):
        return True

    def to_dict(self):
        return {"type": self.type_name, "mask": list(self.mask), "cond": list(self.cond), "pol": self.pol}

    def default_label(self):
        m = ",".join(map(str, self.mask)) if len(self.mask) <= 4 else f"w={len(self.mask)}"
        c = ",".join(map(str, self.cond)) if len(self.cond) <= 6 else f"w={len(self.cond)}"
        return f"flip({m})|{'+' if self.pol > 0 else '-'}prod({c})"


class SignedPermutation(SpinAction):
    """``new s_i = eps_i * old s_{pi(i)}`` for listed destinations i; other spins unchanged.

    ``mapping``: dict dest -> (src, sign) or an iterable of (dest, src, sign). Identity entries
    (dest == src, sign +1) are dropped. The destinations must equal the sources as sets (a
    bijection on the moved coordinates), so the full map is a signed permutation.
    """

    kind = KIND_PERM
    type_name = "perm"

    def __init__(self, mapping):
        if isinstance(mapping, Mapping):
            items = [(int(d), int(v[0]), int(v[1])) for d, v in mapping.items()]
        else:
            items = [(int(d), int(s), int(e)) for d, s, e in mapping]
        seen = set()
        clean = []
        for d, s, e in items:
            if d in seen:
                raise ValueError(f"SignedPermutation: destination {d} listed twice")
            seen.add(d)
            if e not in (1, -1):
                raise ValueError("SignedPermutation: signs must be +1 or -1")
            if d < 0 or s < 0:
                raise ValueError("SignedPermutation: negative index")
            if d == s and e == 1:
                continue
            clean.append((d, s, e))
        clean.sort()
        dests = [d for d, _, _ in clean]
        srcs = [s for _, s, _ in clean]
        if not clean:
            raise ValueError("SignedPermutation: identity map (no moved coordinate)")
        if set(dests) != set(srcs) or len(set(srcs)) != len(srcs):
            raise UnsupportedTransition(
                "SignedPermutation must be a bijection on the moved coordinates "
                f"(destinations {sorted(dests)} vs sources {sorted(srcs)})")
        self.dests = tuple(dests)
        self.srcs = tuple(srcs)
        self.signs = tuple(e for _, _, e in clean)
        self._d = np.asarray(self.dests, dtype=np.int64)
        self._s = np.asarray(self.srcs, dtype=np.int64)
        self._e = np.asarray(self.signs, dtype=np.int8)

    @property
    def mapping(self) -> dict:
        return {d: (s, e) for d, s, e in zip(self.dests, self.srcs, self.signs)}

    def apply(self, spins):
        s, _ = _as_spins(spins)
        out = s.copy()
        out[..., self._d] = s[..., self._s] * self._e.astype(s.dtype)
        return out

    def is_selfloop(self, spins):
        s, single = _as_spins(spins)
        same = np.all(s[..., self._s] * self._e.astype(s.dtype) == s[..., self._d], axis=-1)
        return bool(same) if single else same

    def touched(self):
        return self.dests

    @property
    def is_involution(self):
        mp = self.mapping
        for d, (s, e) in mp.items():
            if s == d:
                continue  # sign flip, always involutive
            back = mp.get(s)
            if back is None or back[0] != d or back[1] * e != 1:
                return False
        return True

    def inverse(self) -> "SignedPermutation":
        """new s_d = e * old s_s is undone by new s_s = e * old s_d."""
        if self.is_involution:
            return self
        return SignedPermutation([(s, d, e) for d, s, e in zip(self.dests, self.srcs, self.signs)])

    def to_dict(self):
        return {"type": self.type_name, "dests": list(self.dests), "srcs": list(self.srcs),
                "signs": list(self.signs)}

    def default_label(self):
        if len(self.dests) == 2 and self.signs == (1, 1) and self.srcs == (self.dests[1], self.dests[0]):
            return f"swap({self.dests[0]},{self.dests[1]})"
        return f"perm[{len(self.dests)}]"


def _gf2_inverse(rows: list[int], n: int) -> list[int] | None:
    """Inverse of an n x n GF(2) matrix given as row bitsets (bit j = column j), or None if singular."""
    aug = [(rows[i], 1 << i) for i in range(n)]
    for c in range(n):
        piv = next((i for i in range(c, n) if (aug[i][0] >> c) & 1), None)
        if piv is None:
            return None
        aug[c], aug[piv] = aug[piv], aug[c]
        pr, pi = aug[c]
        for i in range(n):
            if i != c and (aug[i][0] >> c) & 1:
                aug[i] = (aug[i][0] ^ pr, aug[i][1] ^ pi)
    return [aug[i][1] for i in range(n)]


def _apply_rows(rows: list[int], x: int) -> int:
    y = 0
    for i, r in enumerate(rows):
        if (r & x).bit_count() & 1:
            y |= 1 << i
    return y


class AffineMove(SpinAction):
    """``new s_i = eps_i * prod_{j in sources_i} old s_j`` for the listed destinations i; other spins unchanged.

    ``mapping``: dict dest -> (sources, sign) or an iterable of (dest, sources, sign). With
    ``y = (1 - s)/2`` the move is the affine map ``y' = A y + b`` over GF(2), which must be invertible.
    Every clause then changes sign by one signed spin product, so every switch is set by one stored
    clause. Permutations, reversals, conditional flips (``new s_i = s_i s_j``) and conditional
    reversals are special cases. Entries that leave a spin unchanged are dropped.
    """

    kind = KIND_READOUT
    type_name = "affine"

    def __init__(self, mapping):
        if isinstance(mapping, Mapping):
            items = [(int(d), v[0], int(v[1])) for d, v in mapping.items()]
        else:
            items = [(int(d), src, int(e)) for d, src, e in mapping]
        seen, clean = set(), []
        for d, src, e in items:
            if d < 0:
                raise ValueError("AffineMove: negative index")
            if d in seen:
                raise ValueError(f"AffineMove: destination {d} listed twice")
            seen.add(d)
            if e not in (1, -1):
                raise ValueError("AffineMove: signs must be +1 or -1")
            src = _sorted_unique(src, f"AffineMove sources of {d}")
            if src == (d,) and e == 1:
                continue
            clean.append((d, src, e))
        if not clean:
            raise ValueError("AffineMove: identity map (no changed spin)")
        clean.sort()
        self.dests = tuple(d for d, _, _ in clean)
        self.sources = tuple(src for _, src, _ in clean)
        self.signs = tuple(e for _, _, e in clean)
        self._n = 1 + max(max(self.dests), max(max(src) for src in self.sources))
        rows, _ = self._matrix()
        if _gf2_inverse(rows, self._n) is None:
            raise UnsupportedTransition("AffineMove must be invertible (two states would map to the same state)")
        self._src = [np.asarray(src, dtype=np.int64) for src in self.sources]

    def _matrix(self) -> tuple[list[int], int]:
        rows = [1 << i for i in range(self._n)]
        b = 0
        for d, src, e in zip(self.dests, self.sources, self.signs):
            rows[d] = sum(1 << j for j in src)
            if e < 0:
                b |= 1 << d
        return rows, b

    @property
    def mapping(self) -> dict:
        return {d: (src, e) for d, src, e in zip(self.dests, self.sources, self.signs)}

    def apply(self, spins):
        s, _ = _as_spins(spins)
        out = s.copy()
        for d, src, e in zip(self.dests, self._src, self.signs):
            neg = np.count_nonzero(s[..., src] < 0, axis=-1) & 1
            out[..., d] = (e * (1 - 2 * neg)).astype(s.dtype)
        return out

    def touched(self):
        return self.dests

    def max_index(self):
        return self._n - 1

    @property
    def is_involution(self):
        rows, b = self._matrix()
        for i in range(self._n):                       # A^2 = I, one column at a time
            if _apply_rows(rows, _apply_rows(rows, 1 << i)) != 1 << i:
                return False
        return _apply_rows(rows, b) == b                # A b + b = 0

    def inverse(self) -> "AffineMove":
        """y' = A y + b is undone by y = A^-1 y' + A^-1 b."""
        if self.is_involution:
            return self
        rows, b = self._matrix()
        inv = _gf2_inverse(rows, self._n)
        c = _apply_rows(inv, b)
        return AffineMove([(i, [j for j in range(self._n) if (inv[i] >> j) & 1], -1 if (c >> i) & 1 else 1)
                           for i in range(self._n)])

    def to_dict(self):
        return {"type": self.type_name, "dests": list(self.dests),
                "sources": [list(src) for src in self.sources], "signs": list(self.signs)}

    def default_label(self):
        return f"affine[{len(self.dests)}]"


class TableTransition(SpinAction):
    """Explicit finite transformation on an enumerated domain Omega (finite-table route).

    ``table``: dict state-tuple -> state-tuple; ``domain``: list of state tuples (Omega). Domain
    states missing from the table map to themselves (self-loops). Every image must lie in the
    domain. ``apply`` raises InfeasibleState for a state outside the domain.
    """

    kind = KIND_READOUT
    type_name = "table"

    def __init__(self, table: Mapping, domain):
        dom = [tuple(int(v) for v in st) for st in domain]
        if not dom:
            raise ValueError("TableTransition: empty domain")
        n = len(dom[0])
        if any(len(st) != n for st in dom):
            raise ValueError("TableTransition: domain states of different length")
        if any(v not in (-1, 1) for st in dom for v in st):
            raise ValueError("TableTransition: spins must be -1/+1")
        index = {}
        for i, st in enumerate(dom):
            if st in index:
                raise ValueError(f"TableTransition: duplicate domain state {st}")
            index[st] = i
        dest = np.arange(len(dom), dtype=np.int64)
        for a, b in table.items():
            a = tuple(int(v) for v in a)
            b = tuple(int(v) for v in b)
            if a not in index:
                raise ValueError(f"TableTransition: table key {a} not in the domain")
            if b not in index:
                raise UnsupportedTransition(f"TableTransition: image {b} of {a} leaves the domain")
            dest[index[a]] = index[b]
        self.n_spins = n
        self.domain = tuple(dom)
        self._index = index
        self._states = np.asarray(dom, dtype=np.int8)
        self._dest = dest
        changed = np.any(self._states != self._states[dest], axis=0)
        self._touched = tuple(int(i) for i in np.flatnonzero(changed))

    @classmethod
    def from_destinations(cls, states, dest, like: "TableTransition | None" = None) -> "TableTransition":
        """Table from the destination index of every domain state (``dest[t] == t``: self-loop).

        ``like``: an existing table on the same domain, whose domain and index are shared (no copy).
        """
        dest = np.asarray(dest, dtype=np.int64).reshape(-1)
        self = cls.__new__(cls)
        if like is not None:
            if like._states.shape[0] != dest.size or not np.array_equal(like._states, np.asarray(states)):
                raise ValueError("TableTransition.from_destinations: 'like' has a different domain")
            self.domain, self._index, self._states = like.domain, like._index, like._states
        else:
            S = np.asarray(states)
            if S.ndim != 2 or S.shape[0] == 0 or S.shape[0] != dest.size:
                raise ValueError("TableTransition.from_destinations: states must be (len(dest), N)")
            if np.any((S != 1) & (S != -1)):
                raise ValueError("TableTransition: spins must be -1/+1")
            self._states = S.astype(np.int8)
            self.domain = tuple(map(tuple, self._states.tolist()))
            self._index = {st: i for i, st in enumerate(self.domain)}
            if len(self._index) != len(self.domain):
                raise ValueError("TableTransition: duplicate domain state")
        if dest.size and (dest.min() < 0 or dest.max() >= dest.size):
            raise UnsupportedTransition("TableTransition: a destination leaves the domain")
        self.n_spins = int(self._states.shape[1])
        self._dest = dest.copy()
        changed = np.any(self._states != self._states[dest], axis=0)
        self._touched = tuple(int(i) for i in np.flatnonzero(changed))
        return self

    @property
    def domain_states(self) -> np.ndarray:
        return self._states.copy()

    @property
    def destination_index(self) -> np.ndarray:
        return self._dest.copy()

    def _lookup(self, row) -> int:
        key = tuple(int(v) for v in row)
        i = self._index.get(key)
        if i is None:
            raise InfeasibleState(f"state {key} is outside the TableTransition domain")
        return i

    def apply(self, spins):
        s, single = _as_spins(spins)
        if s.shape[-1] != self.n_spins:
            raise ValueError("spin array length does not match the table")
        if single:
            return self._states[self._dest[self._lookup(s)]].astype(s.dtype, copy=True)
        idx = np.fromiter((self._lookup(row) for row in s), dtype=np.int64, count=s.shape[0])
        return self._states[self._dest[idx]].astype(s.dtype, copy=True)

    def touched(self):
        return self._touched

    def max_index(self):
        return self.n_spins - 1

    @property
    def is_involution(self):
        return bool(np.all(self._dest[self._dest] == np.arange(len(self._dest))))

    def inverse(self) -> "TableTransition":
        """The table that undoes this one (its destinations must be a bijection of the domain)."""
        if self.is_involution:
            return self
        inv = np.full(self._dest.size, -1, dtype=np.int64)
        inv[self._dest] = np.arange(self._dest.size)
        if np.any(inv < 0):
            raise UnsupportedTransition("TableTransition: not a bijection of the domain, no inverse")
        return TableTransition.from_destinations(self._states, inv, like=self)

    def to_dict(self):
        pairs = [[list(self.domain[i]), list(self.domain[j])] for i, j in enumerate(self._dest.tolist()) if i != j]
        return {"type": self.type_name, "domain": [list(st) for st in self.domain], "pairs": pairs}

    def default_label(self):
        moved = int(np.count_nonzero(self._dest != np.arange(len(self._dest))))
        return f"table[{moved}/{len(self.domain)}]"


def descriptor_from_dict(d: dict) -> SpinAction:
    """Inverse of ``to_dict`` for every descriptor type."""
    t = d.get("type")
    if t == "mask":
        return FixedMask(d["mask"])
    if t == "condmask":
        return ConditionalMask(d["mask"], d["cond"], d["pol"])
    if t == "perm":
        return SignedPermutation(list(zip(d["dests"], d["srcs"], d["signs"])))
    if t == "affine":
        return AffineMove(list(zip(d["dests"], d["sources"], d["signs"])))
    if t == "table":
        dom = [tuple(st) for st in d["domain"]]
        table = {tuple(a): tuple(b) for a, b in d["pairs"]}
        return TableTransition(table, dom)
    raise ValueError(f"unknown descriptor type {t!r}")

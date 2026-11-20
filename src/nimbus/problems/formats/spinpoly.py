"""Project schema ``nimbus.spinpoly/1``: a spin-product objective with constraint blocks.

This is a versioned *project* schema, not an external standard. YAML (``.yaml``,
``.yml``) or JSON (``.json``) files hold one mapping::

    schema: nimbus.spinpoly/1          # required, exact string
    name: <str>                         # optional
    description: <str>                  # optional
    n_spins: <int >= 1>                 # required
    index_base: 0 | 1                   # optional (default 0): base of every spin index in the file
    spin_convention: plus_one_selected  # required: s = +1 <-> selected/true (x = (1 + s)/2)
    energy_convention: minus_J | plus_c # required:
                                        #   minus_J: E(s) = const - sum_k J_k prod_{i in S_k} s_i
                                        #            (Methods, Derivation of the transition-programmable architecture; weight = J_k)
                                        #   plus_c:  E(s) = const + sum_k c_k prod_{i in S_k} s_i
    const: <number or "p/q">            # required
    terms:                              # required list (may be empty)
      - {support: [i, j, ...], weight: <number or "p/q">}
    constraints:                        # optional; blocks must be pairwise spin-disjoint
      exact_cardinality: [{spins: [...], k: <int>}]  # exactly k of the block at s = +1
      chains: [[i1, i2, ...]]           # prefix chains: s_{i(t+1)} = +1 requires s_{i(t)} = +1
      or_blocks: [{spins: [...], signs: [+1/-1, ...]}]  # at least one i with s_i == sign_i
      amo_blocks: [[...]]               # at most one spin of the block at +1
      contiguity: [{spins: [...], cyclic: <bool>}]  # selected set = one nonempty interval (cyclic=false,
                                        #   path order) or one nonempty proper arc (cyclic=true, m>=3);
                                        #   (builder nimbus.transitions.builders.contiguity_block)
      free: [...]                       # optional explicit list of unconstrained spins
    labels: [<str>, ...]                # optional, one per spin
    metadata: {...}                     # optional free-form mapping

Validation is strict: unknown keys anywhere are rejected; weights and the constant must be finite
numbers (``bool`` is not a number) or exact rationals written as strings (``"11/8"``); spin indices
are integers inside ``index_base .. index_base + n_spins - 1``; a support is non-empty with
distinct indices (parity reduction is not applied silently); two terms with the same support are
rejected (merge them); constraint blocks must be disjoint (overlap raises UnsupportedProblem).

Internally everything is 0-based; ``weights`` always hold J in the manuscript convention.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from nimbus.core.errors import FormatError, UnsupportedProblem
from nimbus.problems.formats import format_error, read_source

__all__ = ["SCHEMA", "SpinPolyDocument", "parse_spinpoly", "read_spinpoly", "write_spinpoly", "spinpoly_to_dict"]

SCHEMA = "nimbus.spinpoly/1"
SPIN_CONVENTIONS = ("plus_one_selected",)
ENERGY_CONVENTIONS = ("minus_J", "plus_c")

_TOP_KEYS = {"schema", "name", "description", "n_spins", "index_base", "spin_convention", "energy_convention",
             "const", "terms", "constraints", "labels", "metadata"}
_CONSTRAINT_KEYS = {"exact_cardinality", "chains", "or_blocks", "amo_blocks", "contiguity", "free"}


def _contiguous_mask(bits: np.ndarray, cyclic: bool) -> np.ndarray:
    """Row-wise: the ``True`` entries of each row of ``bits`` (B, m) form one nonempty contiguous
    interval (``cyclic=False``) or one nonempty proper arc (``cyclic=True``)."""
    m = bits.shape[1]
    nonempty = bits.any(axis=1)
    if cyclic:
        full = bits.all(axis=1)
        transitions = (bits != np.roll(bits, -1, axis=1)).sum(axis=1)
        return nonempty & ~full & (transitions == 2)
    lo = np.where(bits, np.arange(m)[None, :], m).min(axis=1)
    hi = np.where(bits, np.arange(m)[None, :], -1).max(axis=1)
    count = bits.sum(axis=1)
    return nonempty & (hi - lo + 1 == count)


@dataclass(frozen=True, eq=False)
class SpinPolyDocument:
    """E(s) = const - sum_k weights[k] * prod_{i in supports[k]} s_i, plus constraint blocks (0-based)."""

    n_spins: int
    supports: tuple
    weights: np.ndarray  # J_k, manuscript convention
    const: float
    exact_cardinality: tuple = ()  # ((spins...), k)
    chains: tuple = ()
    or_blocks: tuple = ()  # ((spins...), (signs...))
    amo_blocks: tuple = ()
    contiguity: tuple = ()  # ((spins...), cyclic_bool)
    free: tuple = ()
    labels: tuple | None = None
    name: str | None = None
    description: str | None = None
    metadata: dict = field(default_factory=dict)
    exact: dict | None = None  # optional exact rationals {"const": Fraction, "weights": (Fraction, ...)}
    source: str | None = None
    sha256: str | None = None

    # ---- objective -----------------------------------------------------------------------------
    @property
    def n_terms(self) -> int:
        return len(self.supports)

    @property
    def coeffs(self) -> np.ndarray:
        """c_k = -J_k (E = const + sum c T)."""
        return -np.asarray(self.weights, dtype=np.float64)

    def _spins(self, spins) -> tuple[np.ndarray, bool]:
        s = np.asarray(spins)
        single = s.ndim == 1
        if single:
            s = s[None, :]
        if s.ndim != 2 or s.shape[1] != self.n_spins:
            raise ValueError(f"expected spins of shape (N,) or (B, N) with N={self.n_spins}")
        if not np.isin(s, (-1, 1)).all():
            raise ValueError("spins must be +-1")
        return s.astype(np.int64), single

    def energy(self, spins):
        """Direct evaluation E = const - sum J prod s (float or array)."""
        s, single = self._spins(spins)
        e = np.full(s.shape[0], float(self.const))
        for sup, w in zip(self.supports, self.weights):
            e -= float(w) * np.prod(s[:, list(sup)], axis=1)
        return float(e[0]) if single else e

    # ---- constraints ---------------------------------------------------------------------------
    @property
    def constraint_family(self) -> str:
        kinds = [k for k, v in (("exact_cardinality", self.exact_cardinality), ("chain_precedence", self.chains),
                                ("disjoint_or_cover", self.or_blocks), ("unit_packing", self.amo_blocks),
                                ("contiguity", self.contiguity)) if v]
        if not kinds:
            return "unconstrained"
        return kinds[0] if len(kinds) == 1 else "mixed"

    def is_feasible(self, spins):
        """Independent check of every constraint block (bool or bool array)."""
        s, single = self._spins(spins)
        ok = np.ones(s.shape[0], dtype=bool)
        for spins_, k in self.exact_cardinality:
            ok &= (s[:, list(spins_)] > 0).sum(axis=1) == k
        for ch in self.chains:
            sel = s[:, list(ch)] > 0
            # prefix property: once a position is unselected, all later ones are unselected
            ok &= ~np.any(~sel[:, :-1] & sel[:, 1:], axis=1) if len(ch) > 1 else True
        for spins_, signs in self.or_blocks:
            ok &= np.any(s[:, list(spins_)] == np.asarray(signs)[None, :], axis=1)
        for blk in self.amo_blocks:
            ok &= (s[:, list(blk)] > 0).sum(axis=1) <= 1
        for sp, cyclic in self.contiguity:
            bits = s[:, list(sp)] > 0
            ok &= _contiguous_mask(bits, cyclic)
        return bool(ok[0]) if single else ok

    def enumerate_feasible(self, max_spins: int = 22) -> np.ndarray:
        """All feasible states (int8, |Omega| x N) by exhaustive enumeration (small N only)."""
        if self.n_spins > max_spins:
            raise ValueError(f"enumeration limited to N <= {max_spins}")
        n = self.n_spins
        idx = np.arange(2 ** n, dtype=np.int64)
        bits = (idx[:, None] >> np.arange(n)[None, :]) & 1
        states = (1 - 2 * bits).astype(np.int8)  # bit 0 -> +1
        return states[self.is_feasible(states)]

    # ---- conversion ----------------------------------------------------------------------------
    def to_spin_polynomial(self):
        """``nimbus.problems.base.SpinPolynomial`` (E = const + sum c T) for problem adapters."""
        from nimbus.problems.base import SpinPolynomial

        return SpinPolynomial(n_spins=self.n_spins, supports=tuple(self.supports), coeffs=self.coeffs.copy(),
                              const=float(self.const))

    def to_dict(self, index_base: int = 0) -> dict:
        return spinpoly_to_dict(self, index_base=index_base)


# ---------------------------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------------------------
class _V:
    def __init__(self, name: str):
        self.name = name

    def err(self, where: str, msg: str) -> FormatError:
        return format_error(self.name, None, f"{where}: {msg}")

    def keys(self, obj: Any, allowed: set, required: set, where: str) -> dict:
        if not isinstance(obj, dict):
            raise self.err(where, f"expected a mapping, found {type(obj).__name__}")
        unknown = sorted(set(obj) - allowed)
        if unknown:
            raise self.err(where, f"unknown field(s) {unknown} (allowed: {sorted(allowed)})")
        missing = sorted(required - set(obj))
        if missing:
            raise self.err(where, f"missing required field(s) {missing}")
        return obj

    def integer(self, v: Any, where: str) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise self.err(where, f"expected an integer, found {v!r}")
        return int(v)

    def number(self, v: Any, where: str) -> Fraction:
        if isinstance(v, bool):
            raise self.err(where, f"expected a number, found boolean {v!r}")
        if isinstance(v, int):
            return Fraction(v)
        if isinstance(v, float):
            if not math.isfinite(v):
                raise self.err(where, f"non-finite value {v!r}")
            return Fraction(v)
        if isinstance(v, str):
            try:
                f = Fraction(v.strip())
            except (ValueError, ZeroDivisionError):
                raise self.err(where, f"cannot parse {v!r} as a number or rational 'p/q'") from None
            return f
        raise self.err(where, f"expected a number, found {type(v).__name__}")

    def index_list(self, v: Any, where: str, base: int, n: int, *, nonempty: bool = True) -> tuple:
        if not isinstance(v, list):
            raise self.err(where, f"expected a list of spin indices, found {type(v).__name__}")
        if nonempty and not v:
            raise self.err(where, "empty list")
        out = []
        for k, x in enumerate(v):
            i = self.integer(x, f"{where}[{k}]")
            if i < base or i >= base + n:
                raise self.err(f"{where}[{k}]", f"spin index {i} out of range {base}..{base + n - 1}")
            out.append(i - base)
        if len(set(out)) != len(out):
            dup = next(x for x in out if out.count(x) > 1) + base
            raise self.err(where, f"spin index {dup} repeated")
        return tuple(out)


def _parse_mapping(doc: Any, name: str, *, sha256: str | None = None, source: str | None = None) -> SpinPolyDocument:
    v = _V(name)
    d = v.keys(doc, _TOP_KEYS, {"schema", "n_spins", "spin_convention", "energy_convention", "const", "terms"}, "document")
    if d["schema"] != SCHEMA:
        raise v.err("schema", f"expected {SCHEMA!r}, found {d['schema']!r}")
    n = v.integer(d["n_spins"], "n_spins")
    if n < 1:
        raise v.err("n_spins", "must be >= 1")
    base = v.integer(d.get("index_base", 0), "index_base")
    if base not in (0, 1):
        raise v.err("index_base", "must be 0 or 1")
    if d["spin_convention"] not in SPIN_CONVENTIONS:
        raise v.err("spin_convention", f"must be one of {SPIN_CONVENTIONS}, found {d['spin_convention']!r}")
    conv = d["energy_convention"]
    if conv not in ENERGY_CONVENTIONS:
        raise v.err("energy_convention", f"must be one of {ENERGY_CONVENTIONS}, found {conv!r}")
    const = v.number(d["const"], "const")
    terms = d["terms"]
    if not isinstance(terms, list):
        raise v.err("terms", "expected a list")
    supports = []
    weights = []
    seen: dict[tuple, int] = {}
    for k, t in enumerate(terms):
        where = f"terms[{k}]"
        v.keys(t, {"support", "weight"}, {"support", "weight"}, where)
        sup = tuple(sorted(v.index_list(t["support"], f"{where}.support", base, n)))
        if sup in seen:
            raise v.err(where, f"same support as terms[{seen[sup]}]; merge the two weights into one term")
        seen[sup] = k
        w = v.number(t["weight"], f"{where}.weight")
        supports.append(sup)
        weights.append(w if conv == "minus_J" else -w)

    cons = d.get("constraints") or {}
    v.keys(cons, _CONSTRAINT_KEYS, set(), "constraints")
    card = []
    for k, blk in enumerate(cons.get("exact_cardinality") or []):
        where = f"constraints.exact_cardinality[{k}]"
        v.keys(blk, {"spins", "k"}, {"spins", "k"}, where)
        sp = v.index_list(blk["spins"], f"{where}.spins", base, n)
        kk = v.integer(blk["k"], f"{where}.k")
        if kk < 0 or kk > len(sp):
            raise v.err(f"{where}.k", f"k={kk} outside 0..{len(sp)}")
        card.append((sp, kk))
    chains = []
    raw_chains = cons.get("chains") or []
    if not isinstance(raw_chains, list):
        raise v.err("constraints.chains", "expected a list of chains")
    for k, ch in enumerate(raw_chains):
        chains.append(v.index_list(ch, f"constraints.chains[{k}]", base, n))
    ors = []
    for k, blk in enumerate(cons.get("or_blocks") or []):
        where = f"constraints.or_blocks[{k}]"
        v.keys(blk, {"spins", "signs"}, {"spins", "signs"}, where)
        sp = v.index_list(blk["spins"], f"{where}.spins", base, n)
        sg = blk["signs"]
        if not isinstance(sg, list) or len(sg) != len(sp):
            raise v.err(f"{where}.signs", "expected one sign (+1/-1) per spin")
        signs = tuple(v.integer(x, f"{where}.signs[{j}]") for j, x in enumerate(sg))
        if any(x not in (1, -1) for x in signs):
            raise v.err(f"{where}.signs", "signs must be +1 or -1")
        ors.append((sp, signs))
    amos = []
    raw_amo = cons.get("amo_blocks") or []
    if not isinstance(raw_amo, list):
        raise v.err("constraints.amo_blocks", "expected a list of blocks")
    for k, blk in enumerate(raw_amo):
        amos.append(v.index_list(blk, f"constraints.amo_blocks[{k}]", base, n))
    contig = []
    for k, blk in enumerate(cons.get("contiguity") or []):
        where = f"constraints.contiguity[{k}]"
        v.keys(blk, {"spins", "cyclic"}, {"spins", "cyclic"}, where)
        sp = v.index_list(blk["spins"], f"{where}.spins", base, n)
        cyc = blk["cyclic"]
        if not isinstance(cyc, bool):
            raise v.err(f"{where}.cyclic", "expected true or false")
        if cyc and len(sp) < 3:
            raise v.err(where, "a cyclic contiguity block needs at least 3 spins")
        if not cyc and len(sp) < 1:
            raise v.err(where, "a contiguity block needs at least 1 spin")
        contig.append((sp, bool(cyc)))

    owner: dict[int, str] = {}
    blocks = ([(f"exact_cardinality[{k}]", b[0]) for k, b in enumerate(card)] + [(f"chains[{k}]", b) for k, b in enumerate(chains)]
              + [(f"or_blocks[{k}]", b[0]) for k, b in enumerate(ors)] + [(f"amo_blocks[{k}]", b) for k, b in enumerate(amos)]
              + [(f"contiguity[{k}]", b[0]) for k, b in enumerate(contig)])
    for label, sp in blocks:
        for i in sp:
            if i in owner:
                raise UnsupportedProblem(f"{name}: constraints.{label} and constraints.{owner[i]} share spin {i + base}; "
                                         "constraint blocks must be disjoint")
            owner[i] = label
    if "free" in cons and cons["free"] is not None:
        free = tuple(sorted(v.index_list(cons["free"], "constraints.free", base, n, nonempty=False)))
        clash = [i for i in free if i in owner]
        if clash:
            raise v.err("constraints.free", f"spin {clash[0] + base} is also in constraints.{owner[clash[0]]}")
        if set(free) | set(owner) != set(range(n)):
            missing = sorted(set(range(n)) - set(free) - set(owner))
            raise v.err("constraints.free", f"given explicitly but spins {[i + base for i in missing][:8]} are in no block and not free")
    else:
        free = tuple(i for i in range(n) if i not in owner)

    labels = d.get("labels")
    if labels is not None:
        if not isinstance(labels, list) or len(labels) != n or not all(isinstance(x, str) for x in labels):
            raise v.err("labels", f"expected a list of {n} strings")
        labels = tuple(labels)
    meta = d.get("metadata") or {}
    if not isinstance(meta, dict):
        raise v.err("metadata", "expected a mapping")
    for key in ("name", "description"):
        if key in d and d[key] is not None and not isinstance(d[key], str):
            raise v.err(key, "expected a string")
    return SpinPolyDocument(
        n_spins=n,
        supports=tuple(supports),
        weights=np.array([float(w) for w in weights], dtype=np.float64),
        const=float(const),
        exact_cardinality=tuple(card),
        chains=tuple(chains),
        or_blocks=tuple(ors),
        amo_blocks=tuple(amos),
        contiguity=tuple(contig),
        free=free,
        labels=labels,
        name=d.get("name"),
        description=d.get("description"),
        metadata=dict(meta),
        exact={"const": const, "weights": tuple(weights)},
        source=source,
        sha256=sha256,
    )


def parse_spinpoly(text: str, name: str = "<string>", *, fmt: str = "yaml", sha256: str | None = None,
                   source: str | None = None) -> SpinPolyDocument:
    """Parse schema text. ``fmt`` is ``"yaml"`` or ``"json"``."""
    if fmt == "json":
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            raise format_error(name, exc.lineno, f"invalid JSON: {exc.msg}") from None
    elif fmt == "yaml":
        import yaml

        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            raise format_error(name, mark.line + 1 if mark else None, f"invalid YAML: {exc}") from None
    else:
        raise ValueError("fmt must be 'yaml' or 'json'")
    return _parse_mapping(doc, name, sha256=sha256, source=source)


def read_spinpoly(path: str | os.PathLike) -> SpinPolyDocument:
    src = read_source(path)
    suffix = Path(path).suffix.lower()
    fmt = "json" if suffix == ".json" else "yaml" if suffix in (".yaml", ".yml") else None
    if fmt is None:
        raise format_error(src.name, None, "unknown extension (use .yaml, .yml or .json)")
    return parse_spinpoly(src.text, src.name, fmt=fmt, sha256=src.sha256, source=str(src.path))


# ---------------------------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------------------------
def _num_out(x) -> int | float | str:
    f = Fraction(x) if not isinstance(x, Fraction) else x
    if f.denominator == 1:
        return int(f)
    if f.denominator <= 1024:
        return f"{f.numerator}/{f.denominator}"
    return float(f)


def spinpoly_to_dict(doc: SpinPolyDocument, *, index_base: int = 0) -> dict:
    b = index_base
    exact_w = doc.exact["weights"] if doc.exact and len(doc.exact.get("weights", ())) == doc.n_terms else None
    exact_c = doc.exact["const"] if doc.exact and "const" in doc.exact else None
    out: dict = {"schema": SCHEMA}
    if doc.name:
        out["name"] = doc.name
    if doc.description:
        out["description"] = doc.description
    out.update({
        "n_spins": doc.n_spins,
        "index_base": b,
        "spin_convention": "plus_one_selected",
        "energy_convention": "minus_J",
        "const": _num_out(exact_c if exact_c is not None else Fraction(float(doc.const))),
        "terms": [
            {"support": [i + b for i in sup],
             "weight": _num_out(exact_w[k] if exact_w is not None else Fraction(float(doc.weights[k])))}
            for k, sup in enumerate(doc.supports)
        ],
    })
    cons: dict = {}
    if doc.exact_cardinality:
        cons["exact_cardinality"] = [{"spins": [i + b for i in sp], "k": int(k)} for sp, k in doc.exact_cardinality]
    if doc.chains:
        cons["chains"] = [[i + b for i in ch] for ch in doc.chains]
    if doc.or_blocks:
        cons["or_blocks"] = [{"spins": [i + b for i in sp], "signs": [int(x) for x in sg]} for sp, sg in doc.or_blocks]
    if doc.amo_blocks:
        cons["amo_blocks"] = [[i + b for i in blk] for blk in doc.amo_blocks]
    if doc.contiguity:
        cons["contiguity"] = [{"spins": [i + b for i in sp], "cyclic": bool(cyc)} for sp, cyc in doc.contiguity]
    if cons:
        cons["free"] = [i + b for i in doc.free]
        out["constraints"] = cons
    if doc.labels is not None:
        out["labels"] = list(doc.labels)
    if doc.metadata:
        out["metadata"] = dict(doc.metadata)
    return out


def write_spinpoly(path: str | os.PathLike, doc: SpinPolyDocument, *, index_base: int = 0) -> Path:
    """Write YAML (``.yaml``/``.yml``) or JSON (``.json``) by suffix."""
    p = Path(path)
    data = spinpoly_to_dict(doc, index_base=index_base)
    suffix = p.suffix.lower()
    if suffix == ".json":
        p.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    elif suffix in (".yaml", ".yml"):
        import yaml

        p.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None), encoding="utf-8", newline="\n")
    else:
        raise ValueError("use a .yaml, .yml or .json path")
    return p

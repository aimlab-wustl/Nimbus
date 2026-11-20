"""Generic block-constrained spin-polynomial adapter.

A ``ConstrainedBlocksProblem`` couples an arbitrary objective polynomial (any order, any variables)
with DISJOINT constraint blocks drawn from the following supported families: ``exact_cardinality`` (pair exchanges), ``chain_precedence`` (prefix chains),
``disjoint_or_cover`` (signed-literal OR blocks), ``unit_capacity_packing``/``amo`` (at-most-one
blocks) and ``contiguity`` (consecutive-ones blocks on a path or cycle, builder
:func:`nimbus.transitions.builders.contiguity_block`). Variables in no block are free (single
flips). Two input formats build one:

- the project schema ``nimbus.spinpoly/1`` (:mod:`nimbus.problems.formats.spinpoly`), which has a
  native ``constraints.contiguity`` field;
- the OPB PB24 subset (:mod:`nimbus.problems.formats.opb`), which has **no** native contiguity
  constraint (OPB has no "consecutive ones" relation) -- a contiguity block on an OPB-sourced problem
  can only be added via an explicit *sidecar* list in the config (``problem.contiguity``), using the
  same 1-based variable numbering as the OPB file (``x<k>`` -> sidecar entry spin ``k``). This is
  documented here rather than invented as new OPB syntax.

The runtime builds the transition library by composing :mod:`nimbus.transitions.builders` over the
blocks (never a feasibility filter): ``is_feasible`` re-checks the ORIGINAL constraints
independently of the compiled topology. ``native_objective`` evaluates the OPB/spinpoly objective
directly on the decoded 0/1 assignment ``x = (1+s)/2``, independently of the spin polynomial's own
expansion.
"""

from __future__ import annotations

import itertools

import numpy as np

from nimbus.core.errors import ConfigError, UnsupportedProblem
from nimbus.core.rng import INIT, stream_bits, stream_integers
from nimbus.problems.base import PenaltyFormulation, Problem, SpinPolynomial
from nimbus.transitions import builders as B
from nimbus.transitions.library import TransitionLibrary

PROBLEM_KIND = ("spinpoly", "opb")

__all__ = ["PROBLEM_KIND", "ConstrainedBlocksProblem", "from_config"]

_MAX_EXACT_PENALTY_BLOCK = 18  # Walsh-transform violation polynomial: O(m * 2**m)
_MAX_ENUMERATE_STATES = 1 << 20


# ---------------------------------------------------------------------------------------------
# small shared helpers (no dependency on tests/helpers_core, which is test-only)
# ---------------------------------------------------------------------------------------------
def _all_block_states(m: int) -> np.ndarray:
    idx = np.arange(1 << m, dtype=np.int64)
    bits = (idx[:, None] >> np.arange(m)[None, :]) & 1
    return (1 - 2 * bits).astype(np.int8)


def _contiguous(bits_row: np.ndarray, cyclic: bool) -> bool:
    """``bits_row``: bool array, True where selected. One block, one state."""
    if not bits_row.any():
        return False
    if cyclic:
        if bits_row.all():
            return False
        trans = int(np.count_nonzero(bits_row != np.roll(bits_row, -1)))
        return trans == 2
    idx = np.flatnonzero(bits_row)
    return int(idx[-1] - idx[0] + 1) == idx.size


def _add_polynomials(n_spins: int, parts) -> SpinPolynomial:
    terms: dict[tuple, float] = {}
    const = 0.0
    for poly, scale in parts:
        const += scale * poly.const
        for sup, c in zip(poly.supports, poly.coeffs):
            terms[sup] = terms.get(sup, 0.0) + scale * float(c)
    supports = tuple(sorted(terms))
    coeffs = np.array([terms[s] for s in supports], dtype=np.float64)
    return SpinPolynomial(n_spins=n_spins, supports=supports, coeffs=coeffs, const=const)


def _fwht(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64).copy()
    n = a.shape[0]
    h = 1
    while h < n:
        a = a.reshape(n // (2 * h), 2, h)
        x, y = a[:, 0, :].copy(), a[:, 1, :].copy()
        a[:, 0, :], a[:, 1, :] = x + y, x - y
        a = a.reshape(n)
        h *= 2
    return a


def _exact_block_violation(n_spins: int, block_spins, bad_of_row) -> SpinPolynomial:
    """Exact multilinear penalty polynomial over ``block_spins`` (0/1-free on the rest of the
    problem), zero iff ``bad_of_row(bits_row)`` is False for every row, by Walsh-Hadamard transform
    of the 0/1 "bad" indicator over all ``2**m`` block states (``m = len(block_spins)``)."""
    m = len(block_spins)
    if m > _MAX_EXACT_PENALTY_BLOCK:
        raise ValueError(f"exact violation polynomial limited to blocks of size <= {_MAX_EXACT_PENALTY_BLOCK} "
                         f"(block of {m} spins); use a problem-specific closed form instead")
    states = _all_block_states(m)
    bad = np.array([1.0 if bad_of_row(row > 0) else 0.0 for row in states])
    transform = _fwht(bad) / (1 << m)
    terms: dict[tuple, float] = {}
    const = 0.0
    for s in range(1 << m):
        c = float(transform[s])
        if abs(c) < 1e-12:
            continue
        if s == 0:
            const = c
        else:
            sub = tuple(block_spins[b] for b in range(m) if (s >> b) & 1)
            terms[tuple(sorted(sub))] = c
    supports = tuple(sorted(terms))
    coeffs = np.array([terms[s] for s in supports], dtype=np.float64)
    return SpinPolynomial(n_spins=n_spins, supports=supports, coeffs=coeffs, const=const)


def _cardinality_violation(n_spins: int, block, k: int) -> SpinPolynomial:
    """``(n+ - k)**2`` expanded (same derivation as ``toy_cardinality._cardinality_violation_polynomial``)."""
    block = [int(i) for i in block]
    m = len(block)
    a = m / 2.0 - k
    const = a * a + m / 4.0
    terms: dict[tuple, float] = {}
    if a != 0.0:
        for i in block:
            terms[(i,)] = a
    for i, j in itertools.combinations(sorted(block), 2):
        terms[(i, j)] = 0.5
    supports = tuple(sorted(terms))
    coeffs = np.array([terms[s] for s in supports], dtype=np.float64)
    return SpinPolynomial(n_spins=n_spins, supports=supports, coeffs=coeffs, const=const)


def _chain_violation(n_spins: int, chain) -> SpinPolynomial:
    """Sum over consecutive pairs of the "later selected, earlier not" indicator
    ``(1 - s_t)(1 + s_{t+1}) / 4``, zero iff the chain is prefix-closed."""
    c = [int(i) for i in chain]
    terms: dict[tuple, float] = {}
    const = 0.0
    for t in range(len(c) - 1):
        a, b = c[t], c[t + 1]
        const += 0.25
        terms[(b,)] = terms.get((b,), 0.0) + 0.25
        terms[(a,)] = terms.get((a,), 0.0) - 0.25
        key = tuple(sorted((a, b)))
        terms[key] = terms.get(key, 0.0) - 0.25
    supports = tuple(sorted(terms))
    coeffs = np.array([terms[s] for s in supports], dtype=np.float64)
    return SpinPolynomial(n_spins=n_spins, supports=supports, coeffs=coeffs, const=const)


def _amo_violation(n_spins: int, block) -> SpinPolynomial:
    """Sum over pairs of the "both selected" indicator ``(1+s_i)(1+s_j)/4``, zero iff at most one
    spin of the block is +1."""
    b = [int(i) for i in block]
    terms: dict[tuple, float] = {}
    const = 0.0
    for i, j in itertools.combinations(sorted(b), 2):
        const += 0.25
        terms[(i,)] = terms.get((i,), 0.0) + 0.25
        terms[(j,)] = terms.get((j,), 0.0) + 0.25
        terms[(i, j)] = terms.get((i, j), 0.0) + 0.25
    supports = tuple(sorted(terms))
    coeffs = np.array([terms[s] for s in supports], dtype=np.float64)
    return SpinPolynomial(n_spins=n_spins, supports=supports, coeffs=coeffs, const=const)


# ---------------------------------------------------------------------------------------------
# problem
# ---------------------------------------------------------------------------------------------
class ConstrainedBlocksProblem(Problem):
    """Objective polynomial + disjoint constraint blocks. See module docstring."""

    def __init__(self, n_spins: int, supports, coeffs, const: float, *, exact_cardinality=(), chains=(),
                or_blocks=(), amo_blocks=(), contiguity=(), free=None, name: str = "constrained_blocks",
                source_format: str = "spinpoly", labels=None, native_objective_fn=None,
                native_objective_label: str = "objective"):
        self.n_spins = int(n_spins)
        self.name = name
        self.source_format = source_format
        self.labels = tuple(labels) if labels is not None else None
        self._native_objective_fn = native_objective_fn
        self._native_objective_label = native_objective_label
        self._poly = SpinPolynomial(n_spins=self.n_spins, supports=tuple(supports),
                                    coeffs=np.asarray(coeffs, dtype=np.float64), const=float(const))
        self.exact_cardinality = tuple((tuple(sp), int(k)) for sp, k in exact_cardinality)
        self.chains = tuple(tuple(ch) for ch in chains)
        self.or_blocks = tuple((tuple(sp), tuple(sg)) for sp, sg in or_blocks)
        self.amo_blocks = tuple(tuple(b) for b in amo_blocks)
        self.contiguity = tuple((tuple(sp), bool(cyc)) for sp, cyc in contiguity)
        owner: dict[int, str] = {}
        blocks = ([(f"exact_cardinality[{k}]", sp) for k, (sp, _) in enumerate(self.exact_cardinality)]
                  + [(f"chains[{k}]", ch) for k, ch in enumerate(self.chains)]
                  + [(f"or_blocks[{k}]", sp) for k, (sp, _) in enumerate(self.or_blocks)]
                  + [(f"amo_blocks[{k}]", b) for k, b in enumerate(self.amo_blocks)]
                  + [(f"contiguity[{k}]", sp) for k, (sp, _) in enumerate(self.contiguity)])
        for label, sp in blocks:
            for i in sp:
                if i in owner:
                    raise UnsupportedProblem(f"constrained_blocks: spin {i} is in both {owner[i]} and {label} "
                                             "(constraint blocks must be disjoint)")
                owner[i] = label
        self.free = tuple(sorted(free)) if free is not None else tuple(i for i in range(self.n_spins) if i not in owner)
        clash = [i for i in self.free if i in owner]
        if clash:
            raise UnsupportedProblem(f"constrained_blocks: spin {clash[0]} is both free and in {owner[clash[0]]}")
        missing = sorted(set(range(self.n_spins)) - set(owner) - set(self.free))
        if missing:
            raise UnsupportedProblem(f"constrained_blocks: spins {missing[:8]} are in no block and not listed free")
        present = []
        if self.exact_cardinality:
            present.append("exact_cardinality")
        if self.chains:
            present.append("chain_precedence")
        if self.or_blocks:
            present.append("disjoint_or_cover")
        if self.amo_blocks:
            present.append("unit_capacity_packing")
        if self.contiguity:
            present.append("contiguity")
        self.family = "unconstrained" if not present else (present[0] if len(present) == 1 else "mixed_blocks")

    # ---- objective and constraints -------------------------------------------------------------
    def polynomial(self, instance: int = 0) -> SpinPolynomial:
        if instance != 0:
            raise ValueError("constrained_blocks has a single instance")
        return self._poly

    def is_feasible(self, spins) -> bool:
        s = np.asarray(spins)
        for sp, k in self.exact_cardinality:
            if int(np.count_nonzero(s[list(sp)] == 1)) != k:
                return False
        for ch in self.chains:
            sel = s[list(ch)] > 0
            if np.any(~sel[:-1] & sel[1:]):
                return False
        for sp, signs in self.or_blocks:
            if not np.any(s[list(sp)] == np.asarray(signs)):
                return False
        for b in self.amo_blocks:
            if int(np.count_nonzero(s[list(b)] == 1)) > 1:
                return False
        for sp, cyclic in self.contiguity:
            if not _contiguous(s[list(sp)] > 0, cyclic):
                return False
        return True

    # ---- independent evaluation -------------------------------------------------------------
    def native_objective(self, spins, instance: int = 0) -> float:
        """The OPB/spinpoly objective evaluated directly on the decoded 0/1 assignment, independent
        of the spin-polynomial expansion. Falls back to the polynomial's own evaluation when no
        independent evaluator was supplied (spinpoly input has no other given representation)."""
        if self._native_objective_fn is not None:
            x = ((1 + np.asarray(spins, dtype=np.int64)) // 2)
            return float(self._native_objective_fn(x))
        return self.energy(spins, instance)

    def decode(self, spins) -> dict:
        s = np.asarray(spins, dtype=np.int64)
        x = (1 + s) // 2
        out = {"x": x.tolist()}
        if self.labels is not None:
            out["assignment"] = {lbl: int(v) for lbl, v in zip(self.labels, x)}
        return out

    # ---- initialization, transitions ---------------------------------------------------------
    def random_feasible(self, seed: int, n: int, instance: int = 0) -> np.ndarray:
        out = np.full((int(n), self.n_spins), -1, dtype=np.int8)
        for m in range(int(n)):
            tag = 0
            for sp, k in self.exact_cardinality:
                combos = list(itertools.combinations(sp, k))
                if len(combos) > 200000:
                    raise UnsupportedProblem("random_feasible: exact_cardinality block too large to sample by "
                                             "enumeration")
                idx = int(stream_integers(seed, INIT, 1, len(combos), c1=tag, c2=int(instance), c3=m)[0])
                out[m, list(combos[idx])] = 1
                tag += 1
            for ch in self.chains:
                length = int(stream_integers(seed, INIT, 1, len(ch) + 1, c1=tag, c2=int(instance), c3=m)[0])
                out[m, list(ch[:length])] = 1
                tag += 1
            for sp, signs in self.or_blocks:
                forced = int(stream_integers(seed, INIT, 1, len(sp), c1=tag, c2=int(instance), c3=m)[0])
                tag += 1
                bits = stream_bits(seed, INIT, len(sp), c1=tag, c2=int(instance), c3=m)
                tag += 1
                vals = (1 - 2 * bits.astype(np.int16)).astype(np.int8)
                vals[forced] = signs[forced]
                out[m, list(sp)] = vals
            for b in self.amo_blocks:
                choice = int(stream_integers(seed, INIT, 1, len(b) + 1, c1=tag, c2=int(instance), c3=m)[0])
                tag += 1
                if choice < len(b):
                    out[m, b[choice]] = 1
            for sp, cyclic in self.contiguity:
                mm = len(sp)
                max_len = mm - 1 if cyclic else mm
                length = 1 + int(stream_integers(seed, INIT, 1, max_len, c1=tag, c2=int(instance), c3=m)[0])
                tag += 1
                n_starts = mm if cyclic else mm - length + 1
                start = int(stream_integers(seed, INIT, 1, n_starts, c1=tag, c2=int(instance), c3=m)[0])
                tag += 1
                if cyclic:
                    sel = [sp[(start + t) % mm] for t in range(length)]
                else:
                    sel = sp[start:start + length]
                out[m, list(sel)] = 1
            if self.free:
                bits = stream_bits(seed, INIT, len(self.free), c1=tag, c2=int(instance), c3=m)
                out[m, list(self.free)] = (1 - 2 * bits.astype(np.int16)).astype(np.int8)
        return out

    def library_names(self) -> list:
        return ["default"]

    def transition_library(self, name: str = "default", *, cardinality_pattern: str = "complete",
                           or_tree: str = "path", **options) -> TransitionLibrary:
        if name != "default":
            raise ValueError(f"unknown constrained_blocks library {name!r}; known: ['default']")
        if options:
            raise ValueError(f"unknown transition_library option(s) {sorted(options)}")
        n = self.n_spins
        libs = []
        for sp, _k in self.exact_cardinality:
            if len(sp) > 1:
                libs.append(B.pair_exchanges(sp, pattern=cardinality_pattern, n_spins=n))
        for ch in self.chains:
            if len(ch) > 0:
                libs.append(B.chain_precedence(ch, n_spins=n))
        for sp, signs in self.or_blocks:
            libs.append(B.or_block(list(zip(sp, signs)), tree=or_tree, n_spins=n))
        for b in self.amo_blocks:
            libs.append(B.amo_block(b, n_spins=n))
        for sp, cyclic in self.contiguity:
            libs.append(B.contiguity_block(sp, cyclic=cyclic, n_spins=n))
        if self.free:
            libs.append(B.single_flips(n, spins=self.free))
        libs = [lib for lib in libs if lib.n_groups > 0]
        if not libs:
            return B.single_flips(n, spins=())
        return TransitionLibrary.concat(*libs, name="constrained_blocks", require_disjoint=True)

    def enumerate_feasible(self) -> np.ndarray | None:
        n = self.n_spins
        pieces = []  # (indices, states)
        total = 1
        for sp, k in self.exact_cardinality:
            m = len(sp)
            combos = list(itertools.combinations(range(m), k))
            states = np.full((len(combos), m), -1, dtype=np.int8)
            for row, combo in enumerate(combos):
                states[row, list(combo)] = 1
            pieces.append((sp, states))
            total *= len(combos)
        for ch in self.chains:
            m = len(ch)
            states = np.array([[1] * t + [-1] * (m - t) for t in range(m + 1)], dtype=np.int8).reshape(-1, m)
            pieces.append((ch, states))
            total *= states.shape[0]
        for sp, signs in self.or_blocks:
            m = len(sp)
            all_s = _all_block_states(m)
            ok = np.any(all_s * np.asarray(signs)[None, :] == 1, axis=1)
            pieces.append((sp, all_s[ok]))
            total *= int(ok.sum())
        for b in self.amo_blocks:
            m = len(b)
            all_s = _all_block_states(m)
            ok = np.count_nonzero(all_s == 1, axis=1) <= 1
            pieces.append((b, all_s[ok]))
            total *= int(ok.sum())
        for sp, cyclic in self.contiguity:
            m = len(sp)
            all_s = _all_block_states(m)
            ok = np.array([_contiguous(row > 0, cyclic) for row in all_s])
            pieces.append((sp, all_s[ok]))
            total *= int(ok.sum())
        if self.free:
            all_f = _all_block_states(len(self.free))
            pieces.append((self.free, all_f))
            total *= all_f.shape[0]
        if n > 24 or total > _MAX_ENUMERATE_STATES:
            return None
        out = np.full((1, n), -1, dtype=np.int8)
        for idxs, piece in pieces:
            reps = piece.shape[0]
            out = np.repeat(out, reps, axis=0)
            tiled = np.tile(piece, (out.shape[0] // reps, 1))
            out[:, list(idxs)] = tiled
        return out

    # ---- baselines -----------------------------------------------------------------------------
    def penalty_formulation(self, instance: int = 0, *, lam: float = 6.0, **strengths) -> PenaltyFormulation:
        if strengths:
            raise ValueError(f"unknown penalty strength keys: {sorted(strengths)}")
        objective = self.polynomial(instance)
        parts = []
        for sp, k in self.exact_cardinality:
            parts.append((_cardinality_violation(self.n_spins, sp, k), 1.0))
        for ch in self.chains:
            parts.append((_chain_violation(self.n_spins, ch), 1.0))
        for sp, signs in self.or_blocks:
            def bad(row, sp=sp, signs=signs):
                vals = np.where(row, 1, -1)
                return not np.any(vals == np.asarray(signs))
            parts.append((_exact_block_violation(self.n_spins, sp, bad), 1.0))
        for b in self.amo_blocks:
            parts.append((_amo_violation(self.n_spins, b), 1.0))
        for sp, cyclic in self.contiguity:
            def bad(row, cyclic=cyclic):
                return not _contiguous(row, cyclic)
            parts.append((_exact_block_violation(self.n_spins, sp, bad), 1.0))
        if parts:
            violation = _add_polynomials(self.n_spins, parts)
        else:
            violation = SpinPolynomial(n_spins=self.n_spins, supports=(), coeffs=np.zeros(0), const=0.0)
        run = _add_polynomials(self.n_spins, [(objective, 1.0), (violation, float(lam))])
        return PenaltyFormulation(run=run, objective=objective, violation=violation, n_physical=self.n_spins,
                                  ledger={"lambda": float(lam), "blocks": {
                                      "exact_cardinality": len(self.exact_cardinality), "chains": len(self.chains),
                                      "or_blocks": len(self.or_blocks), "amo_blocks": len(self.amo_blocks),
                                      "contiguity": len(self.contiguity)}})

    def metadata(self):
        md = super().metadata()
        md.update(source_format=self.source_format, free=len(self.free))
        return md


# ---------------------------------------------------------------------------------------------
# from_config: spinpoly / OPB (+ optional contiguity sidecar for OPB)
# ---------------------------------------------------------------------------------------------
def _resolve(base_dir, value: str):
    from pathlib import Path

    if str(value).startswith("data:"):
        from nimbus.experiments.paths import data_root

        return data_root() / str(value)[len("data:"):]
    p = Path(value)
    return p if p.is_absolute() else Path(base_dir) / p


def _from_spinpoly_doc(doc, *, name: str) -> ConstrainedBlocksProblem:
    return ConstrainedBlocksProblem(
        n_spins=doc.n_spins, supports=doc.supports, coeffs=doc.coeffs, const=doc.const,
        exact_cardinality=doc.exact_cardinality, chains=doc.chains, or_blocks=doc.or_blocks,
        amo_blocks=doc.amo_blocks, contiguity=doc.contiguity, free=doc.free, name=name,
        source_format="spinpoly", labels=doc.labels)


_SPINPOLY_TOP = {"kind", "path", "name"}
_OPB_TOP = {"kind", "path", "name", "contiguity"}
_SIDECAR_KEYS = {"spins", "cyclic"}


def from_config(cfg: dict, base_dir) -> ConstrainedBlocksProblem:
    """``cfg`` fields: ``kind: "spinpoly"`` with ``path`` (a ``nimbus.spinpoly/1`` file); or
    ``kind: "opb"`` with ``path`` (an OPB file) and optional ``contiguity`` (a sidecar list
    ``[{spins: [<1-based x-index>, ...], cyclic: <bool>}]`` -- the ONLY way to attach a contiguity
    block to an OPB-sourced problem, since OPB itself has no such constraint). Unknown fields raise
    ``ConfigError``."""
    cfg = dict(cfg)
    kind = cfg.pop("kind", None)
    if kind not in PROBLEM_KIND:
        raise ConfigError(f"constrained_blocks.from_config: expected kind in {PROBLEM_KIND}, got {kind!r}")
    if "path" not in cfg:
        raise ConfigError(f"constrained_blocks ({kind}): 'path' is required")
    path = _resolve(base_dir, str(cfg.pop("path")))
    name = str(cfg.pop("name", kind))
    if kind == "spinpoly":
        unknown = sorted(set(cfg) - (_SPINPOLY_TOP - {"kind", "path", "name"}))
        if unknown:
            raise ConfigError(f"constrained_blocks (spinpoly): unknown field(s) {unknown}")
        from nimbus.problems.formats.spinpoly import read_spinpoly

        doc = read_spinpoly(path)
        return _from_spinpoly_doc(doc, name=name)

    # kind == "opb"
    contiguity_cfg = cfg.pop("contiguity", None)
    if cfg:
        raise ConfigError(f"constrained_blocks (opb): unknown field(s) {sorted(cfg)}")
    from nimbus.problems.formats.opb import objective_spin_terms, read_opb, recognize_constraints

    model = read_opb(path)
    rec = recognize_constraints(model)
    terms, const = objective_spin_terms(model)
    supports = tuple(terms)
    coeffs = np.array([float(c) for c in terms.values()], dtype=np.float64)

    contiguity = []
    if contiguity_cfg is not None:
        if not isinstance(contiguity_cfg, list):
            raise ConfigError("constrained_blocks (opb): 'contiguity' sidecar must be a list")
        for k, blk in enumerate(contiguity_cfg):
            if not isinstance(blk, dict) or set(blk) != _SIDECAR_KEYS:
                raise ConfigError(f"constrained_blocks (opb): contiguity[{k}] must have exactly {_SIDECAR_KEYS}")
            sp = tuple(int(v) - 1 for v in blk["spins"])  # x<k> -> spin k-1, same convention as recognize_constraints
            if any(i < 0 or i >= model.n_vars for i in sp):
                raise ConfigError(f"constrained_blocks (opb): contiguity[{k}] spin index out of range")
            cyc = bool(blk["cyclic"])
            if cyc and len(sp) < 3:
                raise ConfigError(f"constrained_blocks (opb): contiguity[{k}] cyclic block needs >= 3 spins")
            contiguity.append((sp, cyc))

    free = [i for i in rec.free if not any(i in sp for sp, _ in contiguity)]
    labels = tuple(f"x{i + 1}" for i in range(model.n_vars))

    def native_objective_fn(x):
        return model.objective_value(x)

    return ConstrainedBlocksProblem(
        n_spins=model.n_vars, supports=supports, coeffs=coeffs, const=float(const),
        exact_cardinality=rec.exact_cardinality, chains=rec.chains, or_blocks=rec.or_blocks,
        amo_blocks=rec.amo_blocks, contiguity=contiguity, free=free, name=name, source_format="opb",
        labels=labels, native_objective_fn=native_objective_fn)

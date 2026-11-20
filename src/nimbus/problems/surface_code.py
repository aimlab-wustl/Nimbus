"""Surface-code (CSS) decoding problem adapter, code-capacity noise (affine-parity family, fixed switches).

Rotated planar surface code, ``d x d`` data qubits (qubit ``(r, c)`` -> index ``r*d + c``), ``d^2 - 1``
stabilizers (weight 4 in the bulk, 2 on the boundary), one logical qubit, distance ``d``.

Spins are error bits, ``s = 1 - 2 e``:

* ``noise="independent"`` decodes ONE sector (``sector="X"`` by default: X errors, detected by the Z checks
  ``H_Z``): ``n = d^2`` spins, feasible set ``H_Z e = syndrome`` (an affine-parity coset), objective
  ``-log P(e) = w |e| + const`` with ``w = log((1 - p)/p)`` (unary clauses; weights depend on ``p`` only).
* ``noise="depolarizing"``: ``2 n`` spins, ``x_i`` (spin ``i``) and ``z_i`` (spin ``n + i``); P(X)=P(Y)=P(Z)=p/3.
  ``-log P = w3 * #{i : x_i or z_i} + const`` with ``w3 = log(3(1 - p)/p)``; ``x or z = 3/4 - (a + b)/4 - ab/4``
  in spins ``a = 1 - 2x``, ``b = 1 - 2z``, so each qubit has two unary clauses and one pair clause ``a_i b_i``,
  all of weight ``J = w3/4``. Feasible set: ``H_Z x = s_Z`` and ``H_X z = s_X``.

Fixed transitions (``FixedMask``, no gates, constant switches) are the stabilizer generators of the OPPOSITE type
(they commute with the checks, so they preserve the syndrome) and, optionally, the logical operators ("shortcuts":
without them the classes of the logical qubit are unreachable). Library names:

* ``stabilizers``          stabilizer generators only (cannot change the logical class);
* ``stabilizers_logical``  stabilizers + one logical representative per type (default);
* ``stabilizers_logical_all``  stabilizers + all ``d`` straight-line logical representatives per type.

Success is a LOGICAL error rate: with true error ``e`` and correction ``c``, the residual ``e + c`` has
zero syndrome and is a logical failure iff it anticommutes with a logical operator of the other type
(:meth:`SurfaceCodeProblem.logical_failure`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nimbus.core import gf2
from nimbus.core import rng as _rng
from nimbus.core.errors import ConfigError
from nimbus.problems import channels
from nimbus.problems.base import Problem, SpinPolynomial
from nimbus.problems.formats.alist import ParityCheckMatrix
from nimbus.transitions import builders as B
from nimbus.transitions.library import TransitionLibrary

PROBLEM_KIND = "surface_code"

__all__ = ["PROBLEM_KIND", "rotated_surface_code", "SurfaceCodeProblem", "from_config"]

_LIBRARIES = ("stabilizers", "stabilizers_logical", "stabilizers_logical_all")


def rotated_surface_code(d: int) -> dict:
    """Rotated planar surface code. Returns dict(n, HX, HZ (uint8 dense), lx_all, lz_all (lists of qubit-index
    tuples: the straight-line logical representatives of weight d, all stabilizer-equivalent per type)).

    ``lx_all`` are X-type logicals (commute with every Z check, anticommute with ``lz_all``)."""
    if d < 2:
        raise ValueError("d >= 2")
    n = d * d
    xs, zs = [], []
    for i in range(d + 1):
        for j in range(d + 1):
            qs = [r * d + c for r in (i - 1, i) for c in (j - 1, j) if 0 <= r < d and 0 <= c < d]
            if len(qs) == 4:
                (xs if (i + j) % 2 == 0 else zs).append(qs)
            elif len(qs) == 2:
                horizontal_edge = i in (0, d)  # top/bottom boundary plaquettes: X type; left/right: Z type
                if horizontal_edge and (i + j) % 2 == 0:
                    xs.append(qs)
                elif (not horizontal_edge) and (i + j) % 2 == 1:
                    zs.append(qs)

    def dense(rows):
        m = np.zeros((len(rows), n), dtype=np.uint8)
        for k, r in enumerate(rows):
            m[k, r] = 1
        return m

    HX, HZ = dense(xs), dense(zs)
    if (HX.astype(int) @ HZ.T.astype(int) % 2).any():
        raise AssertionError("X and Z stabilizers do not commute")
    if len(xs) + len(zs) != n - 1:
        raise AssertionError(f"expected {n - 1} stabilizers, got {len(xs) + len(zs)}")
    lines = [tuple(r * d + c for c in range(d)) for r in range(d)] + [tuple(r * d + c for r in range(d))
                                                                       for c in range(d)]
    def ok(v, H):
        return not (H.astype(int) @ v % 2).any()

    def vec(ln):
        v = np.zeros(n, dtype=np.uint8)
        v[list(ln)] = 1
        return v

    lx = [ln for ln in lines if ok(vec(ln), HZ)]
    lz = [ln for ln in lines if ok(vec(ln), HX)]
    if not lx or not lz:
        raise AssertionError("no straight-line logical found")
    # all representatives of a type share one orientation (a row line and a column line differ by a
    # logical of the other kind); keep the orientation whose members anticommute with the other type
    def same_orient(lst, first):
        return [l for l in lst if (l[1] - l[0] == 1) == (first[1] - first[0] == 1)]

    lx = same_orient(lx, lx[0])
    lz = [l for l in lz if int(vec(lx[0]) @ vec(l)) % 2 == 1]
    lz = same_orient(lz, lz[0]) if lz else lz
    if not lz:
        raise AssertionError("logical X and Z do not anticommute")
    return {"n": n, "d": d, "HX": HX, "HZ": HZ, "lx_all": lx, "lz_all": lz}


class SurfaceCodeProblem(Problem):
    """Min-weight / min -log P decoding of one surface-code syndrome class (see module docstring)."""

    family = "affine_parity"

    def __init__(self, d: int, p: float, *, noise: str = "independent", sector: str = "X", syndrome=None,
                 library: str = "stabilizers_logical", name: str | None = None):
        if noise not in ("independent", "depolarizing"):
            raise ConfigError("noise must be 'independent' or 'depolarizing'")
        if sector not in ("X", "Z"):
            raise ConfigError("sector must be 'X' or 'Z'")
        if not 0.0 < p < 0.5 or (noise == "depolarizing" and not 0.0 < p < 0.75):
            raise ConfigError(f"physical error rate p={p} out of range")
        if library not in _LIBRARIES:
            raise ConfigError(f"unknown surface_code library {library!r}; choose from {_LIBRARIES}")
        self.d, self.p, self.noise, self.sector = int(d), float(p), noise, sector
        self.code = rotated_surface_code(self.d)
        self.nq = self.code["n"]
        self.n_spins = self.nq if noise == "independent" else 2 * self.nq
        self.n_instances = 1
        self.name = name or f"surface_code_d{d}_{noise}"
        self._library_name = library
        if noise == "independent":
            if sector == "X":
                self._checks = [(self.code["HZ"], 0, self.code["HX"], self.code["lx_all"], self.code["lz_all"])]
            else:
                self._checks = [(self.code["HX"], 0, self.code["HZ"], self.code["lz_all"], self.code["lx_all"])]
        else:  # block 0: x bits (checked by HZ, moved by HX, logical X); block 1: z bits (checked by HX, moved by HZ)
            self._checks = [(self.code["HZ"], 0, self.code["HX"], self.code["lx_all"], self.code["lz_all"]),
                            (self.code["HX"], self.nq, self.code["HZ"], self.code["lz_all"], self.code["lx_all"])]
        self.n_checks = [c[0].shape[0] for c in self._checks]
        self._pcm = [ParityCheckMatrix.from_dense(c[0]) for c in self._checks]
        m = sum(self.n_checks)
        if syndrome is None:
            self._syndrome = np.zeros(m, dtype=np.uint8)
        else:
            s = np.asarray(syndrome, dtype=np.int64)
            if s.shape != (m,) or not np.isin(s, (0, 1)).all():
                raise ConfigError(f"syndrome must be a 0/1 vector of length {m}")
            self._syndrome = s.astype(np.uint8)
        self._masks = None
        # weights
        if noise == "independent":
            self.w = math.log((1.0 - p) / p)
        else:
            self.w = math.log(3.0 * (1.0 - p) / p)

    # ---- bookkeeping -------------------------------------------------------------------------
    @property
    def syndrome_length(self) -> int:
        return int(sum(self.n_checks))

    def _split_syndrome(self, syndrome):
        s = np.asarray(syndrome, dtype=np.uint8)
        out, o = [], 0
        for m in self.n_checks:
            out.append(s[..., o:o + m])
            o += m
        return out

    def syndrome_of(self, bits) -> np.ndarray:
        """Syndrome (concatenated over blocks) of error bits (``n_spins`` long, or batch x n_spins)."""
        b = np.asarray(bits, dtype=np.int64)
        outs = []
        for k, (Hc, off, *_rest) in enumerate(self._checks):
            outs.append((b[..., off:off + self.nq] @ Hc.T.astype(np.int64)) % 2)
        return np.concatenate(outs, axis=-1).astype(np.uint8)

    # ---- objective ---------------------------------------------------------------------------
    def polynomial(self, instance: int = 0) -> SpinPolynomial:
        n, w = self.nq, self.w
        if self.noise == "independent":
            sup = [(i,) for i in range(n)]
            co = [-w / 2.0] * n
            const = n * w / 2.0
        else:
            sup = [(i,) for i in range(n)] + [(n + i,) for i in range(n)] + [(i, n + i) for i in range(n)]
            co = [-w / 4.0] * (3 * n)
            const = 3.0 * n * w / 4.0
        return SpinPolynomial(n_spins=self.n_spins, supports=tuple(sup), coeffs=np.asarray(co, np.float64),
                              const=const)

    def native_objective(self, spins, instance: int = 0) -> float:
        """-log P(error pattern) computed directly from the channel (not from the polynomial)."""
        e = ((1 - np.asarray(spins, dtype=np.int64)) // 2)
        p = self.p
        if self.noise == "independent":
            k = int(e.sum())
            return -(k * math.log(p) + (self.nq - k) * math.log(1 - p))
        x, z = e[: self.nq], e[self.nq:]
        nonid = int(np.sum(x | z))
        return -(nonid * math.log(p / 3.0) + (self.nq - nonid) * math.log(1 - p))

    def energy_to_native(self, energy: float, instance: int = 0) -> float:
        """polynomial energy -> -log P: add the identity-qubit constant ``n (-log(1-p))``."""
        return float(energy) - self.nq * math.log(1 - self.p)

    # ---- feasibility / decode ------------------------------------------------------------------
    def is_feasible(self, spins, syndrome=None) -> bool:
        s = np.asarray(spins, dtype=np.int64)
        y = ((1 - s) // 2).astype(np.uint8)
        target = self._syndrome if syndrome is None else np.asarray(syndrome, np.uint8)
        return bool(np.array_equal(self.syndrome_of(y), target))

    def decode(self, spins) -> dict:
        y = ((1 - np.asarray(spins, dtype=np.int64)) // 2).astype(np.uint8)
        synd = self.syndrome_of(y)
        return {"bits": y, "syndrome": synd, "feasible": bool(np.array_equal(synd, self._syndrome))}

    def _measure_matrix(self) -> list:
        """Per block, the logical operator(s) of the other type used to test the residual's class."""
        out = []
        for (_Hc, off, _Hm, _lm, lmeas) in self._checks:
            v = np.zeros(self.nq, dtype=np.uint8)
            v[list(lmeas[0])] = 1
            out.append((off, v))
        return out

    def logical_failure(self, correction_bits, error_bits) -> np.ndarray:
        """Boolean per row: residual ``e + c`` is a nontrivial logical (assumes zero residual syndrome; a
        residual with nonzero syndrome counts as a failure). Inputs: bits, ``n_spins`` long (or batch)."""
        c = np.atleast_2d(np.asarray(correction_bits, dtype=np.int64))
        e = np.atleast_2d(np.asarray(error_bits, dtype=np.int64))
        r = (c + e) % 2
        fail = self.syndrome_of(r).any(axis=-1)
        for off, v in self._measure_matrix():
            fail = fail | ((r[:, off:off + self.nq] @ v.astype(np.int64)) % 2).astype(bool)
        return fail

    # ---- states -----------------------------------------------------------------------------------
    def particular_bits(self, syndrome=None) -> np.ndarray:
        """A fast particular solution y (uint8[n_spins]) of the syndrome (NOT the true error)."""
        sy = self._syndrome if syndrome is None else np.asarray(syndrome, np.uint8)
        y = np.zeros(self.n_spins, dtype=np.uint8)
        for (pcm, (Hc, off, *_r), sblk) in zip(self._pcm, self._checks, self._split_syndrome(sy)):
            y[off:off + self.nq] = channels.particular_solution(pcm, sblk)
        return y

    def particular_spins(self, syndromes) -> np.ndarray:
        """int8 start spin states, one per syndrome row."""
        sy = np.atleast_2d(np.asarray(syndromes, np.uint8))
        return np.stack([(1 - 2 * self.particular_bits(s).astype(np.int16)).astype(np.int8) for s in sy])

    def _kernel_masks(self) -> list:
        """Null-space basis of the check matrices as index tuples over the full spin vector."""
        if self._masks is None:
            ms = []
            for (Hc, off, *_r), pcm in zip(self._checks, self._pcm):
                ms += [tuple(off + i for i in m) for m in channels.generator_masks(pcm)]
            self._masks = ms
        return self._masks

    def random_feasible(self, seed: int, n: int, instance: int = 0) -> np.ndarray:
        masks = self._kernel_masks()
        part = self.particular_bits()
        bits = np.stack([_rng.stream_bits(seed, _rng.INIT, len(masks), c1=i, c2=instance) for i in range(n)])
        y = channels.encode(masks, self.n_spins, bits, part)
        return (1 - 2 * y.astype(np.int16)).astype(np.int8)

    def enumerate_feasible(self):
        if len(self._kernel_masks()) > 16:
            return None
        masks = self._kernel_masks()
        k = len(masks)
        info = ((np.arange(1 << k, dtype=np.uint32)[:, None] >> np.arange(k, dtype=np.uint32)[None, :]) & 1)
        y = channels.encode(masks, self.n_spins, info.astype(np.uint8), self.particular_bits())
        return (1 - 2 * y.astype(np.int16)).astype(np.int8)

    # ---- transitions ----------------------------------------------------------------------------
    def library_names(self) -> list:
        return list(_LIBRARIES)

    def transition_library(self, name: str | None = None, **options) -> TransitionLibrary:
        name = name or self._library_name
        if name not in _LIBRARIES:
            raise ConfigError(f"unknown surface_code library {name!r}; choose from {_LIBRARIES}")
        if options:
            raise ConfigError(f"surface_code library: unknown option(s) {sorted(options)}")
        masks, labels = [], []
        for (_Hc, off, Hmove, lmove, _lmeas) in self._checks:
            for k in range(Hmove.shape[0]):
                masks.append(tuple(int(off + i) for i in np.flatnonzero(Hmove[k])))
                labels.append(f"stab@{off}#{k}")
        if name != "stabilizers":
            for (_Hc, off, _Hm, lmove, _lmeas) in self._checks:
                reps = lmove if name == "stabilizers_logical_all" else lmove[:1]
                for k, ln in enumerate(reps):
                    masks.append(tuple(off + i for i in ln))
                    labels.append(f"logical@{off}#{k}")
        return B.fixed_masks(masks, self.n_spins, name=name, template="surface_code_" + name, labels=labels,
                             params={"d": self.d, "noise": self.noise, "n_stabilizer_moves": int(
                                 sum(c[2].shape[0] for c in self._checks)), "weights_max": max(len(m) for m in masks)})

    def metadata(self) -> dict[str, Any]:
        md = super().metadata()
        md.update(d=self.d, p=self.p, noise=self.noise, sector=self.sector, library=self._library_name, w=self.w)
        return md


_TOP = {"kind", "d", "p", "noise", "sector", "syndrome", "library", "name"}


def from_config(cfg: dict, base_dir) -> SurfaceCodeProblem:
    unknown = sorted(set(cfg) - _TOP)
    if unknown:
        raise ConfigError(f"surface_code problem: unknown field(s) {unknown}")
    for k in ("d", "p"):
        if k not in cfg:
            raise ConfigError(f"surface_code problem: {k!r} is required")
    return SurfaceCodeProblem(int(cfg["d"]), float(cfg["p"]), noise=cfg.get("noise", "independent"),
                              sector=cfg.get("sector", "X"), syndrome=cfg.get("syndrome"),
                              library=cfg.get("library", "stabilizers_logical"), name=cfg.get("name"))

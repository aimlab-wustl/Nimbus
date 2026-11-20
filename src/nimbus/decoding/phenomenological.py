"""Surface-code memory under phenomenological noise, as a constrained spin problem.

Rotated surface code (:func:`nimbus.problems.surface_code.rotated_surface_code`). A memory experiment has T noisy
measurement rounds followed by one error-free (perfect) round. Before each of the T + 1 rounds every data qubit may
suffer an error, and each of the T noisy rounds has measurement errors with the same probability.

:class:`PhenoCode` is one sector: X errors detected by the Z checks (sector "X") or the dual (sector "Z").
Data errors x(q, t) occur before each of the T + 1 rounds (t = 0..T) and measurement errors m(c, t) on the T noisy rounds
(t = 0..T-1). Detectors D(c, t) = s(c, t) xor s(c, t-1), t = 0..T, with s(c, -1) = 0, s(c, t) = HZ E_t + m(c, t) (E_t the
cumulative error) and s(c, T) = HZ E_T. The observable is the parity of E_T on one logical support.

Spin index: x(q, t) -> t n + q (t = 0..T); m(c, t) -> n (T + 1) + t m + c (t = 0..T-1). Detector D(c, t) -> t m + c.
Moves (fixed masks): the spatial stabilizer at each slice; the time-like move {x(q, t), x(q, t + 1), m(c, t) for c adjacent
to q}; the straight logical lines at each slice (they flip the observable).

open_top=True describes a window of a sliding-window decoder: T noisy rounds and no perfect final round. Data errors
x(q, t), t = 0..T-1, measurement errors m(c, t), t = 0..T-1, detectors D(c, t), t = 0..T-1; m(c, T-1) keeps only its
in-window detector D(c, T-1) (open time boundary). The top time-like moves are truncated to {x(q, T-1), m(c, T-1), c ~ q}.
The logical class of an open window is not defined; the observable L is kept only for index bookkeeping.

:class:`DepolarizingMemory` joins the two sectors for depolarizing data noise: each data qubit gets X, Y or Z with
probability p/3 each before each round, and both check types have measurement errors with probability p on the noisy
rounds. :func:`memory_shots` samples the shots of a memory experiment from a seed key.
"""
from __future__ import annotations

import numpy as np

from nimbus.problems.surface_code import rotated_surface_code

__all__ = ["gf2_rref", "gf2_rank", "PhenoCode", "DepolarizingMemory", "memory_model", "memory_shots"]


def gf2_rref(A):
    """Row-reduce A (uint8) over GF(2). Returns (R, pivots, Tm) with R = Tm @ A mod 2 in reduced row echelon form."""
    A = A.copy() % 2
    r, c = A.shape
    Tm = np.eye(r, dtype=np.uint8)
    piv = []
    row = 0
    for col in range(c):
        if row >= r:
            break
        nz = np.flatnonzero(A[row:, col])
        if nz.size == 0:
            continue
        k = row + nz[0]
        if k != row:
            A[[row, k]] = A[[k, row]]
            Tm[[row, k]] = Tm[[k, row]]
        hit = np.flatnonzero(A[:, col])
        hit = hit[hit != row]
        if hit.size:
            A[hit] ^= A[row]
            Tm[hit] ^= Tm[row]
        piv.append(col)
        row += 1
    return A, piv, Tm


def gf2_rank(A):
    """Rank of A over GF(2)."""
    return len(gf2_rref(A)[1])


class PhenoCode:
    """One sector of the phenomenological memory (see the module docstring).

    Attributes: H (detectors x spins check matrix), L (observable), n, m (qubits, checks), ns (data slices), nx (data
    spins), N (spins), ndet (detectors), sup / SUP / VAL (move supports, padded with N), kind (0 spatial, 1 time-like,
    2 logical), islog."""

    def __init__(self, d: int, T: int | None = None, sector: str = "X", open_top: bool = False):
        """Distance d, T noisy rounds (default d). sector "X": X errors, Z checks (default); "Z": the dual (Z errors,
        X checks, Z-logical moves); the attributes HZ / HX / lz then name the checks / moves / observable support of that
        sector."""
        code = dict(rotated_surface_code(d))
        if sector == "Z":
            code["HZ"], code["HX"] = code["HX"], code["HZ"]
            code["lz_all"], code["lx_all"] = code["lx_all"], code["lz_all"]
        self.d, self.T = d, (d if T is None else T)
        T = self.T
        self.open_top = open_top
        ns = T if open_top else T + 1                     # data slices = detector rounds
        self.ns = ns
        HZ, HX = code["HZ"], code["HX"]
        n, m = code["n"], HZ.shape[0]
        self.n, self.m = n, m
        self.nx = n * ns
        self.N = self.nx + m * T
        self.ndet = m * ns
        H = np.zeros((self.ndet, self.N), np.uint8)
        for t in range(ns):
            H[t * m:(t + 1) * m, t * n:(t + 1) * n] = HZ
        for t in range(T):
            for c in range(m):
                j = self.nx + t * m + c
                H[t * m + c, j] = 1
                if t + 1 < ns:
                    H[(t + 1) * m + c, j] = 1
        self.H = H
        lz = np.array(code["lz_all"][0])
        L = np.zeros(self.N, np.uint8)
        for t in range(ns):
            L[t * n + lz] = 1
        self.L = L
        self.HZ, self.HX, self.lz = HZ, HX, lz
        sup, kind = [], []
        for t in range(ns):
            for row in HX:
                sup.append(t * n + np.flatnonzero(row)); kind.append(0)
        adj = [np.flatnonzero(HZ[:, q]) for q in range(n)]
        for t in range(T):
            for q in range(n):
                xs = [t * n + q, (t + 1) * n + q] if t + 1 < ns else [t * n + q]
                sup.append(np.concatenate([xs, self.nx + t * m + adj[q]])); kind.append(1)
        for t in range(ns):
            for ln in code["lx_all"]:
                sup.append(t * n + np.array(ln)); kind.append(2)
        self.sup = [np.asarray(s, np.int64) for s in sup]
        self.kind = np.array(kind)
        self.islog = self.kind == 2
        Lmax = max(len(s) for s in self.sup)
        self.SUP = np.array([np.concatenate([s, np.full(Lmax - len(s), self.N)]) for s in self.sup], np.int64)
        self.VAL = (self.SUP < self.N)
        self._rr = None                                   # GF(2) elimination of H, computed on first use
        self._hzr = None

    def _rref(self):
        if self._rr is None:
            _, piv, Tm = gf2_rref(self.H)
            self._rr = (piv, Tm)
        return self._rr

    @property
    def piv(self):
        """Pivot columns of the GF(2) elimination of H."""
        return self._rref()[0]

    @property
    def Tm(self):
        """Row transform of the GF(2) elimination of H."""
        return self._rref()[1]

    @property
    def rank_H(self):
        return len(self.piv)

    @property
    def Hsp(self):
        """H as a scipy CSR matrix (int32)."""
        if getattr(self, "_hsp", None) is None:
            from scipy.sparse import csr_matrix
            self._hsp = csr_matrix(self.H.astype(np.int32))
        return self._hsp

    # ------------------------------------------------------------------ checks
    def check(self, rng=None):
        """Null-space, observable and completeness checks, and a direct simulation of the detectors. Returns a dict."""
        M = np.zeros((len(self.sup), self.N), np.uint8)
        for i, s in enumerate(self.sup):
            M[i, s] = 1
        Hs = (M.astype(np.int64) @ self.H.T.astype(np.int64)) % 2
        Ls = (M.astype(np.int64) @ self.L.astype(np.int64)) % 2
        assert not Hs.any(), "a move changes the syndrome"
        if not self.open_top:
            assert (Ls == self.islog).all(), "logical flags wrong"
        HL = self.H if self.open_top else np.vstack([self.H, self.L[None]])
        dim_ker = self.N - gf2_rank(HL)
        rank_moves = gf2_rank(M[~self.islog])
        rng = rng or np.random.default_rng(0)
        e = (rng.random((200, self.N)) < 0.05).astype(np.uint8)
        Ddir = self.simulate_detectors(e)
        assert (Ddir == (e.astype(np.int64) @ self.H.T % 2)).all(), "H differs from direct simulation"
        obs_dir = (e[:, :self.nx].reshape(200, self.ns, self.n).sum(1) % 2)[:, self.lz].sum(1) % 2
        assert (obs_dir == (e.astype(np.int64) @ self.L % 2)).all()
        return dict(N=self.N, ndet=self.ndet, rank_H=self.rank_H, dim_ker_HL=int(dim_ker),
                    rank_nonlogical_moves=int(rank_moves), complete=bool(rank_moves == dim_ker),
                    moves=len(self.sup), n_spatial=int((self.kind == 0).sum()), n_time=int((self.kind == 1).sum()),
                    n_logical=int(self.islog.sum()), max_move_weight=int(self.VAL.sum(1).max()))

    def simulate_detectors(self, e):
        """Detectors from an explicit round-by-round simulation (independent of H)."""
        n, m, T = self.n, self.m, self.T
        x = e[:, :self.nx].reshape(-1, self.ns, n)
        meas = e[:, self.nx:].reshape(-1, T, m)
        E = np.zeros((e.shape[0], n), np.int64)
        prev = np.zeros((e.shape[0], m), np.int64)
        D = []
        for t in range(self.ns):
            E = (E + x[:, t]) % 2
            s = (E @ self.HZ.T.astype(np.int64)) % 2
            if t < T:
                s = (s + meas[:, t]) % 2
            D.append((s + prev) % 2)
            prev = s
        return np.concatenate(D, 1).astype(np.uint8)

    # ------------------------------------------------------------------ sampling and start states
    def sample(self, p, shots, rng):
        """Independent errors (every spin with probability p): (errors, detectors, observable)."""
        e = (rng.random((shots, self.N)) < p).astype(np.uint8)
        D = (e.astype(np.int64) @ self.H.T % 2).astype(np.uint8)
        obs = (e.astype(np.int64) @ self.L % 2).astype(np.uint8)
        return e, D, obs

    def particular(self, D):
        """Particular solutions c0 with H c0 = D (rows), by the GF(2) elimination of H."""
        Y = (D.astype(np.int64) @ self.Tm.T.astype(np.int64)) % 2
        c0 = np.zeros((D.shape[0], self.N), np.uint8)
        c0[:, self.piv] = Y[:, :self.rank_H]
        assert ((c0.astype(np.int64) @ self.H.T % 2) == D).all(), "syndrome not in the column space"
        return c0

    def particular_fast(self, D):
        """Particular solutions c0 with H c0 = D by construction, O(N) per row (no elimination of the large H). In a
        closed code the parity sigma(c) = XOR_t D(c, t) is first produced by data errors on the last slice
        (HZ x = sigma, small elimination of HZ); every remaining detector is then produced by measurement-error strings
        m(c, t) = XOR_{t' <= t} D'(c, t'). Asserts H c0 = D."""
        n, m, T, ns = self.n, self.m, self.T, self.ns
        D = np.asarray(D, np.uint8)
        Dr = D.reshape(-1, ns, m).copy()
        c0 = np.zeros((D.shape[0], self.N), np.uint8)
        if not self.open_top:
            if self._hzr is None:
                _, pv, Tz = gf2_rref(self.HZ)
                self._hzr = (np.array(pv), Tz)
            pv, Tz = self._hzr
            sig = np.bitwise_xor.reduce(Dr, axis=1)
            Y = (sig.astype(np.int64) @ Tz.T.astype(np.int64)) % 2
            xT = np.zeros((D.shape[0], n), np.uint8)
            xT[:, pv] = Y[:, :len(pv)]
            assert ((xT.astype(np.int64) @ self.HZ.T % 2) == sig).all(), "syndrome not in the column space"
            c0[:, T * n:(T + 1) * n] = xT
            Dr[:, T] ^= sig
        c0[:, self.nx:] = np.bitwise_xor.accumulate(Dr[:, :T], axis=1).reshape(D.shape[0], T * m)
        assert ((self.Hsp @ c0.T.astype(np.int32)).T % 2 == D).all(), "particular_fast failed"
        return c0


class DepolarizingMemory:
    """Both sectors of the memory under depolarizing data noise (X, Y, Z each p/3) and measurement errors (p).

    Attributes: cx, cz (the X and Z sectors, :class:`PhenoCode`); O = cx.N (offset of the Z sector in a joint spin
    vector); N = cx.N + cz.N; nx (data spins per sector); SUP / VAL (move supports of both sectors in the joint
    layout, padded with N); logx / logz (logical moves of each sector)."""

    def __init__(self, d, T=None, open_top=False):
        """T noisy rounds (default d); open_top: a sliding window with no perfect final round (see PhenoCode)."""
        self.cx = PhenoCode(d, T, sector="X", open_top=open_top)
        self.cz = PhenoCode(d, T, sector="Z", open_top=open_top)
        cx, cz = self.cx, self.cz
        assert cx.nx == cz.nx
        self.O = cx.N
        self.N = cx.N + cz.N
        N, O = self.N, self.O
        self.nx = cx.nx
        SUPx = np.where(cx.SUP == cx.N, N, cx.SUP)
        SUPz = np.where(cz.SUP == cz.N, N, cz.SUP + O)
        L = max(SUPx.shape[1], SUPz.shape[1])
        pad = lambda A: np.hstack([A, np.full((A.shape[0], L - A.shape[1]), N)])
        self.SUP = np.vstack([pad(SUPx), pad(SUPz)])
        self.VAL = self.SUP < N
        self.logx = np.concatenate([cx.islog, np.zeros(len(SUPz), bool)])
        self.logz = np.concatenate([np.zeros(len(SUPx), bool), cz.islog])

    def sample(self, p, shots, rng, errors=False):
        """Detectors and observables (Dx, Dz, ox, oz); errors=True also returns the error vectors (ex, ez), from the
        same random stream. Dx: detectors of the X sector [shots, cx.ndet]; ox: its observable."""
        cx, cz = self.cx, self.cz
        u = rng.random((shots, self.nx))
        x = (u < 2 * p / 3).astype(np.uint8)                    # X or Y
        z = ((u >= p / 3) & (u < p)).astype(np.uint8)           # Y or Z
        mz = (rng.random((shots, cx.N - cx.nx)) < p).astype(np.uint8)
        mx = (rng.random((shots, cz.N - cz.nx)) < p).astype(np.uint8)
        ex, ez = np.hstack([x, mz]), np.hstack([z, mx])
        Dx = (ex.astype(np.int64) @ cx.H.T % 2).astype(np.uint8)
        Dz = (ez.astype(np.int64) @ cz.H.T % 2).astype(np.uint8)
        ox = (ex.astype(np.int64) @ cx.L % 2).astype(np.int64)
        oz = (ez.astype(np.int64) @ cz.L % 2).astype(np.int64)
        if errors:
            return Dx, Dz, ox, oz, ex, ez
        return Dx, Dz, ox, oz

    def energy(self, S, W, wm):
        """Energy W #{data sites with x or z} + wm #{measurement errors} of joint states S [., N(+1)]."""
        N, O, nx = self.N, self.O, self.nx
        data = (S[:, :nx] | S[:, O:O + nx]).sum(1)
        meas = S[:, nx:O].sum(1) + S[:, O + nx:N].sum(1)
        return W * data + wm * meas


_MODELS = {}


def memory_model(d: int, T: int, open_top: bool = False) -> DepolarizingMemory:
    """Cached :class:`DepolarizingMemory` (distance d, T noisy rounds, open or closed top)."""
    key = (d, T, open_top)
    if key not in _MODELS:
        _MODELS[key] = DepolarizingMemory(d, T, open_top)
    return _MODELS[key]


def memory_shots(d: int, p: float, shots: int, chunk: int = 0, seed: int = 11, rounds: int | None = None):
    """Shots of a memory experiment: (Dx, Dz, ox, oz) of ``memory_model(d, rounds)`` (default rounds = 4 d noisy rounds
    plus the perfect round), drawn from ``numpy.random.default_rng([seed, chunk, d, int(p * 1e5), 5])``. Shots with the
    same (seed, chunk, d, p, shots, rounds) are identical; chunks give independent batches."""
    N = rounds or 4 * d
    return memory_model(d, N).sample(p, shots, np.random.default_rng([seed, chunk, d, int(p * 1e5), 5]))

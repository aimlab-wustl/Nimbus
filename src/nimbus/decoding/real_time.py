"""Real time: bit-exact model of the sliding-window parallel-tempering surface-code decoder.

This module is the software specification of the FPGA decoder that decodes a stream of syndrome rounds in sliding
windows. Every random bit, threshold and decision is reproduced exactly, so the logical error rates computed here are
those of the hardware. The conventions (lane generators, swap unit, start chains) are those of
:mod:`nimbus.decoding.one_shot`; the window geometry is that of
:class:`nimbus.decoding.phenomenological.DepolarizingMemory` and :func:`nimbus.decoding.windows.decode_windows`.

Memory: N noisy rounds plus one perfect round, both check types measured, measurement error rate q = p. Raw syndromes
s_X(c, t) (Z checks, X sector) and s_Z(c, t) (X checks, Z sector), t = 0..N (t = N: the perfect round), are the
cumulative XOR of the detectors.

Window (W = C + B rounds [t0, t0 + T), T = W, open top; the final window is [t0, N) plus the perfect round, closed).
The state is the literals held by the clause units (there are no spin registers):

- data clause cl_d(q, tau), tau = 0..ns-1 (ns = T, or T + 1 in the final window): literals x (literal 2j) and z (2j + 1),
  j = tau n + q; weight W = ln(3 (1 - p) / p);
- measurement clauses cl_mZ(c, tau) (X sector, literal 2 ndc + tau m + c) and cl_mX(c, tau) (Z sector, literal
  2 ndc + m T + tau m + c), tau = 0..T-1; weight wm = ln((1 - p) / p).

Clause outputs: data clause for an X flip = 0 if z = 1, else +1 (x = 0) / -1 (x = 1); Z flip symmetric; measurement
clause = +1 (m = 0) / -1 (m = 1). The neuron of a move reads the outputs of the clauses in its support only:
a = sum of its data-clause outputs, b = sum of its measurement-clause outputs, dE = W a + wm b.

Moves (fixed masks, X sector then Z sector; per sector the spatial stabilizers at each slice, the time-like moves
{x(q, tau), x(q, tau + 1), m(c, tau) for c ~ q} (top slice of an open window truncated to x(q, T-1)), and the logical lines
(all d straight lines) at each slice). Neurons: SL (stabilizer or logical line; b = 0), T2 / T1 (time-like with two / one
adjacent check).

Acceptance (exact two-weight form): the neuron maps (table, a, b, mode) to a threshold index r (r = 0: dE <= 0, always
accept); accept iff r = 0 or ``u < thr[lab][r]``. Threshold row per temperature t (host-written, 16 bits):

- r = 1..KSL: SL levels k W, KSL = max(4, d) (the largest support);
- r = KSL + 1..KSL + 7: T2 levels, ascending: {2W - 2wm, 2wm - W} (mode = 1, p < 1/28: this order; otherwise swapped),
  W, 2wm, 2W, W + 2wm, 2W + 2wm;
- r = KSL + 8..KSL + 12: T1 levels, ascending: W - wm, wm, 2W - wm, W + wm, 2W + wm;
- ``thr[t][r] = min(floor(2^B exp(-beta_t dE_r)), 2^B - 1)``; ``thr[t][0] = 2^B``.

Within each table the thresholds do not increase with r, so u < thr[r] equals the thermometer test r <= c. Valid for
p < 1/4 (W < 2 wm), checked at build.

Schedule: colors from the masks (no two moves of a color share a clause; one table, SL or T, per color; first fit by
support size, then balancing). Hardware schedule (PTWConfig k >= 1, the FPGA layout): one coloring of the physical neurons
of a k-slab layout (:func:`hw_schedule`), shared by both window kinds; the color cycles are (slab j, color c), slab-major,
and the lanes are the physical lanes; the kernel is the same. A sweep runs the colors in order; in color cycle k, every
replica i:

1. each generator bank g (lanes [g LB, (g + 1) LB)) draws nl_g B bits from its GFSR pair (the lane generator of
   :mod:`~nimbus.decoding.one_shot`: y = a XOR b, a_n = a_{n-2281} XOR a_{n-1029}, b_n = b_{n-3217} XOR b_{n-576}); lane j
   gets u_j = bits [j' B, (j' + 1) B) of its bank (j' = j - g LB, first bit = most significant bit). LB = 64 keeps the
   3-term leap (N = 1024 < 1152 bits per bank and cycle). With one bank this is the one-shot generator exactly;
2. lane j tests move r = cls_mat[k, j]; all accepted moves of the color flip their literals together;
   nd += a, nm += b (accepted), class c ^= logical bits (1 = X-sector logical, 2 = Z-sector logical).

After every K-th sweep (K = 1 or 2): swap round q = (s + 1) / K - 1 over the pairs t = q % 2, q % 2 + 2, ...:
i = top USW bits of the pair's xoshiro128**, D = E[holder[t + 1]] - E[holder[t]], exchange labels iff
D <= XI_t[i] = min((theta_t L[i]) >> (LF + TF), 2^XIW - 1), theta_t = round(2^(TF + EF) / (beta_t - beta_t+1)), with the
fixed-point energy E = Wq nd + wq nm, Wq = round(W 2^EF), wq = round(wm 2^EF) (EW-bit registers).

After every sweep (after its swap round): global best = the lowest E over replicas (lowest replica index on ties); if
strictly below the best so far, snapshot it (bestE, literals, class). Occupancy count of the class of holder[0] (beta = 1)
in block bit_length(s) (final window: ML class = argmax of the last block, sweeps [2^(k-1), 2^k)). The best starts as
replica 0's start state. Labels lab[i] = i.

Start state (every replica): m(c, tau) = s(c, t0 + tau) XOR sigma(c), x = z = 0; the final window adds data chains on its
perfect slice for s(c, N) XOR sigma(c) (the fixed minimum-weight chains to the boundary of the one-shot decoder; start =
"gf2" uses the pivot solution of :meth:`PhenoCode.particular_fast` instead). nd, nm are loaded by popcount.

Commit (non-final window, C committed rounds): sigma_new(c) = s(c, t0 + C - 1) XOR m*(c, C - 1) per sector, and 2 logical
frame bits: parity of x* on the X-sector observable line (lz) and of z* on the Z-sector line (lx), summed over the
committed slices tau < C (* = best snapshot). Final window: the same parities over all its slices, moved to the ML class
("ml": XOR (bestC XOR ML)) or not ("me"). Failure = frame bits != the true observables.

Generator seeds per (shot, window) from one 64-bit base (:func:`window_bases`): replica i, bank g:
a <- splitmix64 bits of base ^ (2 (i NB + g) + 1) 0xD1B54A32D192ED03, b <- base ^ (2 (i NB + g) + 2) 0xD1B54A32D192ED03
(the one-shot formula for NB = 1); swap pair t: xoshiro128** state from splitmix64(base ^ (t + 1) 0xABCDEF0123456789),
2 words. PTWConfig.carry (as the hardware): the generators are seeded so at the first window of a shot only and run on
through its later windows (no reseeding cost per window).

Main entry points: :func:`decode_stream` (many shots, window by window; :func:`fpga_config` gives the configuration of
the FPGA design for a distance), :func:`failures`, :func:`window_spec` and :func:`run_window` (one window), and
:func:`decode_batch` (the board's single generator stream over a batch of shots).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from nimbus.decoding.one_shot import GFSR_A, GFSR_B, _chains, ladder, lut, splitmix64_seq, xoshiro_next
from nimbus.decoding.phenomenological import gf2_rref
from nimbus.decoding.windows import schedule
from nimbus.problems.surface_code import rotated_surface_code

__all__ = ["PTWConfig", "WSpec", "window_spec", "hw_schedule", "schedule", "syndromes", "start_state", "window_bases",
           "run_window", "gen_state", "decode_stream", "failures", "gfsr_stream", "to_memory_layout", "frame_parity",
           "measurement_literals", "with_tables", "decode_batch", "ring_from_bits", "fpga_config", "FPGA_LAYOUTS",
           "T2_LEVELS", "T1_LEVELS", "MODE_P"]

M64 = (1 << 64) - 1
RW = 128                                   # GFSR ring: 128 words = 8192 bits per generator
RW_MASK = RW - 1
MODE_P = 1.0 / 28.0                        # T2 order crossover (W / wm = 4/3)
T1_LEVELS = [(1, -1), (0, 1), (2, -1), (1, 1), (2, 1)]
T2_REST = [(1, 0), (0, 2), (2, 0), (1, 2), (2, 2)]

#: Schedule parameters (slabs k, LUTRAM clause units) of the FPGA designs per code distance d.
FPGA_LAYOUTS = {3: (1, False), 5: (2, False), 7: (5, True), 9: (9, True), 11: (11, True), 13: (13, True)}


def T2_LEVELS(mode: int):
    """Positive (a, b) pairs of a time-like move with two adjacent checks, ascending dE (mode = 1: p < 1/28)."""
    return ([(2, -2), (-1, 2)] if mode else [(-1, 2), (2, -2)]) + T2_REST


@dataclass(frozen=True)
class PTWConfig:
    """Machine parameters.

    R: replicas; bmin: smallest inverse temperature of the ladder geomspace(1, bmin, R); K: a swap round after every K
    sweeps; B, USW: uniform bits per lane and per swap decision; LF, TF: fractional bits of the swap log table and of
    theta_t; EF, EW: fractional bits and width of the fixed-point weights; TW, XIW: widths of theta_t and of the swap
    thresholds; LB: lanes per generator bank; k, lram: schedule (see below); carry: generator handling across windows."""
    R: int = 16
    bmin: float = 0.1
    K: int = 1                 # swap round after every K sweeps
    B: int = 16                # lane uniform bits
    USW: int = 12              # swap uniform bits
    LF: int = 16
    TF: int = 3
    EF: int = 9                # fractional bits of Wq, wq
    EW: int = 12               # width of Wq, wq
    TW: int = 20               # width of theta_t
    XIW: int = 20              # swap threshold width (saturated)
    LB: int = 64               # lanes per generator bank
    k: int = 0                 # 0: software schedule (12 colors per window); >= 1: the hardware schedule of a
                               # k-slab layout (hw_schedule; k = 1 = fully parallel), one coloring for both window kinds
    lram: bool = False         # hardware schedule for LUTRAM clause units (one word address per clause unit and color)
    carry: bool = False        # True: the lane and swap generators run on across the windows of a shot (seeded once, at
                               # its first window, from that window's base), as the hardware does; False: reseeded
                               # every window


@dataclass
class WSpec:
    """Moves, colors and tables of one window kind (d, T, final, p, config); built by :func:`window_spec`."""
    d: int
    T: int
    final: bool
    p: float
    cfg: PTWConfig
    n: int
    m: int
    ns: int
    ndc: int
    NL: int
    M: int
    tptr: np.ndarray           # [M + 1] term CSR
    tli: np.ndarray            # [nt] literal of each term
    tdat: np.ndarray           # [nt] 1 = data clause term (partner literal = tli ^ 1), 0 = measurement clause term
    tsp: np.ndarray            # [M] first measurement term of each move (data terms come first)
    mtab: np.ndarray           # [M] 0 = SL, 1 = T2, 2 = T1
    mkind: np.ndarray          # [M] 0 = stabilizer, 1 = time-like, 2 = logical
    mlg: np.ndarray            # [M] logical bits
    cls_mat: np.ndarray        # [NC, Lmax] move per lane, -1 idle
    col_table: np.ndarray      # [NC] 0 = SL color, 1 = T color
    bank_lanes: np.ndarray     # [NB] lanes per generator bank
    KSL: int
    rk: np.ndarray             # [3, 2 KSL + 1, 5] threshold index of (table, a + KSL, b + 2)
    levels: list               # per threshold index r >= 1: (table, a, b, dE)
    mode: int
    W: float
    wm: float
    Wq: int
    wq: int
    betas: np.ndarray
    thr: np.ndarray            # [R, KSL + 13]
    theta: np.ndarray          # [R - 1]
    xi_sw: np.ndarray          # [R - 1, 2^USW]
    chains: list               # [sector] [m, n] chain of each check (the one-shot start chains)
    gf2: list                  # [sector] (pivots, Tm) of the check matrix (PhenoCode.particular_fast)
    obs: list                  # [sector] observable qubits (X sector: lz, Z sector: lx)
    info: dict = field(default_factory=dict)


# ============================================================================================ geometry
def _gf2_rref(A):
    """Pivots and transform of :func:`~nimbus.decoding.phenomenological.gf2_rref` (so start = "gf2" reproduces
    PhenoCode.particular_fast)."""
    _, piv, Tm = gf2_rref(A)
    return np.array(piv, np.int64), Tm


def _color(sites, table):
    """First fit by support size (ties: move index), then balancing of the largest color."""
    M = len(sites)
    cols = []

    def fits(i, c):
        return c["table"] == table[i] and not (c["sites"] & sites[i])

    for i in sorted(range(M), key=lambda i: (-len(sites[i]), i)):
        c = next((c for c in cols if fits(i, c)), None)
        if c is None:
            c = dict(table=table[i], sites=set(), members=set())
            cols.append(c)
        c["sites"] |= sites[i]
        c["members"].add(i)
    for _ in range(200):
        big = max(cols, key=lambda c: len(c["members"]))
        moved = False
        for i in sorted(big["members"]):
            big["sites"] -= sites[i]; big["members"].discard(i)
            tgt = [c for c in cols if c is not big and len(c["members"]) < len(big["members"]) and fits(i, c)]
            if tgt:
                c = min(tgt, key=lambda c: len(c["members"]))
            else:
                c = big
            c["sites"] |= sites[i]; c["members"].add(i)
            moved |= c is not big
            if len(big["members"]) <= M / len(cols):
                break
        if not moved:
            break
    for c in cols:                         # no two moves of one color share a clause
        seen = set()
        for i in c["members"]:
            assert not (seen & sites[i]), "color shares a clause"
            seen |= sites[i]
    return [(c["table"], sorted(c["members"])) for c in cols]


def _rank_table(KSL, mode):
    rk = np.zeros((3, 2 * KSL + 1, 5), np.int64)
    levels = [None] + [("SL", k, 0) for k in range(1, KSL + 1)]
    for a in range(1, KSL + 1):
        rk[0, a + KSL, 2] = a
    for j, (a, b) in enumerate(T2_LEVELS(mode)):
        rk[1, a + KSL, b + 2] = KSL + 1 + j
        levels.append(("T2", a, b))
    for j, (a, b) in enumerate(T1_LEVELS):
        rk[2, a + KSL, b + 2] = KSL + 8 + j
        levels.append(("T1", a, b))
    return rk, levels


_MV = {}


def _moves(d: int, T: int, final: bool) -> dict:
    """Move list of a window (order: per sector stabilizers per slice, time-like, logical lines), with each move's key
    (sector, kind, spatial id, base slice) and literal names (("d", q, t, sector) / ("m", sector, c, t))."""
    key = (d, T, bool(final))
    if key in _MV:
        return _MV[key]
    code = rotated_surface_code(d)
    n = code["n"]
    HX, HZ = np.asarray(code["HX"], np.uint8), np.asarray(code["HZ"], np.uint8)
    m = HZ.shape[0]
    assert HX.shape[0] == m
    ns = T + 1 if final else T
    ndc = n * ns
    NL = 2 * ndc + 2 * m * T
    # sector: (stabilizer moves, checks, logical lines, observable, data literal offset, meas offset, logical bit)
    secs = [(HX, HZ, code["lx_all"], np.array(code["lz_all"][0]), 0, 0, 1),
            (HZ, HX, code["lz_all"], np.array(code["lx_all"][0]), 1, m * T, 2)]
    terms, mtab, mkind, mlg, sites, keys = [], [], [], [], [], []
    for sec, (Hs, Hc, logs, _obs, lo, mo, lb) in enumerate(secs):
        mv = []
        for t in range(ns):
            for i, row in enumerate(Hs):
                mv.append(([t * n + int(q) for q in np.flatnonzero(row)], [], 0, 0, (sec, 0, i, t)))
        for t in range(T):
            for q in range(n):
                xs = [t * n + q, (t + 1) * n + q] if t + 1 < ns else [t * n + q]
                adj = [int(c) for c in np.flatnonzero(Hc[:, q])]
                mv.append((xs, [t * m + c for c in adj], 1, 0, (sec, 1, q, t)))
        for t in range(ns):
            for i, ln in enumerate(logs):
                mv.append(([t * n + int(q) for q in ln], [], 2, lb, (sec, 2, i, t)))
        for dj, mj, kind, lg, ky in mv:
            terms.append([(2 * j + lo, 1) for j in dj] + [(2 * ndc + mo + k, 0) for k in mj])
            mtab.append(0 if not mj else (1 if len(mj) == 2 else 2))
            mkind.append(kind); mlg.append(lg); keys.append(ky)
            sites.append(set(dj) | {ndc + mo + k for k in mj})

    def lname(li):
        if li < 2 * ndc:
            j = li // 2
            return ("d", j % n, j // n, li % 2)
        k = li - 2 * ndc
        return ("m", k // (m * T), k % m, (k % (m * T)) // m)

    out = dict(code=code, n=n, m=m, ns=ns, ndc=ndc, NL=NL, HX=HX, HZ=HZ, secs=secs, terms=terms, mtab=mtab, mkind=mkind,
               mlg=mlg, sites=sites, keys=keys, lname=lname)
    _MV[key] = out
    return out


_HS = {}


def hw_schedule(d: int, T: int, k: int, lram: bool = False) -> dict:
    """Hardware schedule of a k-slab layout. Slice t sits in slab j = min(t // S, k - 1),
    row t - j S (S = ceil(T / k); the perfect slice t = T of a final window may need an extra row S of the last slab).
    A physical neuron = (sector, kind, spatial id, row) serves that move in every slab; one coloring of the physical neurons,
    valid for both window kinds (final-only moves idle in other windows): two neurons of a color never touch the same clause
    in one slab (relative rows; FF copies) or the same physical clause unit (rows mod S; LUTRAM, one word address per unit).
    Sweep = for each slab j, for each color c (row j NC + c of cls); lane = position in the color (sorted).
    Returns phys (list of dicts), NC, S, cls ``[kind][k NC, Lmax]`` (logical move indices of each window kind), col_table."""
    key = (d, T, k, bool(lram))
    if key in _HS:
        return _HS[key]
    S = -(-T // k)
    xr = k * S == T                        # the perfect slice T = row S of the last slab (an extra row, FF clause units)
    mv = [_moves(d, T, False), _moves(d, T, True)]

    def place(t):
        j = min(t // S, k - 1)
        return j, t - j * S

    phys, pidx = [], {}
    inst = [{}, {}]                        # [kind] (p, slab) -> logical move
    for f in (1, 0):                       # final window first: its moves are complete (no truncated top)
        M_ = mv[f]
        for r, (sec, kind, i, t) in enumerate(M_["keys"]):
            j, row = place(t)
            pk = (sec, kind, i, row)
            if pk not in pidx:
                pidx[pk] = len(phys)
                phys.append(dict(key=pk, sec=sec, kind=kind, sid=i, row=row, table=int(M_["mtab"][r]),
                                 lg=int(M_["mlg"][r]), dterms=[], mterms=[]))
            p = pidx[pk]
            P = phys[p]
            assert P["table"] == int(M_["mtab"][r])
            inst[f][(p, j)] = r
            for li, dt in M_["terms"][r]:
                nm = M_["lname"](li)
                if dt:
                    rel = ("d", nm[1], nm[2] - t)          # (q, delta)
                    if rel not in P["dterms"]:
                        P["dterms"].append(rel)
                else:
                    rel = ("m", nm[1], nm[2], 0)
                    assert nm[3] == t
                    if rel not in P["mterms"]:
                        P["mterms"].append(rel)

    def site(P, tm):
        """Clause units a term touches (relative rows; LUTRAM: physical units, rows mod S, extra row X)."""
        if tm[0] == "d":
            rr = P["row"] + tm[2]
            if not lram:
                return {("d", tm[1], rr)}
            if P["row"] == S:
                return {("d", tm[1], "X")}
            if rr < S:
                return {("d", tm[1], rr)}
            out = {("d", tm[1], 0)} if k > 1 else set()           # slabs below the last: row 0 of the next slab
            if xr:
                out.add(("d", tm[1], "X"))                         # last slab: the extra row
            return out
        return {("m", tm[1], tm[2], P["row"])}

    sites = [set().union(*[site(P, tm) for tm in P["dterms"] + P["mterms"]]) for P in phys]
    cols = _color(sites, [int(P["table"] > 0) for P in phys])
    NC = len(cols)
    Lmax = max(len(mem) for _, mem in cols)
    for c, (_, mem) in enumerate(cols):
        for l, p in enumerate(mem):
            phys[p]["col"] = c
            phys[p]["lane"] = l
    cls = []
    for f in (0, 1):
        cm = -np.ones((k * NC, Lmax), np.int64)
        for p, P in enumerate(phys):
            for j in range(k):
                r = inst[f].get((p, j))
                if r is not None:
                    cm[j * NC + P["col"], P["lane"]] = r
        assert sorted(cm[cm >= 0].tolist()) == list(range(len(mv[f]["keys"])))
        cls.append(cm)
    for P in phys:
        P["slabs"] = [[j for j in range(k) if (pidx[P["key"]], j) in inst[f]] for f in (0, 1)]
    out = dict(d=d, T=T, k=k, S=S, xr=xr, lram=bool(lram), NC=NC, Lmax=Lmax, phys=phys, inst=inst, cls=cls,
               col_table=np.array([tb for tb, _ in cols], np.int64), moves=mv, place=place)
    _HS[key] = out
    return out


_WS = {}


def window_spec(d: int, T: int, final: bool, p: float, cfg: PTWConfig = PTWConfig()) -> WSpec:
    """Specification of a window of T rounds (final: plus the perfect round, closed top) at error rate p (cached)."""
    key = (d, T, bool(final), float(p), cfg)
    if key in _WS:
        return _WS[key]
    if not 0 < p < 0.25:
        raise ValueError("the rank acceptance is exact for 0 < p < 1/4 only")
    mvs = _moves(d, T, final)
    code, n, m, ns, ndc, NL = mvs["code"], mvs["n"], mvs["m"], mvs["ns"], mvs["ndc"], mvs["NL"]
    HX, HZ = mvs["HX"], mvs["HZ"]
    secs = mvs["secs"]
    terms, mtab, mkind, mlg, sites = mvs["terms"], mvs["mtab"], mvs["mkind"], mvs["mlg"], mvs["sites"]
    M = len(terms)
    tptr = np.zeros(M + 1, np.int64)
    tptr[1:] = np.cumsum([len(t) for t in terms])
    tli = np.array([li for t in terms for li, _ in t], np.int64)
    tdat = np.array([dt for t in terms for _, dt in t], np.uint8)
    tsp = tptr[:-1] + np.array([sum(dt for _, dt in t) for t in terms], np.int64)
    assert all(all(t[e][1] >= t[e + 1][1] for e in range(len(t) - 1)) for t in terms)
    mtab = np.array(mtab, np.int64)
    if cfg.k > 0:                          # hardware schedule of a slab layout (rows = (slab, color), lanes = physical lanes)
        hs = hw_schedule(d, T, cfg.k, cfg.lram)
        cls_mat = hs["cls"][int(bool(final))].copy()
        col_table = np.tile(hs["col_table"], cfg.k)
        cols = [(int(col_table[r]), [int(x) for x in cls_mat[r] if x >= 0]) for r in range(cls_mat.shape[0])]
        Lmax = cls_mat.shape[1]
    else:
        cols = _color(sites, [int(x > 0) for x in mtab])
        Lmax = max(len(mem) for _, mem in cols)
        cls_mat = -np.ones((len(cols), Lmax), np.int64)
        for k, (_, mem) in enumerate(cols):
            cls_mat[k, :len(mem)] = mem
        col_table = np.array([tb for tb, _ in cols], np.int64)
    LB = cfg.LB
    NB = -(-Lmax // LB)
    bank_lanes = np.array([min(LB, Lmax - g * LB) for g in range(NB)], np.int64)
    if LB * cfg.B >= 2 * min(GFSR_A[1], GFSR_B[1]):
        raise ValueError("LB B bits per bank and cycle need more than 3-term leaps")
    # weights, rank tables, thresholds
    W, wm = float(np.log(3 * (1 - p) / p)), float(np.log((1 - p) / p))
    mode = int(3 * W < 4 * wm)
    KSL = max(4, d)
    rk, levels = _rank_table(KSL, mode)
    lv = [None]
    for tb, a, b in levels[1:]:
        lv.append((tb, a, b, W * a + wm * b))
    for tb, A_, B_ in (("T2", (-2, 3), (-2, 0, 2)), ("T1", (-2, 3), (-1, 1))):   # rank table = sorted positive dE
        mine = [(a, b) for (t_, a, b, _) in lv[1:] if t_ == tb]
        for a in range(*A_):
            for b in B_:
                assert ((W * a + wm * b > 1e-12) == ((a, b) in mine)), (tb, a, b)
        de = [W * a + wm * b for a, b in mine]
        assert all(x <= y + 1e-12 for x, y in zip(de, de[1:])), (tb, "levels not ascending")
    R = cfg.R
    betas = ladder(R, cfg.bmin)
    dE = np.array([0.0] + [x[3] for x in lv[1:]])
    thr = np.floor((1 << cfg.B) * np.exp(-betas[:, None] * dE[None, :])).astype(np.int64)
    thr = np.minimum(thr, (1 << cfg.B) - 1)
    thr[:, 0] = 1 << cfg.B
    Wq, wq = int(round(W * (1 << cfg.EF))), int(round(wm * (1 << cfg.EF)))
    if max(Wq, wq) >= (1 << cfg.EW):
        raise ValueError(f"Wq = {Wq}, wq = {wq} exceed {cfg.EW} bits")
    if R > 1:
        th = np.rint(float(1 << (cfg.TF + cfg.EF)) / (betas[:-1] - betas[1:]))
        theta = np.clip(th, 0, (1 << cfg.TW) - 1).astype(np.int64)
        xi_sw = np.minimum((theta[:, None] * lut(cfg.USW, cfg.LF)[None, :]) >> (cfg.LF + cfg.TF), (1 << cfg.XIW) - 1)
    else:
        theta, xi_sw = np.zeros(0, np.int64), np.zeros((0, 1 << cfg.USW), np.int64)
    chains = [_chains(HZ), _chains(HX)]
    gf2 = [_gf2_rref(HZ), _gf2_rref(HX)]
    obs = [secs[0][3], secs[1][3]]
    info = dict(moves=M, sl=int((mtab == 0).sum()), tl=int((mtab > 0).sum()), t1=int((mtab == 2).sum()),
                colors=len(cols), Lmax=Lmax, NB=NB, sizes=[len(mem) for _, mem in cols], literals=NL,
                data_clauses=ndc, meas_clauses=2 * m * T, theta_sat=bool((theta == (1 << cfg.TW) - 1).any()))
    ws = WSpec(d, T, bool(final), float(p), cfg, n, m, ns, ndc, NL, M, tptr, tli, tdat, tsp, mtab, np.array(mkind),
               np.array(mlg, np.int64), cls_mat, col_table, bank_lanes, KSL, rk, lv, mode, W, wm, Wq, wq, betas, thr,
               theta, xi_sw, chains, gf2, obs, info)
    _WS[key] = ws
    return ws


def to_memory_layout(ws: WSpec, lit: np.ndarray):
    """Literals [S, NL] -> the per-sector layout of DepolarizingMemory: (ex, ez), ex = [x(tau, q) | mZ(tau, c)]."""
    S = lit.shape[0]
    nd2 = 2 * ws.ndc
    mt = ws.m * ws.T
    ex = np.concatenate([lit[:, 0:nd2:2], lit[:, nd2:nd2 + mt]], 1)
    ez = np.concatenate([lit[:, 1:nd2:2], lit[:, nd2 + mt:]], 1)
    return ex.reshape(S, -1), ez.reshape(S, -1)


# ============================================================================================ stream helpers
def syndromes(D: np.ndarray, N: int, m: int) -> np.ndarray:
    """Raw syndromes s(c, t), t = 0..N, from the detectors D [S, (N + 1) m] (t = N: the perfect round). [S, N + 1, m]."""
    return np.bitwise_xor.accumulate(np.asarray(D, np.uint8).reshape(-1, N + 1, m), axis=1)


def start_state(ws: WSpec, s_x, s_z, sig_x, sig_z, sp_x=None, sp_z=None, start="chain") -> np.ndarray:
    """Start literals [S, NL]: m(c, tau) = s(c, t0 + tau) XOR sigma(c) (s_x, s_z: [S, T, m], the window's rounds), x = z = 0;
    final window: perfect-slice data for s(c, N) XOR sigma (sp_x, sp_z [S, m]) by the one-shot chains or ("gf2") the pivot solution."""
    S, T, m, n = s_x.shape[0], ws.T, ws.m, ws.n
    lit = np.zeros((S, ws.NL), np.uint8)
    nd2 = 2 * ws.ndc
    lit[:, nd2:nd2 + m * T] = (s_x[:, :T] ^ sig_x[:, None]).reshape(S, T * m)
    lit[:, nd2 + m * T:] = (s_z[:, :T] ^ sig_z[:, None]).reshape(S, T * m)
    if ws.final:
        for sec, (sp, sg) in enumerate(((sp_x, sig_x), (sp_z, sig_z))):
            r = (sp ^ sg).astype(np.int64)
            if start == "chain":
                x = (r @ ws.chains[sec].astype(np.int64)) % 2
            elif start == "gf2":
                pv, Tm = ws.gf2[sec]
                Y = (r @ Tm.T.astype(np.int64)) % 2
                x = np.zeros((S, n), np.int64)
                x[:, pv] = Y[:, :len(pv)]
            else:
                raise ValueError(start)
            base = 2 * (T * n) + sec
            lit[:, base:base + 2 * n:2] = x
    return lit


def window_bases(seed: int, shot_ids, t0: int, T: int, final: bool) -> np.ndarray:
    """64-bit generator base per shot and window (streams keyed by (seed, shot, t0, T, final))."""
    out = np.empty(len(shot_ids), np.uint64)
    tag = ((int(t0) << 32) | (int(T) << 8) | int(bool(final))) * 0xD6E8FEB86659FD93
    for i, s in enumerate(shot_ids):
        x = (int(seed) * 0x100000001B3 + int(s) * 0x9E3779B97F4A7C15 + tag) & M64
        out[i] = splitmix64_seq(x, 1)[0]
    return out


# ============================================================================================ RNG (word-level lane generator)
_SM_G = np.uint64(0x9E3779B97F4A7C15)
_SM_1 = np.uint64(0xBF58476D1CE4E5B9)
_SM_2 = np.uint64(0x94D049BB133111EB)
_SD_L = np.uint64(0xD1B54A32D192ED03)
_SD_S = np.uint64(0xABCDEF0123456789)
_REV8 = np.array([int(f"{i:08b}"[::-1], 2) for i in range(256)], np.int64)


@njit(cache=True)
def _sm_next(x):
    x = x + _SM_G
    z = x
    z = (z ^ (z >> np.uint64(30))) * _SM_1
    z = (z ^ (z >> np.uint64(27))) * _SM_2
    return x, z ^ (z >> np.uint64(31))


@njit(cache=True)
def _rd64(ring, b):
    """64 stream bits starting at bit position b (bit k of the result = stream bit b + k)."""
    wi = (b >> 6) & RW_MASK
    sh = b & 63
    lo = ring[wi] >> np.uint64(sh)
    if sh == 0:
        return lo
    return lo | (ring[(wi + 1) & RW_MASK] << np.uint64(64 - sh))


@njit(cache=True)
def _seed_ring(ring, seed, p):
    """Seed bits (splitmix64 of seed, as one_shot.bits_from) at stream positions P0 - p .. P0 - 1; returns P0 (first new bit)."""
    for w in range(RW):
        ring[w] = np.uint64(0)
    P0 = ((p + 63) // 64) * 64
    off = P0 - p
    x = seed
    k = 0
    anyb = False
    while k < p:
        x, z = _sm_next(x)
        for b in range(64):
            if k >= p:
                break
            if (z >> np.uint64(b)) & np.uint64(1):
                anyb = True
                pos = off + k
                ring[(pos >> 6) & RW_MASK] |= np.uint64(1) << np.uint64(pos & 63)
            k += 1
    if not anyb:
        ring[(off >> 6) & RW_MASK] |= np.uint64(1) << np.uint64(off & 63)
    return P0


@njit(cache=True)
def _fill(ring, gpos, need, p, q):
    """Generate aligned 64-bit words x_n = x_{n-p} XOR x_{n-q} until gpos >= need; returns gpos."""
    while gpos < need:
        ring[(gpos >> 6) & RW_MASK] = _rd64(ring, gpos - p) ^ _rd64(ring, gpos - q)
        gpos += 64
    return gpos


@njit(cache=True)
def _gfsr_stream(seed, p, q, nbits, out):
    ring = np.zeros(RW, np.uint64)
    g = _seed_ring(ring, seed, p)
    r = g
    for k in range(nbits):
        g = _fill(ring, g, r + 1, p, q)
        out[k] = np.uint8(_rd64(ring, r) & np.uint64(1))
        r += 1


def gfsr_stream(seed: int, which: str, nbits: int) -> np.ndarray:
    """First nbits output bits of lane generator a or b seeded as in the one-shot decoder (test hook)."""
    p, q = GFSR_A if which == "a" else GFSR_B
    out = np.zeros(nbits, np.uint8)
    _gfsr_stream(np.uint64(seed), p, q, nbits, out)
    return out


# ============================================================================================ kernel
@njit(cache=True)
def _window(lit0, base, NSW, R, K, B, USW, tptr, tli, tsp, mtab, mlg, cls_mat, bank_lanes, LB, rk, KSL, thr, xi_sw,
            Wq, wq, ndc, best_lit, outv, counts, rec, rec_on, ra, rb, gp, sw, seed_on):
    NL = lit0.shape[0]
    NC, Lmax = cls_mat.shape
    NB = bank_lanes.shape[0]
    pa, qa, pb, qb = 2281, 1029, 3217, 576
    lit = np.empty((R, NL), np.uint8)
    for i in range(R):
        for l in range(NL):
            lit[i, l] = lit0[l]
    nd0 = 0
    for j in range(ndc):
        nd0 += lit0[2 * j] | lit0[2 * j + 1]
    nm0 = 0
    for l in range(2 * ndc, NL):
        nm0 += lit0[l]
    nd = np.full(R, nd0, np.int64)
    nm = np.full(R, nm0, np.int64)
    cls = np.zeros(R, np.int64)
    lab = np.arange(R)
    holder = np.arange(R)
    # generators (ra, rb [R, NB, RW], gp [R, NB, 4] = gpos a, read pos a, gpos b, read pos b; sw [R - 1, 4]): seeded from
    # base (seed_on), or continued from the previous window of the shot (PTWConfig.carry)
    if seed_on:
        for i in range(R):
            for g in range(NB):
                sa = base ^ (np.uint64(2 * (i * NB + g) + 1) * _SD_L)
                sb = base ^ (np.uint64(2 * (i * NB + g) + 2) * _SD_L)
                gp[i, g, 0] = _seed_ring(ra[i, g], sa, pa); gp[i, g, 1] = gp[i, g, 0]
                gp[i, g, 2] = _seed_ring(rb[i, g], sb, pb); gp[i, g, 3] = gp[i, g, 2]
        for t in range(R - 1):
            x = base ^ (np.uint64(t + 1) * _SD_S)
            x, z0 = _sm_next(x)
            x, z1 = _sm_next(x)
            sw[t, 0] = np.uint32(z0 & np.uint64(0xFFFFFFFF)); sw[t, 1] = np.uint32(z0 >> np.uint64(32))
            sw[t, 2] = np.uint32(z1 & np.uint64(0xFFFFFFFF)); sw[t, 3] = np.uint32(z1 >> np.uint64(32))
    bestE = Wq * nd0 + wq * nm0
    for l in range(NL):
        best_lit[l] = lit0[l]
    bestC = 0
    mask16 = np.uint64((1 << B) - 1)
    for s in range(NSW):
        kb = 0
        x_ = s
        while x_ > 0:
            kb += 1
            x_ >>= 1
        for k in range(NC):
            for i in range(R):
                t = lab[i]
                L_ = lit[i]
                th_ = thr[t]
                for g in range(NB):
                    nb_ = bank_lanes[g] * B
                    gp[i, g, 0] = _fill(ra[i, g], gp[i, g, 0], gp[i, g, 1] + nb_, pa, qa)
                    gp[i, g, 2] = _fill(rb[i, g], gp[i, g, 2], gp[i, g, 3] + nb_, pb, qb)
                for j in range(Lmax):
                    r = cls_mat[k, j]
                    if r < 0:
                        continue
                    g = j // LB
                    o = (j - g * LB) * B
                    v = (_rd64(ra[i, g], gp[i, g, 1] + o) ^ _rd64(rb[i, g], gp[i, g, 3] + o)) & mask16
                    vi = np.int64(v)
                    u = ((_REV8[vi & 255] << 8) | _REV8[(vi >> 8) & 255]) >> (16 - B)
                    a = 0
                    b = 0
                    e1 = tsp[r]                                  # data-clause terms, then measurement-clause terms
                    for e in range(tptr[r], e1):                 # neuron input = sum of its clause outputs
                        li = tli[e]
                        a += (1 - np.int64(L_[li ^ 1])) * (1 - 2 * np.int64(L_[li]))
                    for e in range(e1, tptr[r + 1]):
                        b += 1 - 2 * np.int64(L_[tli[e]])
                    idx = rk[mtab[r], a + KSL, b + 2]
                    acc = np.uint8(1) if (idx == 0) | (u < th_[idx]) else np.uint8(0)
                    for e in range(tptr[r], tptr[r + 1]):
                        L_[tli[e]] ^= acc
                    nd[i] += acc * a
                    nm[i] += acc * b
                    cls[i] ^= acc * mlg[r]
                for g in range(NB):
                    nb_ = bank_lanes[g] * B
                    gp[i, g, 1] += nb_
                    gp[i, g, 3] += nb_
        if R > 1 and (s + 1) % K == 0:
            qd = (s + 1) // K - 1
            t = qd % 2
            while t < R - 1:
                w = xoshiro_next(sw[t])
                ii = np.int64(w >> np.uint64(32 - USW))
                i1 = holder[t + 1]; i0 = holder[t]
                D = (Wq * nd[i1] + wq * nm[i1]) - (Wq * nd[i0] + wq * nm[i0])
                if D <= xi_sw[t, ii]:
                    holder[t] = i1; holder[t + 1] = i0
                    lab[i1] = t; lab[i0] = t + 1
                t += 2
        bi = 0
        be = Wq * nd[0] + wq * nm[0]
        for i in range(1, R):
            e_ = Wq * nd[i] + wq * nm[i]
            if e_ < be:
                be = e_; bi = i
        if be < bestE:
            bestE = be
            bestC = cls[bi]
            for l in range(NL):
                best_lit[l] = lit[bi, l]
        counts[kb, cls[holder[0]]] += 1
        if rec_on:
            for i in range(R):
                rec[s, i, 0] = Wq * nd[i] + wq * nm[i]; rec[s, i, 1] = lab[i]; rec[s, i, 2] = cls[i]
    outv[0] = bestE
    outv[1] = bestC
    emax = 0
    for i in range(R):
        e_ = Wq * nd[i] + wq * nm[i]
        if e_ > emax:
            emax = e_
    outv[2] = emax


@njit(parallel=True, cache=True)
def _run_many(lit0, bases, NSW, R, K, B, USW, tptr, tli, tsp, mtab, mlg, cls_mat, bank_lanes, LB, rk, KSL, thr, xi_sw,
              Wq, wq, ndc, best_lit, outv, counts, ra, rb, gp, sw, seed_on):
    S = lit0.shape[0]
    for s in prange(S):
        rec = np.zeros((1, 1, 3), np.int64)
        _window(lit0[s], bases[s], NSW, R, K, B, USW, tptr, tli, tsp, mtab, mlg, cls_mat, bank_lanes, LB, rk, KSL, thr,
                xi_sw, Wq, wq, ndc, best_lit[s], outv[s], counts[s], rec, False, ra[s], rb[s], gp[s], sw[s], seed_on)


def gen_state(S: int, R: int, NB: int) -> dict:
    """Generator state of S shots (lane rings and positions, swap xoshiro words) carried across windows (PTWConfig.carry)."""
    return dict(ra=np.zeros((S, R, NB, RW), np.uint64), rb=np.zeros((S, R, NB, RW), np.uint64),
                gp=np.zeros((S, R, NB, 4), np.int64), sw=np.zeros((S, max(R - 1, 1), 4), np.uint32), fresh=True)


def run_window(ws: WSpec, lit0: np.ndarray, bases: np.ndarray, nsw: int, rec: bool = False, state: dict = None):
    """Decode one window for all shots (rows of lit0). nsw = sweeps (a power of two). Returns best_lit [S, NL], bestE,
    bestC, counts [S, log2(nsw) + 1, 4] (occupancy of the beta = 1 class per sweep block), Emax (largest final replica E);
    rec = True (one shot): also the per-sweep record [nsw, R, 3] = (E, label, class) of every replica.
    state = None: the generators are seeded from bases; state = gen_state(...): they continue from it (seeded from bases
    while state["fresh"]), and it is updated in place."""
    cfg = ws.cfg
    if nsw & (nsw - 1):
        raise ValueError("nsw must be a power of two")
    S = lit0.shape[0]
    KB = nsw.bit_length() - 1
    lit0 = np.ascontiguousarray(lit0, np.uint8)
    best = np.zeros((S, ws.NL), np.uint8)
    outv = np.zeros((S, 3), np.int64)
    counts = np.zeros((S, KB + 1, 4), np.int64)
    NB = ws.bank_lanes.shape[0]
    st = gen_state(S, cfg.R, NB) if state is None else state
    assert st["ra"].shape[:3] == (S, cfg.R, NB), "generator state does not match the window's banks"
    seed_on = bool(st["fresh"])
    st["fresh"] = False
    a = (ws.tptr, ws.tli, ws.tsp, ws.mtab, ws.mlg, ws.cls_mat, ws.bank_lanes, cfg.LB, ws.rk, ws.KSL, ws.thr, ws.xi_sw,
         ws.Wq, ws.wq, ws.ndc)
    if rec:
        assert S == 1
        r = np.zeros((nsw, cfg.R, 3), np.int64)
        _window(lit0[0], np.uint64(bases[0]), nsw, cfg.R, cfg.K, cfg.B, cfg.USW, *a, best[0], outv[0], counts[0], r, True,
                st["ra"][0], st["rb"][0], st["gp"][0], st["sw"][0], seed_on)
    else:
        _run_many(lit0, np.asarray(bases, np.uint64), nsw, cfg.R, cfg.K, cfg.B, cfg.USW, *a, best, outv, counts,
                  st["ra"], st["rb"], st["gp"], st["sw"], seed_on)
        r = None
    return dict(best_lit=best, bestE=outv[:, 0].copy(), bestC=outv[:, 1].copy(), Emax=outv[:, 2].copy(), counts=counts,
                ml=counts[:, KB].argmax(1), rec=r)


# ============================================================================================ stream
def frame_parity(ws: WSpec, lit: np.ndarray, sec: int, nsl: int) -> np.ndarray:
    """Parity of the sector's data literal on its observable line over slices tau < nsl."""
    idx = np.array([2 * (t * ws.n + q) + sec for t in range(nsl) for q in ws.obs[sec]], np.int64)
    return (lit[:, idx].sum(1) % 2).astype(np.int64)


def measurement_literals(ws: WSpec, lit: np.ndarray, sec: int) -> np.ndarray:
    """Measurement-error literals of sector sec as [S, T, m]."""
    nd2, mt = 2 * ws.ndc, ws.m * ws.T
    return lit[:, nd2 + sec * mt: nd2 + (sec + 1) * mt].reshape(-1, ws.T, ws.m)


def decode_stream(d: int, p: float, Dx: np.ndarray, Dz: np.ndarray, N: int, C: int, B: int, nsw: int,
                  cfg: PTWConfig = PTWConfig(), seed: int = 1, shot_ids=None, start: str = "chain",
                  debug: bool = False):
    """Streaming decode of S shots (detectors Dx, Dz [S, (N + 1) m], the layout of DepolarizingMemory.sample). Windows
    (C, B), nsw sweeps each.
    Returns ({"ml": (bx, bz), "me": (bx, bz)} logical frame bits, info); debug: info["windows"] has every window's spec,
    start literals, result and committed sigma. start: "chain" (the one-shot chains on the perfect slice), "gf2" (the
    pivot solution of PhenoCode.particular_fast), or a tuple of both (keys "<start>:ml" for all but the first)."""
    S = len(Dx)
    shot_ids = np.arange(S) if shot_ids is None else np.asarray(shot_ids)
    ws0 = window_spec(d, 1, True, p, cfg)
    m = ws0.m
    sx, sz = syndromes(Dx, N, m), syndromes(Dz, N, m)
    sig = [np.zeros((S, m), np.uint8), np.zeros((S, m), np.uint8)]
    fr = [np.zeros(S, np.int64), np.zeros(S, np.int64)]
    sched = schedule(N, C, B)
    wins = []
    emax = 0
    starts = (start,) if isinstance(start, str) else tuple(start)
    res = {}
    gst = None                 # generator state carried across windows (cfg.carry)
    for k, (t0, Tw, final) in enumerate(sched):
        ws = window_spec(d, Tw, final, p, cfg)
        # the start choice matters only on the final window's perfect slice: non-final windows are run once
        for st in (starts if final else starts[:1]):
            lit0 = start_state(ws, sx[:, t0:t0 + Tw], sz[:, t0:t0 + Tw], sig[0], sig[1], sx[:, N], sz[:, N], st)
            if cfg.carry and gst is None:
                gst = gen_state(S, cfg.R, ws.bank_lanes.shape[0])
            stw = None if not cfg.carry else (gst if (not final or st == starts[-1]) else copy.deepcopy(gst))
            o = run_window(ws, lit0, window_bases(seed, shot_ids, t0, Tw, final), nsw, state=stw)
            bl = o["best_lit"]
            emax = max(emax, int(o["Emax"].max()), int(o["bestE"].max()))
            if not final:
                Cc = sched[k + 1][0] - t0
                for sec, s_ in ((0, sx), (1, sz)):
                    sig[sec] = s_[:, t0 + Cc - 1] ^ measurement_literals(ws, bl, sec)[:, Cc - 1]
                    fr[sec] = fr[sec] ^ frame_parity(ws, bl, sec, Cc)
            else:
                px, pz = fr[0] ^ frame_parity(ws, bl, 0, ws.ns), fr[1] ^ frame_parity(ws, bl, 1, ws.ns)
                sh = o["bestC"] ^ o["ml"]
                pre = "" if st == starts[0] else f"{st}:"
                res.update({pre + "me": (px, pz), pre + "ml": (px ^ (sh & 1), pz ^ (sh >> 1))})
            if debug:
                wins.append(dict(t0=t0, T=Tw, final=final, start=st, ws=ws, lit0=lit0, out=o,
                                 sigma=(sig[0].copy(), sig[1].copy())))
    return res, dict(windows=wins, Emax=emax, sched=sched)


def failures(res: dict, ox: np.ndarray, oz: np.ndarray) -> dict:
    """Per-shot logical failure (bool) of each decision of :func:`decode_stream` against the true observables."""
    return {v: (bx != ox) | (bz != oz) for v, (bx, bz) in res.items()}


# ============================================================================================ board stream
def with_tables(ws: WSpec, thr, theta, Wq: int, wq: int, mode: int) -> WSpec:
    """The window spec with runtime tables as the host writes them to the board (thr [R, KSL + 12] or [R, KSL + 13] with
    column 0, theta [R - 1], fixed-point weights Wq, wq, rank mode); xi and the rank table follow from them."""
    cfg = ws.cfg
    thr = np.asarray(thr, np.int64)
    if thr.shape[1] == ws.KSL + 12:
        thr = np.concatenate([np.full((thr.shape[0], 1), 1 << cfg.B, np.int64), thr], 1)
    assert thr.shape == ws.thr.shape, (thr.shape, ws.thr.shape)
    theta = np.asarray(theta, np.int64)
    w2 = copy.copy(ws)
    w2.thr = thr
    w2.theta = theta
    w2.xi_sw = (np.minimum((theta[:, None] * lut(cfg.USW, cfg.LF)[None, :]) >> (cfg.LF + cfg.TF), (1 << cfg.XIW) - 1)
                if len(theta) else ws.xi_sw)
    w2.Wq, w2.wq, w2.mode = int(Wq), int(wq), int(mode)
    w2.rk, w2.levels = _rank_table(ws.KSL, int(mode))
    return w2


def ring_from_bits(hist: np.ndarray) -> tuple:
    """Lane generator ring (RW words) and stream position from an explicit history (P bits, oldest first), as _seed_ring
    places the seed bits: what the hardware holds after the host shifted these bits in."""
    P = len(hist)
    ring = np.zeros(RW, np.uint64)
    P0 = ((P + 63) // 64) * 64
    off = P0 - P
    for k in np.flatnonzero(hist):
        pos = off + int(k)
        ring[(pos >> 6) & RW_MASK] |= np.uint64(1) << np.uint64(pos & 63)
    return ring, P0


def decode_batch(d: int, p: float, sx: np.ndarray, sz: np.ndarray, nsw: int, cfg: PTWConfig, seed: int = 1,
                 state: dict = None, tables: tuple = None) -> tuple:
    """The board's stream: the shots of a batch decoded one after the other, window by window ((C, B) = (d, d)), on one
    generator stream: seeded once (state None: from window_bases(seed, [0], 0, T, False), i.e. as decode_stream seeds
    shot 0's first window) and carried across every window and shot (an extension of PTWConfig.carry from the windows of a
    shot to the shots of a batch). sx, sz: raw syndromes [S, N + 1, m] (t = N: the perfect round). tables: (thr, theta,
    Wq, wq, mode) as written by the host (default: window_spec's for p). Returns (records, state): one dict per window in
    stream order (shot, win, fin, best_e, best_c, dec_ml, counts of the last block, sigma bits [2 m] after the commit,
    frame and fr_ml as 2-bit values, bit 0 = X sector) and the generator state after the batch."""
    S, N1, m = sx.shape
    N = N1 - 1
    sched = schedule(N, d, d)
    KB = nsw.bit_length() - 1
    recs = []
    for s in range(S):
        sig = [np.zeros((1, m), np.uint8), np.zeros((1, m), np.uint8)]
        fr = [0, 0]
        for w, (t0, Tw, final) in enumerate(sched):
            ws = window_spec(d, Tw, final, p, cfg)
            if tables is not None:
                ws = with_tables(ws, *tables)
            lit0 = start_state(ws, sx[s:s + 1, t0:t0 + Tw], sz[s:s + 1, t0:t0 + Tw], sig[0], sig[1], sx[s:s + 1, N],
                               sz[s:s + 1, N])
            if state is None:
                state = gen_state(1, cfg.R, ws.bank_lanes.shape[0])
            o = run_window(ws, lit0, window_bases(seed, [0], 0, Tw, False), nsw, state=state)
            bl = o["best_lit"]
            ml, bc = int(o["ml"][0]), int(o["bestC"][0])
            if not final:
                for sec, s_ in ((0, sx), (1, sz)):
                    sig[sec] = (s_[s:s + 1, t0 + d - 1] ^ measurement_literals(ws, bl, sec)[:, d - 1]).astype(np.uint8)
                    fr[sec] ^= int(frame_parity(ws, bl, sec, d)[0])
                frml = 0
            else:
                for sec in (0, 1):
                    fr[sec] ^= int(frame_parity(ws, bl, sec, ws.ns)[0])
                sh = bc ^ ml
                frml = (fr[0] ^ (sh & 1)) | ((fr[1] ^ (sh >> 1)) << 1)
            recs.append(dict(shot=s, win=w, fin=int(final), best_e=int(o["bestE"][0]), best_c=bc, dec_ml=ml,
                             counts=[int(x) for x in o["counts"][0, KB]], sigma=np.concatenate([sig[0][0], sig[1][0]]),
                             frame=fr[0] | (fr[1] << 1), fr_ml=frml))
    return recs, state


def fpga_config(d: int, R: int = 16, bmin: float = 0.1, K: int = 1) -> PTWConfig:
    """Configuration of the FPGA design for distance d: the hardware schedule of its slab layout (FPGA_LAYOUTS) with
    generators carried across the windows of a shot."""
    k, lram = FPGA_LAYOUTS[d]
    return PTWConfig(R=R, bmin=bmin, K=K, k=k, lram=lram, carry=True)

"""One syndrome per shot: bit-exact model of the parallel-tempering (replica-exchange) surface-code decoder.

This module is the software specification of the FPGA decoder that decodes one syndrome per shot. Every random bit,
threshold and decision is reproduced exactly, so the logical error rates computed here are those of the hardware.

Workload: rotated surface code, depolarizing code-capacity noise, move library ``stabilizers_logical_all`` (stabilizer
moves and all d straight logical lines per type). Energy W = #{q : x_q or z_q} (one clause x_q OR z_q per qubit, weight
w3 = ln(3 (1 - p) / p)). The input of the neuron of move r (an X move on qubit set S; Z moves symmetric) is its energy change
dW_r = sum over q in S of [z_q = 0] (1 - 2 x_q).

Machine: NS replicas, each with its own state, energy register E (= W), 2-bit class register c (XOR of the accepted logical
moves: 1 = X logical, 2 = Z logical; relative to the start state), temperature label lab (0 = beta 1) and its own random
generator. holder[t] = replica with label t.

Start state (every replica): c0 = XOR over the fired checks of a fixed minimum-weight chain from the check to the boundary
(breadth-first search in the matching graph of each check type, deterministic tie-break). Labels lab[i] = i.

Sweep s = 0 .. Nsw - 1 consists of one cycle per color class, in the fixed order X-stabilizer colors, X logicals,
Z-stabilizer colors, Z logicals (colors from the move masks: moves of one color share no clause). In color cycle k, every
replica i:

1. draws N = Lmax * B bits from its generator (all lanes, used or not); lane j gets u_j = bits [jB, (j + 1)B) (first bit =
   most significant bit);
2. lane j < |class k| tests move r = class_k[j]: accept iff dW_r <= 0 or ``u_j < T[lab_i][dW_r]``, with
   ``T[t][k] = floor(2^B exp(-beta_t w3 k))``;
3. all accepted moves of the color flip together (they are independent); E += sum dW, c ^= logical bits.

After each color cycle: per replica best (E < bestE strictly -> bestE, bestC = c); occupancy count of c[holder[0]] in the
block of the sweep (block(s) = bit_length(s) for s < Nsw // 2 and bit_length(Nsw - 1) from Nsw // 2 on: blocks double, and
the last block is the second half of the run, sweeps Nsw // 2 .. Nsw - 1, also when Nsw is not a power of two).

After sweep s, if (s + 1) % K == 0 and s is not the last sweep: swap round q = (s + 1) // K - 1 over the pairs
t = q % 2, q % 2 + 2, ... < NS - 1. Pair t draws i = top USW bits of its xoshiro128** generator,
D = E[holder[t + 1]] - E[holder[t]], and exchanges the labels iff D <= XI_t[i], with XI_t[i] = (theta_t L[i]) >> (LF + TF),
L[i] = round(-ln((i + 1/2) / 2^USW) 2^LF) and theta_t = round(2^TF / ((beta_t - beta_t+1) w3)). This is the Metropolis rule
on (beta_t - beta_t+1)(E_t - E_t+1); the energies are exact (the engine drains its energy pipeline before a round).

Decisions of a run of Nsw = 2^k sweeps: maximum likelihood (ML) = argmax of the counts of block k (sweeps [2^(k-1), 2^k);
ties: lowest class); minimum energy = bestC of the replica with the lowest bestE (ties: lowest replica). One long run gives
every budget 2^k.

Lane generator (per replica): y_n = a_n XOR b_n, a_n = a_{n-2281} XOR a_{n-1029}, b_n = b_{n-3217} XOR b_{n-576} (primitive
trinomials of Mersenne-exponent degree). N bits per cycle in parallel: every new bit is an XOR of at most 3 old bits per
generator when N < 1152 (one LUT6 per output bit in total). Seeds: splitmix64.

Main entry points:

- :func:`build_spec` compiles the move library and the hardware tables for (d, p, :class:`PTConfig`);
- :func:`c0_from_syndrome` gives the start state of every shot;
- :func:`run_pt` decodes many shots at a power-of-two sweep budget and returns the class counts per sweep, from which
  :func:`ml_class_by_budget` and :func:`min_energy_class_by_budget` give the decision at every smaller budget;
- :func:`run_fixed_budget` runs any sweep count, optionally with the early stop of the hardware;
- :func:`run_shot` decodes one shot with a per-cycle record (the reference trace for register-transfer-level simulation).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from nimbus.compiler import compile_model
from nimbus.problems.surface_code import SurfaceCodeProblem

__all__ = ["PTConfig", "PTSpec", "build_spec", "c0_from_syndrome", "seed_states", "run_pt", "run_shot", "run_fixed_budget",
           "ml_class_by_budget", "min_energy_class_by_budget", "xoshiro_next", "splitmix64_seq", "bits_from", "GFSR_A",
           "GFSR_B", "XIW", "ladder", "make_tables", "swap_thresholds", "with_tables", "TW_THETA", "FPGA_REPLICAS",
           "FPGA_SWEEPS"]

GFSR_A = (2281, 1029)
GFSR_B = (3217, 576)
XIW = 10                       # swap threshold width (saturated)
TW_THETA = 20                  # width of the runtime-loadable swap constants theta_t

#: Replicas NS of the FPGA designs, per code distance d.
FPGA_REPLICAS = {3: 16, 5: 16, 7: 64, 9: 56, 11: 40, 13: 32, 15: 24}
#: Sweep budget per shot used on the board, per code distance d (not always a power of two; see run_fixed_budget).
FPGA_SWEEPS = {3: 46, 5: 128, 7: 216, 9: 304, 11: 430, 13: 512, 15: 1722}


@dataclass(frozen=True)
class PTConfig:
    """Machine parameters.

    NS: replicas; bmin: smallest inverse temperature of the geometric ladder beta = geomspace(1, bmin, NS); K: a swap
    round after every K sweeps (a K larger than the sweep count disables replica exchange); B: uniform bits per lane;
    USW: uniform bits per swap decision; LF, TF: fractional bits of the swap log table and of theta_t."""
    NS: int = 16
    bmin: float = 0.1
    K: int = 1                 # swap round after every K sweeps
    B: int = 16                # lane uniform bits
    USW: int = 12              # swap uniform bits
    LF: int = 16
    TF: int = 12


@dataclass
class PTSpec:
    """Moves, color classes and hardware tables for one (d, p, config); built by :func:`build_spec`."""
    d: int
    p: float
    cfg: PTConfig
    prob: object
    n: int
    R: int
    sup: np.ndarray            # [R, L] qubit supports, padded with n (x moves) / n + 1 (z moves) dummies
    blk: np.ndarray            # [R] 0 = x move, 1 = z move
    logbit: np.ndarray         # [R] 0, 1 (X logical), 2 (Z logical)
    classes: list              # color classes in sweep order (arrays of move ids)
    class_kind: list
    cls_mat: np.ndarray        # [C, Lmax] move per lane, -1 idle
    betas: np.ndarray
    kmax: int
    thr: np.ndarray            # [NS, kmax + 1] lane thresholds (k = 0 unused)
    xi_sw: np.ndarray          # [NS - 1, 2^USW]
    theta_sw: np.ndarray
    chains: list               # per block: [m_b, n] uint8 chain of each check
    w3: float
    N: int                     # lane bits per replica per cycle
    model: object = None
    info: dict = field(default_factory=dict)


def lut(U: int, LF: int) -> np.ndarray:
    """Swap log table L[i] = round(-ln((i + 1/2) / 2^U) 2^LF), i < 2^U."""
    i = np.arange(1 << U, dtype=np.float64)
    return np.rint(-np.log((i + 0.5) / float(1 << U)) * float(1 << LF)).astype(np.int64)


def ladder(NS: int, bmin: float) -> np.ndarray:
    """Default temperature ladder: beta_t = geomspace(1, bmin, NS)."""
    return np.geomspace(1.0, bmin, NS) if NS > 1 else np.array([1.0])


def make_tables(w3: float, betas, B: int, kmax: int, TF: int = 12, TW: int = TW_THETA):
    """Runtime-loadable tables (the registers the host writes to the FPGA):
    thr [NS, kmax + 1], thr[t, k] = floor(2^B exp(-beta_t w3 k)) for k >= 1, saturated to 2^B - 1 (k = 0 unused, = 2^B);
    theta [NS - 1], theta_t = round(2^TF / ((beta_t - beta_t+1) w3)), saturated to [0, 2^TW - 1].
    For p < 3/4 and a strictly decreasing ladder with beta > 0 no saturation occurs."""
    betas = np.asarray(betas, np.float64)
    ks = np.arange(kmax + 1)
    thr = np.floor((1 << B) * np.exp(-betas[:, None] * w3 * ks[None, :])).astype(np.int64)
    thr[:, 1:] = np.minimum(thr[:, 1:], (1 << B) - 1)
    thr[:, 0] = 1 << B
    if len(betas) > 1:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            th = np.rint((1 << TF) / ((betas[:-1] - betas[1:]) * w3))
        th = np.where(np.isfinite(th), th, (1 << TW) - 1)
        theta = np.clip(th, 0, (1 << TW) - 1).astype(np.int64)
    else:
        theta = np.zeros(0, np.int64)
    return thr, theta


def swap_thresholds(theta: np.ndarray, cfg: PTConfig) -> np.ndarray:
    """XI_t[i] = (theta_t L[i]) >> (LF + TF), saturated to 2^XIW - 1 (|D| <= n < 2^XIW)."""
    if len(theta) == 0:
        return np.zeros((0, 1 << cfg.USW), np.int64)
    Ls = lut(cfg.USW, cfg.LF)
    return np.minimum((np.asarray(theta, np.int64)[:, None] * Ls[None, :]) >> (cfg.LF + cfg.TF), (1 << XIW) - 1)


def with_tables(spec: "PTSpec", thr: np.ndarray, theta: np.ndarray) -> "PTSpec":
    """Copy of spec that uses the given runtime tables (as written to the board): thr [NS, >= kmax + 1] or [NS, kmax]
    (k = 1..kmax), theta [NS - 1]. The thresholds must not increase with k (the hardware's thermometer acceptance relies on it)."""
    import dataclasses
    NS, km = spec.cfg.NS, spec.kmax
    thr = np.asarray(thr, np.int64)
    if thr.shape == (NS, km):
        thr = np.concatenate([np.full((NS, 1), 1 << spec.cfg.B, np.int64), thr], 1)
    thr = thr[:, :km + 1].copy()
    if thr.shape != (NS, km + 1):
        raise ValueError(f"thr shape {thr.shape}, expected ({NS}, {km + 1})")
    if (thr[:, 1:] >= (1 << spec.cfg.B)).any() or (thr < 0).any():
        raise ValueError("thresholds must lie in [0, 2^B)")
    if (np.diff(thr[:, 1:], axis=1) > 0).any():
        raise ValueError("thresholds must not increase with k")
    theta = np.asarray(theta, np.int64).copy()
    if theta.shape != (max(NS - 1, 0),) or (theta < 0).any() or (theta >= (1 << TW_THETA)).any():
        raise ValueError("theta must have NS - 1 entries in [0, 2^TW)")
    return dataclasses.replace(spec, thr=thr, theta_sw=theta, xi_sw=swap_thresholds(theta, spec.cfg))


def _greedy_color(moves, sups):
    cols = []
    for r in moves:
        for c in cols:
            if all(not (set(sups[r]) & set(sups[x])) for x in c):
                c.append(r)
                break
        else:
            cols.append([r])
    return cols


def _chains(H: np.ndarray) -> np.ndarray:
    """Minimum-weight chain from each check to the boundary (matching graph: qubit = edge between its checks, or check-boundary)."""
    m, n = H.shape
    B = m                                     # boundary node
    adj = [[] for _ in range(m + 1)]
    for q in range(n):
        cs = np.flatnonzero(H[:, q])
        if len(cs) == 2:
            a, b = int(cs[0]), int(cs[1])
        elif len(cs) == 1:
            a, b = int(cs[0]), B
        else:
            continue
        adj[a].append((b, q)); adj[b].append((a, q))
    for lst in adj:
        lst.sort()
    out = np.zeros((m, n), np.uint8)
    for j in range(m):
        prev = {j: None}
        dq = deque([j])
        while dq:
            u = dq.popleft()
            if u == B:
                break
            for v, q in adj[u]:
                if v not in prev:
                    prev[v] = (u, q)
                    dq.append(v)
        u = B
        while prev[u] is not None:
            pu, q = prev[u]
            out[j, q] ^= 1
            u = pu
        syn = (H @ out[j]) % 2
        assert syn[j] == 1 and syn.sum() == 1, "chain does not reproduce a unit syndrome"
    return out


def build_spec(d: int, p: float, cfg: PTConfig = PTConfig(), library: str = "stabilizers_logical_all") -> PTSpec:
    """Decoder specification for distance d and physical error rate p (the tables use w3 = ln(3 (1 - p) / p)).

    ``library`` is a move library of :class:`~nimbus.problems.surface_code.SurfaceCodeProblem`:
    ``"stabilizers_logical_all"`` (stabilizers and all d straight logical lines per type, the hardware library) or
    ``"stabilizers_logical"`` (one logical line per type)."""
    prob = SurfaceCodeProblem(d, p, noise="depolarizing", library=library)
    n = prob.nq
    model = compile_model(prob.structure(), prob.transition_library(), identities=())
    R = model.n_groups
    ro = np.asarray(model.readout_clause_of_spin)
    spin_of_clause = {int(k): s for s, k in enumerate(ro)}
    gp, rc = np.asarray(model.route_ptr), np.asarray(model.route_clause)
    lx = np.zeros(n, np.uint8); lx[list(prob.code["lx_all"][0])] = 1
    lz = np.zeros(n, np.uint8); lz[list(prob.code["lz_all"][0])] = 1
    sups, blk, logbit, clauses = [], [], [], []
    for r in range(R):
        cl = set(int(k) for k in rc[gp[r]:gp[r + 1]])
        sp = sorted(spin_of_clause[k] for k in cl if k in spin_of_clause)
        b = 0 if sp[0] < n else 1
        if any((s >= n) != bool(b) for s in sp):
            raise ValueError(f"move {r} mixes x and z spins")
        q = [s - b * n for s in sp]
        f = np.zeros(n, np.uint8); f[q] = 1
        lg = int((f @ lz) % 2) if b == 0 else int((f @ lx) % 2)    # nontrivial logical iff it anticommutes with the other type
        sups.append(q); blk.append(b); logbit.append((1 if b == 0 else 2) if lg else 0); clauses.append(cl)
    # color classes from the masks: per kind (X stabilizers, X logicals, Z stabilizers, Z logicals), greedy on shared qubits
    classes, kinds = [], []
    for b in (0, 1):
        for islog in (False, True):
            mv = [r for r in range(R) if blk[r] == b and bool(logbit[r]) == islog]
            for c in _greedy_color(mv, sups):
                classes.append(np.array(sorted(c), np.int64)); kinds.append(("X" if b == 0 else "Z") + ("log" if islog else "stab"))
    for c in classes:      # no two moves of a color share a clause (checked on the compiled clause sets)
        seen = set()
        for r in c:
            if seen & clauses[r]:
                raise ValueError("color class shares a clause")
            seen |= clauses[r]
    assert sorted(np.concatenate(classes).tolist()) == list(range(R))
    L = max(len(s) for s in sups)
    sup = np.array([s + [n + b] * (L - len(s)) for s, b in zip(sups, blk)], dtype=np.int64)
    Lmax = max(len(c) for c in classes)
    cls_mat = -np.ones((len(classes), Lmax), dtype=np.int64)
    for k, c in enumerate(classes):
        cls_mat[k, :len(c)] = c
    w3 = float(prob.w)
    betas = ladder(cfg.NS, cfg.bmin)
    kmax = L
    thr, theta_sw = make_tables(w3, betas, cfg.B, kmax, cfg.TF)
    xi_sw = swap_thresholds(theta_sw, cfg)
    chains = [_chains(np.asarray(Hc, np.uint8)) for (Hc, _off, *_r) in prob._checks]
    N = Lmax * cfg.B
    if N >= 2 * min(GFSR_A[1], GFSR_B[1]):
        raise ValueError(f"N = {N} lane bits per cycle needs more than 3-term leaps")
    fanin = [int(np.max(ch.sum(0))) for ch in chains]
    info = dict(n=n, R=R, L=L, M=int(model.n_clauses), C=len(classes), Lmax=Lmax, class_sizes=[len(c) for c in classes],
                class_kinds=kinds, w3=w3, N=N, c0_xor_fanin=fanin, chain_weight_max=[int(ch.sum(1).max()) for ch in chains])
    return PTSpec(d, p, cfg, prob, n, R, sup, np.array(blk), np.array(logbit), classes, kinds, cls_mat, betas, kmax, thr, xi_sw,
                  theta_sw, chains, w3, N, model, info)


def c0_from_syndrome(spec: PTSpec, syn: np.ndarray) -> np.ndarray:
    """Start states [S, 2n] (x | z) = XOR over fired checks of their chains."""
    syn = np.atleast_2d(np.asarray(syn, np.uint8))
    m0 = spec.chains[0].shape[0]
    x = (syn[:, :m0].astype(np.int64) @ spec.chains[0]) % 2
    z = (syn[:, m0:].astype(np.int64) @ spec.chains[1]) % 2
    return np.concatenate([x, z], 1).astype(np.uint8)


# ============================================================================================ RNG
def splitmix64_seq(seed: int, count: int) -> np.ndarray:
    """``count`` successive splitmix64 outputs of ``seed`` (uint64)."""
    out = np.zeros(count, dtype=np.uint64)
    x = seed & 0xFFFFFFFFFFFFFFFF
    for k in range(count):
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        out[k] = z ^ (z >> 31)
    return out


def _bits_from(seed: int, nbits: int) -> np.ndarray:
    """Seed history of a lane generator: nbits splitmix64 bits (LSB first); never all zero."""
    z = splitmix64_seq(seed, (nbits + 63) // 64)
    b = np.unpackbits(z.view(np.uint8), bitorder="little")[:nbits]
    if not b.any():
        b[0] = 1
    return b.astype(np.uint8)


def seed_states(spec: PTSpec, shots: int, seed: int):
    """Generator states per shot: ga [S, NS, 2281], gb [S, NS, 3217] (bit history, oldest first), sw [S, NS-1, 4] (xoshiro)."""
    NS = spec.cfg.NS
    ga = np.zeros((shots, NS, GFSR_A[0]), np.uint8)
    gb = np.zeros((shots, NS, GFSR_B[0]), np.uint8)
    sw = np.zeros((shots, max(NS - 1, 0), 4), np.uint32)
    for s in range(shots):
        base = (seed * 0x100000001B3 + s * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        for i in range(NS):
            ga[s, i] = _bits_from(base ^ (2 * i + 1) * 0xD1B54A32D192ED03, GFSR_A[0])
            gb[s, i] = _bits_from(base ^ (2 * i + 2) * 0xD1B54A32D192ED03, GFSR_B[0])
        for t in range(NS - 1):
            z = splitmix64_seq(base ^ (t + 1) * 0xABCDEF0123456789, 2)
            sw[s, t] = [z[0] & 0xFFFFFFFF, z[0] >> 32, z[1] & 0xFFFFFFFF, z[1] >> 32]
    return ga, gb, sw


@njit(cache=True)
def _rotl(x, k):
    return ((x << np.uint64(k)) | (x >> np.uint64(32 - k))) & np.uint64(0xFFFFFFFF)


@njit(cache=True)
def xoshiro_next(st):
    """Output of xoshiro128** for state st (uint32[4]); steps st in place."""
    m = np.uint64(0xFFFFFFFF)
    s0 = np.uint64(st[0]); s1 = np.uint64(st[1]); s2 = np.uint64(st[2]); s3 = np.uint64(st[3])
    res = (_rotl((s1 * np.uint64(5)) & m, 7) * np.uint64(9)) & m
    t = (s1 << np.uint64(9)) & m
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = _rotl(s3, 11)
    st[0] = np.uint32(s0); st[1] = np.uint32(s1); st[2] = np.uint32(s2); st[3] = np.uint32(s3)
    return res


@njit(cache=True)
def _gfsr_block(hist, head, p, q, N, out):
    """Append N bits to the circular history hist (length p, oldest at head) and write them to out."""
    h2 = head - q                                       # position of x_{n-q}; head holds x_{n-p} (oldest)
    if h2 < 0:
        h2 += p
    for j in range(N):
        b = hist[head] ^ hist[h2]
        hist[head] = b
        out[j] = b
        head += 1
        if head == p:
            head = 0
        h2 += 1
        if h2 == p:
            h2 = 0
    return head


# ============================================================================================ kernel
@njit(cache=True)
def _shot(c0x, c0z, n, sup, blk, logbit, cls_mat, thr, kmax, xi_sw, NS, K, B, USW, NSW, ga, gb, heads, sw,
          counts, best_at, rec, rec_on, fin_E, fin_C, fin_lab, fin_bestE, fin_bestC, fin_X, fin_Z, cs, bs, es):
    L = sup.shape[1]
    NCL, Lmax = cls_mat.shape
    N = Lmax * B
    pa, qa, pb, qb = 2281, 1029, 3217, 576
    X = np.zeros((NS, n + 2), np.uint8)
    Z = np.zeros((NS, n + 2), np.uint8)
    W0 = 0
    for q in range(n):
        if c0x[q] | c0z[q]:
            W0 += 1
    for i in range(NS):
        for q in range(n):
            X[i, q] = c0x[q]; Z[i, q] = c0z[q]
        X[i, n + 1] = 1
        Z[i, n] = 1
    E = np.full(NS, W0, np.int64)
    C = np.zeros(NS, np.int64)
    lab = np.arange(NS)
    holder = np.arange(NS)
    bestE = np.full(NS, W0, np.int64)
    bestC = np.zeros(NS, np.int64)
    ba = np.zeros(N, np.uint8); bb = np.zeros(N, np.uint8)
    cyc = 0
    for s in range(NSW):
        kb = 0
        x = s if s < (NSW >> 1) else NSW - 1
        while x > 0:
            kb += 1
            x >>= 1
        for k in range(NCL):
            for i in range(NS):
                heads[i, 0] = _gfsr_block(ga[i], heads[i, 0], pa, qa, N, ba)
                heads[i, 1] = _gfsr_block(gb[i], heads[i, 1], pb, qb, N, bb)
                nacc = 0
                for j in range(Lmax):
                    r = cls_mat[k, j]
                    if r < 0:
                        continue
                    u = 0
                    for t in range(B):
                        u = (u << 1) | (ba[j * B + t] ^ bb[j * B + t])
                    dW = 0
                    for m in range(L):
                        q = sup[r, m]
                        if blk[r] == 0:
                            own = X[i, q]; par = Z[i, q]
                        else:
                            own = Z[i, q]; par = X[i, q]
                        if par == 0:
                            dW += 1 - 2 * np.int64(own)
                    if dW <= 0 or u < thr[lab[i], dW]:
                        for m in range(L):
                            q = sup[r, m]
                            if blk[r] == 0:
                                X[i, q] ^= 1
                            else:
                                Z[i, q] ^= 1
                        E[i] += dW
                        C[i] ^= logbit[r]
                        nacc += 1
                if rec_on:
                    rec[cyc, i, 0] = E[i]; rec[cyc, i, 1] = C[i]; rec[cyc, i, 2] = nacc; rec[cyc, i, 3] = lab[i]
            for i in range(NS):
                if E[i] < bestE[i]:
                    bestE[i] = E[i]; bestC[i] = C[i]
            counts[kb, C[holder[0]]] += 1
            if cs.shape[0] == NSW:
                cs[s, C[holder[0]]] += 1
            cyc += 1
        if NS > 1 and (s + 1) % K == 0 and s != NSW - 1:
            qd = (s + 1) // K - 1
            t = qd % 2
            while t < NS - 1:
                w = xoshiro_next(sw[t])
                ii = np.int64(w >> np.uint64(32 - USW))
                D = E[holder[t + 1]] - E[holder[t]]
                if D <= xi_sw[t, ii]:
                    a = holder[t]; holder[t] = holder[t + 1]; holder[t + 1] = a
                    lab[holder[t]] = t; lab[holder[t + 1]] = t + 1
                t += 2
        if bs.shape[0] == NSW:
            bi = 0
            for i in range(1, NS):
                if bestE[i] < bestE[bi]:
                    bi = i
            bs[s] = bestC[bi]
        nn = s + 1
        if nn & (nn - 1) == 0:
            kk = 0
            x = nn
            while x > 1:
                kk += 1
                x >>= 1
            bi = 0
            for i in range(1, NS):
                if bestE[i] < bestE[bi]:
                    bi = i
            best_at[kk] = bestC[bi]
            # early stop at a power-of-two checkpoint (after its swap round): leading class >= es[1] / 32 of the block counts
            if es[0] > 0 and nn >= es[0] and s != NSW - 1:
                tot = counts[kb, 0] + counts[kb, 1] + counts[kb, 2] + counts[kb, 3]
                mx = max(max(counts[kb, 0], counts[kb, 1]), max(counts[kb, 2], counts[kb, 3]))
                if 32 * mx >= es[1] * tot:
                    es[2] = nn
                    break
    for i in range(NS):
        fin_E[i] = E[i]; fin_C[i] = C[i]; fin_lab[i] = lab[i]; fin_bestE[i] = bestE[i]; fin_bestC[i] = bestC[i]
        for q in range(n):
            fin_X[i, q] = X[i, q]; fin_Z[i, q] = Z[i, q]


@njit(parallel=True, cache=True)
def _run_many(c0, n, sup, blk, logbit, cls_mat, thr, kmax, xi_sw, NS, K, B, USW, NSW, ga, gb, heads, sw, counts, best_at, cs, bs):
    S = c0.shape[0]
    for s in prange(S):
        rec = np.zeros((1, 1, 4), np.int64)
        fe = np.zeros(NS, np.int64); fc = np.zeros(NS, np.int64); fl = np.zeros(NS, np.int64)
        be = np.zeros(NS, np.int64); bc = np.zeros(NS, np.int64)
        fx = np.zeros((NS, n), np.uint8); fz = np.zeros((NS, n), np.uint8)
        _shot(c0[s, :n], c0[s, n:], n, sup, blk, logbit, cls_mat, thr, kmax, xi_sw, NS, K, B, USW, NSW, ga[s], gb[s], heads[s],
              sw[s], counts[s], best_at[s], rec, False, fe, fc, fl, be, bc, fx, fz, cs[s], bs[s], np.array([0, 0, NSW]))


def run_pt(spec: PTSpec, c0: np.ndarray, log2sweeps: int, seed: int = 1, states=None):
    """All shots (rows of c0), 2^log2sweeps sweeps, independent generators per shot. Returns counts [S, k+1, 4] (per sweep block),
    best_at [S, k+1] (min-energy class at 2^k sweeps), ml [S, k+1], per-sweep counts cs [S, NSW, 4] (uint8) and the
    minimum-energy class after every sweep bs [S, NSW]."""
    cfg = spec.cfg
    S = c0.shape[0]
    ga, gb, sw = seed_states(spec, S, seed) if states is None else states
    heads = np.zeros((S, cfg.NS, 2), np.int64)
    Kb = log2sweeps
    counts = np.zeros((S, Kb + 1, 4), np.int64)
    best_at = np.zeros((S, Kb + 1), np.int64)
    cs = np.zeros((S, 1 << Kb, 4), np.uint8)
    bs = np.zeros((S, 1 << Kb), np.int8)
    _run_many(np.ascontiguousarray(c0, dtype=np.uint8), spec.n, spec.sup, spec.blk, spec.logbit, spec.cls_mat, spec.thr,
              spec.kmax, spec.xi_sw, cfg.NS, cfg.K, cfg.B, cfg.USW, 1 << Kb, ga, gb, heads, sw, counts, best_at, cs, bs)
    return dict(counts=counts, best_at=best_at, ml=counts.argmax(2), cs=cs, bs=bs)


def run_shot(spec: PTSpec, c0row: np.ndarray, nsweeps: int, ga1, gb1, heads1, sw1, gmin: int = 0, th32: int = 0):
    """One shot with a per-cycle record (bit-exactness reference for hardware
    simulation). Generator states advance in place.
    gmin > 0 enables early stopping at power-of-two sweep counts >= gmin (leading class >= th32 / 32 of the block counts);
    the result's ``stop`` is the number of sweeps run."""
    cfg = spec.cfg
    NS, n = cfg.NS, spec.n
    kb = max(1, int(nsweeps).bit_length())
    counts = np.zeros((kb + 1, 4), np.int64)
    best_at = np.zeros(kb + 1, np.int64)
    rec = np.zeros((nsweeps * spec.cls_mat.shape[0], NS, 4), np.int64)
    fe = np.zeros(NS, np.int64); fc = np.zeros(NS, np.int64); fl = np.zeros(NS, np.int64)
    be = np.zeros(NS, np.int64); bc = np.zeros(NS, np.int64)
    fx = np.zeros((NS, n), np.uint8); fz = np.zeros((NS, n), np.uint8)
    c0row = np.ascontiguousarray(c0row, dtype=np.uint8)
    es = np.array([gmin, th32, nsweeps], np.int64)
    _shot(c0row[:n], c0row[n:], n, spec.sup, spec.blk, spec.logbit, spec.cls_mat, spec.thr, spec.kmax, spec.xi_sw, NS, cfg.K,
          cfg.B, cfg.USW, nsweeps, ga1, gb1, heads1, sw1, counts, best_at, rec, True, fe, fc, fl, be, bc, fx, fz,
          np.zeros((1, 4), np.uint8), np.zeros(1, np.int8), es)
    stop = int(es[2])
    return dict(E=fe, C=fc, lab=fl, bestE=be, bestC=bc, X=fx, Z=fz, counts=counts, best_at=best_at,
                rec=rec[: stop * spec.cls_mat.shape[0]], stop=stop)


bits_from = _bits_from


# ============================================================================================ budgets and early stop
def ml_class_by_budget(cs: np.ndarray, budgets) -> np.ndarray:
    """Maximum-likelihood class at each budget g (sweeps) from the per-sweep counts ``cs`` [S, Nsw, 4] of :func:`run_pt`:
    argmax of the beta = 1 class counts over sweeps [g // 2, g), as the hardware decides at budget g. Returns
    [len(budgets), S]."""
    S = cs.shape[0]
    cum = np.concatenate([np.zeros((S, 1, 4), np.int32), cs.astype(np.int32).cumsum(1)], 1)
    return np.stack([(cum[:, g] - cum[:, g // 2]).argmax(1) for g in budgets])


def min_energy_class_by_budget(bs: np.ndarray, budgets) -> np.ndarray:
    """Minimum-energy class after g sweeps for each budget g, from ``bs`` [S, Nsw] of :func:`run_pt`. [len(budgets), S]."""
    return np.stack([bs[:, g - 1] for g in budgets])


@njit(parallel=True, cache=True)
def _run_many_es(c0, n, sup, blk, logbit, cls_mat, thr, kmax, xi_sw, NS, K, B, USW, NSW, ga, gb, heads, sw, counts,
                 stops, gmin, th32):
    S = c0.shape[0]
    for s in prange(S):
        rec = np.zeros((1, 1, 4), np.int64)
        best_at = np.zeros(counts.shape[1], np.int64)
        fe = np.zeros(NS, np.int64); fc = np.zeros(NS, np.int64); fl = np.zeros(NS, np.int64)
        be = np.zeros(NS, np.int64); bc = np.zeros(NS, np.int64)
        fx = np.zeros((NS, n), np.uint8); fz = np.zeros((NS, n), np.uint8)
        es = np.array([gmin, th32, NSW])
        _shot(c0[s, :n], c0[s, n:], n, sup, blk, logbit, cls_mat, thr, kmax, xi_sw, NS, K, B, USW, NSW, ga[s], gb[s],
              heads[s], sw[s], counts[s], best_at, rec, False, fe, fc, fl, be, bc, fx, fz,
              np.zeros((1, 4), np.uint8), np.zeros(1, np.int8), es)
        stops[s] = es[2]


def run_fixed_budget(spec: PTSpec, c0: np.ndarray, nsweeps: int, seed: int = 1, gmin: int = 0, th32: int = 0):
    """All shots (rows of c0) with exactly ``nsweeps`` sweeps (any positive count), as the board runs them.

    The decision is the ML class of the last count block, which is the second half of the run (sweeps
    [nsweeps // 2, nsweeps)). gmin > 0 enables the hardware's early stop: at every power-of-two sweep count g >= gmin
    (after its swap round) the shot stops when the leading class holds at least th32 / 32 of the counts of the current
    block; the decision is then taken from that block. Generators are seeded as in :func:`run_pt` with the same seed,
    so at a power-of-two budget without early stop the decisions equal ``run_pt(...)["counts"][:, k].argmax(1)``.

    Returns (ml [S], stops [S]): the decided class and the number of sweeps run per shot."""
    cfg = spec.cfg
    S, NSW = c0.shape[0], int(nsweeps)
    ga, gb, sw = seed_states(spec, S, seed)
    heads = np.zeros((S, cfg.NS, 2), np.int64)
    kb = max(1, NSW.bit_length())
    counts = np.zeros((S, kb + 1, 4), np.int64)
    stops = np.zeros(S, np.int64)
    _run_many_es(np.ascontiguousarray(c0, dtype=np.uint8), spec.n, spec.sup, spec.blk, spec.logbit, spec.cls_mat,
                 spec.thr, spec.kmax, spec.xi_sw, cfg.NS, cfg.K, cfg.B, cfg.USW, NSW, ga, gb, heads, sw, counts, stops,
                 gmin, th32)
    s1 = stops - 1
    blk = np.array([max(1, int(x) if x < (NSW >> 1) else NSW - 1).bit_length() for x in s1])
    ml = counts[np.arange(S), blk].argmax(1)
    return ml, stops

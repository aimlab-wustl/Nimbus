"""CPU reference decoders for the surface-code experiments (wrappers around PyMatching, ldpc, stim and qecsim).

These packages are optional (the ``refs`` extra) and are imported only when a decoder is built.

One syndrome per shot (code capacity, :class:`~nimbus.problems.surface_code.SurfaceCodeProblem`; corrections are error
bits of ``n_spins`` length):

- :func:`matching_correction`: minimum-weight perfect matching (PyMatching), X and Z decoded separately;
- :func:`bposd_correction`: BP-OSD (ldpc; min-sum, scaling 0.625, 50 iterations, OSD-CS order 10); under depolarizing
  noise on the joint X / Y / Z columns;
- :func:`union_find_correction`: union-find (ldpc, peeling), X and Z decoded separately;
- :func:`tensor_network_failures`: approximate maximum-likelihood decoding by tensor-network contraction (qecsim matrix
  product state decoder, bond dimension chi).

Memory experiment (phenomenological depolarizing noise, :class:`~nimbus.decoding.phenomenological.DepolarizingMemory`;
priors: X, Y, Z each p/3 on data, measurement flips p):

- :func:`memory_baselines`: block decoders on the whole memory: "pm" (PyMatching per sector, marginal 2p/3), "pmc"
  (PyMatching correlated matching on the detector error model with Y = X ^ Z hyperedges), "bpm" (BP-weighted matching:
  product-sum BP on the joint X / Y / Z matrix, posteriors folded into per-sector edge probabilities, then PyMatching per
  sector), "bp" (BP-OSD on the joint matrix);
- :func:`memory_union_find_failures`: union-find per sector;
- :data:`WINDOW_DECODERS`: the same decoders as window decoders for :func:`nimbus.decoding.windows.decode_windows`
  (sliding windows, real time): "pm", "pmc", "bpm", "bp" (ldpc's default min-sum scaling, 1.0 in ldpc 2.4) and
  "bp625" (min-sum scaling 0.625).
"""
from __future__ import annotations

import time

import numpy as np

__all__ = ["matching_correction", "bposd_decoder", "bposd_correction", "union_find_correction", "qecsim_lattice_map",
           "tensor_network_failures", "joint_check_matrix", "detector_error_model", "memory_baselines",
           "memory_union_find_failures", "window_matching_decoder", "window_correlated_matching_decoder",
           "window_bp_matching_decoder", "window_bposd_decoder", "WINDOW_DECODERS"]


# ============================================================================================ one syndrome per shot
def matching_correction(prob, syn):
    """PyMatching correction [S, n_spins] of syndromes syn; one matching graph per check type."""
    import pymatching
    out = np.zeros((syn.shape[0], prob.n_spins), np.uint8)
    for (Hc, off, *_r), s in zip(prob._checks, prob._split_syndrome(syn)):
        m = pymatching.Matching.from_check_matrix(Hc)
        out[:, off:off + prob.nq] = m.decode_batch(s)
    return out


def bposd_decoder(prob, osd_order=10, max_iter=50):
    """The ldpc BpOsdDecoder used by :func:`bposd_correction` (built once; ``decode(syndrome)`` per shot). Under
    depolarizing noise its columns are X_i, Z_i, Y_i (rows: Z checks, then X checks), each with prior p/3."""
    from ldpc import BpOsdDecoder
    n = prob.nq
    if prob.noise == "independent":
        Hc = prob._checks[0][0]
        return BpOsdDecoder(Hc, error_rate=prob.p, max_iter=max_iter, bp_method="minimum_sum", ms_scaling_factor=0.625,
                            osd_method="osd_cs", osd_order=osd_order)
    HZ, HX = prob._checks[0][0], prob._checks[1][0]
    mz, mx = HZ.shape[0], HX.shape[0]
    Hj = np.zeros((mz + mx, 3 * n), np.uint8)       # columns: X_i, Z_i, Y_i
    Hj[:mz, :n] = HZ
    Hj[mz:, n:2 * n] = HX
    Hj[:mz, 2 * n:] = HZ
    Hj[mz:, 2 * n:] = HX
    return BpOsdDecoder(Hj, error_rate=prob.p, channel_probs=[prob.p / 3] * (3 * n), max_iter=max_iter,
                        bp_method="minimum_sum", ms_scaling_factor=0.625, osd_method="osd_cs", osd_order=osd_order)


def bposd_correction(prob, syn, osd_order=10, max_iter=50):
    """BP-OSD correction [S, n_spins] of syndromes syn (see :func:`bposd_decoder`)."""
    n = prob.nq
    out = np.zeros((syn.shape[0], prob.n_spins), np.uint8)
    dec = bposd_decoder(prob, osd_order, max_iter)
    if prob.noise == "independent":
        for k in range(syn.shape[0]):
            out[k] = dec.decode(syn[k])
        return out
    for k in range(syn.shape[0]):
        r = dec.decode(syn[k])
        out[k, :n] = r[:n] ^ r[2 * n:]
        out[k, n:] = r[n:2 * n] ^ r[2 * n:]
    return out


def union_find_correction(prob, syn):
    """Union-find correction [S, n_spins] (ldpc UnionFindDecoder, peeling), one decoder per check type."""
    from ldpc.union_find_decoder import UnionFindDecoder
    out = np.zeros((syn.shape[0], prob.n_spins), np.uint8)
    for (Hc, off, *_r), s in zip(prob._checks, prob._split_syndrome(syn)):
        dec = UnionFindDecoder(Hc, uf_method="peeling")
        for k in range(s.shape[0]):
            out[k, off:off + prob.nq] = dec.decode(s[k])
    return out


def qecsim_lattice_map(prob, code):
    """Qubit map between this package's rotated code and a qecsim RotatedPlanarCode of the same distance: they differ
    by a lattice symmetry and possibly an X / Z swap, found by matching the stabilizer sets. Returns (perm, swap): qubit
    q here is qubit perm[q] there; swap = True if the X and Z stabilizers are exchanged."""
    d, n = prob.d, prob.nq
    S = code.stabilizers
    qx = {tuple(np.flatnonzero(r[:n])) for r in S if r[:n].any()}
    qz = {tuple(np.flatnonzero(r[n:])) for r in S if r[n:].any()}
    ox = [np.flatnonzero(r) for r in prob.code["HX"]]
    oz = [np.flatnonzero(r) for r in prob.code["HZ"]]
    maps = [lambda r, c: (r, c), lambda r, c: (c, d - 1 - r), lambda r, c: (d - 1 - r, d - 1 - c),
            lambda r, c: (d - 1 - c, r), lambda r, c: (r, d - 1 - c), lambda r, c: (d - 1 - r, c),
            lambda r, c: (c, r), lambda r, c: (d - 1 - c, d - 1 - r)]
    for f in maps:
        perm = np.array([f(i // d, i % d)[0] * d + f(i // d, i % d)[1] for i in range(n)])
        mx = {tuple(sorted(perm[s])) for s in ox}
        mz = {tuple(sorted(perm[s])) for s in oz}
        if mx == qx and mz == qz:
            return perm, False
        if mx == qz and mz == qx:
            return perm, True
    raise RuntimeError("no lattice map found")


def to_qecsim(prob, code, e):
    """Error bits e [S, 2n] (x | z) as qecsim binary symplectic vectors on ``code`` (see :func:`qecsim_lattice_map`)."""
    n = prob.nq
    perm, swap = qecsim_lattice_map(prob, code)
    out = []
    for k in range(len(e)):
        x, z = np.zeros(n, np.uint8), np.zeros(n, np.uint8)
        x[perm] = e[k, :n]; z[perm] = e[k, n:]
        out.append(np.concatenate([z, x] if swap else [x, z]))
    return out


def tensor_network_failures(prob, e, chi=16):
    """Logical failure [S] (bool) of the qecsim matrix-product-state decoder (RotatedPlanarMPSDecoder, bond dimension
    chi) on the depolarizing errors e [S, 2n] of ``prob``."""
    from qecsim import paulitools as pt
    from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarMPSDecoder
    code = RotatedPlanarCode(prob.d, prob.d)
    dec = RotatedPlanarMPSDecoder(chi=chi)
    fails = np.zeros(len(e), bool)
    for k, bsf in enumerate(to_qecsim(prob, code, e)):
        rec = dec.decode(code, pt.bsp(bsf, code.stabilizers.T))
        res = rec ^ bsf
        fails[k] = bool(pt.bsp(res, code.stabilizers.T).any() or pt.bsp(res, code.logicals.T).any())
    return fails


# ============================================================================================ memory, block decoders
def joint_check_matrix(mem):
    """Joint check matrix of the space-time code. Columns: X(q, t), Y(q, t), Z(q, t), mZ (Z-check measurements),
    mX (X-check measurements); rows: X-sector detectors (Z checks), then Z-sector detectors (X checks).
    Returns (H, Lx, Lz, (nx, nm1, nm2)): the two observables over the columns and the column block sizes."""
    cx, cz = mem.cx, mem.cz
    nx, nm1, nm2 = cx.nx, cx.N - cx.nx, cz.N - cz.nx
    Hx, Hz = cx.H, cz.H
    zx = np.zeros((Hx.shape[0], nx), np.uint8); zz = np.zeros((Hz.shape[0], nx), np.uint8)
    H = np.vstack([np.hstack([Hx[:, :nx], Hx[:, :nx], zx, Hx[:, nx:], np.zeros((Hx.shape[0], nm2), np.uint8)]),
                   np.hstack([zz, Hz[:, :nx], Hz[:, :nx], np.zeros((Hz.shape[0], nm1), np.uint8), Hz[:, nx:]])])
    Lx = np.concatenate([cx.L[:nx], cx.L[:nx], np.zeros(nx), np.zeros(nm1 + nm2)]).astype(np.int64)
    Lz = np.concatenate([np.zeros(nx), cz.L[:nx], cz.L[:nx], np.zeros(nm1 + nm2)]).astype(np.int64)
    return H, Lx, Lz, (nx, nm1, nm2)


def _dem_lines(mem, p, fx, fz):
    cx, cz = mem.cx, mem.cz
    q3 = p / 3
    lines = []
    for i in range(cx.nx):
        lines += [f"error({q3!r}) {fx(i)}", f"error({q3!r}) {fz(i)}", f"error({q3!r}) {fx(i)} ^ {fz(i)}"]
    lines += [f"error({p!r}) {fx(j)}" for j in range(cx.nx, cx.N)]
    lines += [f"error({p!r}) {fz(j)}" for j in range(cz.nx, cz.N)]
    return "\n".join(lines)


def detector_error_model(mem, p, per_variable=False):
    """stim DetectorErrorModel of the memory: X, Y, Z (p/3 each) per data site, Y decomposed as X ^ Z; measurement
    errors (p) on both check types. Detectors: the X sector, then the Z sector. per_variable=False: L0 = X-sector
    observable, L1 = Z-sector observable. per_variable=True: every sector variable j carries its own observable (X sector
    L_j, Z sector L_{cx.N + j}), so a decoder's prediction is the configuration itself."""
    import stim
    cx, cz = mem.cx, mem.cz
    ndx, NX = cx.H.shape[0], cx.N
    Hxc, Hzc = cx.H.T.astype(bool), cz.H.T.astype(bool)
    if per_variable:
        fx = lambda j: " ".join(f"D{k}" for k in np.flatnonzero(Hxc[j])) + f" L{j}"
        fz = lambda j: " ".join(f"D{ndx + k}" for k in np.flatnonzero(Hzc[j])) + f" L{NX + j}"
    else:
        fx = lambda j: " ".join(f"D{k}" for k in np.flatnonzero(Hxc[j])) + (" L0" if cx.L[j] else "")
        fz = lambda j: " ".join(f"D{ndx + k}" for k in np.flatnonzero(Hzc[j])) + (" L1" if cz.L[j] else "")
    return stim.DetectorErrorModel(_dem_lines(mem, p, fx, fz))


def _sector_weights(code, p):
    wd, wm = np.log((1 - 2 * p / 3) / (2 * p / 3)), np.log((1 - p) / p)
    return np.concatenate([np.full(code.nx, wd), np.full(code.N - code.nx, wm)])


def _fold_posteriors(log_prob_ratios, nx, nm1):
    """BP posteriors of the joint columns -> per-sector edge probabilities (p_x = P_X + P_Y - 2 P_X P_Y, data columns
    first, then the sector's measurement columns)."""
    P = np.clip(1.0 / (1.0 + np.exp(np.asarray(log_prob_ratios))), 1e-12, 1 - 1e-12)
    PX, PY, PZ = P[:nx], P[nx:2 * nx], P[2 * nx:3 * nx]
    pxe = np.clip(np.concatenate([PX + PY - 2 * PX * PY, P[3 * nx:3 * nx + nm1]]), 1e-12, 1 - 1e-12)
    pze = np.clip(np.concatenate([PZ + PY - 2 * PZ * PY, P[3 * nx + nm1:]]), 1e-12, 1 - 1e-12)
    return pxe, pze


def memory_baselines(mem, p, Dx, Dz, ox, oz, which=("pm", "pmc", "bpm", "bp")):
    """Block decoders on the whole memory (shots Dx, Dz, ox, oz of ``mem.sample``). Returns ({name: failures [S]},
    {name: seconds per shot}); "pm" is always run. Names as in the module docstring."""
    import pymatching
    from scipy.sparse import csc_matrix
    cx, cz = mem.cx, mem.cz
    nsh = len(Dx)
    f, t = {}, {}
    wx = _sector_weights(cx, p)
    t0 = time.perf_counter()
    mx = pymatching.Matching.from_check_matrix(csc_matrix(cx.H), weights=wx)
    mz = pymatching.Matching.from_check_matrix(csc_matrix(cz.H), weights=wx)
    cxh = mx.decode_batch(Dx)
    czh = mz.decode_batch(Dz)
    fx = (cxh.astype(np.int64) @ cx.L % 2) != ox
    fz = (czh.astype(np.int64) @ cz.L % 2) != oz
    f["pm"] = fx | fz
    t["pm"] = (time.perf_counter() - t0) / nsh
    D = np.hstack([Dx, Dz])
    obs = np.stack([ox, oz], 1)
    if "pmc" in which:
        t0 = time.perf_counter()
        mc = pymatching.Matching.from_detector_error_model(detector_error_model(mem, p), enable_correlations=True)
        pr = mc.decode_batch(D, enable_correlations=True)
        f["pmc"] = (pr.astype(np.int64) != obs).any(1)
        t["pmc"] = (time.perf_counter() - t0) / nsh
    if "bpm" in which or "bp" in which:
        H, Lx, Lz, (nx, nm1, nm2) = joint_check_matrix(mem)
        prior = np.concatenate([np.full(3 * nx, p / 3), np.full(nm1 + nm2, p)])
        Hs = csc_matrix(H)
    if "bpm" in which:
        from ldpc import BpDecoder
        t0 = time.perf_counter()
        bpd = BpDecoder(Hs, error_channel=list(prior), bp_method="product_sum", max_iter=30, schedule="parallel")
        Hxs, Hzs = csc_matrix(cx.H), csc_matrix(cz.H)
        f_b = np.zeros(nsh, bool)
        for i in range(nsh):
            bpd.decode(D[i])
            pxe, pze = _fold_posteriors(bpd.log_prob_ratios, nx, nm1)
            m1 = pymatching.Matching.from_check_matrix(Hxs, weights=np.log((1 - pxe) / pxe))
            m2 = pymatching.Matching.from_check_matrix(Hzs, weights=np.log((1 - pze) / pze))
            a = m1.decode(Dx[i]).astype(np.int64); b = m2.decode(Dz[i]).astype(np.int64)
            f_b[i] = ((a @ cx.L) % 2 != ox[i]) | ((b @ cz.L) % 2 != oz[i])
        f["bpm"] = f_b
        t["bpm"] = (time.perf_counter() - t0) / nsh
    if "bp" in which:
        from ldpc import BpOsdDecoder
        t0 = time.perf_counter()
        dec = BpOsdDecoder(Hs, error_channel=list(prior), bp_method="minimum_sum", max_iter=50,
                           osd_method="osd_cs", osd_order=10)
        f_bp = np.zeros(nsh, bool)
        for i in range(nsh):
            c = dec.decode(D[i]).astype(np.int64)
            f_bp[i] = ((c @ Lx) % 2 != ox[i]) | ((c @ Lz) % 2 != oz[i])
        f["bp"] = f_bp
        t["bp"] = (time.perf_counter() - t0) / nsh
    return f, t


def memory_union_find_failures(mem, Dx, Dz, ox, oz):
    """Logical failure [S] of union-find (ldpc, peeling) on each sector of the whole memory."""
    from ldpc.union_find_decoder import UnionFindDecoder
    f = np.zeros(len(Dx), bool)
    for c, D, o in ((mem.cx, Dx, ox), (mem.cz, Dz, oz)):
        dec = UnionFindDecoder(c.H, uf_method="peeling")
        for i in range(len(D)):
            f[i] |= int(dec.decode(D[i]).astype(np.int64) @ c.L % 2) != o[i]
    return f


# ============================================================================================ window decoders
def _cached(build):
    cache = {}

    def get(w):
        if id(w) not in cache:
            cache[id(w)] = build(w)
        return cache[id(w)]
    return get


def window_matching_decoder(p):
    """Window decoder: PyMatching on each sector (marginal data prior 2p/3, measurement prior p). Variant "pm"."""
    import pymatching
    from scipy.sparse import csc_matrix
    get = _cached(lambda c: pymatching.Matching.from_check_matrix(csc_matrix(c.H), weights=_sector_weights(c, p)))

    def dec(w, Dxw, Dzw, final, k, t0):
        return {"pm": (get(w.cx).decode_batch(Dxw).astype(np.uint8), get(w.cz).decode_batch(Dzw).astype(np.uint8))}
    return dec


def window_correlated_matching_decoder(p):
    """Window decoder: PyMatching correlated matching on the window's detector error model (Y = X ^ Z hyperedges); every
    sector variable carries its own fault id, so the decoder returns the configuration. Variant "pmc"."""
    import pymatching
    get = _cached(lambda w: pymatching.Matching.from_detector_error_model(detector_error_model(w, p, per_variable=True),
                                                                         enable_correlations=True))

    def dec(w, Dxw, Dzw, final, k, t0):
        c = get(w).decode_batch(np.hstack([Dxw, Dzw]), enable_correlations=True).astype(np.uint8)
        return {"pmc": (c[:, :w.cx.N], c[:, w.cx.N:w.cx.N + w.cz.N])}
    return dec


def _joint(w, p):
    from scipy.sparse import csc_matrix
    H, _, _, (nx, nm1, nm2) = joint_check_matrix(w)
    prior = np.concatenate([np.full(3 * nx, p / 3), np.full(nm1 + nm2, p)])
    return csc_matrix(H), prior, nx, nm1, nm2


def window_bp_matching_decoder(p):
    """Window decoder: product-sum BP (30 iterations) on the window's joint X / Y / Z matrix, posteriors folded per
    sector, then PyMatching per sector with those weights. Variant "bpm"."""
    import pymatching
    from ldpc import BpDecoder
    from scipy.sparse import csc_matrix

    def build(w):
        Hs, prior, nx, nm1, nm2 = _joint(w, p)
        bpd = BpDecoder(Hs, error_channel=list(prior), bp_method="product_sum", max_iter=30, schedule="parallel")
        return bpd, csc_matrix(w.cx.H), csc_matrix(w.cz.H), nx, nm1
    get = _cached(build)

    def dec(w, Dxw, Dzw, final, k, t0):
        bpd, Hxs, Hzs, nx, nm1 = get(w)
        D = np.hstack([Dxw, Dzw])
        ex = np.zeros((len(D), w.cx.N), np.uint8)
        ez = np.zeros((len(D), w.cz.N), np.uint8)
        for i in range(len(D)):
            bpd.decode(D[i])
            pxe, pze = _fold_posteriors(bpd.log_prob_ratios, nx, nm1)
            ex[i] = pymatching.Matching.from_check_matrix(Hxs, weights=np.log((1 - pxe) / pxe)).decode(Dxw[i])
            ez[i] = pymatching.Matching.from_check_matrix(Hzs, weights=np.log((1 - pze) / pze)).decode(Dzw[i])
        return {"bpm": (ex, ez)}
    return dec


def window_bposd_decoder(p, ms_scaling=None, name="bp"):
    """Window decoder: BP-OSD (min-sum, 50 iterations, OSD-CS order 10) on the window's joint matrix. ms_scaling=None
    uses the library default min-sum scaling (1.0 in ldpc 2.4). Variant ``name``."""
    from ldpc import BpOsdDecoder
    kw = {} if ms_scaling is None else dict(ms_scaling_factor=ms_scaling)

    def build(w):
        Hs, prior, nx, nm1, nm2 = _joint(w, p)
        return BpOsdDecoder(Hs, error_channel=list(prior), bp_method="minimum_sum", max_iter=50, osd_method="osd_cs",
                            osd_order=10, **kw), nx, nm1
    get = _cached(build)

    def dec(w, Dxw, Dzw, final, k, t0):
        bpo, nx, nm1 = get(w)
        D = np.hstack([Dxw, Dzw])
        ex = np.zeros((len(D), w.cx.N), np.uint8)
        ez = np.zeros((len(D), w.cz.N), np.uint8)
        for i in range(len(D)):
            c = bpo.decode(D[i]).astype(np.uint8)
            cX, cY, cZ = c[:nx], c[nx:2 * nx], c[2 * nx:3 * nx]
            ex[i] = np.concatenate([cX ^ cY, c[3 * nx:3 * nx + nm1]])
            ez[i] = np.concatenate([cY ^ cZ, c[3 * nx + nm1:]])
        return {name: (ex, ez)}
    return dec


#: Factories p -> window decoder, keyed by the variant name each decoder returns.
WINDOW_DECODERS = {"pm": window_matching_decoder, "pmc": window_correlated_matching_decoder,
                   "bpm": window_bp_matching_decoder, "bp": window_bposd_decoder,
                   "bp625": lambda p: window_bposd_decoder(p, 0.625, "bp625")}

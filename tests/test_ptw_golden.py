"""Real-time bit-exact decoder (nimbus.decoding.real_time): lane generator = the one-shot generator, window geometry =
DepolarizingMemory, exact rank acceptance, no noise, block equivalence (W >= N), determinism, hardware schedules, carried
generators, and a replay of its commits through windows.decode_windows (consistency checks, starts and failures)."""
import numpy as np
import pytest

from nimbus.decoding import one_shot as A
from nimbus.decoding import real_time as g
from nimbus.decoding import windows as wn
from nimbus.decoding.phenomenological import memory_model


def _shots(d, p, S, seed=4):
    N = 4 * d
    return (N,) + tuple(memory_model(d, N).sample(p, S, np.random.default_rng([seed, d, int(p * 1e5)])))


@pytest.mark.parametrize("which", ["a", "b"])
def test_gfsr_matches_one_shot(which):
    p, q = A.GFSR_A if which == "a" else A.GFSR_B
    seed = 0x5EED ^ 77
    hist = A.bits_from(seed, p).copy()
    ref = np.zeros(4000, np.uint8)
    A._gfsr_block(hist, 0, p, q, 4000, ref)
    assert (g.gfsr_stream(seed, which, 4000) == ref).all()


def _memory_index(ws, li):
    O = (ws.ndc + ws.m * ws.T)                 # spins per sector in the DepolarizingMemory layout
    nd2, mt = 2 * ws.ndc, ws.m * ws.T
    if li < nd2:
        return li // 2 + (O if li % 2 else 0)
    k = li - nd2
    return ws.ndc + k if k < mt else O + ws.ndc + (k - mt)


@pytest.mark.parametrize("final", [False, True])
def test_geometry_matches_memory_model(final):
    d, T = 3, 6
    ws = g.window_spec(d, T, final, 0.03)
    dp = memory_model(d, T, not final)
    assert ws.M == len(dp.SUP) and ws.NL == dp.N
    for r in range(ws.M):
        mine = sorted(_memory_index(ws, li) for li in ws.tli[ws.tptr[r]:ws.tptr[r + 1]])
        assert mine == sorted(dp.SUP[r][dp.VAL[r]].tolist())
        assert ws.mlg[r] == (int(dp.logx[r]) | (int(dp.logz[r]) << 1))
    if not final:                              # sizes of the d = 3 open window
        assert (ws.ndc, 2 * ws.m * T, ws.NL, ws.M) == (54, 48, 156, 192)
        assert (ws.info["sl"], ws.info["tl"]) == (84, 108)
    assert ws.info["colors"] == 12
    for k in range(ws.cls_mat.shape[0]):       # one table per color, no shared clause
        mv = ws.cls_mat[k][ws.cls_mat[k] >= 0]
        assert len({int(ws.mtab[r] > 0) for r in mv}) == 1 and int(ws.mtab[mv[0]] > 0) == ws.col_table[k]
        cl = [li if li >= 2 * ws.ndc else li // 2 for r in mv for li in ws.tli[ws.tptr[r]:ws.tptr[r + 1]]]
        assert len(cl) == len(set(cl))


@pytest.mark.parametrize("p", [0.01, 0.03, 0.04, 0.1, 0.2])
def test_rank_acceptance_exact(p):
    ws = g.window_spec(5, 4, False, p)
    assert ws.mode == int(p < g.MODE_P)
    W, wm, Bq = ws.W, ws.wm, 1 << ws.cfg.B
    cases = [(0, a, 0) for a in range(-ws.KSL, ws.KSL + 1)]
    cases += [(1, a, b) for a in range(-2, 3) for b in (-2, 0, 2)] + [(2, a, b) for a in range(-2, 3) for b in (-1, 1)]
    for tab, a, b in cases:
        idx = ws.rk[tab, a + ws.KSL, b + 2]
        dE = W * a + wm * b
        if dE <= 1e-12:
            assert idx == 0
        else:
            exp = np.minimum(np.floor(Bq * np.exp(-ws.betas * dE)), Bq - 1).astype(np.int64)
            assert (ws.thr[:, idx] == exp).all()
    for lo, hi in ((1, ws.KSL), (ws.KSL + 1, ws.KSL + 7), (ws.KSL + 8, ws.KSL + 12)):   # thermometer order per table
        assert (np.diff(ws.thr[:, lo:hi + 1], axis=1) <= 0).all()


def test_no_noise_no_correction():
    d, S = 3, 8
    N = 4 * d
    m = (d * d - 1) // 2
    Dx = np.zeros((S, (N + 1) * m), np.uint8)
    res, info = g.decode_stream(d, 0.04, Dx, Dx.copy(), N, d, d, 64, seed=2, debug=True)
    for v, (bx, bz) in res.items():
        assert not bx.any() and not bz.any(), v
    for w in info["windows"]:
        assert not w["out"]["best_lit"].any() and not w["out"]["bestE"].any()
        assert not w["sigma"][0].any() and not w["sigma"][1].any()


def test_window_at_least_N_is_block_decode():
    d, p, S, nsw = 3, 0.04, 12, 64
    N, Dx, Dz, ox, oz = _shots(d, p, S)
    assert g.schedule(N, N - 2, 5) == [(0, N, True)]
    r1, _ = g.decode_stream(d, p, Dx, Dz, N, N - 2, 5, nsw, seed=9)
    # block decode by hand: one closed window over all N rounds, sigma = 0
    ws = g.window_spec(d, N, True, p)
    sx, sz = g.syndromes(Dx, N, ws.m), g.syndromes(Dz, N, ws.m)
    z = np.zeros((S, ws.m), np.uint8)
    o = g.run_window(ws, g.start_state(ws, sx[:, :N], sz[:, :N], z, z, sx[:, N], sz[:, N]),
                     g.window_bases(9, np.arange(S), 0, N, True), nsw)
    bl = o["best_lit"]
    px, pz = g.frame_parity(ws, bl, 0, ws.ns), g.frame_parity(ws, bl, 1, ws.ns)
    sh = o["bestC"] ^ o["ml"]
    assert (r1["me"][0] == px).all() and (r1["me"][1] == pz).all()
    assert (r1["ml"][0] == px ^ (sh & 1)).all() and (r1["ml"][1] == pz ^ (sh >> 1)).all()


def test_determinism_and_K2():
    d, p, S = 3, 0.04, 6
    N, Dx, Dz, ox, oz = _shots(d, p, S)
    a, _ = g.decode_stream(d, p, Dx, Dz, N, d, d, 32, seed=5)
    b, _ = g.decode_stream(d, p, Dx, Dz, N, d, d, 32, seed=5)
    assert all((a[v][i] == b[v][i]).all() for v in a for i in (0, 1))
    ws = g.window_spec(d, 2 * d, False, p)
    m = ws.m
    sx, sz = g.syndromes(Dx, N, m), g.syndromes(Dz, N, m)
    z = np.zeros((1, m), np.uint8)
    lit0 = g.start_state(ws, sx[:1, :2 * d], sz[:1, :2 * d], z, z)
    r1 = g.run_window(ws, lit0, g.window_bases(5, [0], 0, 2 * d, False), 64, rec=True)["rec"]
    r2 = g.run_window(ws, lit0, g.window_bases(5, [0], 0, 2 * d, False), 64, rec=True)["rec"]
    r3 = g.run_window(ws, lit0, g.window_bases(6, [0], 0, 2 * d, False), 64, rec=True)["rec"]
    assert (r1 == r2).all() and not (r1 == r3).all()
    assert (np.diff(r1[:, :, 1], axis=0) != 0).any()            # K = 1: labels move
    ws2 = g.window_spec(d, 2 * d, False, p, g.PTWConfig(K=2))
    r4 = g.run_window(ws2, lit0, g.window_bases(5, [0], 0, 2 * d, False), 64, rec=True)["rec"]
    lab = r4[:, :, 1]
    assert (lab[2::2] == lab[1:-1:2]).all()                     # K = 2: no swap round after even sweeps
    assert (lab[1:] != lab[:-1]).any()
    for r in (r1, r4):                                          # labels stay a permutation
        assert (np.sort(r[:, :, 1], 1) == np.arange(ws.cfg.R)).all()


@pytest.mark.parametrize("start", ["chain", "gf2"])
def test_replay_through_window_decoding(start):
    """Commits of the bit-exact decoder fed to windows.decode_windows (check=True): its starts equal particular_fast
    (non-final windows always, the final window for start = "gf2"), every commit clears its rounds, the final correction
    meets every detector, bestE matches the literals, and the frame-bit failures equal the failures of the same
    corrections."""
    d, p, S = 3, 0.04, 16
    N, Dx, Dz, ox, oz = _shots(d, p, S)
    res, info = g.decode_stream(d, p, Dx, Dz, N, d, d, 32, seed=3, start=start, debug=True)
    W = info["windows"]

    def dec(w, Dxw, Dzw, final, k, t0):
        e = W[k]
        assert e["t0"] == t0 and e["final"] == final
        ex0, ez0 = g.to_memory_layout(e["ws"], e["lit0"])
        if not final or start == "gf2":
            assert (ex0 == w.cx.particular_fast(Dxw)).all() and (ez0 == w.cz.particular_fast(Dzw)).all()
        ex, ez = g.to_memory_layout(e["ws"], e["out"]["best_lit"])
        nd = (ex[:, :w.nx] | ez[:, :w.nx]).sum(1)
        nm = ex[:, w.nx:].sum(1) + ez[:, w.nx:].sum(1)
        assert (e["ws"].Wq * nd + e["ws"].wq * nm == e["out"]["bestE"]).all()
        return {"me": (ex, ez)} if final else {"c": (ex, ez)}

    corr = wn.decode_windows(d, N, wn.schedule(N, d, d), Dx, Dz, dec, check=True)
    assert (wn.correction_failures(d, N, corr, ox, oz)["me"] == g.failures(res, ox, oz)["me"]).all()


@pytest.mark.parametrize("d,k,lram", [(3, 1, False), (5, 2, False), (7, 5, True), (7, 4, False)])
def test_hw_schedule(d, k, lram):
    """Hardware schedule (FPGA layouts): every move once per sweep in both window kinds, no two moves of a (slab, color)
    cycle share a clause, one table per color, lanes = physical lanes; LUTRAM: one word address per unit and color."""
    T = 2 * d
    hs = g.hw_schedule(d, T, k, lram)
    for f in (False, True):
        ws = g.window_spec(d, T, f, 0.04, g.PTWConfig(k=k, lram=lram))
        cm = ws.cls_mat
        assert cm.shape == (k * hs["NC"], hs["Lmax"])
        assert sorted(cm[cm >= 0].tolist()) == list(range(ws.M))
        for r in range(cm.shape[0]):
            mv = cm[r][cm[r] >= 0]
            if not len(mv):
                continue
            assert {int(ws.mtab[x] > 0) for x in mv} == {int(ws.col_table[r])}
            cl = [li if li >= 2 * ws.ndc else li // 2 for x in mv for li in ws.tli[ws.tptr[x]:ws.tptr[x + 1]]]
            assert len(cl) == len(set(cl))
    if lram:
        S = hs["S"]
        for c in range(hs["NC"]):
            units = [(tm[1], (P["row"] + tm[2]) % S) for P in hs["phys"] if P["col"] == c for tm in P["dterms"]]
            assert len(units) == len(set(units))
    # the default (software) schedule is unchanged by the hardware option
    assert g.window_spec(d, T, False, 0.04).info["colors"] == 12


def test_carry_generators():
    """PTWConfig.carry: the first window of a shot is seeded as without carry; later windows continue the generators (as
    the hardware), and run_window with an explicit state reproduces decode_stream window by window."""
    d, p, S, nsw = 3, 0.04, 4, 16
    N, Dx, Dz, ox, oz = _shots(d, p, S)
    c0, c1 = g.PTWConfig(k=1), g.PTWConfig(k=1, carry=True)
    _, i0 = g.decode_stream(d, p, Dx, Dz, N, d, d, nsw, c0, seed=4, debug=True)
    _, i1 = g.decode_stream(d, p, Dx, Dz, N, d, d, nsw, c1, seed=4, debug=True)
    w0, w1 = i0["windows"], i1["windows"]
    assert (w0[0]["out"]["bestE"] == w1[0]["out"]["bestE"]).all()
    assert any((a["out"]["counts"] != b["out"]["counts"]).any() for a, b in zip(w0[1:], w1[1:]))
    st = g.gen_state(1, c1.R, w1[0]["ws"].bank_lanes.shape[0])
    for e in w1:
        b = g.window_bases(4, [1], e["t0"], e["T"], e["final"])
        o = g.run_window(e["ws"], e["lit0"][1:2], b, nsw, rec=True, state=st)
        assert o["bestE"][0] == e["out"]["bestE"][1] and (o["best_lit"][0] == e["out"]["best_lit"][1]).all()


def test_fpga_config():
    c = g.fpga_config(7, R=24)
    assert (c.k, c.lram, c.carry, c.R, c.K, c.bmin) == (5, True, True, 24, 1, 0.1)

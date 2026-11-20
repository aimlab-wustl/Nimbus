"""Sliding-window decoding of the phenomenological depolarizing memory (nimbus.decoding.windows, phenomenological):
open-window model, schedule, commit and syndrome-forwarding logic, particular solutions and memory shots."""
import importlib.util

import numpy as np
import pytest

from nimbus.decoding import windows as wn
from nimbus.decoding.phenomenological import PhenoCode, memory_model, memory_shots

HAVE_PM = importlib.util.find_spec("pymatching") is not None


@pytest.mark.parametrize("d,T", [(3, 2), (3, 5), (5, 4)])
def test_open_window_model(d, T):
    c = PhenoCode(d, T, open_top=True)
    r = c.check()
    assert r["complete"] and r["rank_H"] == c.ndet          # full row rank: any window syndrome is decodable
    assert c.ndet == c.m * T and c.nx == c.n * T
    top = np.flatnonzero(c.H[:, c.nx + (T - 1) * c.m])
    assert list(top) == [(T - 1) * c.m]                     # top measurement error: one in-window detector


@pytest.mark.parametrize("d,T", [(3, 3), (5, 2)])
def test_closed_memory_model(d, T):
    r = PhenoCode(d, T).check()
    assert r["complete"] and r["n_logical"] == (T + 1) * d


@pytest.mark.parametrize("N,C,B", [(12, 3, 3), (12, 3, 2), (12, 2, 3), (12, 3, 5), (12, 5, 1), (12, 4, 8)])
def test_schedule(N, C, B):
    s = wn.schedule(N, C, B)
    assert s[-1][2] and s[-1][0] + s[-1][1] == N
    assert all(not f and Tw == C + B and t0 + Tw < N for t0, Tw, f in s[:-1])
    assert [t0 for t0, _, _ in s] == [C * k for k in range(len(s))]


def _sample(d, N, p, shots, seed=3):
    G = memory_model(d, N)
    return G.sample(p, shots, np.random.default_rng(seed), errors=True)


@pytest.mark.parametrize("C,B", [(3, 3), (3, 2), (2, 3), (3, 5), (1, 0)])
def test_commit_forwarding_oracle_and_particular(C, B):
    d, N = 3, 12
    G = memory_model(d, N)
    Dx, Dz, ox, oz, ex, ez = _sample(d, N, 0.08, 64)
    sched = wn.schedule(N, C, B)

    def oracle(w, Dxw, Dzw, final, k, t0):
        vx, dx, _ = wn.window_maps(G.cx, w.cx, t0)
        vz, dz, _ = wn.window_maps(G.cz, w.cz, t0)
        a, b = ex[:, vx], ez[:, vz]
        assert ((a.astype(np.int64) @ w.cx.H.T % 2) == Dxw).all()   # residual = syndrome of the true in-window error
        assert ((b.astype(np.int64) @ w.cz.H.T % 2) == Dzw).all()
        return {"o": (a, b)}
    (cx, cz), = wn.decode_windows(d, N, sched, Dx, Dz, oracle).values()
    assert (cx == ex).all() and (cz == ez).all()
    assert not wn.correction_failures(d, N, {"o": (cx, cz)}, ox, oz)["o"].any()
    # any valid window configuration: decode_windows asserts that the total correction cancels every detector
    (cx, cz), = wn.decode_windows(d, N, sched, Dx, Dz, wn.particular_decoder()).values()
    assert ((cx.astype(np.int64) @ G.cx.H.T % 2) == Dx).all() and ((cz.astype(np.int64) @ G.cz.H.T % 2) == Dz).all()


@pytest.mark.skipif(not HAVE_PM, reason="needs pymatching (refs extra)")
def test_pm_windows_valid():
    from nimbus.decoding.baselines import window_matching_decoder
    d, N = 3, 9
    Dx, Dz, ox, oz, _, _ = _sample(d, N, 0.05, 200)
    blk = wn.decode_windows(d, N, [(0, N, True)], Dx, Dz, window_matching_decoder(0.05))
    for C, B in [(3, 3), (2, 2), (9, 0), (2, 10)]:
        r = wn.decode_windows(d, N, wn.schedule(N, C, B), Dx, Dz, window_matching_decoder(0.05))   # asserts validity
        assert set(r) == {"pm"}
        if C + B >= N:                                       # one final window: the block decoder
            assert (r["pm"][0] == blk["pm"][0]).all() and (r["pm"][1] == blk["pm"][1]).all()


@pytest.mark.parametrize("d,T,open_top", [(3, 3, False), (5, 4, False), (3, 4, True), (5, 6, True)])
def test_particular_fast(d, T, open_top):
    c = PhenoCode(d, T, open_top=open_top)
    e = (np.random.default_rng(1).random((50, c.N)) < 0.1).astype(np.uint8)
    D = (e.astype(np.int64) @ c.H.T % 2).astype(np.uint8)
    if open_top:                                            # an open window has full row rank: any D is valid
        D[:10] = np.random.default_rng(2).random((10, c.ndet)) < 0.3
    c0 = c.particular_fast(D)
    assert ((c0.astype(np.int64) @ c.H.T % 2) == D).all()
    if not open_top:
        assert ((c.particular(D).astype(np.int64) @ c.H.T % 2) == D).all()


def test_memory_shots_reproducible():
    a = memory_shots(3, 0.03, 5, chunk=2)
    b = memory_shots(3, 0.03, 5, chunk=2)
    c = memory_shots(3, 0.03, 5, chunk=3)
    assert all((x == y).all() for x, y in zip(a, b)) and not all((x == y).all() for x, y in zip(a, c))
    G = memory_model(3, 12)
    assert a[0].shape == (5, G.cx.ndet) and a[1].shape == (5, G.cz.ndet)
    assert (a[0] == G.sample(0.03, 5, np.random.default_rng([11, 2, 3, 3000, 5]))[0]).all()
    # direct round-by-round simulation of the sampled errors gives the same detectors
    Dx, Dz, ox, oz, ex, ez = G.sample(0.05, 20, np.random.default_rng(0), errors=True)
    assert (G.cx.simulate_detectors(ex) == Dx).all() and (G.cz.simulate_detectors(ez) == Dz).all()

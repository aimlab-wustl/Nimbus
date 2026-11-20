"""One-syndrome-per-shot board driver (hardware/host/pt_board.py) without hardware: tables and seeds equal the golden model's, and a full driver
session against the register-level mock (hardware/host/pt_mock.py) reproduces a direct golden run bit for bit."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hardware" / "host"))
import pt_board as D  # noqa: E402
import pt_host as H  # noqa: E402
from pt_mock import PTBoardMock, ProtocolError, RecordingMMIO  # noqa: E402
from nimbus.decoding import one_shot as G  # noqa: E402
from nimbus.decoding.code_capacity import sample_shots  # noqa: E402


@pytest.mark.parametrize("p,NS,B,kmax", [(0.15, 16, 16, 11), (0.01, 8, 12, 5), (0.19, 64, 16, 15), (0.12, 4, 14, 3)])
def test_tables_match_golden(p, NS, B, kmax):
    w3 = D.w3_depolarizing(p)
    t1, h1 = D.make_tables(w3, D.ladder(NS, 0.1), B, kmax)
    t2, h2 = G.make_tables(w3, G.ladder(NS, 0.1), B, kmax)
    assert (t1 == t2).all() and (h1 == h2).all()
    # the default spec tables are these
    s = G.build_spec(3, p, G.PTConfig(NS=NS, B=B))
    t3, h3 = D.make_tables(D.w3_depolarizing(p), D.ladder(NS, 0.1), B, s.kmax)
    assert (t3 == s.thr).all() and (h3 == s.theta_sw).all()


def test_seed_states_match_golden():
    s = G.build_spec(3, 0.12, G.PTConfig(NS=5))
    ga, gb, sw = G.seed_states(s, 1, 11)
    da, db, dsw = D.seed_states(5, 11)
    assert (ga[0] == da).all() and (gb[0] == db).all() and (sw[0] == dsw).all()


def test_records_roundtrip():
    rng = np.random.default_rng(1)
    S = 9
    r = dict(ml=rng.integers(0, 4, S), me=rng.integers(0, 4, S), stopped=rng.integers(0, 2, S).astype(bool),
             emin=rng.integers(0, 600, S), sweeps=rng.integers(1, 1 << 20, S), cyc_start=np.full(S, 3),
             cyc_run=rng.integers(0, 1 << 30, S), counts=rng.integers(0, 1 << 20, (S, 4)), t_done=rng.integers(0, 1 << 32, S))
    u = D.unpack_records(D.pack_records(r))
    for k in r:
        assert (np.asarray(u[k]).astype(np.int64) == np.asarray(r[k]).astype(np.int64)).all(), k


def test_cycle_model_matches_xsim():
    # values measured in xsim (hardware/sim/run_pt_board.sh): d = 5, NS = 8, K = 2, budget 64, early stop gmin 4
    assert D.cycle_model([4, 8], [True, True], 64, 2, 8, 4).tolist() == [41, 77]
    # d = 3, NS = 4, K = 1, budget 8, no early stop: 3 + 80 + 2 cycles
    assert D.cycle_model([8], [False], 8, 1, 4, 0).tolist() == [85]


def _session(d=3, NS=4, K=2, B=16, nsw=16, shots=8, seed=5, gmin=0, th32=29):
    mock = PTBoardMock(d, NS, K, B, p_gen=0.12)
    brd = D.PTBoard(RecordingMMIO(mock))
    e, syn = sample_shots(mock.spec.prob, shots, seed)
    brd.set_noise(0.15)
    sp1 = mock.tables_spec()
    brd.seed(seed)
    brd.configure(nsw, gmin, th32)
    h = shots // 2
    r1 = brd.run_batch(syn[:h])
    brd.set_noise(0.10)                    # table switch with RECALC, same generator streams
    sp2 = mock.tables_spec()
    r2 = brd.run_batch(syn[h:])
    return mock, brd, syn, (sp1, sp2), (r1, r2)


@pytest.mark.parametrize("gmin", [0, 4])
def test_driver_session_bit_exact(gmin):
    mock, brd, syn, (sp1, sp2), (r1, r2) = _session(nsw=32, gmin=gmin, th32=24)
    spec = mock.spec
    ga, gb, sw = G.seed_states(spec, 1, 5)
    ga, gb, sw = ga[0], gb[0], sw[0]
    heads = np.zeros((spec.cfg.NS, 2), np.int64)
    c0 = G.c0_from_syndrome(spec, syn)
    got = {k: np.concatenate([r1[k], r2[k]]) for k in ("ml", "me", "sweeps", "emin", "counts")}
    for s in range(len(syn)):
        sp = sp1 if s < len(syn) // 2 else sp2
        r = G.run_shot(sp, c0[s], 32, ga, gb, heads, sw, gmin, 24 if gmin else 0)
        kb = int(r["stop"]).bit_length() - 1
        assert got["ml"][s] == int(np.argmax(r["counts"][max(1, int(r["stop"]) - 1).bit_length()]))
        assert got["me"][s] == r["bestC"][np.argmin(r["bestE"])]   # lowest-energy class at the end (ties: lowest replica)
        assert got["sweeps"][s] == r["stop"]
        assert got["emin"][s] == r["bestE"].min()
        assert (got["counts"][s] == r["counts"][kb]).all()


class _Arr:
    """numpy-like word window over a mock, as PYNQ's MMIO.array (slices only)."""
    def __init__(self, m):
        self.m = m

    def __setitem__(self, sl, vals):
        for i, v in zip(range(sl.start, sl.stop), vals):
            self.m.write(4 * i, int(v))

    def __getitem__(self, sl):
        return np.array([self.m.read(4 * i) for i in range(sl.start, sl.stop)], np.uint32)


class _ArrMMIO:
    def __init__(self, m):
        self.m, self.array = m, _Arr(m)

    def read(self, off):
        return self.m.read(off)

    def write(self, off, v):
        self.m.write(off, v)


def test_host_check_on_saved_board_output(tmp_path):
    mock = PTBoardMock(3, 4, 2, 16)
    b = D.PTBoard(_ArrMMIO(mock))                                # PYNQ path: bulk memory access through .array
    e, syn = sample_shots(mock.spec.prob, 8, 2)
    b.set_noise(0.15); b.seed(3); b.configure(16, 4, 26)
    res = b.run(syn, batch=4)                                     # two batches, one generator stream
    out = str(tmp_path / "res.npz")
    D.save_results(out, res, b.meta(), dict(syn=syn), csv=True)
    r2, meta, z = H.load_board_npz(out)
    gold = H.golden_for(r2, meta, z["syn"], 3)
    rep = H.compare(r2, gold)
    assert rep["mismatched_shots"] == 0, rep
    assert (tmp_path / "res.csv").exists()


def test_protocol_errors():
    mock = PTBoardMock(3, 4, 2, 16)
    brd = D.PTBoard(mock)
    with pytest.raises(RuntimeError):
        brd.run_batch(np.zeros((4, brd.M), np.uint8))            # not seeded
    brd.set_noise(0.15); brd.seed(1); brd.configure(8)
    mock.write(D.THETA, 1234)                                     # theta change without RECALC
    with pytest.raises(ProtocolError):
        brd.run_batch(np.zeros((4, brd.M), np.uint8))
    with pytest.raises(ValueError, match="multiple of 4"):
        brd.run_batch(np.zeros((3, brd.M), np.uint8))             # shot count not a multiple of 4
    with pytest.raises(ValueError):
        brd.configure(1)                                          # below 2
    with pytest.raises(ValueError):
        brd.set_tables(np.tile(np.arange(1, brd.KM + 1), (brd.NS, 1)), np.zeros(brd.NS - 1))   # increasing in k

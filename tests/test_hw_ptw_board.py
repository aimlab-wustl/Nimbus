"""Real-time board driver (hardware/host/ptw_board.py) against the register-level mock (hardware/host/ptw_mock.py, golden model inside)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("hardware/host", "hardware/gen"):
    sys.path.insert(0, str(ROOT / p))
from nimbus.decoding import real_time as g  # noqa: E402
from nimbus.decoding.phenomenological import memory_model  # noqa: E402
import ptw_board as D  # noqa: E402
import ptw_host as H  # noqa: E402
from ptw_mock import PTWBoardMock, ProtocolError  # noqa: E402


@pytest.fixture(scope="module")
def mock_info(tmp_path_factory):
    import gen_ptw
    out = tmp_path_factory.mktemp("ptw_d3")
    gen_ptw.main(["--d", "3", "--R", "4", "--k", "1", "--out", str(out)])
    return json.loads((out / "ptw_info.json").read_text())


@pytest.mark.parametrize("p", [0.02, 0.03, 0.04, 0.1, 0.2])
@pytest.mark.parametrize("d,R", [(3, 16), (5, 16), (9, 24), (11, 24)])
def test_tables_equal_golden(p, d, R):
    ws = g.window_spec(d, 2 * d, False, p, g.PTWConfig(R=R))
    thr, theta, Wq, wq, mode = D.make_tables(p, R, ws.KSL)
    assert (thr == ws.thr[:, 1:]).all() and (theta == ws.theta).all()
    assert (Wq, wq, mode) == (ws.Wq, ws.wq, ws.mode)


def test_batch_base_equals_window_base():
    for seed in (1, 7, 123456789):
        for TR in (6, 10, 22):
            assert D.batch_base(seed, TR) == int(g.window_bases(seed, [0], 0, TR, False)[0])


def test_decode_batch_one_shot_equals_decode_stream():
    d, N, p = 3, 12, 0.04
    cfg = g.PTWConfig(R=4, k=1, carry=True)
    Dx, Dz, ox, oz = memory_model(d, N).sample(p, 1, np.random.default_rng(3))
    res, dbg = g.decode_stream(d, p, Dx, Dz, N, d, d, 16, cfg, seed=9, debug=True)
    m = 4
    recs, _ = g.decode_batch(d, p, g.syndromes(Dx, N, m), g.syndromes(Dz, N, m), 16, cfg, seed=9)
    assert [r["best_e"] for r in recs] == [int(w["out"]["bestE"][0]) for w in dbg["windows"]]
    assert recs[-1]["fr_ml"] == int(res["ml"][0][0]) | (int(res["ml"][1][0]) << 1)


def _session(info, S=5, nsw=8, T=0):
    mock = PTWBoardMock(info)
    brd = D.PTWBoard(mock)
    z = H.make_shots(3, 0.04, S, 11)
    brd.set_noise(0.03)
    brd.configure(nsw, T)
    res = brd.run(z["syn"], seed=4, batch=2)
    return mock, brd, z, res


def test_driver_mock_session_bit_exact(mock_info):
    mock, brd, z, res = _session(mock_info)
    meta = brd.meta()
    gold = H.golden_for(meta, res, z["syn"], len(z["syn"]))
    rep = H.compare(res, gold, meta)
    assert rep["mismatched_windows"] == 0 and rep["windows"] == 15
    assert list(res["b_shots"]) == [2, 2, 1] and list(res["b_seed"]) == [4, 5, 6]
    assert (res["cyc_win"] == meta["cyc_model"]).all()


def test_timed_mock_and_save_load(mock_info, tmp_path):
    mock, brd, z, res = _session(mock_info, S=3, T=200)
    assert (res["d_take"][res["win"] < 2] == mock.TR + 3).all() and (res["lag"] == 0).all()
    f = tmp_path / "r.npz"
    meta = brd.meta(); meta.update(reseed=1)
    D.save_results(str(f), res, meta, dict(syn=z["syn"], obs=z["obs"], clock_mhz=250.0))
    r2, m2, z2 = H.load_board_npz(str(f))
    gold = H.golden_for(m2, r2, z2["syn"], 3)
    assert H.compare(r2, gold, m2)["mismatched_windows"] == 0
    ts = D.timing_summary(r2, m2, 250.0)
    assert ts["misses"] == 0 and ts["slack_us"]["min"] > 0


def test_protocol_errors(mock_info):
    mock = PTWBoardMock(mock_info)
    brd = D.PTWBoard(mock)
    z = H.make_shots(3, 0.04, 1, 1)
    with pytest.raises(RuntimeError):
        brd.run_batch(z["syn"])                             # not seeded
    brd.seed(1)
    mock.write(D.THETA, 5)                                  # theta after FILL without RECALC
    with pytest.raises(ProtocolError):
        brd.run_batch(z["syn"])
    with pytest.raises(ProtocolError):
        brd.configure(8, 20)
        mock.write(D.CTRL, D.CTRL_RECALC)
        brd.run_batch(z["syn"])                             # too tight for the mock's timing

"""Options used by the robustness checks: memory length (M d noisy rounds), beta_min of the real-time decoder, BP-OSD with
min-sum scaling 0.625 as a window decoder, and fixed-budget / early-stop runs of the one-shot decoder."""
import importlib.util

import numpy as np
import pytest

from nimbus.decoding import real_time as g
from nimbus.decoding import windows as wn
from nimbus.decoding.code_capacity import sample_shots
from nimbus.decoding.one_shot import PTConfig, build_spec, c0_from_syndrome, run_fixed_budget, run_pt
from nimbus.decoding.phenomenological import memory_model, memory_shots

HAVE_REFS = all(importlib.util.find_spec(m) is not None for m in ("pymatching", "ldpc"))


def test_rounds_and_beta_min():
    d, p = 3, 0.03
    D4 = memory_shots(d, p, 2)
    D8 = memory_shots(d, p, 2, rounds=8 * d)
    assert D4[0].shape[1] == memory_model(d, 4 * d).cx.ndet and D8[0].shape[1] == memory_model(d, 8 * d).cx.ndet
    for N, (Dx, Dz, ox, oz) in ((4 * d, D4), (8 * d, D8)):
        for bmin in (0.1, 0.2):
            cfg = g.fpga_config(d, bmin=bmin)
            res, _ = g.decode_stream(d, p, Dx, Dz, N, d, d, 1 << 6, cfg, seed=1)
            assert set(g.failures(res, ox, oz)) == {"ml", "me"}
            assert np.isclose(g.window_spec(d, 2 * d, False, p, cfg).betas[-1], bmin)


@pytest.mark.skipif(not HAVE_REFS, reason="needs pymatching and ldpc (refs extra)")
def test_window_baselines_long_memory():
    from nimbus.decoding.baselines import WINDOW_DECODERS
    d, p, N = 3, 0.03, 24
    Dx, Dz, ox, oz = memory_shots(d, p, 3, rounds=N)
    f = {}
    for q in ("pm", "bp625"):
        f.update(wn.correction_failures(d, N, wn.decode_windows(d, N, wn.schedule(N, d, d), Dx, Dz,
                                                                WINDOW_DECODERS[q](p)), ox, oz))
    assert set(f) == {"pm", "bp625"} and all(v.shape == (3,) for v in f.values())


def test_fixed_budget_equals_power_of_two_run_and_early_stop():
    d, p = 3, 0.12
    spec = build_spec(d, p, PTConfig(NS=16, bmin=0.1, K=2, B=16))
    e, syn = sample_shots(spec.prob, 40, 7)
    c0 = c0_from_syndrome(spec, syn)
    ml, stops = run_fixed_budget(spec, c0, 64, seed=5)
    res = run_pt(spec, c0, 6, seed=5)                         # power-of-two budget: last block = sweeps [32, 64)
    assert (stops == 64).all() and (ml == res["counts"][:, 6].argmax(1)).all()
    ml2, stops2 = run_fixed_budget(spec, c0, 64, seed=5, gmin=16, th32=16)
    assert (stops2 < 64).any() and (stops2 >= 16).all()
    ml3, stops3 = run_fixed_budget(spec, c0, 46, seed=5)      # non-power-of-two budget (d = 3 board budget)
    assert (stops3 == 46).all() and set(np.unique(ml3)) <= {0, 1, 2, 3}

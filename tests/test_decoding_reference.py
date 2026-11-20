"""Bit identity of nimbus.decoding with stored reference outputs.

The files in tests/fixtures/decoding/ hold small outputs of the implementation this package was ported from (same
seeds): shots, start states, per-sweep class counts and decisions of both bit-exact decoders, early-stop runs, and the
corrections / failures of the CPU reference decoders. Every array must be reproduced exactly.
"""
from pathlib import Path

import numpy as np
import pytest

from nimbus.decoding import baselines as bl
from nimbus.decoding import real_time as g
from nimbus.decoding import windows as wn
from nimbus.decoding.code_capacity import class_failures, sample_shots
from nimbus.decoding.one_shot import (FPGA_REPLICAS, PTConfig, build_spec, c0_from_syndrome, run_fixed_budget,
                                      run_pt)
from nimbus.decoding.phenomenological import memory_model, memory_shots
from nimbus.problems.surface_code import SurfaceCodeProblem

FIX = Path(__file__).resolve().parent / "fixtures" / "decoding"
ROBUST = {"b0.1_K2_n24_g8_t20": dict(bmin=0.1, K=2, nsw=24, gmin=8, th32=20),
          "b0.05_K1_n24": dict(bmin=0.05, K=1, nsw=24, gmin=0, th32=0)}


def _have(*mods):
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in mods)


def one_shot_case(d, p, S, l2, chunk, refs=True, tn=True):
    """Arrays of one code-capacity case, keyed as in the reference file."""
    out = {}
    key = f"d{d}"
    spec = build_spec(d, p, PTConfig(NS=FPGA_REPLICAS[d], bmin=0.1, K=2, B=16))
    prob = spec.prob
    e, syn = sample_shots(prob, S, 7 + 1000 * chunk)
    c0 = c0_from_syndrome(spec, syn)
    res = run_pt(spec, c0, l2, seed=1 + 7919 * chunk)
    out.update({f"{key}_e": e, f"{key}_syn": syn, f"{key}_c0": c0, f"{key}_counts": res["counts"],
                f"{key}_best_at": res["best_at"], f"{key}_cs": res["cs"], f"{key}_bs": res["bs"],
                f"{key}_fail_cls": class_failures(prob, c0, e)})
    if refs:
        out.update({f"{key}_c_pm": bl.matching_correction(prob, syn), f"{key}_c_bo": bl.bposd_correction(prob, syn),
                    f"{key}_c_uf": bl.union_find_correction(prob, syn)})
    if tn and d <= 5:
        out[f"{key}_f_tn"] = bl.tensor_network_failures(prob, e, 8)
    for v, o in ROBUST.items():
        sp = build_spec(d, p, PTConfig(NS=FPGA_REPLICAS[d], bmin=o["bmin"], K=o["K"], B=16))
        ml, stops = run_fixed_budget(sp, c0, o["nsw"], seed=5, gmin=o["gmin"], th32=o["th32"])
        out[f"{key}_rob_{v}_ml"], out[f"{key}_rob_{v}_stops"] = ml, stops
    if d == 3:
        sp1 = build_spec(d, p, PTConfig(NS=16, bmin=0.1, K=1 << 30, B=16), library="stabilizers_logical")
        out[f"{key}_one_noswap_counts"] = run_pt(sp1, c0_from_syndrome(sp1, syn[:6]), 4, seed=3)["counts"]
    return out


def real_time_case(d, p, S, chunk, M, K, refs=True, software=True):
    out = {}
    key = f"d{d}_n{M}"
    N = M * d
    Dx, Dz, ox, oz = memory_shots(d, p, S, chunk=chunk, rounds=N)
    out.update({f"{key}_Dx": Dx, f"{key}_Dz": Dz, f"{key}_ox": ox, f"{key}_oz": oz})
    gseed = (11 << 40) ^ (chunk << 24) ^ (d << 16) ^ int(p * 1e5)
    cfgs = {"g_hwc_r24": (g.fpga_config(d, R=24), ("chain",)), "g_hwc_bm": (g.fpga_config(d, 16, bmin=0.05), ("chain",))}
    if software:
        cfgs.update({"g": (g.PTWConfig(R=16, bmin=0.1, K=1), ("chain", "gf2")),
                     "g_k2": (g.PTWConfig(R=16, bmin=0.1, K=2), ("chain",))})
    for name, (cfg, starts) in cfgs.items():
        res, info = g.decode_stream(d, p, Dx, Dz, N, d, d, 1 << K, cfg, seed=gseed, start=starts)
        for v, (bx, bz) in res.items():
            out[f"{key}_{name}_{v}_bx"], out[f"{key}_{name}_{v}_bz"] = bx, bz
        out[f"{key}_{name}_emax"] = np.array(info["Emax"])
    if refs:
        sched = wn.schedule(N, d, d)
        for q, fac in bl.WINDOW_DECODERS.items():
            (cx, cz), = wn.decode_windows(d, N, sched, Dx, Dz, fac(p)).values()
            out[f"{key}_w_{q}_cx"], out[f"{key}_w_{q}_cz"] = cx, cz
        G = memory_model(d, N)
        fb, _ = bl.memory_baselines(G, p, Dx, Dz, ox, oz, ("pm", "pmc", "bpm", "bp"))
        for q, f in fb.items():
            out[f"{key}_blk_{q}"] = f
        out[f"{key}_blk_uf"] = bl.memory_union_find_failures(G, Dx, Dz, ox, oz)
    (cx, cz), = wn.decode_windows(d, N, wn.schedule(N, 2, 3), Dx, Dz, wn.particular_decoder()).values()
    out[f"{key}_part_cx"], out[f"{key}_part_cz"] = cx, cz
    return out


def compare(ref, got):
    """Names of the reference arrays that ``got`` does not reproduce exactly (values compared as integers)."""
    bad = []
    for k, v in got.items():
        assert k in ref.files, f"no reference for {k}"
        r = ref[k]
        if r.shape != np.shape(v) or not np.array_equal(r.astype(np.int64), np.asarray(v).astype(np.int64)):
            bad.append(k)
    return bad


REFS = _have("pymatching", "ldpc", "stim")


def test_one_shot_reference():
    ref = np.load(FIX / "one_shot_reference.npz")
    got = one_shot_case(3, 0.12, 24, 5, 40, refs=REFS, tn=_have("qecsim"))
    pi = SurfaceCodeProblem(5, 0.1, noise="independent")
    got["ind_e"], got["ind_syn"] = sample_shots(pi, 16, 3)
    assert compare(ref, got) == []


def test_real_time_reference():
    ref = np.load(FIX / "real_time_reference.npz")
    assert compare(ref, real_time_case(3, 0.03, 20, 0, 4, 6, refs=REFS)) == []


def test_board_stream_reference():
    ref = np.load(FIX / "real_time_reference.npz")
    d, p, N = 3, 0.03, 12
    Dx, Dz, _, _ = memory_shots(d, p, 20, chunk=0, rounds=N)
    m = (d * d - 1) // 2
    recs, _ = g.decode_batch(d, p, g.syndromes(Dx[:2], N, m), g.syndromes(Dz[:2], N, m), 16, g.fpga_config(3), seed=7)
    got = {f"batch_{f}": np.array([r[f] for r in recs]) for f in
           ("shot", "win", "fin", "best_e", "best_c", "dec_ml", "frame", "fr_ml", "counts", "sigma")}
    assert compare(ref, got) == []


def test_budget_decisions_match_counts():
    """At power-of-two budgets the windowed ML decision from per-sweep counts equals the block counts of run_pt."""
    from nimbus.decoding.one_shot import min_energy_class_by_budget, ml_class_by_budget
    spec = build_spec(3, 0.12, PTConfig(NS=16, bmin=0.1, K=2, B=16))
    e, syn = sample_shots(spec.prob, 12, 9)
    res = run_pt(spec, c0_from_syndrome(spec, syn), 5, seed=2)
    budgets = [2, 4, 8, 16, 32]
    assert (ml_class_by_budget(res["cs"], budgets) == res["counts"][:, 1:].argmax(2).T).all()
    assert (min_energy_class_by_budget(res["bs"], budgets) == res["best_at"][:, 1:].T).all()


@pytest.mark.skipif(not REFS, reason="needs pymatching, ldpc and stim (refs extra)")
def test_window_decoders_valid_and_named():
    d, N, p = 3, 9, 0.05
    Dx, Dz, ox, oz = memory_shots(d, p, 10, rounds=N)
    for q, fac in bl.WINDOW_DECODERS.items():
        r = wn.decode_windows(d, N, wn.schedule(N, d, d), Dx, Dz, fac(p))   # asserts validity
        assert set(r) == {q}

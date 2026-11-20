"""Iterations-to-solution analysis of the graph-matching schedule studies (experiments/graph_matching/analysis)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

GM = Path(__file__).resolve().parents[1] / "experiments" / "graph_matching"
sys.path.insert(0, str(GM))
sys.path.insert(0, str(GM / "analysis"))
common = pytest.importorskip("common")
S = pytest.importorskip("sched_its")


def _shard(path, task_id, inst, variant, kappa, m, budgets, F, dup_of=None, lib="complete", role="tuning", n=10):
    recs, arrays = [], {}
    for b in budgets:
        recs.append(dict(instance=inst, alias=inst, lib=lib, variant=variant, kappa=kappa, m=m, budget_log2=b, role=role, n=n,
                         dup_of=dup_of))
        if dup_of is None:
            arrays[f"{inst}|{variant}|b{b}|bestF"] = np.asarray(F[b], float)
    common.write_shard(common.shard_path(path, task_id), recs, arrays)


def test_tts_definition():
    assert common.tts_at(64, 1.0) == 64
    assert common.tts_at(64, 0.0) == math.inf
    assert common.tts_at(100, 0.5) == pytest.approx(100 * math.log(0.01) / math.log(0.5))


def test_success_uses_098_of_psi_ref():
    F = np.array([0.979, 0.98, 1.0, np.nan])
    assert S.success(F, 1.0) == 0.5


def test_shards_selection_and_dup_resolution(tmp_path):
    ref = {f"i{k}": 1.0 for k in range(6)}
    good = {6: np.ones(8), 8: np.ones(8)}                   # every replica succeeds: ITS = 2^6 at the smallest budget
    poor = {6: np.r_[np.ones(2), np.zeros(6)], 8: np.r_[np.ones(4), np.zeros(4)]}
    for k in range(6):
        _shard(tmp_path, 10 * k, f"i{k}", "k2_m1", 2.0, 1.0, (6, 8), good)
        _shard(tmp_path, 10 * k + 1, f"i{k}", "k4_m1", 4.0, 1.0, (6, 8), poor)
        _shard(tmp_path, 10 * k + 2, f"i{k}", "k8_m1", 8.0, 1.0, (6, 8), {}, dup_of="k2_m1")
    rows = S.load_shards(tmp_path)
    assert len(rows) == 6 * 3 * 2
    dup = [r for r in rows if r["variant"] == "k8_m1"][0]
    assert np.array_equal(dup["bestF"], np.ones(8))
    res = S.rank_settings(rows, ref, budgets=[6, 8], boot=50)
    assert res[0]["setting"] == "k2_m1" and res[0]["its"] == 64 and res[0]["budget_log2_star"] == 6
    assert res[1]["setting"] == "k8_m1"                    # same ITS, tie broken towards kappa = 2
    assert S.select(res, kappa2_rule=True)["setting"] == "k2_m1"
    per = S.per_instance([r for r in rows if r["variant"] == "k4_m1"], ref)
    o = common.optimal_median_its(per)
    assert o["budget_log2_star"] == 6 and o["opt"] == pytest.approx(64 * math.log(0.01) / math.log(0.75))

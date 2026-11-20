"""Scripts of the L = 10 method comparison (experiments/graph_matching): penalty-machine CLI, setting selection,
export to the domain-solver format."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

GM = Path(__file__).resolve().parents[1] / "experiments" / "graph_matching"
sys.path.insert(0, str(GM))
sys.path.insert(0, str(GM / "analysis"))
sys.path.insert(0, str(GM / "domain_solvers"))


@pytest.fixture(scope="module")
def inst(tmp_path_factory):
    import stage_instances

    out = tmp_path_factory.mktemp("inst")
    recs = stage_instances.synthetic([5], 2, 0.05, 0, out)           # index 0: training, index 1: test
    (out / "manifest.json").write_text(json.dumps(recs))
    return out


def test_penalty_cli_records(inst, tmp_path):
    import penalty_baselines as PB

    choice = tmp_path / "choice.json"
    choice.write_text(json.dumps({"nhoim": {"5": dict(lam_c=1, mult=1.0, Tend_d0=0.3, ratio=30, Tstart_d0=9.0)}}))
    outs = []
    for k in range(2):
        out = tmp_path / f"test{k}.jsonl"
        PB.main(["test", "--kind", "nhoim", "--n", "5", "--choice", str(choice), "--names", "syn_n5_s0.05_o0_01",
                 "--budgets", "10", "12", "--replicas", "8", "--threads", "1", "--inst", str(inst), "--out", str(out)])
        outs.append([json.loads(x) for x in out.read_text().splitlines()])
    for a, b in zip(*outs):
        a.pop("wall_s"), b.pop("wall_s")
        assert a == b
    for r in outs[0]:
        F = np.array([np.nan if x is None else x for x in r["bestF"]])
        ok = np.isfinite(F)
        assert r["p1"] == pytest.approx(np.mean(ok & (np.where(ok, F, 0) >= r["F_ref"] * (1 - 1e-9))))
        assert r["tests"] <= 2 ** r["budget_log2"] and r["n_aux"] == 0 and r["stage"] == "test"
    PB.main(["tune", "--kind", "neurosa", "--names", "syn_n5_s0.05_o0_00", "--budgets", "10", "--replicas", "4",
             "--threads", "1", "--grid", '{"lam_c": [1], "Tend_d0": [0.3], "ratio": [10], "mult": [1.0]}',
             "--inst", str(inst), "--out", str(tmp_path / "tune.jsonl")])
    rows = [json.loads(x) for x in (tmp_path / "tune.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["stage"] == "tuning" and rows[0]["n_aux"] > 0
    with pytest.raises(SystemExit):     # selection only on training instances
        PB.main(["tune", "--kind", "nhoim", "--names", "syn_n5_s0.05_o0_01", "--budgets", "10", "--inst", str(inst),
                 "--out", str(tmp_path / "x.jsonl")])


def test_selection_rule_picks_smallest_median_its():
    import select_settings as SS

    def per(p):
        return {f"i{k}": {b: (2.0 ** b, p) for b in range(6, 21, 2)} for k in range(6)}

    rows = [dict(set="synthetic", method="complete", setting=s, kappa=k, m=m, instances=6, complete=True)
            for s, k, m in (("a", 2, 1), ("b", 4, 1), ("c", 2, 0.5))]
    for r, p in zip(rows, (0.5, 0.9, 0.1)):
        o = SS.C.optimal_median_its(per(p))
        r.update(median_its=o["opt"], log2_median=np.log2(o["opt"]), budget_log2_star=o["budget_log2_star"],
                 mean_log2_its=float(np.log2(o["opt"])))
    c = SS.choose(rows, "complete")
    assert (c["kappa"], c["m"], c["budget_log2_star"]) == (4, 1, 6)
    assert "budget low (2^6)" in c["edges"]


def test_export_format_scores_six_psi(inst):
    from export_mat import solver_format

    from nimbus.problems.formats.matching_tensor import evaluate_F, load_tensor

    m = json.loads((inst / "manifest.json").read_text())[1]
    t = load_tensor(inst / m["tensor"])
    ind, val = solver_format(t)
    rng = np.random.default_rng(1)
    for _ in range(5):
        p = rng.permutation(t.n)
        x = np.zeros(t.n * t.n)
        x[np.arange(t.n) * t.n + p] = 1
        score = float(np.sum(val * x[ind[:, 0]] * x[ind[:, 1]] * x[ind[:, 2]]))
        assert score == pytest.approx(6 * float(evaluate_F(t, p)), rel=1e-9)

"""nimbus.analysis.report: summaries over stored run directories (owned by experiments-cli)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.quick

from nimbus.analysis import report
from nimbus.experiments.runner import solve

TOY = {
    "schema": "nimbus.run/1", "name": "toy_analysis",
    "problem": {"kind": "toy_cardinality"},
    "transitions": {"library": "pair_exchanges"},
    "annealing": {"schedule": "geometric", "params": {"tau_start": 2.0, "tau_end": 0.05}, "attempts": 300},
    "run": {"seed": 1, "replicas": 2},
    "backend": {"name": "emulation"},
}


def test_load_run(tmp_path):
    solve(TOY, output_dir=tmp_path / "run")
    loaded = report.load_run(tmp_path / "run")
    assert loaded["manifest"]["status"] == "ok"
    assert len(loaded["rows"]) == 2


def test_load_run_missing_dir_is_empty(tmp_path):
    loaded = report.load_run(tmp_path / "nope")
    assert loaded["manifest"] is None
    assert loaded["rows"] == []


def test_summarize_single_run(tmp_path):
    solve(TOY, output_dir=tmp_path / "run")
    out = report.summarize_runs([tmp_path / "run"])
    assert out["n_runs"] == 1
    entry = out["runs"][0]
    assert entry["status"] == "ok"
    assert entry["n_feasible"] == 2
    assert entry["best_feasible_objective"] == -7.0
    assert out["pooled"]["best_feasible_objective"] == -7.0


def test_tts_from_target_hits(tmp_path):
    # the optimum of the toy is -7; a target of -7 is hit by some replicas, the run does not stop on it
    cfg = dict(TOY, run=dict(TOY["run"], replicas=4, targets=-7.0, stop_on_target=False))
    solve(cfg, output_dir=tmp_path / "run")
    rows = report.load_run(tmp_path / "run")["rows"]
    hits = sum(1 for r in rows if r["target_hit"])
    assert all(r["target_hit"] is not None for r in rows)
    entry = report.summarize_runs([tmp_path / "run"])["runs"][0]
    assert entry["success_criterion"] == "target"
    assert entry["n_target_hits"] == hits and entry["p_success"] == hits / 4
    assert "tts_0.99" in entry
    # an unreachable target: no hits, TTS unbounded, although every replica ended feasible
    cfg = dict(TOY, run=dict(TOY["run"], replicas=4, targets=-100.0))
    solve(cfg, output_dir=tmp_path / "run2")
    entry = report.summarize_runs([tmp_path / "run2"])["runs"][0]
    assert entry["n_target_hits"] == 0 and entry["p_success"] == 0.0 and entry["n_feasible"] == 4
    assert entry["tts_0.99"] == float("inf")


def test_summarize_without_target_uses_feasible_fraction(tmp_path):
    solve(TOY, output_dir=tmp_path / "run")
    entry = report.summarize_runs([tmp_path / "run"])["runs"][0]
    assert entry["success_criterion"] == "feasible" and entry["n_target_hits"] is None
    assert entry["p_success"] == entry["p_feasible"]


def test_summarize_multiple_runs(tmp_path):
    for seed in (1, 2):
        cfg = dict(TOY, run=dict(TOY["run"], seed=seed))
        solve(cfg, output_dir=tmp_path / f"run{seed}")
    out = report.summarize_runs([tmp_path / "run1", tmp_path / "run2"])
    assert out["n_runs"] == 2
    assert out["pooled"]["n_feasible"] == 4

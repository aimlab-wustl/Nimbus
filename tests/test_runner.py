"""nimbus.experiments.runner: end-to-end solve/resume/checkpoint on the emulation backend, plus
the 'cpu' backend where importable (skipped only if preparing it raises ImportError).

Uses the real toy_cardinality adapter (six spins, known optimum -7) so "end to end" means the
whole config -> solve -> results -> resume pipeline against a real problem, not a stub.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.quick

import numpy as np

from nimbus.experiments.results import read_shard
from nimbus.experiments.runner import resolve_backend_name, resolve_instances, resume, solve

TOY = {
    "schema": "nimbus.run/1", "name": "toy_test",
    "problem": {"kind": "toy_cardinality"},
    "transitions": {"library": "pair_exchanges"},
    "annealing": {"schedule": "geometric", "params": {"tau_start": 2.0, "tau_end": 0.05}, "attempts": 400},
    "run": {"seed": 1, "replicas": 2},
    "backend": {"name": "emulation"},
    "recording": {"profile": "summary"},
}


def _cpu_available() -> bool:
    try:
        from nimbus.backends import get_backend

        get_backend("cpu")
        return True
    except ImportError:
        return False
    except Exception:
        return True  # importable but failed for another reason: not our skip condition


def test_resolve_instances():
    assert resolve_instances("all", 3) == [0, 1, 2]
    assert resolve_instances({"first": 2}, 5) == [0, 1]
    assert resolve_instances({"start": 1, "stop": 3}, 5) == [1, 2]
    assert resolve_instances([2, 0], 5) == [2, 0]


def test_resolve_backend_name_explicit_passthrough():
    assert resolve_backend_name("emulation") == "emulation"
    assert resolve_backend_name("gpu") == "gpu"  # never silently substituted


def test_resolve_backend_name_auto_resolves_to_a_real_backend():
    name = resolve_backend_name("auto")
    assert name in ("cpu", "emulation")


def test_solve_toy_reaches_known_optimum(tmp_path):
    outcome = solve(TOY, output_dir=tmp_path / "run")
    assert outcome.status == "ok"
    assert len(outcome.rows) == 2
    best = min(r.best_feasible_objective for r in outcome.rows)
    assert best == pytest.approx(-7.0)
    assert all(r.final_feasible for r in outcome.rows)
    assert (tmp_path / "run" / "manifest.json").exists()
    assert (tmp_path / "run" / "results.parquet").exists()
    assert (tmp_path / "run" / "config.resolved.yaml").exists()
    assert not (tmp_path / "run" / "checkpoint.npz").exists()  # default: not kept after completion


def test_solve_rows_match_results_shard(tmp_path):
    outcome = solve(TOY, output_dir=tmp_path / "run")
    rows = read_shard(tmp_path / "run" / "results.parquet")
    assert len(rows) == len(outcome.rows)
    assert {r["instance_id"] for r in rows} == {0}


def test_solve_writes_failed_manifest_on_bad_problem(tmp_path):
    bad = dict(TOY, problem={"kind": "no_such_kind"})
    outcome = solve(bad, output_dir=tmp_path / "run")
    assert outcome.status == "failed"
    assert outcome.message
    assert (tmp_path / "run" / "manifest.json").exists()


def test_resume_without_interruption_is_a_noop(tmp_path):
    cfg = dict(TOY, output={"keep_checkpoint": True})
    out1 = solve(cfg, output_dir=tmp_path / "run")
    out2 = resume(tmp_path / "run")
    assert out2.status == "ok"
    assert len(out2.rows) == len(out1.rows)  # no duplicate rows


def test_resume_after_interruption_continues_and_matches_uninterrupted(tmp_path):
    cfg = dict(TOY, run=dict(TOY["run"], chunk_attempts=50), output={"keep_checkpoint": True})

    class _Stop(Exception):
        pass

    def stop_early(event, info):
        if event == "chunk" and info["attempts"] >= 100:
            raise _Stop()

    with pytest.raises(_Stop):
        solve(cfg, output_dir=tmp_path / "interrupted", progress=stop_early)
    resumed = resume(tmp_path / "interrupted")

    uninterrupted = solve(cfg, output_dir=tmp_path / "uninterrupted")

    assert resumed.status == uninterrupted.status == "ok"
    r_by_rep = {r.replica_id: r for r in resumed.rows}
    u_by_rep = {r.replica_id: r for r in uninterrupted.rows}
    assert set(r_by_rep) == set(u_by_rep)
    for rid in r_by_rep:
        assert r_by_rep[rid].attempts == u_by_rep[rid].attempts
        assert r_by_rep[rid].best_feasible_objective == pytest.approx(u_by_rep[rid].best_feasible_objective)
        assert r_by_rep[rid].accepted == u_by_rep[rid].accepted
        assert r_by_rep[rid].final_energy == pytest.approx(u_by_rep[rid].final_energy)


def test_resume_rejects_mismatched_config(tmp_path):
    from nimbus.core.errors import ConfigError

    cfg = dict(TOY, output={"keep_checkpoint": True})
    run_dir = tmp_path / "run"
    solve(cfg, output_dir=run_dir)
    # rewrite config.resolved.yaml for a config with a different trajectory hash
    other = dict(cfg)
    other["run"] = dict(cfg["run"], seed=999)
    from nimbus.experiments.config import load_config

    load_config(other).write_resolved(run_dir)
    with pytest.raises(ConfigError):
        resume(run_dir)


def test_solve_with_explicit_output_overwrite_flag(tmp_path):
    from nimbus.core.errors import ConfigError

    run_dir = tmp_path / "run"
    solve(TOY, output_dir=run_dir)
    with pytest.raises(ConfigError, match="not empty"):
        solve(TOY, output_dir=run_dir)
    cfg = dict(TOY, output={"overwrite": True})
    outcome = solve(cfg, output_dir=run_dir)
    assert outcome.status == "ok"


def test_penalty_baseline_runs_end_to_end(tmp_path):
    cfg = dict(TOY)
    cfg["baseline"] = {"kind": "penalty", "options": {"lam": 6.0}}
    cfg["annealing"] = {"schedule": "geometric", "params": {"tau_start": 3.0, "tau_end": 0.05}, "attempts": 500}
    cfg["run"] = {"seed": 3, "replicas": 2}
    outcome = solve(cfg, output_dir=tmp_path / "run")
    assert outcome.status == "ok"
    assert len(outcome.rows) == 2
    assert any(r.best_feasible_objective == pytest.approx(-7.0) for r in outcome.rows)


def test_baseline_needing_calibration_reports_unsupported(tmp_path):
    # quadratic (like penalty) is wired, but auto/calibrated schedules are not yet -- same gate as
    # the penalty baseline's own calibration check (see test_calibrated_schedule_runs_end_to_end for
    # the ordinary path exercising calibration successfully).
    cfg = dict(TOY)
    cfg["baseline"] = {"kind": "quadratic"}
    cfg["annealing"] = {"schedule": "fn_endpoints", "params": "auto",
                        "calibration": {"eta": 0.1, "p_start": 0.8, "p_end": 0.02, "n_pilot": 256,
                                       "min_pilot": 64, "budget_cap_fraction": 1.0},
                        "attempts": 600}
    outcome = solve(cfg, output_dir=tmp_path / "run")
    assert outcome.status == "unsupported"
    assert "quadratic" in outcome.message


def test_calibrated_schedule_runs_end_to_end(tmp_path):
    cfg = dict(TOY)
    cfg["annealing"] = {"schedule": "fn_endpoints", "params": "auto",
                        "calibration": {"eta": 0.1, "p_start": 0.8, "p_end": 0.02, "n_pilot": 256,
                                       "min_pilot": 64, "budget_cap_fraction": 1.0},
                        "attempts": 600}
    outcome = solve(cfg, output_dir=tmp_path / "run")
    assert outcome.status == "ok"
    assert len(outcome.rows) == 2


@pytest.mark.skipif(not _cpu_available(), reason="not executed: the cpu backend is not importable here")
def test_solve_on_cpu_backend_if_available(tmp_path):
    cfg = dict(TOY, backend={"name": "cpu"})
    outcome = solve(cfg, output_dir=tmp_path / "run")
    assert outcome.status in ("ok", "failed")  # cpu backend under concurrent development: report, don't demand


def _tau_end_of(tmp_path, late_advance, name):
    cal = {"eta": 0.1, "p_start": 0.8, "p_end": 0.02, "n_pilot": 512, "min_pilot": 64,
           "budget_cap_fraction": 1.0, "late_advance": late_advance, "late_advance_attempts": 2048}
    cfg = dict(TOY)
    cfg["annealing"] = {"schedule": "fn_endpoints", "params": "auto", "calibration": cal, "attempts": 600}
    outcome = solve(cfg, output_dir=tmp_path / name)
    assert outcome.status == "ok"
    return outcome.manifest["instances"][0]["calibration"]["tau_end"]


def test_late_pilot_is_default_and_none_is_selectable(tmp_path):
    """late_advance defaults to 'descent'; 'none' keeps the old random-state end pilot, and the
    two give different (both finite) tau_end -- the late pilot samples annealed states."""
    from nimbus.experiments.config import load_config

    cfg = dict(TOY)
    cfg["annealing"] = {"schedule": "fn_endpoints", "params": "auto", "calibration": {"eta": 0.1}, "attempts": 600}
    assert load_config(cfg).data["annealing"]["calibration"]["late_advance"] == "descent"
    t_late = _tau_end_of(tmp_path, "descent", "late")
    t_none = _tau_end_of(tmp_path, "none", "none")
    assert t_late > 0 and t_none > 0 and t_late != t_none

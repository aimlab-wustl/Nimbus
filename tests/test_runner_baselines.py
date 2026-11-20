"""nimbus.experiments.runner: the generic quadratic baseline wired into the solve() pipeline,
following the 'penalty' baseline path's own conventions.

A small, fast, end-to-end solve() against the toy_cardinality adapter -- not a stub -- checking that
results actually land in results.parquet/manifest.json with the right schema.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.quick

from nimbus.experiments.runner import solve

TOY = {
    "schema": "nimbus.run/1", "name": "toy_test",
    "problem": {"kind": "toy_cardinality"},
    "transitions": {"library": "pair_exchanges"},
    "annealing": {"schedule": "geometric", "params": {"tau_start": 2.0, "tau_end": 0.05}, "attempts": 400},
    "run": {"seed": 1, "replicas": 2},
    "backend": {"name": "emulation"},
    "recording": {"profile": "summary"},
}


# ============================================================================================
# quadratic
# ============================================================================================
def test_quadratic_baseline_runs_end_to_end(tmp_path):
    cfg = dict(TOY)
    cfg["baseline"] = {"kind": "quadratic", "options": {"lam": 6.0}}
    cfg["annealing"] = {"schedule": "geometric", "params": {"tau_start": 3.0, "tau_end": 0.05}, "attempts": 500}
    cfg["run"] = {"seed": 3, "replicas": 2}
    outcome = solve(cfg, output_dir=tmp_path / "run")
    assert outcome.status == "ok"
    assert len(outcome.rows) == 2
    assert outcome.manifest["timings"]
    assert (tmp_path / "run" / "results.parquet").exists()

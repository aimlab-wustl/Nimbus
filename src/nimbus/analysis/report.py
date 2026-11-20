"""Summaries/tables built from stored run directories only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nimbus.analysis import stats as _stats
from nimbus.experiments.manifest import read_manifest
from nimbus.experiments.results import read_shard

__all__ = ["load_run", "summarize_runs"]


def load_run(run_dir) -> dict:
    """``{manifest, rows}`` for one completed (or partial) run directory."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir) if (run_dir / "manifest.json").exists() else None
    shard = run_dir / "results.parquet"
    rows = read_shard(shard) if shard.exists() else []
    return {"run_dir": str(run_dir), "manifest": manifest, "rows": rows}


def summarize_runs(run_dirs, *, kind: str = "summary") -> dict:
    """Per-run and pooled summary: status, row counts, feasible-best statistics, TTS at 0.99 using
    the attempt budget per replica. The success fraction is the fraction of replicas that hit the target
    when the run set one (whether or not it stopped on the target), otherwise the fraction of replicas
    that ended with a feasible state."""
    runs = [load_run(d) for d in run_dirs]
    out = {"n_runs": len(runs), "runs": []}
    all_feasible = []
    for r in runs:
        rows = r["rows"]
        feas = [row["best_feasible_objective"] for row in rows if row.get("best_feasible_objective") is not None]
        n_total_attempts = max((row.get("attempts") or 0) for row in rows) if rows else 0
        with_target = [row for row in rows if row.get("target_hit") is not None]
        if with_target:
            n_hits = sum(1 for row in with_target if row["target_hit"])
            p_success = n_hits / len(with_target)
            criterion = "target"
        else:
            n_hits = None
            p_success = (len(feas) / len(rows)) if rows else float("nan")
            criterion = "feasible"
        entry = {
            "run_dir": r["run_dir"],
            "status": (r["manifest"] or {}).get("status"),
            "n_rows": len(rows),
            "n_feasible": len(feas),
            "p_feasible": (len(feas) / len(rows)) if rows else float("nan"),
            "success_criterion": criterion,
            "n_target_hits": n_hits,
            "p_success": p_success,
            "best_feasible_objective": min(feas) if feas else None,
            "attempts_per_replica": n_total_attempts,
        }
        if rows and np.isfinite(p_success):
            entry["tts_0.99"] = _stats.tts(p_success, n_total_attempts, 0.99).tts
        out["runs"].append(entry)
        all_feasible.extend(feas)
    out["pooled"] = {"n_feasible": len(all_feasible),
                     "best_feasible_objective": min(all_feasible) if all_feasible else None}
    return out


def _main(argv=None) -> int:  # pragma: no cover - thin CLI wrapper, exercised via nimbus.cli
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    args = p.parse_args(argv)
    print(json.dumps(summarize_runs(args.run_dirs), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

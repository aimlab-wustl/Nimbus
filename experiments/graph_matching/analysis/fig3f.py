"""Fig. 3f: iterations to solution at L = 10 on 20 synthetic test instances and 40 WILLOW test pairs.

  python analysis/fig3f.py --shards RUNS/willow_l10/test \
      --penalty RUNS/willow_l10/btest/*.jsonl RUNS/willow_l10/test_neurosa_willow.jsonl --out ../../source_data/fig3

Target: the certified optimum. For every method and problem set at its selected setting (``configs/willow_sym``): per
instance and budget ``t`` the iterations to solution ``ITS_i(t) = t ln(0.01) / ln(1 - p_i(t))`` (``p_i(t)`` = fraction of
replicas at the optimum; ``p = 1`` gives ``t``, ``p = 0`` gives infinity); the reported value is the minimum over budgets of
the median over instances, with a 95% percentile bootstrap interval over instances (5000 resamples, seed 20261015; the same
resample at every budget, then the minimum). Iterations: proposed moves for this work, single-spin update tests for the
penalty machines.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_runs as LR  # noqa: E402

C = LR.C
CFG = HERE.parent / "configs" / "willow_sym"
LABEL = {"complete": "this work, complete library", "path": "this work, path library",
         "nhoim": "N-HOIM with penalty", "neurosa": "NeuroSA with quadratized penalty"}
SETS = (("synthetic", "test", 20), ("willow", "willow_test", 40))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", nargs="+", required=True, help="folder(s) of the test shards of this work")
    ap.add_argument("--penalty", nargs="+", required=True, help="penalty-machine test records (JSON lines; globs allowed)")
    ap.add_argument("--choice", default=str(CFG / "choice.json"))
    ap.add_argument("--baseline-choice", nargs=2, default=[str(CFG / "baseline_choice_synthetic.json"),
                                                            str(CFG / "baseline_choice_willow.json")],
                    metavar=("SYNTHETIC", "WILLOW"))
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ch = json.loads(Path(a.choice).read_text())
    bch = {"synthetic": json.loads(Path(a.baseline_choice[0]).read_text()),
           "willow": json.loads(Path(a.baseline_choice[1]).read_text())}
    files = sorted(f for p in a.penalty for f in glob.glob(p))
    rows = LR.load_sched_shards(a.shards)
    out = []
    for set_, stage, n_inst in SETS:
        for key in ("complete", "path", "nhoim", "neurosa"):
            if key in ("complete", "path"):
                c = ch[set_][key]
                per = LR.ours_points(rows, set_, key, c["kappa"], c["m"])
                reps, setting = 256, f"m={c['m']:g}, kappa={c['kappa']:g}"
            else:
                par = bch[set_][key]["10"]
                per, reps = LR.penalty_points(files, key, stage, par)
                setting = f"lambda={par['lam_c']:g}, T_end={par['Tend_d0']:g}, ratio={par['ratio']:g}"
            assert len(per) == n_inst, (set_, key, len(per))
            o = C.optimal_median_its(per, boot=a.boot, seed=20261015)
            out.append({"problem_set": set_, "method": LABEL[key], "setting": setting, "replicas": reps,
                        "instances": o["instances"], "solved_at_best_budget": o["solved_at_star"],
                        "its_median_iterations": o["opt"], "its_ci_low": o["lo"], "its_ci_high": o["hi"],
                        "best_budget_log2": o["budget_log2_star"],
                        "largest_budget_iterations": max(t for v in per.values() for (t, _) in v.values())})
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "fig3f.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) and math.isfinite(v) else v) for k, v in r.items()})
    for r in out:
        print(f"{r['problem_set']:9s} {r['method']:34s} {r['its_median_iterations']:.4g} "
              f"[{r['its_ci_low']:.4g}, {r['its_ci_high']:.4g}] at 2^{r['best_budget_log2']}")
    print("wrote", dst / "fig3f.csv")


if __name__ == "__main__":
    main()

"""Fig. 3c: probability that a replica reaches the certified optimum versus iterations per replica, 40 WILLOW test pairs.

  python analysis/fig3c.py --shards RUNS/willow_l10/test \
      --nhoim RUNS/willow_l10/btest/test_nhoim_willow_*.jsonl --neurosa RUNS/willow_l10/test_neurosa_willow.jsonl \
      --domain RUNS/willow_domain/willow_domain_solvers.csv --out ../../source_data/fig3

Methods and settings (``configs/willow_sym/choice.json``, ``baseline_choice_willow.json``): this work with the complete
and the path library at the selected ``(m, kappa)``; N-HOIM with penalty and NeuroSA with quadratized penalty at their
selected settings. Per budget: mean over pairs of the per-pair fraction of replicas at the certified optimum, with a 95%
percentile bootstrap interval over pairs (5000 resamples). Domain solvers: mean over pairs of the fraction of restarts
that return the certified optimum (``domain_solvers/willow_domain_summary.py``).
Writes ``fig3c.csv`` and ``fig3c_domain_solvers.csv``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_runs as LR  # noqa: E402

CFG = HERE.parent / "configs" / "willow_sym"
LABEL = {"complete": "this work, complete library", "path": "this work, path library",
         "nhoim": "N-HOIM with penalty", "neurosa": "NeuroSA with quadratized penalty"}
DOMAIN = {"adapt_bcagm3": "Adapt-BCAGM3", "bcagm3": "BCAGM3", "rrwhm": "RRWHM"}


def curves(per, method, replicas, setting, rng):
    out = []
    insts = sorted(per)
    for lb in sorted(set.intersection(*[set(per[i]) for i in insts])):
        mean, lo, hi = LR.boot_mean([per[i][lb][1] for i in insts], rng=rng)
        out.append({"method": method, "setting": setting, "budget_log2": lb,
                    "iterations_per_replica": int(round(np.median([per[i][lb][0] for i in insts]))),
                    "replicas": replicas, "pairs": len(insts), "p_optimum": mean, "p_optimum_ci_low": lo,
                    "p_optimum_ci_high": hi})
    return out


def _files(patterns):
    return sorted(f for p in patterns for f in glob.glob(p))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", nargs="+", required=True, help="folder(s) of the test shards of this work")
    ap.add_argument("--nhoim", nargs="+", required=True, help="N-HOIM test records (JSON lines; glob patterns allowed)")
    ap.add_argument("--neurosa", nargs="+", required=True, help="NeuroSA test records")
    ap.add_argument("--domain", required=True, help="willow_domain_solvers.csv from willow_domain_summary.py")
    ap.add_argument("--choice", default=str(CFG / "choice.json"))
    ap.add_argument("--baseline-choice", default=str(CFG / "baseline_choice_willow.json"))
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ch = json.loads(Path(a.choice).read_text())["willow"]
    bch = json.loads(Path(a.baseline_choice).read_text())
    rows = LR.load_sched_shards(a.shards)
    rng = np.random.default_rng(a.seed)
    out = []
    for lib in ("complete", "path"):
        c = ch[lib]
        per = LR.ours_points(rows, "willow", lib, c["kappa"], c["m"])
        assert len(per) == 40, (lib, len(per))
        out += curves(per, LABEL[lib], 256, f"m={c['m']:g}, kappa={c['kappa']:g}", rng)
    for kind, files in (("nhoim", a.nhoim), ("neurosa", a.neurosa)):
        par = bch[kind]["10"]
        per, reps = LR.penalty_points(_files(files), kind, "willow_test", par)
        assert len(per) == 40, (kind, len(per))
        out += curves(per, LABEL[kind], reps, f"lambda={par['lam_c']:g}, T_end={par['Tend_d0']:g}, ratio={par['ratio']:g}", rng)
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    cols = list(out[0])
    with open(dst / "fig3c.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(a.domain, newline="", encoding="utf-8") as fh:
        dom = {r["solver"]: r for r in csv.DictReader(fh)}
    with open(dst / "fig3c_domain_solvers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["solver", "pairs", "restarts_per_pair", "p_restart_optimum", "p_restart_optimum_ci_low",
                    "p_restart_optimum_ci_high"])
        for key, lab in DOMAIN.items():
            d = dom[key]
            w.writerow([lab, d["instances"], d["restarts_per_instance"], d["p_restart_opt_mean"], d["p_restart_opt_mean_lo"],
                        d["p_restart_opt_mean_hi"]])
    print("wrote", dst / "fig3c.csv", "and", dst / "fig3c_domain_solvers.csv")


if __name__ == "__main__":
    main()

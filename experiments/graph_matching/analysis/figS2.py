"""Fig. S2a, b: per-replica results of the control runs (anneal, no cooling, random walk).

  python analysis/figS2.py --per-replica RUNS/control/per_replica --summary RUNS/control/summary.jsonl \
      --out-a ../../source_data/figS2/figS2a.csv --out-b ../../source_data/figS2/figS2b.csv

Reads the per-replica files of ``control_random_walk.py`` and checks that their means equal the summary lines. Writes
one row per replica: instance index (1-3 per L), L, variant, replica, best objective / Psi_ref (S2a), and in addition
the fraction of proposals accepted over the run (S2b). Medians over replicas (the bars of S2a) are listed in
``--out-medians`` if given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sched_its as S  # noqa: E402

NAMES = [f"syn_n{n}_s0.05_o0_0{i}" for n in (20, 30) for i in (6, 7, 8)]
ARMS = {"anneal": "anneal", "hot": "no cooling", "walk": "random walk"}
BUDGET = {20: 18, 30: 20}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-replica", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out-a", required=True)
    ap.add_argument("--out-b", required=True)
    ap.add_argument("--out-medians", default=None)
    a = ap.parse_args(argv)
    summ = {(r["instance"], r["arm"]): r for r in map(json.loads, open(a.summary, encoding="utf-8"))}
    rows, med = [], []
    for name in NAMES:
        L, idx = int(name.split("_")[1][1:]), int(name[-2:]) - 5
        for arm, label in ARMS.items():
            z = np.load(Path(a.per_replica) / f"{name}__{arm}.npz")
            b, acc = np.asarray(z["best_over_ref"], float), np.asarray(z["acceptance"], float)
            s = summ[(name, arm)]
            assert np.isclose(b.mean(), s["best_over_ref_mean"], rtol=0, atol=1e-12), (name, arm)
            assert np.isclose(acc.mean(), s["acceptance"], rtol=1e-9), (name, arm)   # equal attempts per replica
            for k, (x, y) in enumerate(zip(b, acc)):
                rows.append(dict(L=L, instance_index=idx, instance=name, variant=label, iterations_log2=BUDGET[L], replica=k,
                                 psi_over_psi_ref=float(x), acceptance_fraction=float(y)))
            med.append(dict(L=L, instance_index=idx, instance=name, variant=label, replicas=len(b),
                            median_psi_over_psi_ref=float(np.median(b)), mean_psi_over_psi_ref=float(b.mean()),
                            max_psi_over_psi_ref=float(b.max()), fraction_at_least_098=float((b >= 0.98).mean()),
                            acceptance_fraction_run=s["acceptance"]))
    cols = ["L", "instance_index", "instance", "variant", "iterations_log2", "replica", "psi_over_psi_ref"]
    S.write_csv(a.out_a, rows, cols)
    S.write_csv(a.out_b, rows, cols + ["acceptance_fraction"])
    if a.out_medians:
        S.write_csv(a.out_medians, med, list(med[0]))
    walk = [r["psi_over_psi_ref"] for r in rows if r["variant"] == "random walk"]
    hot = [m["mean_psi_over_psi_ref"] for m in med if m["variant"] == "no cooling"]
    print(f"random walk: largest Psi/Psi_ref over all replicas = {max(walk):.4f}; "
          f"no cooling: largest mean over replicas = {max(hot):.4f}")


if __name__ == "__main__":
    main()

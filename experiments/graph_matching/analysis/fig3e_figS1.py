"""Fig. 3e and Supplementary Fig. S1b, S1c: synthetic instances with the automatically calibrated schedule.

  python analysis/fig3e_figS1.py --sigma005 BUNDLES/sigma005.npz --sigma010 BUNDLES/sigma010.npz --out ../../source_data

Inputs: the bundles of the runs ``configs/auto_schedule/tasks_sigma005.csv`` and ``tasks_sigma010.csv`` (``extract_shards.py``)
and ``configs/psi_ref.csv``. Only the 20 test instances per size are used.

Per instance and budget: the per-replica best objective ``Psi`` (256 replicas). Quantities:

* ``psi_over_psi_ref``: mean over replicas of ``Psi / Psi_ref``, then mean over instances (Fig. 3e, Fig. S1c);
* ``p_reach_098``: fraction of replicas with ``Psi >= 0.98 Psi_ref``, then mean over instances (Fig. S1b).

95% confidence intervals: percentile bootstrap over instances (4,000 resamples, seed 1).

Outputs: ``fig3/fig3e.csv`` (sigma = 0.05, L = 10, 15, 20, 30, complete and path libraries, budgets up to the largest
budget run with the complete library), ``figS1/figS1b.csv`` and ``figS1/figS1c.csv`` (sigma = 0.10, complete library).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import psi_ref_table  # noqa: E402
from extract_shards import load_bundle  # noqa: E402


def bootstrap_mean(x, B=4000, seed=1):
    """Mean and 95% percentile bootstrap interval over instances."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (B, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_instance(bundle, tag, ref):
    """(L, library, budget) -> {instance: (mean Psi/Psi_ref, P(Psi >= 0.98 Psi_ref))}, in record order."""
    recs, get = load_bundle(bundle)
    g = {}
    for r in recs:
        if "budget_log2" not in r or "bestF_max" not in r or "mode" in r:
            continue
        if r.get("role") != "test" or r.get("family") != "synthetic" or tag not in r["instance"]:
            continue
        if r.get("lib") not in ("complete", "path"):
            continue
        F = get(r["_shard"], f"{r['instance']}_b{r['budget_log2']}_bestF")
        F = np.where(np.isfinite(F), F, 0.0)
        R = ref[r["instance"]]
        p = float((F >= 0.98 * R - 1e-9 * abs(R)).mean())
        g.setdefault((r["n"], r["lib"], r["budget_log2"]), {})[r["instance"]] = (F.mean() / R, p)
    return g


def table(g, col):
    rows = []
    for (n, lib, b), d in sorted(g.items()):
        v = [x[col] for x in d.values()]
        m, lo, hi = bootstrap_mean(v)
        rows.append(dict(L=n, library=lib, iterations_log2=b, iterations=2 ** b, instances=len(d), value=m, ci95_lo=lo,
                         ci95_hi=hi))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sigma005", required=True, help="bundle of the sigma = 0.05 runs")
    ap.add_argument("--sigma010", required=True, help="bundle of the sigma = 0.10 runs")
    ap.add_argument("--psi-ref", default=None, help="Psi_ref table (default configs/psi_ref.csv)")
    ap.add_argument("--out", required=True, help="source_data folder")
    a = ap.parse_args(argv)
    ref = psi_ref_table(a.psi_ref)
    out = Path(a.out)
    (out / "fig3").mkdir(parents=True, exist_ok=True)
    (out / "figS1").mkdir(parents=True, exist_ok=True)

    s05 = table(per_instance(a.sigma005, "_s0.05_", ref), 0)
    keep = []
    for n in (10, 15, 20, 30):
        bmax = s05[(s05.L == n) & (s05.library == "complete")].iterations_log2.max()
        keep.append(s05[(s05.L == n) & (s05.iterations_log2 <= bmax)])
    e = pd.concat(keep).rename(columns={"value": "psi_over_psi_ref_mean"})
    e.to_csv(out / "fig3" / "fig3e.csv", index=False, float_format="%.6g")

    g10 = per_instance(a.sigma010, "_s0.1_", ref)
    g10 = {k: v for k, v in g10.items() if k[1] == "complete"}
    b = table(g10, 1).rename(columns={"value": "p_reach_098_mean"})
    b.to_csv(out / "figS1" / "figS1b.csv", index=False, float_format="%.6g")
    c = table(g10, 0).rename(columns={"value": "psi_over_psi_ref_mean"})
    c.to_csv(out / "figS1" / "figS1c.csv", index=False, float_format="%.6g")
    print("wrote", len(e), len(b), len(c), "rows")


if __name__ == "__main__":
    main()

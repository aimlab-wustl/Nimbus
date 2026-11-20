"""Fig. 4g and Supplementary Table S3: batched versus static move libraries at L = 20.

  python analysis/fig4g_tableS3.py --batching RUNS/batching --sigma005 BUNDLES/sigma005.npz \
      --libraries BUNDLES/libraries.npz --control RUNS/library_control --out ../../source_data

Inputs:
* ``RUNS/batching``: shards of ``configs/batching/tasks_batching.csv`` (batch of K informed shortcuts replaced every D
  iterations; K = 20, 40, 80; D = 2^12, 2^14, 2^16; budgets 2^18 and 2^20);
* ``libraries.npz``: the static library (backbone + the first batch of K informed shortcuts, backbone share 0.5);
* ``sigma005.npz``: the complete library with uniform proposals;
* ``RUNS/library_control/ctrl``: the complete library with backbone share 0.5, run with the calibration of batch 0 of
  each K (``library_control.py``, calibrations ``k20``, ``k40``, ``k80``).

Per instance: ``P_s`` = fraction of the 256 replicas with ``Psi >= 0.98 Psi_ref`` (20 test instances). Means over
instances with 95% percentile bootstrap intervals over instances (5,000 resamples); paired differences (batched minus
static) resample the instances jointly. The bootstrap draws come from one generator (seed 20261011) per table, in the
order of the rows, so the intervals are reproducible.

Outputs: ``fig4/fig4g.csv`` (heat-map cells) and ``tables/tableS3.csv``.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import psi_ref_table, read_shard  # noqa: E402
from extract_shards import load_bundle  # noqa: E402

BACKBONE, COMPLETE = 19, 190


class Boot:
    """Percentile bootstrap of the mean over instances, drawing from one seeded generator in call order."""

    def __init__(self, seed=20261011, B=5000):
        self.rng = np.random.default_rng(seed)
        self.B = B

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return (float("nan"),) * 3
        if x.size == 1 or np.all(x == x[0]):
            return float(x.mean()), float(x.mean()), float(x.mean())
        m = x[self.rng.integers(0, x.size, (self.B, x.size))].mean(1)
        return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_static(bundles):
    """(library label, budget, instance) -> (bestF, wall seconds) of the L = 20 test runs of the given bundles."""
    static = {}
    for path in bundles:
        R, g = load_bundle(path)
        for r in R:
            if r.get("n") == 20 and r.get("family") == "synthetic" and r.get("role") == "test" and "budget_log2" in r \
                    and "bestF_max" in r and (r.get("lib") == "complete" or str(r.get("lib", "")).startswith("path+informed")):
                v = g(r["_shard"], f"{r['instance']}_b{r['budget_log2']}_bestF")
                if v is not None:
                    static[(r["lib"], r["budget_log2"], r["instance"])] = (v, r.get("wall_s"))
    return static


def load_batching(folder):
    recs, arr = [], {}
    for f in sorted(glob.glob(str(Path(folder) / "shard_*.npz"))):
        rs, ar = read_shard(f)
        recs += rs
        arr.update(ar)
    return recs, arr


def load_control(folder):
    ctrl, recs = {}, []
    for f in sorted(glob.glob(str(Path(folder) / "ctrl" / "ctrl_*.npz"))):
        z = np.load(f, allow_pickle=False)
        for r in json.loads(str(z["records"])):
            ctrl[(r["cal"], r["budget_log2"], r["instance"])] = z[f"b{r['budget_log2']}_bestF"]
            recs.append(r)
    return ctrl, recs


def batching_table(recs, arr, static, ref):
    """Rows per (K, budget): static, each D, complete (uniform proposals)."""
    boot = Boot()
    p_s = lambda F, i: float((F >= 0.98 * ref[i]).mean())
    rows = []
    for k in sorted({r["k"] for r in recs}):
        for lb in sorted({r["budget_log2"] for r in recs}):
            insts = sorted({r["instance"] for r in recs if r["k"] == k and r["budget_log2"] == lb})
            st = np.array([p_s(static[(f"path+informed{k}@share0.5", lb, i)][0], i) for i in insts])
            cp = np.array([p_s(static[("complete", lb, i)][0], i) for i in insts])
            m, lo, hi = boot(st)
            rows.append(dict(K=k, iterations_log2=lb, condition="static", dwell_log2="", instances=len(insts),
                             p_reach_098_mean=m, ci95_lo=lo, ci95_hi=hi))
            for ld in sorted({r["dwell_log2"] for r in recs if r["k"] == k and r["budget_log2"] == lb}):
                rr = {r["instance"]: r for r in recs if r["k"] == k and r["budget_log2"] == lb and r["dwell_log2"] == ld}
                x = np.array([p_s(arr[f"{i}_k{k}_b{lb}_D{ld}_bestF"], i) for i in insts])
                m, lo, hi = boot(x)
                ds, dslo, dshi = boot(x - st)
                dc, dclo, dchi = boot(x - cp)
                r0 = rr[insts[0]]
                rows.append(dict(K=k, iterations_log2=lb, condition="batched", dwell_log2=ld, instances=len(insts),
                                 p_reach_098_mean=m, ci95_lo=lo, ci95_hi=hi, static_p_reach_098_mean=float(st.mean()),
                                 diff_vs_static=ds, diff_vs_static_ci95_lo=dslo, diff_vs_static_ci95_hi=dshi,
                                 wilcoxon_p_vs_static=float(stats.wilcoxon(x, st).pvalue) if np.any(x != st) else 1.0,
                                 diff_vs_complete=dc, diff_vs_complete_ci95_lo=dclo, diff_vs_complete_ci95_hi=dchi,
                                 batches=r0["n_batches"], windows=r0["windows"]))
            m, lo, hi = boot(cp)
            rows.append(dict(K=k, iterations_log2=lb, condition="complete_uniform", dwell_log2="", instances=len(insts),
                             p_reach_098_mean=m, ci95_lo=lo, ci95_hi=hi))
    return rows


def control_table(ctrl, ctrl_recs, static, dw, ref):
    """Rows per (budget, calibration) of the complete library with backbone share 0.5. The comparisons made here (with
    the complete library, the static library and the batched runs) fix the order of the bootstrap draws."""
    boot = Boot()
    p_s = lambda F, i: float((F >= 0.98 * ref[i]).mean())
    rows = []
    for lb in (18, 20):
        for cal in ("own", "k20", "k40", "k80"):
            insts = sorted({k[2] for k in ctrl if k[0] == cal and k[1] == lb})
            if not insts:
                continue
            x = np.array([p_s(ctrl[(cal, lb, i)], i) for i in insts])
            m, lo, hi = boot(x)
            row = dict(iterations_log2=lb, calibration=cal, instances=len(insts), p_reach_098_mean=m, ci95_lo=lo,
                       ci95_hi=hi)
            cp = np.array([p_s(static[("complete", lb, i)][0], i) for i in insts if ("complete", lb, i) in static])
            if len(cp) == len(insts):
                boot(x - cp)
            if cal != "own":
                k = int(cal[1:])
                st = [static.get((f"path+informed{k}@share0.5", lb, i)) for i in insts]
                if all(v is not None for v in st):
                    boot(x - np.array([p_s(v[0], i) for v, i in zip(st, insts)]))
                for ld in (12, 14, 16):
                    if all((k, ld, lb, i) in dw for i in insts):
                        y = np.array([p_s(dw[(k, ld, lb, i)], i) for i in insts])
                        boot(y - x)
            rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fig. 4g and Table S3 source data.")
    ap.add_argument("--batching", required=True, help="shard folder of the batched runs")
    ap.add_argument("--sigma005", required=True)
    ap.add_argument("--libraries", required=True)
    ap.add_argument("--control", required=True, help="output folder of library_control.py (holds ctrl/)")
    ap.add_argument("--psi-ref", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ref = psi_ref_table(a.psi_ref)
    out = Path(a.out)
    recs, arr = load_batching(a.batching)
    static = load_static([a.sigma005, a.libraries])
    rows = batching_table(recs, arr, static, ref)
    dw = {(r["k"], r["dwell_log2"], r["budget_log2"], r["instance"]):
          arr[f"{r['instance']}_k{r['k']}_b{r['budget_log2']}_D{r['dwell_log2']}_bestF"] for r in recs}
    ctrl, crecs = load_control(a.control)
    crow = control_table(ctrl, crecs, static, dw, ref)

    cells = []
    for r in rows:
        name = {"static": "static", "complete_uniform": "complete, uniform proposals"}.get(r["condition"])
        if name is None:
            name = f"batched, D = 2^{r['dwell_log2']}"
        cells.append(dict(iterations_log2=r["iterations_log2"], K=r["K"], condition=name,
                          loaded_exchange_fraction=(BACKBONE + r["K"]) / COMPLETE, instances=r["instances"],
                          p_reach_098_mean=r["p_reach_098_mean"], ci95_lo=r["ci95_lo"], ci95_hi=r["ci95_hi"]))
    for r in crow:
        if r["calibration"] == "own":
            continue
        k = int(r["calibration"][1:])
        cells.append(dict(iterations_log2=r["iterations_log2"], K=k, condition="complete, backbone share 0.5",
                          loaded_exchange_fraction=(BACKBONE + k) / COMPLETE, instances=r["instances"],
                          p_reach_098_mean=r["p_reach_098_mean"], ci95_lo=r["ci95_lo"], ci95_hi=r["ci95_hi"]))
    order = ["static", "batched, D = 2^12", "batched, D = 2^14", "batched, D = 2^16", "complete, backbone share 0.5",
             "complete, uniform proposals"]
    g = pd.DataFrame(cells)
    g["_o"] = g.condition.map(order.index)
    g = g.sort_values(["iterations_log2", "K", "_o"]).drop(columns="_o")
    (out / "fig4").mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "fig4" / "fig4g.csv", index=False, float_format="%.6g")

    t = pd.DataFrame([dict(K=r["K"], iterations_log2=r["iterations_log2"], dwell_log2=r["dwell_log2"],
                           instances=r["instances"], static_p_reach_098=r["static_p_reach_098_mean"],
                           batched_p_reach_098=r["p_reach_098_mean"], difference=r["diff_vs_static"],
                           difference_ci95_lo=r["diff_vs_static_ci95_lo"], difference_ci95_hi=r["diff_vs_static_ci95_hi"],
                           wilcoxon_p=r["wilcoxon_p_vs_static"])
                      for r in rows if r["condition"] == "batched"])
    t = t.sort_values(["K", "iterations_log2", "dwell_log2"])
    (out / "tables").mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "tables" / "tableS3.csv", index=False, float_format="%.6g")
    print("fig4g", len(g), "cells; tableS3", len(t), "rows")


if __name__ == "__main__":
    main()

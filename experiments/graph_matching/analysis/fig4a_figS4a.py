"""Fig. 4a (L = 20, 2^20 iterations) and Supplementary Fig. S4a (L = 30, 2^22 iterations): probability of reaching
0.98 Psi_ref versus the number of exchanges in the move library.

  python analysis/fig4a_figS4a.py --sigma005 BUNDLES/sigma005.npz --libraries BUNDLES/libraries.npz \
      --control RUNS/library_control --out ../../source_data

Inputs:
* ``libraries.npz``: the backbone plus K informed or random shortcuts with backbone share 0.5
  (``configs/auto_schedule/tasks_libraries.csv``);
* ``sigma005.npz``: the path library (backbone alone) and the complete library with uniform proposals
  (``configs/auto_schedule/tasks_sigma005.csv``), on the same 20 test instances;
* ``RUNS/library_control/ctrl`` and ``ctrl_n30``: the complete library with backbone share 0.5 (``library_control.py``,
  calibration ``own``).

Per instance: the fraction of the 256 replicas with ``Psi >= 0.98 Psi_ref``; then the mean over the 20 instances with a
95% percentile bootstrap interval over instances (4,000 resamples, seed 1).
Outputs: ``fig4/fig4a.csv`` and ``figS4/figS4a.csv`` (columns: library, K, exchanges, iterations_log2, instances,
p_reach_098_mean, ci95_lo, ci95_hi).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import psi_ref_table  # noqa: E402
from extract_shards import load_bundle  # noqa: E402

FRONT = {20: 20, 30: 22}      # L -> budget (log2 iterations) of the panel


def bootstrap_mean(x, B=4000, seed=1):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (B, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def p098(F, R, tol=True):
    F = np.where(np.isfinite(F), F, 0.0)
    return float((F >= 0.98 * R - (1e-9 * abs(R) if tol else 0.0)).mean())


def rows_for(n, lib_b, s05_b, ctrl_dir, ref):
    B = FRONT[n]
    per = {}
    recs, get = load_bundle(lib_b)
    insts = set()
    for r in recs:
        sp = r.get("lib_spec") or {}
        if r.get("n") != n or r.get("budget_log2") != B or sp.get("share") != 0.5 or "bestF_max" not in r:
            continue
        F = get(r["_shard"], f"{r['instance']}_b{B}_bestF")
        per.setdefault((sp["extra"], int(sp["k"])), {})[r["instance"]] = p098(F, ref[r["instance"]])
        insts.add(r["instance"])
    recs, get = load_bundle(s05_b)
    for r in recs:
        if r.get("n") != n or r.get("budget_log2") != B or r.get("lib") not in ("path", "complete") or "mode" in r:
            continue
        if r["instance"] not in insts or "bestF_max" not in r:
            continue
        F = get(r["_shard"], f"{r['instance']}_b{B}_bestF")
        per.setdefault((r["lib"], 0), {})[r["instance"]] = p098(F, ref[r["instance"]])
    sub = "ctrl" if n == 20 else f"ctrl_n{n}"
    files = sorted(f for d in ctrl_dir for f in glob.glob(str(Path(d) / sub / "ctrl_*_own.npz")))
    for f in files:
        z = np.load(f, allow_pickle=False)
        for r in json.loads(str(z["records"])):
            if r["budget_log2"] == B:
                per.setdefault(("share", 0), {})[r["instance"]] = p098(z[f"b{B}_bestF"], ref[r["instance"]], tol=False)
    out = []
    order = {"informed": 0, "random": 1, "path": 2, "complete": 3, "share": 4}
    for (lib, k), d in sorted(per.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])):
        v = [d[i] for i in sorted(d)]
        m, lo, hi = bootstrap_mean(v)
        exch = (n - 1) + k if lib in ("informed", "random") else (n - 1 if lib == "path" else n * (n - 1) // 2)
        out.append(dict(L=n, library=lib, K=k if lib in ("informed", "random") else "", exchanges=exch,
                        iterations_log2=B, instances=len(d), p_reach_098_mean=m, ci95_lo=lo, ci95_hi=hi))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fig. 4a and Fig. S4a source data.")
    ap.add_argument("--sigma005", required=True)
    ap.add_argument("--libraries", required=True)
    ap.add_argument("--control", required=True, nargs="+",
                    help="output folder(s) of library_control.py (holding ctrl/ and ctrl_n30/)")
    ap.add_argument("--psi-ref", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ref = psi_ref_table(a.psi_ref)
    out = Path(a.out)
    for n, sub, name in ((20, "fig4", "fig4a"), (30, "figS4", "figS4a")):
        (out / sub).mkdir(parents=True, exist_ok=True)
        df = rows_for(n, a.libraries, a.sigma005, a.control, ref)
        df.to_csv(out / sub / f"{name}.csv", index=False, float_format="%.6g")
        print(name, len(df), "rows")


if __name__ == "__main__":
    main()

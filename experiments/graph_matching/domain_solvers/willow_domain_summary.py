"""Summarize the domain solvers on the 40 WILLOW test pairs (L = 10, certified optima; run_refs.m with 500 restarts).

  python willow_domain_summary.py INST refs_willow.mat OUTDIR

Writes ``willow_domain_per_instance.csv`` and ``willow_domain_solvers.csv``. Per solver: mean over pairs (95% bootstrap
interval over pairs, 4000 resamples, one generator with seed 0 used in a fixed order) of: the best restart reaches the
optimum (``P_best_restart_reaches_opt``), the fraction of restarts at the optimum (``p_restart_opt_mean``, used in
Fig. 3c), the fraction of restarts with ``Psi >= 0.98 Psi*`` and the median restart quality ``Psi / Psi*``; and the
median and mean time per restart (ms; MATLAB, one thread).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_refs import load_refs  # noqa: E402


def main(argv=None):
    inst, mat, out = (argv or sys.argv[1:])[:3]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per = []
    for r in load_refs(mat, inst):
        F = np.atleast_1d(r["F"])
        t = 1e3 * np.atleast_1d(r["times"])
        fo = r["meta"]["F_opt"]
        q = F / fo
        per.append(dict(instance=r["instance"], solver=r["solver"], F_opt=fo, restarts=F.size, best_F=F.max(),
                        best_over_opt=q.max(), best_hits_opt=bool(q.max() >= 1 - 1e-9),
                        p_restart_opt=float(np.mean(q >= 1 - 1e-9)), p_restart_098=float(np.mean(q >= 0.98)),
                        median_q=float(np.median(q)), median_ms=float(np.median(t)), mean_ms=float(t.mean())))
    pi = pd.DataFrame(per)
    pi.to_csv(out / "willow_domain_per_instance.csv", index=False, float_format="%.6g")
    rng = np.random.default_rng(0)

    def boot(x):
        b = [rng.choice(x, len(x)).mean() for _ in range(4000)]
        return np.percentile(b, [2.5, 97.5])

    agg = []
    for s, g in pi.groupby("solver"):
        d = dict(solver=s, instances=len(g), restarts_per_instance=int(g.restarts.iloc[0]))
        for k, col in (("P_best_restart_reaches_opt", "best_hits_opt"), ("p_restart_opt_mean", "p_restart_opt"),
                       ("p_restart_098_mean", "p_restart_098"), ("median_restart_quality_mean", "median_q")):
            x = g[col].astype(float).values
            lo, hi = boot(x)
            d[k] = x.mean()
            d[k + "_lo"] = lo
            d[k + "_hi"] = hi
        d["median_restart_quality_median_over_pairs"] = g.median_q.median()
        d["best_over_opt_min"] = g.best_over_opt.min()
        d["median_ms_per_restart"] = g.median_ms.median()
        d["mean_ms_per_restart"] = g.mean_ms.mean()
        agg.append(d)
    ag = pd.DataFrame(agg)
    ag.to_csv(out / "willow_domain_solvers.csv", index=False, float_format="%.5g")
    print(ag.T.to_string())


if __name__ == "__main__":
    main()

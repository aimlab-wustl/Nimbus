"""Fig. 4d (iterations to solution at L = 10), Fig. 4e (rank correlation at fixed K) and Supplementary Table S4.

    python analysis/fig4de_tableS4.py --auto RUNS/library_its/auto --tuned RUNS/library_its/tuned \
        --gap RUNS/landscape/gap.json --depth RUNS/landscape/depth --out-fig source_data/fig4 --out-table source_data/tables

Inputs: test runs of ``library_its.py`` with the automatically calibrated schedule (``--auto``) and with the settings
selected on the training instances (``--tuned``), and the landscape outputs of ``landscape.py``.

Iterations to solution of an instance at budget t: ``ITS_i(t) = t ln(0.01) / ln(1 - p_i(t))`` with p_i(t) the fraction
of the 256 replicas that reach the certified optimum (random shortcuts: median over the ten draws of each instance).
ITS of a library = minimum over budgets of the median over the 20 test instances. 95% CI: percentile bootstrap over
instances (5,000 resamples, the same resampled instances at every budget, then the minimum).

Fig. 4e: for each instance and K, the 11 libraries with K shortcuts (one informed, ten random) form a cell; the
Spearman correlation between log10 ITS_i at a common budget (the budget that minimizes the informed library's median)
and each landscape property is computed per cell. Bars: median over the 80 cells; 95% CI: bootstrap over cells
(2,000 resamples, continuing the random stream of Fig. 4b, 4c).

Table S4 ratios: ratio of the ITS of two libraries; 95% CI: bootstrap that resamples the same instances for both
(5,000 resamples, ``numpy.random.default_rng(20261023)``, automatic schedule first).
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fig4bc  # noqa: E402
from common import tts_at  # noqa: E402
from library_its import CLASSES, KS, load, matrix, optimal  # noqa: E402

QUANT = ["difficulty", "d_rel", "n_minima", "t_rel"]
QNAME = {"difficulty": "catoni_difficulty", "d_rel": "critical_depth", "n_minima": "local_minima",
         "t_rel": "relaxation_time_infinite_temperature"}
LIB = {"path": ("path", 0), "complete": ("complete", 36)}


def lib_of(cls):
    if cls in LIB:
        return LIB[cls]
    return ("informed" if cls.startswith("inf") else "random"), int(cls[3:])


def class_its(rows):
    by_c = defaultdict(list)
    for r in rows:
        by_c[r["cls"]].append(r)
    tab, M = [], {}
    for cls in CLASSES:
        if cls not in by_c:
            continue
        insts, buds, T = matrix(by_c[cls])
        M[cls] = (insts, buds, T)
        o = optimal(T, buds, boot=5000)
        col = T[:, o["j"]]
        r0 = by_c[cls][0]
        tab.append(dict(cls=cls, m=r0["m"], kappa=r0["kappa"], its=o["its"], lo=o["lo"], hi=o["hi"],
                        budget_log2=o["budget_log2"], solved=int(np.isfinite(col).sum()), instances=len(insts)))
    return tab, M


def fixed_k_cells(rows, M, depth):
    from scipy.stats import spearmanr

    df = pd.DataFrame(rows)
    df["T"] = [tts_at(2.0 ** b, p) for b, p in zip(df.budget_log2, df.P_s)]
    cells = []
    for k in KS:
        fi = M[f"inf{k}"]
        b = optimal(fi[2], fi[1])["budget_log2"]
        sub = df[df.cls.isin([f"inf{k}", f"rnd{k}"])]
        X = sub[sub.budget_log2 == b][["instance", "lib", "k", "seed", "T"]]
        J = X.merge(depth[["instance", "lib", "k", "seed", "n_minima", "d_rel", "difficulty", "t_rel"]],
                    on=["instance", "lib", "k", "seed"])
        J["logT"] = np.log10(J["T"].replace(math.inf, 1e15))
        for inst, c in J.groupby("instance"):
            for q in ("difficulty", "d_rel", "n_minima", "t_rel"):
                if c[q].nunique() > 1 and c.logT.nunique() > 1:
                    cells.append(dict(k=k, instance=inst, quantity=q, rho=float(spearmanr(c.logT, c[q]).statistic),
                                      budget_log2=b))
    return pd.DataFrame(cells)


def its_matrices(rows):
    df = pd.DataFrame(rows)
    df["T"] = [tts_at(2.0 ** b, p) for b, p in zip(df.budget_log2, df.P_s)]
    return {c: g.groupby(["instance", "budget_log2"]).T.median().unstack().sort_index() for c, g in df.groupby("cls")}


def ratios(rows, tag, rng):
    M = its_matrices(rows)
    n = len(M["path"])

    def its(c, ix):
        return np.median(M[c].values[ix], axis=0).min()
    out = []
    for a, b in [("path", "complete")] + [(f"inf{k}", f"rnd{k}") for k in KS]:
        r = its(a, np.arange(n)) / its(b, np.arange(n))
        bs = [its(a, ix) / its(b, ix) for ix in rng.integers(0, n, (5000, n))]
        out.append(dict(schedule=tag, num=a, den=b, ratio=r, lo=float(np.percentile(bs, 2.5)),
                        hi=float(np.percentile(bs, 97.5))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--auto", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--gap", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--out-fig", required=True)
    ap.add_argument("--out-table", required=True)
    a = ap.parse_args(argv)
    auto = [r for r in load([a.auto]) if r["role"] == "test"]
    tuned = [r for r in load([a.tuned]) if r["role"] == "test"]
    depth = fig4bc.load_depth(a.depth)
    _, _, _, rng = fig4bc.bootstrap_stream(fig4bc.load_gap(a.gap), depth)
    tab_a, M_a = class_its(auto)
    tab_t, _ = class_its(tuned)
    of, ot = Path(a.out_fig), Path(a.out_table)
    of.mkdir(parents=True, exist_ok=True)
    ot.mkdir(parents=True, exist_ok=True)

    # Fig. 4d
    d = []
    for x in tab_a:
        lib, k = lib_of(x["cls"])
        d.append(dict(L=10, library=lib, K=k, exchanges=9 + k, m=x["m"], kappa=x["kappa"], its_iterations=x["its"],
                      its_ci95_lo=x["lo"], its_ci95_hi=x["hi"], budget_log2_at_minimum=x["budget_log2"],
                      instances_solved_at_minimum=x["solved"], instances=x["instances"]))
    pd.DataFrame(d).to_csv(of / "fig4d.csv", index=False)

    # Fig. 4e
    cells = fixed_k_cells(auto, M_a, depth)
    e = []
    for q in QUANT:
        r = cells[cells.quantity == q].rho.values
        mm, lo, hi = fig4bc.boot_median(r, rng)
        e.append(dict(property=QNAME[q], spearman_rho_median=mm, ci95_lo=lo, ci95_hi=hi,
                      fraction_positive=float(np.mean(r > 0)), cells=len(r)))
    pd.DataFrame(e).to_csv(of / "fig4e.csv", index=False)

    # Table S4
    rrng = np.random.default_rng(20261023)
    R = ratios(auto, "automatic", rrng) + ratios(tuned, "optimized", rrng)
    rows = []
    for tag, tab in (("automatic", tab_a), ("optimized", tab_t)):
        for x in tab:
            lib, k = lib_of(x["cls"])
            rows.append(dict(section="library", item=f"{lib}" + (f", K={k}" if lib in ("informed", "random") else ""),
                             schedule=tag, m=x["m"], kappa=x["kappa"], unit="iterations", value=x["its"], ci95_lo=x["lo"],
                             ci95_hi=x["hi"], budget_log2_at_minimum=x["budget_log2"]))
    for x in R:
        item = "path / complete" if x["num"] == "path" else f"informed / random, K={x['num'][3:]}"
        rows.append(dict(section="ratio", item=item, schedule=x["schedule"], m="", kappa="", unit="ratio", value=x["ratio"],
                         ci95_lo=x["lo"], ci95_hi=x["hi"], budget_log2_at_minimum=""))
    pd.DataFrame(rows).to_csv(ot / "tableS4.csv", index=False)
    print("wrote", of / "fig4d.csv", of / "fig4e.csv", ot / "tableS4.csv")


if __name__ == "__main__":
    main()

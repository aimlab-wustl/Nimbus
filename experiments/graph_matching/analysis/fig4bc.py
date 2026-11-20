"""Fig. 4b (relaxation time at infinite temperature) and Fig. 4c (critical depth and Catoni difficulty at L = 10).

    python analysis/fig4bc.py --gap RUNS/landscape/gap.json --depth RUNS/landscape/depth --out source_data/fig4

Inputs: the outputs of ``landscape.py gap`` and ``landscape.py depth`` (one JSON per test instance).
Points: median over the 20 test instances; for random shortcuts the median over the draws of each instance is taken
first. 95% CI: percentile bootstrap of the median over instances (2,000 resamples). All bootstrap draws of Fig. 4b, 4c
and 4e come from one random stream (``numpy.random.default_rng(0)``) used in that order (see ``bootstrap_stream``).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
NBOOT = 2000


def load_gap(path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(Path(path).read_text()))


def load_depth(folder) -> pd.DataFrame:
    D = pd.DataFrame([r for f in sorted(glob.glob(str(Path(folder) / "*.json"))) for r in json.loads(Path(f).read_text())])
    D["d_rel"] = D.d_star / D.F_opt
    D.loc[D.lib == "complete", "seed"] = 0
    return D


def boot_median(v, rng, n=NBOOT):
    v = np.asarray(v, float)
    m = np.median(v)
    bs = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def per_instance(df, val):
    """Random shortcuts: median over the draws of each instance first."""
    return df.groupby(["instance", "lib", "k"])[val].median().reset_index()


def summary(df, val, rng, extra=()):
    rows = []
    for key, g in df.groupby(["lib", "k", *extra]):
        m, lo, hi = boot_median(g[val], rng)
        rows.append(dict(zip(["lib", "k", *extra], key if isinstance(key, tuple) else (key,)), median=m, lo=lo, hi=hi,
                         instances=len(g)))
    return pd.DataFrame(rows)


def exchanges(lib, k, L):
    return L * (L - 1) // 2 if lib == "complete" else (L - 1) + int(k)


def bootstrap_stream(gap: pd.DataFrame, depth: pd.DataFrame):
    """Summaries of Fig. 4b and 4c; returns ``(b, c_depth, c_difficulty, rng)`` with the stream advanced past them."""
    rng = np.random.default_rng(SEED)
    g = gap.copy()
    g.loc[g.lib == "complete", "k"] = 0
    b = summary(g, "t_rel", rng, extra=("n",))
    d = depth.copy()
    d.loc[d.lib == "complete", "k"] = 0
    c1 = summary(per_instance(d, "d_rel"), "d_rel", rng)
    c2 = summary(per_instance(d, "difficulty"), "difficulty", rng)
    return b, c1, c2, rng


def tidy(s, L_col=None, L=10):
    rows = []
    for _, r in s.iterrows():
        Lr = int(r[L_col]) if L_col else L
        k = int(r.k) if r.lib != "complete" else Lr * (Lr - 1) // 2 - (Lr - 1)
        rows.append(dict(L=Lr, library=r.lib, K=k, exchanges=exchanges(r.lib, r.k, Lr), median=r["median"],
                         ci95_lo=r.lo, ci95_hi=r.hi, instances=int(r.instances)))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gap", required=True, help="gap.json of landscape.py gap")
    ap.add_argument("--depth", required=True, help="folder with the JSON files of landscape.py depth")
    ap.add_argument("--out", required=True, help="output folder (source_data/fig4)")
    a = ap.parse_args(argv)
    b, c1, c2, _ = bootstrap_stream(load_gap(a.gap), load_depth(a.depth))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tb = tidy(b, L_col="n").rename(columns={"median": "relaxation_time_median_iterations",
                                            "ci95_lo": "relaxation_time_ci95_lo_iterations",
                                            "ci95_hi": "relaxation_time_ci95_hi_iterations"})
    tb.sort_values(["L", "library", "K"]).to_csv(out / "fig4b.csv", index=False)
    tc = pd.concat([tidy(c1).assign(quantity="critical_depth_over_psi_opt"),
                    tidy(c2).assign(quantity="catoni_difficulty")])
    tc = tc[["quantity", "L", "library", "K", "exchanges", "median", "ci95_lo", "ci95_hi", "instances"]]
    tc.to_csv(out / "fig4c.csv", index=False)
    print("wrote", out / "fig4b.csv", out / "fig4c.csv")


if __name__ == "__main__":
    main()

"""Fig. 6a, 6b and Fig. S9a-d: per-window operation ratios, acceptance and sharing ratio from the event runs.

  python analysis/event_windows.py --runs RUNS/event --out ../../source_data

Reads every ``<task>.npz`` written by ``run_event.py`` and writes
  fig6/fig6a.csv           cumulative ratio event-driven / recomputation per window, four workloads (pooled line and
                           interquartile range of the per-replica ratio)
  fig6/fig6b.csv           acceptance a and sharing ratio rho per window, four workloads, with anneal progress
  fig6/fig6_summary.csv    per workload: break-even point, end-of-run ratios, rho range, acceptance floor, checks
  figS9/figS9a.csv         per-window ratio event-driven / recomputation, four workloads, 95% bootstrap interval
  figS9/figS9b.csv         the same for the synthetic family
  figS9/figS9c.csv         cumulative ratio, synthetic family (as fig6a)
  figS9/figS9d.csv         a and rho per window, synthetic family (as fig6b)
  figS9/figS9_summary.csv  per setting of the synthetic family (as fig6_summary)

Definitions (per window w; pooled = summed over all replicas of all instances of a workload):
  a      = accepted state-changing moves / iterations          (self-loops excluded: they cause no update)
  f_in   = connection reads of the proposed moves / iterations (weighted connections of the proposed move;
           recomputation)
  f_out  = input updates / accepted state-changing moves        (event-driven updating)
  rho    = f_out / f_in
  ratio  = input updates / connection reads = a * rho           (event-driven / recomputation, per iteration)
  cumulative ratio = sum of input updates up to the window end / sum of connection reads up to the window end
Break-even point: the window end after which the pooled cumulative ratio stays below 1 (0 if it is below 1 in every
window). Acceptance floor: accepted moves with zero energy change per iteration, pooled over the last octave (last
eight windows). Recompute-all: every move's input recomputed at every iteration, W = sum over moves of their weighted
connections per iteration.

Bootstrap intervals of the per-window ratio (Fig. S9a, b): replicas resampled with replacement within each instance
(instances fixed), 2,000 resamples, one random stream ``default_rng(11)`` used for the workloads in the order of the
sorted file names. The interquartile ranges (Fig. 6a, Fig. S9c) are over replicas of the per-replica cumulative ratio.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

APPS = ["m30c", "m50p", "s11", "pair2d"]
SYN = ["syn_k2_rowswap_path_q16", "syn_k2_rowswap_path_q9", "syn_k2_flip", "syn_k3_flip", "syn_k4_flip",
       "syn_k2_onehot"]
LABEL = {"m30c": "matching, complete library, L = 30", "m50p": "matching, path library, L = 50",
         "s11": "surface code, d = 11, p_err = 0.12", "pair2d": "2D +-J spin glass, single flips",
         "syn_k2_rowswap_path_q16": "16 x 16 permutation, path row exchanges",
         "syn_k2_rowswap_path_q9": "9 x 9 permutation, path row exchanges",
         "syn_k2_flip": "single flips, order-2 terms", "syn_k3_flip": "single flips, order-3 terms",
         "syn_k4_flip": "single flips, order-4 terms", "syn_k2_onehot": "one-hot exchanges, order-2 terms"}
NBOOT, BOOT_SEED = 2000, 11


def load(folders):
    """workload -> list of runs (in sorted file-name order), and the workload order of first appearance."""
    files = sorted(f for d in folders for f in glob.glob(str(Path(d) / "*.npz")))
    data = defaultdict(list)
    for f in files:
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        data[m["task"].split("__")[0]].append(dict(meta=m, **{k: z[k] for k in z.files if k != "meta"}))
    return data


def analyse(wl, runs):
    m0 = runs[0]["meta"]
    F = {k: i for i, k in enumerate(m0["win_fields"])}
    edges = runs[0]["edges"]
    for r in runs:
        assert np.array_equal(r["edges"], edges), "all instances of a workload need the same windows"
    win = np.concatenate([r["win"] for r in runs], axis=0).astype(np.float64)   # [replicas, windows, fields]
    A, C = win[..., F["attempts"]], win[..., F["accepted_changed"]]
    N, Fin, Ev = win[..., F["neutral_accepted"]], win[..., F["fanin_weighted"]], win[..., F["input_corrections"]]
    W_all = float(np.mean([r["meta"]["routes_weighted"] for r in runs for _ in range(r["win"].shape[0])]))
    sA, sC, sN, sFin, sEv = (x.sum(axis=0) for x in (A, C, N, Fin, Ev))
    with np.errstate(all="ignore"):
        a = sC / sA
        f_in = sFin / sA
        f_out = sEv / sC
        rho = f_out / f_in
        ratio = (sEv / sA) / f_in
        ident = np.where(sC > 0, np.abs(ratio - a * rho) / np.maximum(ratio, 1e-300), np.nan)
        ratio_all = (sEv / sA) / W_all
        cratio = np.cumsum(sEv) / np.cumsum(sFin)
        r_cratio = np.cumsum(Ev, axis=1) / np.cumsum(Fin, axis=1)
        q25, q75 = np.nanpercentile(r_cratio, 25, axis=0), np.nanpercentile(r_cratio, 75, axis=0)
    lo_e, hi_e = edges[:-1], edges[1:]
    mid = np.where(lo_e > 0, np.sqrt(np.maximum(lo_e, 1) * hi_e), hi_e / 2.0)
    B = int(edges[-1])

    def crossing(rr):
        below = rr < 1
        if np.all(below):
            return 0
        if not below[-1]:
            return None
        return int(hi_e[np.where(~below)[0].max()])

    late = slice(len(sA) - 8, None)
    with np.errstate(all="ignore"):
        a_late = sC[late].sum() / sA[late].sum()
        rho_late = (sEv[late].sum() / max(sC[late].sum(), 1)) / (sFin[late].sum() / sA[late].sum())
        floor = sN[late].sum() / sA[late].sum()
    be = crossing(np.nan_to_num(cratio, nan=0.0))
    ok = np.isfinite(rho)
    summ = dict(workload=wl, label=LABEL.get(wl, wl), instances=len(runs), replicas=int(A.shape[0]),
                budget_log2=int(m0["budget_log2"]), moves_R=int(np.median([r["meta"]["R"] for r in runs])),
                break_even_iteration=be, break_even_fraction_of_run=None if be is None else be / B,
                cum_ratio_end=float(cratio[-1]), ratio_end=float(np.nan_to_num(ratio[-1])),
                ratio_vs_recompute_all_max=float(np.nanmax(ratio_all)), rho_min=float(np.nanmin(rho[ok])),
                rho_max=float(np.nanmax(rho[ok])), rho_late=float(rho_late), a_first_window=float(a[0]),
                a_late=float(a_late), acceptance_floor_late=float(floor),
                floor_below_one_over_rho=bool(floor < 1 / rho_late) if math.isfinite(rho_late) else None,
                identity_max_rel_dev=float(np.nanmax(ident)) if np.isfinite(ident).any() else float("nan"))
    windows = pd.DataFrame(dict(workload=wl, win_lo=lo_e.astype(np.int64), win_hi=hi_e.astype(np.int64), mid=mid, a=a,
                                rho=rho, f_in=f_in, f_out=f_out, ratio=ratio, cum_ratio=cratio, cum_ratio_q25=q25,
                                cum_ratio_q75=q75, ratio_vs_recompute_all=ratio_all))
    return summ, windows


def bootstrap(data, rng):
    """95% percentile interval of the pooled per-window ratio: replicas resampled within each instance."""
    out = {}
    for wl, runs in data.items():
        F = {k: i for i, k in enumerate(runs[0]["meta"]["win_fields"])}
        ev = [r["win"][..., F["input_corrections"]].astype(np.float64) for r in runs]
        fin = [r["win"][..., F["fanin_weighted"]].astype(np.float64) for r in runs]
        K = ev[0].shape[1]
        bev, bfin = np.zeros((NBOOT, K)), np.zeros((NBOOT, K))
        for e, f in zip(ev, fin):
            M = e.shape[0]
            cnt = rng.multinomial(M, np.full(M, 1.0 / M), size=NBOOT)
            bev += cnt @ e
            bfin += cnt @ f
        with np.errstate(all="ignore"):
            lo, hi = np.nanpercentile(bev / bfin, [2.5, 97.5], axis=0)
        out[wl] = (lo, hi)
    return out


def progress(w, budget_log2):
    """Anneal progress of each window: 0 at the first plotted window, 1 at the end (log iterations)."""
    l0 = np.log2(w.mid.values.min())
    return np.clip((np.log2(w.mid.values) - l0) / (budget_log2 - l0), 0, 1)


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
    print("wrote", path, len(df), "rows")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, help="folder(s) with the <task>.npz files of run_event.py")
    ap.add_argument("--out", required=True, help="source_data folder")
    a = ap.parse_args(argv)
    data = load(a.runs)
    ci = bootstrap(data, np.random.default_rng(BOOT_SEED))
    summ, win = {}, {}
    for wl in APPS + SYN:
        if wl in data:
            summ[wl], win[wl] = analyse(wl, data[wl])
            win[wl]["ratio_ci_low"], win[wl]["ratio_ci_high"] = ci[wl]
    out = Path(a.out)
    for group, fig in ((APPS, "fig6"), (SYN, "figS9")):
        ks = [k for k in group if k in win]
        if not ks:
            continue
        cum = pd.concat([win[k][["workload", "win_lo", "win_hi", "cum_ratio", "cum_ratio_q25", "cum_ratio_q75"]]
                         for k in ks])
        plane = []
        for k in ks:
            w = win[k][(win[k].a > 0) & np.isfinite(win[k].rho)]
            plane.append(w[["workload", "win_lo", "win_hi", "mid", "a", "rho"]].assign(
                progress=progress(w, summ[k]["budget_log2"])))
        per = pd.concat([win[k][["workload", "win_lo", "win_hi", "mid", "a", "rho", "ratio", "ratio_ci_low",
                                 "ratio_ci_high"]] for k in ks])
        sm = pd.DataFrame([summ[k] for k in ks])
        if fig == "fig6":
            write(cum, out / "fig6" / "fig6a.csv")
            write(pd.concat(plane), out / "fig6" / "fig6b.csv")
            write(sm, out / "fig6" / "fig6_summary.csv")
            write(per, out / "figS9" / "figS9a.csv")
        else:
            write(per, out / "figS9" / "figS9b.csv")
            write(cum, out / "figS9" / "figS9c.csv")
            write(pd.concat(plane), out / "figS9" / "figS9d.csv")
            write(sm, out / "figS9" / "figS9_summary.csv")


if __name__ == "__main__":
    main()

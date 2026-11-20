"""Setting selection on the training instances for the L = 10 method comparison (Fig. 3c, 3f; Supplementary Table S2).

  python select_settings.py --shards RUNS/willow_l10/select_* --penalty RUNS/willow_l10/btune/*.jsonl \
      --out configs/willow_sym/choice.json --baseline-out configs/willow_sym

Training instances: the 6 synthetic L = 10 training instances and the 20 WILLOW training pairs, 32 replicas, seed 11.
Target: the certified optimum. For each problem set, method and setting, with ``p_i(t)`` the fraction of replicas of
instance ``i`` at the optimum after ``t`` iterations:

    ITS_i(t) = t ln(0.01) / ln(1 - p_i(t))       (p = 1 -> t, p = 0 -> infinity)
    ITS      = min over budgets t of the median over instances of ITS_i(t)

The selected setting has the smallest ITS (the minimum is joint over settings and budgets; only the setting is kept).
Ties: smaller mean of log2 ITS_i at the best budget (unsolved instances count as 2^64), then the setting closest (sum of
|log2| ratios) to the reference setting (``m = 1, kappa = 2`` for this work). Only settings run on every instance at every
budget of the method's grid are ranked. Budget grids (log2 iterations): complete library 6-20, path library 8-24, penalty
machines 14-24. ``edges`` lists grid boundaries at which the selection lies (the grids were widened until it was empty).

Writes the choice JSON (all four methods per problem set) and, with ``--baseline-out``, ``baseline_choice_<set>.json``
in the format of ``penalty_baselines.py test --choice``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))
import common as C  # noqa: E402
import l10_runs as LR  # noqa: E402

BUD = {"complete": set(range(6, 21, 2)), "path": set(range(8, 25, 2)), "nhoim": set(range(14, 25, 2)),
       "neurosa": set(range(14, 25, 2))}
N_INST = {"synthetic": 6, "willow": 20}
REFERENCE = {"complete": dict(kappa=2, m=1), "path": dict(kappa=2, m=1),
             "nhoim": dict(lam_c=2, mult=1.0, Tend_d0=0.3, ratio=30), "neurosa": dict(lam_c=2, mult=1.0, Tend_d0=0.3, ratio=10)}
AXES = {"complete": ("kappa", "m"), "path": ("kappa", "m"), "nhoim": ("lam_c", "Tend_d0", "ratio"),
        "neurosa": ("lam_c", "mult", "Tend_d0", "ratio")}
HARD_LOW = {"kappa": 1.0, "mult": 1.0}   # natural lower bounds: kappa < 1 heats up; mult < 1 is not an exact reduction


def ours_points(folders):
    pts = defaultdict(lambda: defaultdict(dict))
    meta = {}
    for r in LR.load_sched_shards(folders):
        if r["role"] != "tuning" or r["budget_log2"] not in BUD[r["lib"]]:
            continue
        key = (LR.problem_set(r["alias"]), r["lib"], r["variant"])
        pts[key][r["alias"]][r["budget_log2"]] = (2.0 ** r["budget_log2"], LR.p_opt(r["bestF"], r["F_opt"]))
        meta[key] = dict(kappa=r["kappa"], m=r["m"])
    return pts, meta


def penalty_points(files):
    pts = defaultdict(lambda: defaultdict(dict))
    meta = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["stage"] != "tuning" or r["budget_log2"] not in BUD[r["kind"]]:
                continue
            F = np.array([np.nan if x is None else x for x in r["bestF"]], dtype=float)
            par = (r["lam_c"], r["mult"], r["Tend_d0"], r["ratio"])
            key = (LR.problem_set(r["instance"]), r["kind"], "lam{:g}_mult{:g}_Te{:g}_r{:g}".format(*par))
            pts[key][r["instance"]][r["budget_log2"]] = (float(r["tests"]), LR.p_opt(F, float(r["F_ref"])))   # last row wins
            meta[key] = dict(lam_c=r["lam_c"], mult=r["mult"], Tend_d0=r["Tend_d0"], ratio=r["ratio"])
    return pts, meta


def summarise(pts, meta):
    rows = []
    for key, per in pts.items():
        complete = len(per) == N_INST[key[0]] and all(len(b) == len(BUD[key[1]]) for b in per.values())
        row = dict(set=key[0], method=key[1], setting=key[2], **meta[key], instances=len(per), complete=complete)
        if complete:
            o = C.optimal_median_its(per)
            bstar = o["budget_log2_star"]
            v = np.array([C.tts_at(*per[i][bstar]) for i in per])
            lv = np.where(np.isfinite(v), np.log2(np.maximum(v, 1.0)), 64.0)
            row.update(median_its=o["opt"], log2_median=float(np.log2(o["opt"])) if np.isfinite(o["opt"]) else float("inf"),
                       budget_log2_star=bstar, mean_log2_its=float(lv.mean()), solved_at_star=int(np.isfinite(v).sum()))
        rows.append(row)
    return rows


def choose(rows, method):
    axes, ref = AXES[method], REFERENCE[method]
    rows = [r for r in rows if r["complete"]]

    def dist(r):
        return sum(abs(math.log2(float(r[a]) / float(ref[a]))) for a in axes)

    rows.sort(key=lambda r: (r["median_its"], round(r["mean_log2_its"], 9), dist(r)))
    c = rows[0]
    edges = []
    for a in axes:
        vals = sorted({float(r[a]) for r in rows})
        lo_hard = HARD_LOW.get(a)
        if float(c[a]) == vals[0] and len(vals) > 1 and not (lo_hard is not None and vals[0] <= lo_hard):
            edges.append(f"{a} low ({vals[0]:g})")
        if float(c[a]) == vals[-1] and len(vals) > 1:
            edges.append(f"{a} high ({vals[-1]:g})")
    if c["budget_log2_star"] == min(BUD[method]):
        edges.append(f"budget low (2^{min(BUD[method])})")
    if c["budget_log2_star"] == max(BUD[method]):
        edges.append(f"budget high (2^{c['budget_log2_star']})")
    return dict({k: c[k] for k in axes}, setting=c["setting"], its_training=c["median_its"], log2=c["log2_median"],
                budget_log2_star=c["budget_log2_star"], edges=edges, settings_tried=len(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", nargs="+", required=True, help="folders of the selection shards of this work (globs)")
    ap.add_argument("--penalty", nargs="+", required=True, help="tuning records of the penalty machines (globs)")
    ap.add_argument("--out", required=True, help="choice JSON")
    ap.add_argument("--baseline-out", default=None, help="folder for baseline_choice_<set>.json")
    a = ap.parse_args(argv)
    folders = sorted(f for p in a.shards for f in glob.glob(p))
    files = sorted(f for p in a.penalty for f in glob.glob(p))
    po, mo = ours_points(folders)
    pb, mb = penalty_points(files)
    rows = summarise(po, mo) + summarise(pb, mb)
    choice = defaultdict(dict)
    for set_ in ("synthetic", "willow"):
        for meth in ("complete", "path", "nhoim", "neurosa"):
            g = [r for r in rows if r["set"] == set_ and r["method"] == meth]
            if not any(r["complete"] for r in g):
                print("no complete data for", set_, meth)
                continue
            choice[set_][meth] = choose(g, meth)
            print(set_, meth, choice[set_][meth])
    Path(a.out).write_text(json.dumps(choice, indent=1))
    print("wrote", a.out)
    if a.baseline_out:
        for set_ in ("synthetic", "willow"):
            d = {}
            for kind in ("nhoim", "neurosa"):
                c = choice.get(set_, {}).get(kind)
                if c is None or c["edges"]:
                    continue
                par = dict(lam_c=c["lam_c"], mult=float(c.get("mult", 1.0)), Tend_d0=c["Tend_d0"], ratio=c["ratio"])
                par["Tstart_d0"] = par["Tend_d0"] * par["ratio"]
                d[kind] = {"10": par}
            f = Path(a.baseline_out) / f"baseline_choice_{set_}.json"
            f.write_text(json.dumps(d, indent=1))
            print("wrote", f)


if __name__ == "__main__":
    main()

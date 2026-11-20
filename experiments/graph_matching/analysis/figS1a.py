"""Supplementary Fig. S1a: keypoint accuracy on the 40 WILLOW test pairs (the runs of Fig. 3c).

  python analysis/figS1a.py --shards RUNS/willow_l10/test --inst INST --out ../../source_data/figS1

Keypoint accuracy of an assignment: fraction of the 10 keypoints assigned as in the human annotation (``<pair>.truth.npz``).
Per library (selected setting) and budget: accuracy of the replica with the highest objective (best replica) and the mean
accuracy over the 256 replicas, each averaged over pairs with a 95% percentile bootstrap interval over pairs (5000
resamples, seed 0 for every interval); ``optimum_accuracy`` is the accuracy of the certified optimum (mean over pairs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_runs as LR  # noqa: E402

CFG = HERE.parent / "configs" / "willow_sym"


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--inst", required=True, help="staged instance directory (manifest.json, *.truth.npz)")
    ap.add_argument("--choice", default=str(CFG / "choice.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ch = json.loads(Path(a.choice).read_text())["willow"]
    mlist = json.loads((Path(a.inst) / "manifest.json").read_text())
    man = {m["name"]: m for m in mlist}
    for role in ("test", "tuning"):
        for i, m in enumerate([x for x in mlist if x["family"] == "willow" and x["role"] == role]):
            man[f"@willow:{role}:{i}"] = m
    per = {}
    n_check = n_ok = 0
    for r in LR.load_sched_shards(a.shards):
        if LR.problem_set(r["alias"]) != "willow":
            continue
        c = ch[r["lib"]]
        if (float(r["kappa"]), float(r["m"])) != (float(c["kappa"]), float(c["m"])):
            continue
        m = man[r["alias"]]
        tr = np.load(Path(a.inst) / m["truth"])
        gt, valid = tr["gt"], tr["valid"].astype(bool)
        perm = r["perm"].astype(np.int64)
        acc = (perm[:, valid] == gt[valid]).mean(1)
        at_opt = r["bestF"] >= m["F_opt"] * (1 - 1e-9)
        if at_opt.any():
            n_check += 1
            n_ok += int(np.all(perm[at_opt] == np.asarray(m["opt_perm"])))
        per.setdefault((r["lib"], r["budget_log2"]), {})[m["name"]] = dict(
            acc_mean=float(acc.mean()), acc_best=float(acc[int(np.argmax(r["bestF"]))]), p_opt=float(at_opt.mean()),
            opt_acc=float(m["opt_acc"]))
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    lines = ["library,budget_log2,pairs,accuracy_mean_over_replicas,accuracy_mean_ci_low,accuracy_mean_ci_high,"
             "accuracy_best_replica,accuracy_best_ci_low,accuracy_best_ci_high,p_optimum,optimum_accuracy,"
             "optimum_accuracy_ci_low,optimum_accuracy_ci_high"]
    for (lib, lb) in sorted(per):
        v = list(per[(lib, lb)].values())
        am, ab, po, oa = (boot([x[k] for x in v]) for k in ("acc_mean", "acc_best", "p_opt", "opt_acc"))
        lines.append(f"{lib},{lb},{len(v)},{am[0]:.4f},{am[1]:.4f},{am[2]:.4f},{ab[0]:.4f},{ab[1]:.4f},{ab[2]:.4f},"
                     f"{po[0]:.4f},{oa[0]:.4f},{oa[1]:.4f},{oa[2]:.4f}")
    (dst / "figS1a.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"replicas at the certified optimum carry the enumerated optimal assignment in {n_ok} of {n_check} runs")
    print("wrote", dst / "figS1a.csv")


if __name__ == "__main__":
    main()

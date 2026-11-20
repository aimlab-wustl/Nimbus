"""Readers shared by the L = 10 method-comparison analyses (Fig. 3c, 3f, Supplementary Fig. S1a).

* this work: result shards of the ``matching_sched`` test tasks (``configs/willow_sym/tasks_test.csv``);
* penalty machines: JSON-lines records of ``penalty_baselines.py test``.

Success of a replica: its best objective ``Psi`` (best feasible objective for the penalty machines) reaches the certified
optimum, ``Psi >= Psi* - 1e-9 |Psi*|``. A replica of a penalty machine that never visited a feasible state fails.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

PAR = ("lam_c", "mult", "Tend_d0", "ratio")


def problem_set(name: str) -> str:
    return "willow" if "willow" in name else "synthetic"


def load_sched_shards(folders):
    """Records of ``matching_sched`` shards with the arrays ``bestF``, ``hit_attempt`` and ``perm`` attached."""
    rows = []
    for folder in folders:
        for f in sorted(glob.glob(str(Path(folder) / "shard_*.npz"))):
            recs, arr = C.read_shard(f)
            for r in recs:
                src = r["dup_of"] or r["variant"]
                k = f"{r['alias']}|{src}|b{r['budget_log2']}"
                rows.append(dict(r, bestF=arr[k + "|bestF"], hit_attempt=arr[k + "|hit_attempt"], perm=arr.get(k + "|perm")))
    return rows


def p_opt(bestF, F_opt) -> float:
    F = np.asarray(bestF, dtype=float)
    F = np.where(np.isfinite(F), F, -np.inf)
    return float(np.mean(F >= F_opt - 1e-9 * abs(F_opt)))


def ours_points(rows, set_, lib, kappa, m):
    """instance alias -> {budget_log2: (iterations, P(opt))} for this work at the setting ``(kappa, m)``."""
    per = defaultdict(dict)
    for r in rows:
        if (problem_set(r["alias"]) == set_ and r["lib"] == lib and float(r["kappa"]) == float(kappa)
                and float(r["m"]) == float(m)):
            per[r["alias"]][r["budget_log2"]] = (2.0 ** r["budget_log2"], p_opt(r["bestF"], r["F_opt"]))
    return dict(per)


def penalty_points(files, kind, stage, par):
    """instance -> {budget_log2: (iterations, P(opt))} for one penalty machine at the setting ``par``.

    The iterations are the single-spin tests actually run (``tests``). Repeated rows of the same key: the last one wins.
    """
    per, reps = defaultdict(dict), set()
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["stage"] != stage or r["kind"] != kind or r["n"] != 10:
                continue
            if any(float(r[k]) != float(par.get(k, 1.0)) for k in PAR):
                continue
            per[r["instance"]][r["budget_log2"]] = (float(r["tests"]), p_opt(np.array(
                [np.nan if x is None else x for x in r["bestF"]], dtype=float), float(r["F_ref"])))
            reps.add(r["replicas"])
    return dict(per), (min(reps) if reps else None)


def boot_mean(x, B=5000, rng=None):
    """Mean and 95% percentile bootstrap interval over instances."""
    x = np.asarray(x, dtype=float)
    if np.all(x == x[0]):
        return float(x.mean()), float(x.mean()), float(x.mean())
    idx = rng.integers(0, x.size, size=(B, x.size))
    mm = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))

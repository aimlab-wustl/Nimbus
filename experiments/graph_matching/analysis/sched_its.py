"""Shared analysis of ``matching_sched`` result shards: success probabilities, iterations to solution (ITS) and the
selection of the schedule factors ``m`` and ``kappa`` on training instances.

Definitions (Methods, Benchmark problems and simulation methodology):

* success of a replica: its best objective ``Psi >= eta * Psi_ref`` (``eta = 0.98``; at ``L = 10`` ``Psi_ref`` is the
  certified optimum);
* ``p_i(t)``: fraction of the replicas of instance ``i`` that succeed within ``t`` iterations;
* ``ITS_i(t) = t ln(0.01) / ln(1 - p_i(t))`` (``t`` if ``p = 1``, infinite if ``p = 0``);
* ITS of a method on a problem set: ``min_t median_i ITS_i(t)``, one budget ``t`` for all instances
  (:func:`common.optimal_median_its`), with a 95% percentile bootstrap interval over instances.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

ETA = 0.98


def load_psi_ref(path=None) -> dict:
    """Instance name -> Psi_ref (``configs/psi_ref.csv`` by default)."""
    return C.psi_ref_table(path)


def load_shards(folders) -> list:
    """Records of all ``matching_sched`` shards under ``folders`` (recursively), each with its ``bestF`` array.

    Records are keyed by (library, instance, variant, budget); a later shard supersedes an earlier one (a rerun task).
    A record whose temperatures equal an earlier variant of the same instance (``dup_of``) takes that variant's array.
    """
    if isinstance(folders, (str, Path)):
        folders = [folders]
    recs = []
    for d in folders:
        for f in sorted(glob.glob(str(Path(d) / "**" / "shard_*.npz"), recursive=True)):
            z = np.load(f, allow_pickle=False)
            arr = {k: z[k] for k in z.files if k != "records"}
            for r in json.loads(str(z["records"])):
                recs.append((r, arr))
    out = {}
    for r, arr in recs:
        key = f"{r['alias']}|{r['dup_of'] or r['variant']}|b{r['budget_log2']}|bestF"
        if key in arr:
            out[(r["lib"], r["instance"], r["variant"], r["budget_log2"])] = dict(r, bestF=np.asarray(arr[key], float))
    for r, arr in recs:
        k = (r["lib"], r["instance"], r["variant"], r["budget_log2"])
        if k not in out and r["dup_of"]:
            s = out.get((r["lib"], r["instance"], r["dup_of"], r["budget_log2"]))
            if s is not None:
                out[k] = dict(r, bestF=s["bestF"])
    return list(out.values())


def success(F, ref, eta=ETA) -> float:
    """Fraction of replicas with ``F >= eta * ref`` (relative tolerance 1e-9)."""
    F = np.where(np.isfinite(F), F, -np.inf)
    return float((F >= eta * ref - 1e-9 * abs(ref)).mean())


def per_instance(rows, ref: dict, eta=ETA, budgets=None) -> dict:
    """rows of one (setting, problem set) -> instance -> {budget_log2: (iterations, success probability)}."""
    per = defaultdict(dict)
    for r in rows:
        if budgets is not None and r["budget_log2"] not in budgets:
            continue
        per[r["instance"]][r["budget_log2"]] = (2.0 ** r["budget_log2"], success(r["bestF"], ref[r["instance"]], eta))
    return dict(per)


def its_matrix(per: dict):
    """(budgets common to all instances, ITS[instances, budgets]) with instances sorted by name."""
    insts = sorted(per)
    buds = sorted(set.intersection(*[set(per[i]) for i in insts]))
    return insts, buds, np.array([[C.tts_at(*per[i][b]) for b in buds] for i in insts])


def rank_settings(rows, ref: dict, *, n_train=6, budgets=None, boot=2000) -> list:
    """Rank the settings (variants) of one library on the training instances by ITS (smallest first).

    Ties: the lower mean log2 ITS at the setting's own best budget, then the setting closest to ``kappa = 2, m = 1``.
    Settings not run on all ``n_train`` instances and all ``budgets`` are skipped.
    """
    by_v = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)
    res = []
    for v, rs in by_v.items():
        per = per_instance(rs, ref)
        if len(per) < n_train:
            continue
        common = sorted(set.intersection(*[set(per[i]) for i in per]))
        if budgets is not None and any(b not in common for b in budgets):
            continue
        o = C.optimal_median_its(per, boot=boot)
        _, b_, T = its_matrix(per)
        j = b_.index(o["budget_log2_star"])
        mean_log2 = float(np.mean([math.log2(x) if np.isfinite(x) else 64.0 for x in T[:, j]]))
        res.append(dict(setting=v, kappa=rs[0]["kappa"], m=rs[0]["m"], instances=len(per), its=o["opt"],
                        budget_log2_star=o["budget_log2_star"], ci_lo=o["lo"], ci_hi=o["hi"],
                        solved_at_star=o["solved_at_star"], mean_log2_at_star=mean_log2,
                        budgets=f"{b_[0]}..{b_[-1]}"))
    res.sort(key=lambda x: (x["its"], x["mean_log2_at_star"], abs(math.log2(x["kappa"] / 2)) + abs(math.log2(x["m"]))))
    for k, x in enumerate(res):
        x["rank"] = k + 1
    return res


def select(res: list, kappa2_rule: bool = False) -> dict:
    """The selected setting of a ranked list. With ``kappa2_rule`` the best setting with ``kappa = 2`` is kept when its
    training ITS lies inside the 95% bootstrap interval of the best setting overall."""
    b = res[0]
    rule = "smallest ITS"
    if kappa2_rule and b["kappa"] != 2:
        k2 = [x for x in res if x["kappa"] == 2]
        if k2 and b["ci_lo"] <= k2[0]["its"] <= b["ci_hi"]:
            b, rule = k2[0], "best kappa = 2 setting (inside the interval of the best setting)"
    return dict(b, rule=rule)


def write_csv(path, rows, cols):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", path)

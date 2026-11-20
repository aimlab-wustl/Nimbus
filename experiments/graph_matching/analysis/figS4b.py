"""Fig. S4b: iterations to solution at L = 20 for informed and random shortcuts (K = 40, 80) and the two complete
references, each with its schedule selected on six training instances; ratio random / informed with a paired bootstrap
interval.

  python analysis/figS4b.py --test RUNS/library_its_l20/test RUNS/tts_per_size/test \
      --out ../../source_data/figS4/figS4b.csv --ratio-out ../../source_data/figS4/figS4b_ratio.csv \
      [--train RUNS/library_its_l20/training RUNS/tts_per_size/training --selection-out RUNS/library_its_l20/selection.csv]

Libraries (backbone share 0.5 for all libraries with shortcuts; one random draw per instance, seed = instance index):
informed / random K = 40 and 80; complete library with uniform proposals (its runs are those of the per-size study,
``tts_per_size``, at L = 20); complete library with backbone share 0.5 (the backbone plus all 171 shortcuts).
ITS: target Psi >= 0.98 Psi_ref, minimum over the budgets 2^12 to 2^22 of the median over the 20 test instances; 95% CI:
percentile bootstrap over instances (5,000 resamples, seed 20261015). Ratio: ITS(random) / ITS(informed) with a 95%
percentile interval from 5,000 paired bootstrap resamples of the instances (seed 20261023).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import sched_its as S  # noqa: E402

L = 20
BUDGETS = [12, 14, 16, 18, 20, 22]
#: key -> (library description in the records, label, shortcuts K)
LIBS = {
    "inf40": ("path+informed40@share0.5", "informed", 40),
    "rnd40": ("path+random40@share0.5", "random", 40),
    "inf80": ("path+informed80@share0.5", "informed", 80),
    "rnd80": ("path+random80@share0.5", "random", 80),
    "complete": ("complete", "complete, uniform proposals", 171),
    "ctrl": ("path+informed171@share0.5", "complete, backbone share 0.5", 171),
}
COLS = ["library", "shortcuts_K", "m", "kappa", "its_iterations", "its_ci_lo", "its_ci_hi", "budget_log2_at_min",
        "instances_solved_at_budget", "instances"]


def selection(train_dirs, ref, choice, out=None):
    rows = [r for r in S.load_shards(train_dirs) if r["role"] == "tuning" and r["n"] == L]
    table = []
    for key, (desc, label, _) in LIBS.items():
        res = S.rank_settings([r for r in rows if r["lib"] == desc], ref, budgets=BUDGETS)
        if not res:
            print(key, "no training results")
            continue
        sel = S.select(res)
        want = choice[key]
        same = (sel["m"], sel["kappa"]) == (want["m"], want["kappa"])
        print(f"{label} K = {LIBS[key][2]}: selected m = {sel['m']:g}, kappa = {sel['kappa']:g}; configs: m = {want['m']:g}, "
              f"kappa = {want['kappa']:g} -> {'same' if same else 'DIFFERENT'}")
        for x in res:
            table.append(dict(x, library=key, selected="yes" if x["setting"] == sel["setting"] else ""))
    if out:
        S.write_csv(out, table, ["library", "rank", "setting", "m", "kappa", "instances", "its", "ci_lo", "ci_hi",
                                 "budget_log2_star", "solved_at_star", "mean_log2_at_star", "budgets", "selected"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--train", nargs="*", default=None)
    ap.add_argument("--choice", default=str(C.CONFIGS / "library_its_l20" / "choice.json"))
    ap.add_argument("--psi-ref", default=None)
    ap.add_argument("--psi-ref-training", default=str(C.CONFIGS / "psi_ref_training.csv"),
                    help="Psi_ref of the training instances (full precision; used by the selection)")
    ap.add_argument("--selection-out", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio-out", required=True)
    a = ap.parse_args(argv)
    ref = S.load_psi_ref(a.psi_ref)
    choice = json.loads(Path(a.choice).read_text())
    if a.train:
        selection(a.train, S.load_psi_ref(a.psi_ref_training), choice, a.selection_out)
    rows = [r for r in S.load_shards(a.test) if r["role"] == "test" and r["n"] == L]
    out, mats = [], {}
    for key, (desc, label, K) in LIBS.items():
        c = choice[key]
        per = S.per_instance([r for r in rows if r["lib"] == desc and r["variant"] == c["variant"]], ref, budgets=BUDGETS)
        if len(per) < 20 or any(len(d) < len(BUDGETS) for d in per.values()):
            raise SystemExit(f"{key}: test results incomplete ({len(per)} instances)")
        o = C.optimal_median_its(per, boot=5000)
        insts, buds, T = S.its_matrix(per)
        mats[key] = (insts, T)
        out.append(dict(library=label, shortcuts_K=K, m=c["m"], kappa=c["kappa"], its_iterations=o["opt"], its_ci_lo=o["lo"],
                        its_ci_hi=o["hi"], budget_log2_at_min=o["budget_log2_star"],
                        instances_solved_at_budget=o["solved_at_star"], instances=o["instances"]))
        print(f"{label} K = {K}: ITS {o['opt']:.4g} [{o['lo']:.4g}, {o['hi']:.4g}] at 2^{o['budget_log2_star']}")
    S.write_csv(a.out, out, COLS)
    rng = np.random.default_rng(20261023)
    ratios = []
    for K in (40, 80):
        (ii, Ti), (ir, Tr) = mats[f"inf{K}"], mats[f"rnd{K}"]
        assert ii == ir

        def its(T, ix):
            return np.median(T[ix], axis=0).min()

        full = np.arange(len(ii))
        r = its(Tr, full) / its(Ti, full)
        bs = [its(Tr, ix) / its(Ti, ix) for ix in rng.integers(0, len(ii), (5000, len(ii)))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ratios.append(dict(shortcuts_K=K, its_informed=float(its(Ti, full)), its_random=float(its(Tr, full)),
                           ratio_random_over_informed=float(r), ratio_ci_lo=float(lo), ratio_ci_hi=float(hi)))
        print(f"K = {K}: random / informed = {r:.3g} [{lo:.3g}, {hi:.3g}]")
    S.write_csv(a.ratio_out, ratios, list(ratios[0]))


if __name__ == "__main__":
    main()

"""Fig. 3d: iterations to solution versus the number of keypoints L, complete library, per-size selected schedule.

  python analysis/fig3d.py --test RUNS/tts_per_size/test --out ../../source_data/fig3/fig3d.csv \
      [--train RUNS/tts_per_size/training --selection-out RUNS/tts_per_size/selection.csv]

Test: 20 synthetic test instances per L (sigma = 0.05), 256 replicas, the setting of ``configs/tts_per_size/choice.json``.
Target: Psi >= 0.98 Psi_ref (``configs/psi_ref.csv``; certified optimum at L = 10). ITS: minimum over the test budgets of
the median over instances; 95% CI: percentile bootstrap over instances (5,000 resamples, seed 20261015).

With ``--train`` the selection on the six training instances per L is recomputed (all settings ranked by training ITS,
2,000 bootstrap resamples; the best setting with kappa = 2 is kept when it lies inside the interval of the best
setting) and compared with ``choice.json``. The training targets use ``configs/psi_ref_training.csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import sched_its as S  # noqa: E402

#: training budget grid (log2 iterations) that every ranked setting must cover, per L
TRAIN_BUDGETS = {10: [6, 8, 10, 12, 14, 16, 18], 15: [12, 14, 16, 18, 20], 20: [12, 14, 16, 18, 20, 22],
                 25: [12, 14, 16, 18, 20, 22], 30: [14, 16, 18, 20, 22]}
COLS = ["L", "library", "m", "kappa", "its_iterations", "its_ci_lo", "its_ci_hi", "budget_log2_at_min",
        "instances_solved_at_budget", "instances"]


def selection(train_dirs, ref, choice, out=None):
    rows = [r for r in S.load_shards(train_dirs) if r["role"] == "tuning" and r["lib"] == "complete"]
    table, ok = [], True
    for L in sorted(TRAIN_BUDGETS):
        res = S.rank_settings([r for r in rows if r["n"] == L], ref, budgets=TRAIN_BUDGETS[L])
        if not res:
            continue
        sel = S.select(res, kappa2_rule=True)
        want = choice.get(str(L))
        same = want is not None and (sel["m"], sel["kappa"]) == (want["m"], want["kappa"])
        ok &= same
        print(f"L = {L}: selected m = {sel['m']:g}, kappa = {sel['kappa']:g} ({sel['rule']}); "
              f"configs: m = {want['m']:g}, kappa = {want['kappa']:g} -> {'same' if same else 'DIFFERENT'}")
        for x in res:
            table.append(dict(x, L=L, selected="yes" if x["setting"] == sel["setting"] else ""))
    if out:
        S.write_csv(out, table, ["L", "rank", "setting", "m", "kappa", "instances", "its", "ci_lo", "ci_hi",
                                 "budget_log2_star", "solved_at_star", "mean_log2_at_star", "budgets", "selected"])
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test", nargs="+", required=True, help="folders with the test shards")
    ap.add_argument("--train", nargs="*", default=None, help="folders with the training shards (selection check)")
    ap.add_argument("--choice", default=str(C.CONFIGS / "tts_per_size" / "choice.json"))
    ap.add_argument("--psi-ref", default=None, help="Psi_ref table (default configs/psi_ref.csv)")
    ap.add_argument("--psi-ref-training", default=str(C.CONFIGS / "psi_ref_training.csv"),
                    help="Psi_ref of the training instances (full precision; used by the selection)")
    ap.add_argument("--selection-out", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ref = S.load_psi_ref(a.psi_ref)
    choice = json.loads(Path(a.choice).read_text())
    if a.train:
        selection(a.train, S.load_psi_ref(a.psi_ref_training), choice, a.selection_out)
    rows = [r for r in S.load_shards(a.test) if r["role"] == "test" and r["lib"] == "complete"]
    out = []
    for L in sorted(int(k) for k in choice):
        c = choice[str(L)]
        rs = [r for r in rows if r["n"] == L and r["variant"] == c["variant"]]
        per = S.per_instance(rs, ref)
        if len(per) < 20:
            raise SystemExit(f"L = {L}: test results incomplete ({len(per)} instances)")
        o = C.optimal_median_its(per, boot=5000)
        out.append(dict(L=L, library="complete", m=c["m"], kappa=c["kappa"], its_iterations=o["opt"], its_ci_lo=o["lo"],
                        its_ci_hi=o["hi"], budget_log2_at_min=o["budget_log2_star"],
                        instances_solved_at_budget=o["solved_at_star"], instances=o["instances"]))
        print(f"L = {L}: ITS {o['opt']:.4g} [{o['lo']:.4g}, {o['hi']:.4g}] at 2^{o['budget_log2_star']}")
    S.write_csv(a.out, out, COLS)


if __name__ == "__main__":
    main()

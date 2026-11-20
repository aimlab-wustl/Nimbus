"""Supplementary Table S2: the schedule settings selected on the training instances, read from the selection outputs.

  python analysis/tableS2.py --out ../../source_data/tables/tableS2.csv

Inputs (written by the selection steps; see README):

* ``configs/willow_sym/choice.json``: L = 10, synthetic and WILLOW, this work (complete and path libraries) and the two
  penalty machines (``select_settings.py``);
* ``configs/tts_per_size/choice.json``: complete library, synthetic L = 15-30 (``analysis/fig3d.py --train``);
* ``configs/library_its_l20/choice.json``: libraries at L = 20 (``analysis/figS4b.py --train``);
* ``configs/library_its_l10/choice.json``: libraries at L = 10, target Psi* (``library_its.py choose``).

This work: start factor ``m`` and minimum ratio ``kappa``. Penalty machines: penalty strength (lambda-bar) and end
temperature, both in units of Psi*/L, ratio of start to end temperature, and the quadratization multiplier (NeuroSA
style).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
COLS = ["group", "problem_set", "L", "method", "m", "kappa", "penalty_strength_per_psi_unit",
        "end_temperature_per_psi_unit", "start_to_end_ratio", "quadratization_multiplier", "source"]

METHOD_L10 = {"complete": "This work, complete", "path": "This work, path", "nhoim": "N-HOIM, penalty",
              "neurosa": "NeuroSA, quadratized penalty"}
LIB_L20 = {"inf40": "Informed, K=40", "rnd40": "Random, K=40", "inf80": "Informed, K=80", "rnd80": "Random, K=80",
           "complete": "Complete, uniform proposals", "ctrl": "Complete, backbone share 0.5"}
LIB_L10 = {"path": "Path", "inf4": "Informed, K=4", "rnd4": "Random, K=4", "inf9": "Informed, K=9",
           "rnd9": "Random, K=9", "inf18": "Informed, K=18", "rnd18": "Random, K=18", "inf27": "Informed, K=27",
           "rnd27": "Random, K=27", "complete": "Complete, backbone share 0.5"}


def _g(x):
    return "" if x is None else f"{float(x):g}"


def rows(configs=CONFIGS):
    out = []
    w = json.loads((configs / "willow_sym" / "choice.json").read_text())
    for set_, label in (("synthetic", "Synthetic"), ("willow", "WILLOW")):
        for meth, name in METHOD_L10.items():
            c = w[set_][meth]
            r = dict(group="method comparison", problem_set=label, L=10, method=name,
                     source="configs/willow_sym/choice.json")
            if meth in ("complete", "path"):
                r.update(m=_g(c["m"]), kappa=_g(c["kappa"]))
            else:
                r.update(penalty_strength_per_psi_unit=_g(c["lam_c"]), end_temperature_per_psi_unit=_g(c["Tend_d0"]),
                         start_to_end_ratio=_g(c["ratio"]), quadratization_multiplier=_g(c.get("mult", 1.0)))
            out.append(r)
    t = json.loads((configs / "tts_per_size" / "choice.json").read_text())
    for L in (15, 20, 25, 30):
        c = t[str(L)]
        out.append(dict(group="per-size schedule", problem_set="Synthetic", L=L, method="This work, complete",
                        m=_g(c["m"]), kappa=_g(c["kappa"]), source="configs/tts_per_size/choice.json"))
    for key, cfg, L, group in (("l20", "library_its_l20", 20, "library experiments"),
                               ("l10", "library_its_l10", 10, "library experiments, target Psi*")):
        d = json.loads((configs / cfg / "choice.json").read_text())
        for k, name in (LIB_L20 if key == "l20" else LIB_L10).items():
            c = d[k]
            out.append(dict(group=group, problem_set="Synthetic", L=L, method=name, m=_g(c["m"]), kappa=_g(c["kappa"]),
                            source=f"configs/{cfg}/choice.json"))
    return [{c: r.get(c, "") for c in COLS} for r in out]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rs = rows()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rs)
    print(len(rs), "rows ->", a.out)


if __name__ == "__main__":
    main()

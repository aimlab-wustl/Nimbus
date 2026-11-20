"""Build the reference values Psi_ref from the domain-solver results and the certified optima.

  # test instances (configs/psi_ref.csv; values stored with 6 significant digits)
  python psi_ref_table.py --inst INST --mat RUNS/refs/refs_test_*.mat --out ../configs/psi_ref.csv
  # training instances used in the setting selection (configs/psi_ref_training.csv; full precision)
  python psi_ref_table.py --inst INST --mat RUNS/refs/refs_tuning_*.mat --training --out ../configs/psi_ref_training.csv

Psi_ref of a test instance:

* ``certified_optimum``: the optimum from enumerating all L! assignments (L <= 10; ``F_opt`` in the manifest);
* ``domain_solvers``: the best objective over all restarts of Adapt-BCAGM3, BCAGM3 and RRWHM (each returned assignment is
  evaluated with the package's objective; the solvers' own scores are checked against it).

``--training``: every instance in the ``.mat`` files gets the domain-solver value at full precision (kind
``domain_solvers``; at L = 10 it equals the certified optimum to within 1e-14), as used by the setting selection.
``--keep``: entries of an existing table that are not recomputed here are copied from it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_refs import load_refs  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inst", required=True)
    ap.add_argument("--mat", nargs="*", default=[], help="run_refs.m outputs")
    ap.add_argument("--training", action="store_true", help="domain-solver values only, full precision")
    ap.add_argument("--keep", default=None, help="existing table: entries not recomputed here are copied from it")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ref = {}
    if not a.training:
        for m in json.loads((Path(a.inst) / "manifest.json").read_text()):
            if m.get("F_opt") is not None:
                ref[m["name"]] = (float(m["F_opt"]), "certified_optimum")
    best = {}
    for mp in a.mat:
        for r in load_refs(mp, a.inst):
            best[r["instance"]] = max(best.get(r["instance"], -np.inf), float(np.max(r["F"])))
    for nm, v in best.items():
        if nm not in ref:
            ref[nm] = (v, "domain_solvers")
    if a.keep:
        with open(a.keep, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["instance"] not in ref:
                    ref[r["instance"]] = (float(r["psi_ref"]), r["kind"])
    fmt = repr if a.training else (lambda v: f"{v:.6g}")
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "psi_ref", "kind"])
        for nm in sorted(ref):
            w.writerow([nm, fmt(ref[nm][0]), ref[nm][1]])
    print(len(ref), "instances ->", a.out)


if __name__ == "__main__":
    main()

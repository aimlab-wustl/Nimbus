"""Fig. S3a-e: size of the compiled machine at 16 bits per weight and per maintained neuron input.

  python analysis/figS3.py --sizes RUNS/size/sizes.jsonl --out ../../source_data

Reads the counts written by ``program_size.py`` (one line per case) and writes
  figS3/figS3a.csv   surface code: connections per code distance d and library
  figS3/figS3b.csv   matching: connections per number of keypoints L and library
  figS3/figS3c.csv   matching: objective and auxiliary clause counts per L and library
  figS3/figS3d.csv   matching: bits per segment and library (total and the segments of the share bars)
  figS3/figS3e.csv   surface code: the same
Segments (bits): connections to objective clauses, connections to auxiliary clauses (readout-only clauses included),
switch details, weights (16 bits per objective clause), objective-clause signs (1 bit each), auxiliary-clause signs
(1 bit each), maintained move-neuron inputs (16 bits per neuron). The first four are the program, the last three
the run-time state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BITS = 16
SEG = ["bits_pi_obj", "bits_pi_gate", "bits_switch", "bits_weights", "bits_signs_obj", "bits_signs_gate",
       "bits_inputs"]


def load(path):
    d = pd.DataFrame([json.loads(line) for line in open(path, encoding="utf-8") if line.strip()])
    d = d.drop_duplicates(["problem", "size", "lib"]).copy()
    d["bits_pi_gate"] = d.bits_pi_gate + d.bits_pi_read
    d["bits_signs_obj"] = d.M_obj
    d["bits_signs_gate"] = d.M_gate + d.M_read
    d["bits_weights"] = BITS * d.M_obj
    d["bits_inputs"] = BITS * d.R
    d["bits_total"] = sum(d[k] for k in SEG)
    return d


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("wrote", path, len(df), "rows")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--out", required=True, help="source_data folder")
    a = ap.parse_args(argv)
    d = load(a.sizes)
    mat = d[d.problem == "matching"].sort_values(["lib", "size"])
    sur = d[d.problem == "surface"].sort_values(["lib", "size"])
    out = Path(a.out) / "figS3"
    base = ["size", "lib", "R", "routes"]
    write(sur[base], out / "figS3a.csv")
    write(mat[base], out / "figS3b.csv")
    write(mat[["size", "lib", "R", "M_obj", "M_gate"]], out / "figS3c.csv")
    write(mat[["size", "lib", "bits_total"] + SEG], out / "figS3d.csv")
    write(sur[["size", "lib", "bits_total"] + SEG], out / "figS3e.csv")


if __name__ == "__main__":
    main()

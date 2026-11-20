"""Export matching tensors into the input format of the MATLAB hypergraph-matching solvers (BCAGM3, Adapt-BCAGM3, RRWHM).

  python export_mat.py INST MATDIR [--role test] [--family synthetic|willow] [--names NAME ...]

Format: super-symmetrized (all 6 orderings of every tensor entry), lexicographically sorted triples of 0-based linear
assignment indices ``i*L + a`` (source ``i``, target ``a``) as int32 ``indH3`` (N x 3) with values ``valH3`` (N x 1) and the
size ``nP``. With this format the solvers' own score of an assignment is 6 Psi.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from nimbus.problems.formats.matching_tensor import MatchingTensor, load_tensor


def solver_format(t: MatchingTensor):
    n = t.n
    code = (t.triples[:, :, 0].astype(np.int64) * n + t.triples[:, :, 1])  # (m, 3) assignment codes
    rows = np.concatenate([code[:, list(p)] for p in itertools.permutations(range(3))], axis=0)
    vals = np.tile(t.values, 6)
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
    return rows[order].astype(np.int32), vals[order].astype(np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inst", help="staged instance directory (manifest.json)")
    ap.add_argument("dst", help="output folder, one <name>.mat per instance")
    ap.add_argument("--role", choices=["test", "tuning"], default=None)
    ap.add_argument("--family", choices=["synthetic", "willow"], default=None)
    ap.add_argument("--names", nargs="*", default=None)
    a = ap.parse_args(argv)
    src, dst = Path(a.inst), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    man = json.loads((src / "manifest.json").read_text())
    man = [r for r in man if (a.role is None or r["role"] == a.role) and (a.family is None or r["family"] == a.family)
           and (a.names is None or r["name"] in a.names)]
    for r in man:
        t = load_tensor(src / r["tensor"])
        ind, val = solver_format(t)
        sio.savemat(dst / (r["name"] + ".mat"), {"indH3": ind, "valH3": val.reshape(-1, 1), "nP": np.int32(t.n)},
                    do_compression=True)
    print(f"{len(man)} files -> {dst}")


if __name__ == "__main__":
    main()

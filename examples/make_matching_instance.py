"""Write a synthetic graph-matching instance (third-order triangle affinity tensor) as an .npz file.

Two point sets: n random source points and a rotated, scaled, noisy and shuffled copy as the target points (the
synthetic protocol of the paper). The tensor holds triangle affinities computed from the coordinates only; the true
assignment is written to a separate ``<name>.truth.npz`` that no solver reads.

    python examples/make_matching_instance.py --n 8 --sigma 0.05 --seed 0 --out examples/problems/matching_n8.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nimbus.problems.formats.matching_tensor import point_pair_tensor, save_tensor, similarity_deformation_pair


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=8, help="points per set (rows of the assignment)")
    ap.add_argument("--sigma", type=float, default=0.05, help="position noise of the target points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="examples/problems/matching_n8.npz")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    P1, P2, truth, _valid = similarity_deformation_pair(a.n, a.sigma, rng)
    tensor = point_pair_tensor(P1, P2, seed=a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_tensor(out, tensor)
    np.savez_compressed(out.with_name(out.stem + ".truth.npz"), truth=truth)
    print(f"wrote {out}: n = {tensor.n}, {tensor.nnz} tensor entries")


if __name__ == "__main__":
    main()

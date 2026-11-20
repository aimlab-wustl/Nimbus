"""Write a 2D +-J spin glass on an L x L torus in the ``nimbus.spinpoly/1`` format.

Every spin is free (no constraint block), so the move library is single spin flips. The couplings are +1 or -1 with
equal probability; E(s) = -sum_<ij> J_ij s_i s_j.

    python examples/make_spin_glass.py --L 8 --seed 0 --out examples/problems/spin_glass_8x8.spinpoly.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--L", type=int, default=8, help="linear size of the torus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="examples/problems/spin_glass_8x8.spinpoly.yaml")
    a = ap.parse_args(argv)
    L = a.L
    rng = np.random.default_rng(a.seed)
    terms = []
    for x in range(L):
        for y in range(L):
            i = x * L + y
            for j in (((x + 1) % L) * L + y, x * L + (y + 1) % L):
                terms.append({"support": sorted([i, j]), "weight": float(rng.choice([-1.0, 1.0]))})
    doc = {"schema": "nimbus.spinpoly/1", "name": f"spin_glass_{L}x{L}",
           "description": f"2D +-J spin glass, {L} x {L} torus, seed {a.seed}; all spins free",
           "n_spins": L * L, "spin_convention": "plus_one_selected", "energy_convention": "minus_J",
           "const": 0, "terms": terms}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=110), encoding="utf-8")
    print(f"wrote {out}: {L * L} spins, {len(terms)} couplings")


if __name__ == "__main__":
    main()

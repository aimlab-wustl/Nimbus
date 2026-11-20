"""Pre-generate one-syndrome-per-shot cores (pt_core.sv, pt_params.svh, pt_info.json) into hardware/cores/pt_d<d>_ns<NS>,
so that a Vivado build needs no Python. The seven cores of the published designs:

python hardware/gen/gen_sizes.py 3x16 5x16 7x64 9x56 11x40 13x32 15x24 [--K 2 --B 16] [--out-dir DIR]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_pt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cfg", nargs="+"); ap.add_argument("--K", type=int, default=2); ap.add_argument("--B", type=int, default=16)
ap.add_argument("--p", type=float, default=0.12); ap.add_argument("--bmin", type=float, default=0.1)
ap.add_argument("--out-dir", default=str(ROOT / "hardware" / "cores"))
a = ap.parse_args()
for c in a.cfg:
    d, ns = (int(x) for x in c.split("x"))
    gen_pt.main(["--d", str(d), "--NS", str(ns), "--K", str(a.K), "--B", str(a.B), "--p", str(a.p), "--bmin", str(a.bmin),
                 "--out", str(Path(a.out_dir) / f"pt_d{d}_ns{ns}")])

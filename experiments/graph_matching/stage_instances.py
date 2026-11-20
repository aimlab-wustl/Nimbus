"""Write the graph-matching instance set (deterministic from seeds).

  python -m nimbus.experiments.fetch --set matching            # downloads WILLOW-ObjectClass into data/willow
  python stage_instances.py --out INST                           # all sets used in the paper (about 425 MB)
  python stage_instances.py --out INST --sizes 10 --sigmas 0.05 --no-willow     # a subset

Synthetic instances: for every number of keypoints L and deformation level sigma, 26 instances (indices 0-5:
training, 6-25: test), named ``syn_n{L}_s{sigma}_o0_{k:02d}``. Keypoints of image 1 are uniform in the unit square;
image 2 is a similarity transform of image 1 plus Gaussian noise of standard deviation sigma, with its points in random
order (``nimbus.problems.formats.matching_tensor.similarity_deformation_pair``, seed ``1000 + 1000 L + k``).

WILLOW-ObjectClass pairs: 12 pairs per class (Car, Duck, Face, Motorbike, Winebottle), L = 10, 20 training and 40 test
pairs. Images are split in file order (every third image is a training image), and a pair is drawn within one split, so
no image appears in both roles. The tensor is built from the keypoint coordinates only; the annotation order of image 2
is hidden by a random shuffle (seed 77 + class index).

Third-order tensor: ``point_pair_tensor`` (triangles of each image, similarity of their angles). Every instance writes
``<name>.npz`` (tensor) and ``<name>.truth.npz`` (ground-truth assignment, used only for keypoint accuracy).
``manifest.json`` lists every instance with role, size, objective of the ground truth and, for L <= 10, the certified
optimum ``F_opt`` from enumerating all L! assignments.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nimbus.experiments import paths
from nimbus.problems.formats.matching_tensor import (evaluate_F, point_pair_tensor, save_tensor,
                                                    similarity_deformation_pair)
from nimbus.problems.matching import enumerate_optimum

WILLOW_CLASSES = ("Car", "Duck", "Face", "Motorbike", "Winebottle")


def _finish(rec: dict, tensor, gt, valid, out: Path, enum_max: int) -> dict:
    name = rec["name"]
    save_tensor(out / f"{name}.npz", tensor)
    np.savez_compressed(out / f"{name}.truth.npz", gt=gt, valid=valid)
    rec.update(n=int(tensor.n), nnz=int(tensor.nnz), F_gt=float(evaluate_F(tensor, gt)), tensor=f"{name}.npz",
               truth=f"{name}.truth.npz")
    if tensor.n <= enum_max:
        t0 = time.time()
        r = enumerate_optimum(tensor)
        rec.update(F_opt=r["F_opt"], n_optimal=r["n_optimal"], opt_perm=r["perm"], t_enum=round(time.time() - t0, 2),
                   opt_acc=float(np.mean(np.array(r["perm"])[valid] == gt[valid])))
    return rec


def synthetic(ns, count, sigma, outliers, out: Path, seed0=1000, enum_max=10):
    recs = []
    for n in ns:
        for k in range(count):
            seed = seed0 + 1000 * n + k
            rng = np.random.default_rng(seed)
            P1, P2, gt, valid = similarity_deformation_pair(n, sigma, rng, n_outliers=outliers)
            t = point_pair_tensor(P1, P2, seed=seed)
            rec = {"name": f"syn_n{n}_s{sigma:g}_o{outliers}_{k:02d}", "family": "synthetic", "sigma": sigma,
                   "outliers": outliers, "seed": seed, "role": "tuning" if k < max(1, count // 4) else "test"}
            recs.append(_finish(rec, t, gt, valid, out, enum_max))
    return recs


def willow_points(root: Path, cls: str):
    import scipy.io as sio

    files = sorted((root / "WILLOW-ObjectClass" / cls).glob("*.mat"))
    return [(f.stem, sio.loadmat(f)["pts_coord"].T.astype(np.float64)) for f in files]  # (10, 2)


def willow(pairs_per_class, out: Path, seed0=77, enum_max=10, data_root=None, classes=WILLOW_CLASSES):
    root = Path(data_root) if data_root else paths.data_root() / "willow"
    recs = []
    for ci, cls in enumerate(classes):
        imgs = willow_points(root, cls)
        if not imgs:
            raise SystemExit(f"no WILLOW files under {root / 'WILLOW-ObjectClass' / cls}; "
                             "run: python -m nimbus.experiments.fetch --set matching")
        split = {nm: ("tuning" if i % 3 == 0 else "test") for i, (nm, _) in enumerate(imgs)}
        rng = np.random.default_rng(seed0 + ci)
        done = set()
        npairs = {"tuning": pairs_per_class // 3, "test": pairs_per_class - pairs_per_class // 3}
        for role, cnt in npairs.items():
            pool = [i for i, (nm, _) in enumerate(imgs) if split[nm] == role]
            k = 0
            while k < cnt:
                i, j = sorted(rng.choice(pool, 2, replace=False).tolist())
                if (i, j) in done:
                    continue
                done.add((i, j))
                (n1, P1), (n2, P2) = imgs[i], imgs[j]
                shuf = rng.permutation(len(P2))  # hide the annotation order of the target
                P2s = P2[shuf]
                gt = np.empty(len(P2), np.int64)
                gt[shuf] = np.arange(len(P2))
                t = point_pair_tensor(P1, P2s, seed=int(rng.integers(1 << 30)))
                rec = {"name": f"willow_{cls}_{n1}_{n2}", "family": "willow", "class": cls, "images": [n1, n2],
                       "role": role}
                recs.append(_finish(rec, t, gt, np.ones(len(gt), bool), out, enum_max))
                k += 1
    return recs


def _merge(out: Path, recs):
    mp = out / "manifest.json"
    old = json.loads(mp.read_text()) if mp.exists() else []
    names = {r["name"] for r in recs}
    mp.write_text(json.dumps([r for r in old if r["name"] not in names] + recs, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write the graph-matching instance set.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 20, 25, 30, 40, 50])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--count", type=int, default=26)
    ap.add_argument("--no-willow", action="store_true")
    ap.add_argument("--willow-root", default=None, help="folder holding WILLOW-ObjectClass/ (default: data/willow)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for sg in a.sigmas:
        recs = synthetic(a.sizes, a.count, sg, 0, out)
        _merge(out, recs)
        print(f"{len(recs)} synthetic instances (sigma {sg:g}) -> {out}")
    if not a.no_willow:
        recs = willow(12, out, data_root=a.willow_root)
        _merge(out, recs)
        print(f"{len(recs)} WILLOW pairs -> {out}")


if __name__ == "__main__":
    main()

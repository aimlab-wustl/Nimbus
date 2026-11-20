"""Merge the result shards of one run folder into one compact bundle (records + per-replica arrays).

  python extract_shards.py --run RUNS/auto_schedule --subs sigma005 sigma010 libraries --dest RUNS/bundles

For every sub-folder ``<sub>`` this writes ``DEST/<sub>.npz``: ``records`` (JSON list; each record gets ``_shard``, the
shard id) and every array of every shard under the key ``<shard>/<name>``. Boolean arrays are bit-packed (key suffix
``@bits``, length in ``<shard>/<name>@len``). Per-replica assignments (``*_perm``) are dropped. The analysis scripts read
these bundles with :func:`load_bundle`.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_shard  # noqa: E402


def bundle(folder: Path, dest: Path):
    recs_all, arrays = [], {}
    files = sorted(glob.glob(str(folder / "shard_*.npz")))
    for f in files:
        sid = Path(f).stem.split("_")[-1]
        recs, arrs = read_shard(f)
        for r in recs:
            r = dict(r)
            r["_shard"] = sid
            recs_all.append(r)
        for k, v in arrs.items():
            if k.endswith("_perm"):
                continue
            if v.dtype == bool:
                arrays[f"{sid}/{k}@bits"] = np.packbits(v, axis=-1)
                arrays[f"{sid}/{k}@len"] = np.int64(v.shape[-1])
            else:
                arrays[f"{sid}/{k}"] = v
    np.savez_compressed(dest, records=json.dumps(recs_all), **arrays)
    return len(files), len(recs_all)


def load_bundle(path):
    """``(records, get)``; ``get(shard, name)`` returns the array (bool arrays unpacked) or None."""
    z = np.load(path, allow_pickle=False)
    recs = json.loads(str(z["records"]))
    files = set(z.files)

    def get(shard, name):
        k = f"{shard}/{name}"
        if k in files:
            return z[k]
        if k + "@bits" in files:
            n = int(z[k + "@len"])
            return np.unpackbits(z[k + "@bits"], axis=-1, count=n).astype(bool)
        return None
    return recs, get


def main():
    ap = argparse.ArgumentParser(description="Merge result shards into one bundle per sub-folder.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--subs", nargs="+", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for s in a.subs:
        nf, nr = bundle(Path(a.run) / s, dest / f"{s}.npz")
        print(f"{s}: {nf} shards, {nr} records -> {dest / (s + '.npz')}")


if __name__ == "__main__":
    main()

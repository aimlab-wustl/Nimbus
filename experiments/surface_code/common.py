"""Shared constants and helpers of the surface-code experiment scripts.

* Code distances, physical error rates and the FPGA designs of the paper (replicas and sweep budgets per d).
* Run outputs: every task writes ``<tag>.npz`` (per-shot logical failures) and ``<tag>.json`` (summary); a task whose
  ``.npz`` exists is skipped, so an interrupted job is resumed by running the same command again.
* :func:`wilson` (95% Wilson score interval, z = 1.96, as plotted in the paper) and :func:`write_csv`.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
REPO = HERE.parents[1]
SOURCE = REPO / "source_data"

#: one syndrome per shot: code distances and the physical error rate of the main figure
DS_ONE_SHOT = [3, 5, 7, 9, 11, 13, 15]
P_ONE_SHOT = 0.12
#: replicas of the one-shot FPGA designs
R_ONE_SHOT = {3: 16, 5: 16, 7: 64, 9: 56, 11: 40, 13: 32, 15: 24}
#: sweep budget per shot run on the board (d = 5 runs 128 sweeps; the software model records every budget up to the
#: next power of two, so one run gives all of them)
SWEEPS_BOARD = {3: 46, 5: 128, 7: 216, 9: 304, 11: 430, 13: 512, 15: 1722}
#: larger budget used near the threshold (p >= 0.14, d >= 5): 2^12 sweeps
L2_NEAR_THRESHOLD = 12

#: real time: code distances, the physical error rate of the main figure, replicas of the FPGA designs
DS_REAL_TIME = [5, 7, 9, 11]
P_REAL_TIME = 0.03
R_REAL_TIME = {3: 16, 5: 16, 7: 16, 9: 24, 11: 24}
K_REAL_TIME = 12                                  # 2^12 sweeps per window
#: the CPU baselines among which the best is taken (PyMatching without correlations is shown but not a candidate)
BEST_OF = ("pmc", "bpm", "bp")
BASELINE_NAMES = {"pm": "PyMatching", "pmc": "correlated PyMatching", "bpm": "BP+matching", "bp": "BP-OSD",
                  "uf": "union-find", "tn": "tensor-network ML", "bposd": "BP-OSD"}


def wilson(k, n, z=1.96):
    """95% Wilson score interval of k successes in n trials."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def write_csv(path, rows, cols=None):
    """Write a list of dicts as CSV (columns in the order of ``cols`` or of the first row)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = cols or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in cols})
    print("wrote", path, f"({len(rows)} rows)")


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, np.integer):
        return int(v)
    return v


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save_task(out, tag, arrays: dict, rec: dict):
    """Write ``<tag>.npz`` (compressed) and ``<tag>.json`` into ``out``; the ``.npz`` is written last via a temporary
    file, so its presence marks a finished task."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(rec) + "\n")
    tmp = out / f"{tag}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out / f"{tag}.npz")


def done(out, tag) -> bool:
    return (Path(out) / f"{tag}.npz").exists()


def task_lines(path):
    """Non-empty, non-comment lines of a task table."""
    return [s.strip() for s in Path(path).read_text().splitlines() if s.strip() and not s.lstrip().startswith("#")]


def select_tasks(lines, worker=0, workers=1):
    return [s for i, s in enumerate(lines) if i % workers == worker]


def cpu_model() -> str:
    """CPU model name (Linux /proc/cpuinfo, else platform.processor())."""
    import platform
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"

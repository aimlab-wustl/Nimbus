"""One syndrome per shot (code-capacity depolarizing noise): runs of the paper.

  python one_shot.py accuracy --d 9 --p 0.12 --chunk 40 --shots 1000 --out RUNS/one_shot           [--l2 12]
  python one_shot.py library  --d 15 --p 0.12 --chunk 40 --shots 1000 --variant one --out RUNS/library
  python one_shot.py union-find --dir RUNS/one_shot [--p 0.12]
  python one_shot.py tasks configs/tasks_one_shot.txt --out RUNS/one_shot [--worker j --workers K]

accuracy
    The software model of the FPGA design for d (bit-exact: replicas R per d, a replica-exchange round every second
    sweep, 16 uniform bits per lane, beta from 1 to 0.1) against tensor-network ML decoding (qecsim matrix product
    state, bond dimension 16), PyMatching (X and Z decoded separately) and BP-OSD (joint X / Y / Z columns, OSD-CS order
    10), all on the same shots. The model runs 2^k sweeps (k = ceil(log2) of the board budget, or ``--l2``); the decision
    at a budget g uses the beta = 1 class counts of sweeps [g/2, g), as on the board, so one run gives every budget
    g <= 2^k. Output ``afin_d<d>_p<p>_c<chunk>[_l<l2>].npz``: grid (budgets), f_ml / f_me [budget, shot] (most visited
    class / lowest-energy class), f_pm, f_bo, f_tn [shot] (logical failures). With ``--l2`` the reference decoders are
    skipped (they are in the run without it).
library
    Move library and replica exchange on the same shots: ``all`` (stabilizer moves, backbone and all shortcuts: all d
    straight logical operators per type, the FPGA library), ``one`` (stabilizer moves and the backbone: one logical
    operator per type), ``noswap`` (the FPGA library without replica exchange: the beta = 1 replica alone), ``stab``
    (stabilizer moves only: the class never changes, so the decision is the class of the start state). Output
    ``abl_d<d>_p<p>_c<chunk>_<variant>_l<l2>.npz``: grid, f_ml [budget, shot].
union-find
    Union-find (ldpc, peeling; X and Z decoded separately) on the shots of every accuracy run in ``--dir`` (PyMatching
    is recomputed and checked against the stored failures, which confirms that the shots are the same). Output
    ``<tag>_uf.npz`` (f_uf) and ``<tag>_uf.json``.

Shots: ``sample_shots(problem, shots, 7 + 1000 * chunk)``; decoder seed ``1 + 7919 * chunk``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

from nimbus.decoding import baselines as BL  # noqa: E402
from nimbus.decoding.code_capacity import class_failures, sample_shots  # noqa: E402
from nimbus.decoding.one_shot import (PTConfig, build_spec, c0_from_syndrome, min_energy_class_by_budget,  # noqa: E402
                                      ml_class_by_budget, run_pt)

SEED, GSEED = 7, 1
LIBRARY = {"all": "stabilizers_logical_all", "one": "stabilizers_logical", "noswap": "stabilizers_logical_all",
           "stab": "stabilizers_logical_all"}


def budget_grid(l2, extra=()):
    """Budgets (even sweep counts, about four per octave, from 4 to 2^l2) at which the decision is recorded."""
    nsw = 1 << l2
    g = sorted({int(round(2 ** (j / 4) / 2)) * 2 for j in range(8, 4 * l2 + 1)} | {nsw, *extra})
    return [x for x in g if 4 <= x <= nsw]


def spec_for(d, p, K=2, library="stabilizers_logical_all"):
    return build_spec(d, p, PTConfig(NS=C.R_ONE_SHOT[d], bmin=0.1, K=K, B=16), library=library)


def accuracy(d, p, chunk, shots, out, l2=0, chi=16, no_tn=False):
    tag = f"afin_d{d}_p{p}_c{chunk}" + (f"_l{l2}" if l2 else "")
    if C.done(out, tag):
        return f"skip {tag}"
    spec = spec_for(d, p)
    prob = spec.prob
    e, syn = sample_shots(prob, shots, SEED + 1000 * chunk)
    sec = {}
    t0 = time.time()
    c0 = c0_from_syndrome(spec, syn)
    k = l2 or int(np.ceil(np.log2(C.SWEEPS_BOARD[d])))
    res = run_pt(spec, c0, k, seed=GSEED + 7919 * chunk)
    sec["hw"] = time.time() - t0
    fail_cls = class_failures(prob, c0, e)
    ar = np.arange(len(e))
    grid = budget_grid(k, (C.SWEEPS_BOARD[d],))
    f_ml = np.stack([fail_cls[ar, c] for c in ml_class_by_budget(res["cs"], grid)])
    f_me = np.stack([fail_cls[ar, c] for c in min_energy_class_by_budget(res["bs"], grid)])
    arrays = dict(grid=np.array(grid), f_ml=f_ml, f_me=f_me)
    ler = {}
    if not l2:
        t0 = time.time()
        arrays["f_pm"] = prob.logical_failure(BL.matching_correction(prob, syn), e)
        sec["pm"] = time.time() - t0
        t0 = time.time()
        arrays["f_bo"] = prob.logical_failure(BL.bposd_correction(prob, syn), e)
        sec["bposd"] = time.time() - t0
        arrays["f_tn"] = np.zeros(0, bool)
        if not no_tn:
            t0 = time.time()
            arrays["f_tn"] = BL.tensor_network_failures(prob, e, chi)
            sec["tn"] = time.time() - t0
        ler = dict(pm=float(arrays["f_pm"].mean()), bposd=float(arrays["f_bo"].mean()),
                   tn=float(arrays["f_tn"].mean()) if arrays["f_tn"].size else None)
    gi = grid.index(C.SWEEPS_BOARD[d]) if C.SWEEPS_BOARD[d] in grid else len(grid) - 1
    rec = dict(d=d, p=p, chunk=chunk, shots=len(e), R=C.R_ONE_SHOT[d], nsw_board=C.SWEEPS_BOARD[d], l2=k, chi=chi,
               ler=dict(hw_ml=float(f_ml[gi].mean()), hw_me=float(f_me[gi].mean()), **ler),
               sec_per_shot={q: v / len(e) for q, v in sec.items()})
    C.save_task(out, tag, arrays, rec)
    return json.dumps(rec)


def library(d, p, chunk, shots, variant, out, l2=12):
    tag = f"abl_d{d}_p{p}_c{chunk}_{variant}_l{l2}"
    if C.done(out, tag):
        return f"skip {tag}"
    K = (1 << 30) if variant == "noswap" else 2               # a swap interval longer than the run: no exchange
    spec = spec_for(d, p, K=K, library=LIBRARY[variant])
    prob = spec.prob
    e, syn = sample_shots(prob, shots, SEED + 1000 * chunk)
    c0 = c0_from_syndrome(spec, syn)
    fail_cls = class_failures(prob, c0, e)
    t0 = time.time()
    if variant == "stab":
        grid = [1]
        f_ml = fail_cls[:, :1].T.copy()
    else:
        res = run_pt(spec, c0, l2, seed=GSEED + 7919 * chunk)
        grid = budget_grid(l2)
        ar = np.arange(len(e))
        f_ml = np.stack([fail_cls[ar, c] for c in ml_class_by_budget(res["cs"], grid)])
    rec = dict(d=d, p=p, chunk=chunk, shots=len(e), var=variant, l2=l2, R=C.R_ONE_SHOT[d], sec=time.time() - t0,
               ler_last=float(f_ml[-1].mean()))
    C.save_task(out, tag, dict(grid=np.array(grid), f_ml=f_ml), rec)
    return json.dumps(rec)


def union_find(dirp, p=0.0):
    for js in sorted(Path(dirp).glob("afin_d*_p*_c*.json")):
        if js.stem.endswith(("_uf",)) or "_l" in js.stem.split("_c")[-1]:
            continue
        r = json.loads(js.read_text())
        d, pp, chunk, S = r["d"], r["p"], r["chunk"], r["shots"]
        if p and abs(pp - p) > 1e-9:
            continue
        tag = js.stem + "_uf"
        if C.done(dirp, tag):
            continue
        prob = spec_for(d, pp).prob
        e, syn = sample_shots(prob, S, SEED + 1000 * chunk)
        f_pm = prob.logical_failure(BL.matching_correction(prob, syn), e)
        if not np.array_equal(f_pm, np.load(js.with_suffix(".npz"))["f_pm"]):
            raise SystemExit(f"{js.name}: regenerated shots differ (PyMatching failures do not match)")
        t0 = time.perf_counter()
        f_uf = prob.logical_failure(BL.union_find_correction(prob, syn), e)
        sec = (time.perf_counter() - t0) / S
        C.save_task(dirp, tag, dict(f_uf=f_uf), dict(d=d, p=pp, chunk=chunk, shots=S, ler_uf=float(f_uf.mean()),
                                                     ler_pm=float(f_pm.mean()), sec_uf=sec))
        print(tag, "union-find", round(float(f_uf.mean()), 4), "PyMatching", round(float(f_pm.mean()), 4), flush=True)


def run_line(line, out):
    """One line of a task table: ``accuracy|library d p chunk shots [variant] [l2]``."""
    f = line.split()
    kind, d, p, chunk, shots = f[0], int(f[1]), float(f[2]), int(f[3]), int(f[4])
    if kind == "accuracy":
        return accuracy(d, p, chunk, shots, out, l2=int(f[5]) if len(f) > 5 else 0)
    return library(d, p, chunk, shots, f[5], out, l2=int(f[6]) if len(f) > 6 else 12)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("accuracy")
    a2 = sub.add_parser("library")
    for q in (a1, a2):
        q.add_argument("--d", type=int, required=True)
        q.add_argument("--p", type=float, required=True)
        q.add_argument("--chunk", type=int, required=True)
        q.add_argument("--shots", type=int, default=1000)
        q.add_argument("--out", required=True)
    a1.add_argument("--l2", type=int, default=0, help="2^l2 sweeps instead of the board budget (reference decoders skipped)")
    a1.add_argument("--chi", type=int, default=16)
    a1.add_argument("--no-tn", action="store_true")
    a2.add_argument("--variant", required=True, choices=list(LIBRARY))
    a2.add_argument("--l2", type=int, default=12)
    a3 = sub.add_parser("union-find")
    a3.add_argument("--dir", required=True)
    a3.add_argument("--p", type=float, default=0.0)
    a4 = sub.add_parser("tasks")
    a4.add_argument("table")
    a4.add_argument("--out", required=True)
    a4.add_argument("--worker", type=int, default=0)
    a4.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "accuracy":
        print(accuracy(a.d, a.p, a.chunk, a.shots, a.out, a.l2, a.chi, a.no_tn))
    elif a.cmd == "library":
        print(library(a.d, a.p, a.chunk, a.shots, a.variant, a.out, a.l2))
    elif a.cmd == "union-find":
        union_find(a.dir, a.p)
    else:
        for line in C.select_tasks(C.task_lines(a.table), a.worker, a.workers):
            print(run_line(line, a.out), flush=True)


if __name__ == "__main__":
    main()

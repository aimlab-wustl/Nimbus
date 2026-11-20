"""Complete library with backbone share 0.5 (reference of Figs. 4a, 4g and Supplementary Fig. S4a).

The library holds all ``L(L - 1)/2`` row exchanges; the backbone of ``L - 1`` path exchanges is repeated in the proposal
list so that it receives half of all proposals (``shortcut_library`` with all shortcuts in informed order and
``share = 0.5``). For each of the 20 synthetic test instances (sigma = 0.05; indices 6-25), 256 replicas, seed 11:

* ``--n 20``: budgets 2^18 and 2^20, four calibrations: ``own`` (the automatically calibrated schedule, m = 1,
  kappa = 2, on this library) and ``k20``, ``k40``, ``k80`` (the calibration of batch 0 of the batched runs with
  K = 20, 40, 80, so the comparison with those runs is at equal temperatures);
* ``--n 30``: budgets 2^20 and 2^22, ``own`` calibration only.

Every (instance, calibration) writes ``ctrl_<i>_<cal>.npz`` (JSON records + ``b<B>_bestF``) into ``OUT/ctrl`` (L = 20) or
``OUT/ctrl_n30``; existing files are skipped.

  python library_control.py --out RUNS/library_control --n 20 [--worker j --workers K] [--backend cpu] [--inst INST]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import common as C  # noqa: E402

SEED = 11
CALS = ("own", "k20", "k40", "k80")


def tasks(n=20):
    cals = CALS if n == 20 else ("own",)
    return [(i, c) for i in range(6, 26) for c in cals]


def main(argv=None):
    from nimbus.annealing import schedule as S
    from nimbus.annealing.protocol import calibrate_temperatures
    from nimbus.backends import get_backend
    from nimbus.compiler import compile_model
    from nimbus.core.model import bind_weights
    from nimbus.problems.matching import shortcut_library

    ap = argparse.ArgumentParser(description="Complete library with backbone share 0.5.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inst", default=None)
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--n", type=int, choices=[20, 30], default=20)
    ap.add_argument("--budgets", type=int, nargs="*", default=None, help="log2 budgets (default 18 20 for L = 20, 20 22 for L = 30)")
    a = ap.parse_args(argv)
    inst = C.inst_dir(a.inst)
    be = get_backend(a.backend)
    n = a.n
    out = Path(a.out) / ("ctrl" if n == 20 else f"ctrl_n{n}")
    out.mkdir(parents=True, exist_ok=True)
    budgets = tuple(a.budgets) if a.budgets else ((18, 20) if n == 20 else (20, 22))
    n_extra = n * (n - 1) // 2 - (n - 1)
    spec = {"base": "path", "extra": "informed", "k": n_extra, "share": 0.5}
    cache = {}
    for i, cal in tasks(n)[a.worker::a.workers]:
        f = out / f"ctrl_{i:02d}_{cal}.npz"
        if f.exists():
            continue
        name = f"syn_n{n}_s0.05_o0_{i:02d}"
        t0 = time.perf_counter()
        if i not in cache:
            cache.clear()
            m, t, prob = C.load_instance(inst, name)
            model, binding, info = C.compile_instance(prob, t, spec)
            t1 = time.perf_counter()
            own = calibrate_temperatures(model, binding, prob, SEED, min_ratio=2.0, min_ratio_mode="lower_end")
            t_own = time.perf_counter() - t1
            plan = C.prepare_plan(be, a.backend, model, a.threads)
            cache[i] = (t, prob, model, binding, info, own, t_own, plan)
        t, prob, model, binding, info, own, t_own, plan = cache[i]
        if cal == "own":
            (tau_s, tau_e), t_cal = own, t_own
        else:
            k = int(cal[1:])
            lib0, _ = shortcut_library(prob, {"base": "path", "extra": "informed", "k": k, "share": 0.5}, t)
            m0 = compile_model(prob.structure(), lib0, identities=tuple(prob.identities()))
            b0 = bind_weights(m0, prob.term_weights(0), prob.const(0))
            t1 = time.perf_counter()
            tau_s, tau_e = calibrate_temperatures(m0, b0, prob, SEED, min_ratio=2.0, min_ratio_mode="lower_end")
            t_cal = time.perf_counter() - t1
        reps = 256
        arrays, recs = {}, []
        spins0 = prob.random_feasible(SEED, reps, 0)
        kw = dict(replica_id=np.arange(reps, dtype=np.int64), instance_id=np.zeros(reps, dtype=np.int64),
                  is_feasible=prob.is_feasible)
        warm = be.init_state(plan, binding, spins0, **kw)
        be.run(plan, warm, S.fn_endpoints(tau_s, tau_e, 0.1, 64), seed=SEED)
        for lb in budgets:
            B = 2 ** lb
            st = be.init_state(plan, binding, spins0, **kw)
            t1 = time.perf_counter()
            res = be.run(plan, st, S.fn_endpoints(tau_s, tau_e, 0.1, B), seed=SEED)
            wall = time.perf_counter() - t1
            s = res.state
            bestF = -np.asarray(s.best_energy, dtype=np.float64)
            arrays[f"b{lb}_bestF"] = bestF
            recs.append(dict(instance=name, n=n, lib="complete_share0.5", copies=info.get("copies"),
                             R=int(model.n_groups), R_nodup=info.get("R_nodup"), cal=cal, tau_start=tau_s, tau_end=tau_e,
                             t_calibrate=t_cal, budget_log2=lb, replicas=reps, seed=SEED, backend=a.backend, wall_s=wall,
                             acc_rate=float(np.sum(s.counters["accepted"]) / (reps * B)),
                             bestF_max=float(bestF.max()), bestF_median=float(np.median(bestF)),
                             routes=int(model.n_routes)))
        tmp = f.with_name(f.name + f".tmp{os.getpid()}.npz")
        np.savez_compressed(tmp, records=np.array(json.dumps(recs)), **arrays)
        os.replace(tmp, f)
        print(f"{name} {cal}: tau {tau_s:.3g} -> {tau_e:.3g}, {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"worker {a.worker}/{a.workers} done", flush=True)


if __name__ == "__main__":
    main()

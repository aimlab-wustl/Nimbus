"""Control runs of Fig. S2: anneal, no cooling and random walk on the same moves, starts and random numbers.

  python control_random_walk.py --inst INST --out RUNS/control/summary.jsonl --per-replica RUNS/control/per_replica

Six synthetic test instances (sigma = 0.05): instances 1-3 at L = 20 (``syn_n20_s0.05_o0_06`` to ``_08``, 2^18 iterations)
and at L = 30 (``syn_n30_s0.05_o0_06`` to ``_08``, 2^20 iterations), complete library, 64 replicas, seed 11, the same
random feasible starts in every variant:

* ``anneal``: the automatically calibrated schedule (late-pilot calibration, ``m = 1``, ``kappa = 2``, FN schedule);
* ``hot`` (no cooling): constant temperature at the calibrated start temperature for the whole run;
* ``walk`` (random walk): infinite temperature, every proposed exchange is accepted.

Per instance and variant, ``<name>__<variant>.npz`` holds per replica ``best_over_ref`` (best objective / Psi_ref) and
``acceptance`` (accepted proposals that changed the state / proposals); one JSON line per (instance, variant) holds the
summary. Psi_ref: ``configs/psi_ref.csv``. Runs on one CPU node (16 threads; about 15 minutes in total).
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

NAMES = [f"syn_n{n}_s0.05_o0_0{i}" for n in (20, 30) for i in (6, 7, 8)]
BUDGET = {10: 14, 15: 16, 20: 18, 25: 20, 30: 20}     # log2 iterations per replica, by L


def main(argv=None):
    from nimbus.annealing import schedule as S
    from nimbus.annealing.protocol import calibrate_temperatures
    from nimbus.backends import get_backend
    from nimbus.compiler import compile_model
    from nimbus.core.model import bind_weights

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inst", default=None, help="staged instance directory (default: $NIMBUS_MATCHING_INST)")
    ap.add_argument("--names", nargs="+", default=NAMES)
    ap.add_argument("--arms", nargs="+", default=["anneal", "walk", "hot"], choices=["anneal", "walk", "hot"])
    ap.add_argument("--budget-log2", type=int, default=None, help="override the budget of every instance (tests)")
    ap.add_argument("--replicas", type=int, default=64)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--psi-ref", default=None)
    ap.add_argument("--out", required=True, help="summary JSON lines (appended)")
    ap.add_argument("--per-replica", default=None, help="folder for the per-replica npz files")
    a = ap.parse_args(argv)
    inst = C.inst_dir(a.inst)
    if a.per_replica:
        Path(a.per_replica).mkdir(parents=True, exist_ok=True)
    ref = C.psi_ref_table(a.psi_ref)
    be = get_backend("cpu")
    with open(a.out, "a") as out:
        for name in a.names:
            m, t, prob = C.load_instance(inst, name)
            model = compile_model(prob.structure(), prob.transition_library("complete"),
                                  identities=tuple(prob.identities()))
            binding = bind_weights(model, prob.term_weights(0), prob.const(0))
            ts, te = calibrate_temperatures(model, binding, prob, a.seed, min_ratio=2.0, min_ratio_mode="lower_end")
            ts_raw, te_raw = calibrate_temperatures(model, binding, prob, a.seed, raw=True)
            plan = be.prepare(model, threads=a.threads, strategy="auto", layout="auto")
            lb = a.budget_log2 if a.budget_log2 is not None else BUDGET[m["n"]]
            B = 2 ** lb
            F_ref = ref[name]
            arms = {"anneal": S.fn_endpoints(ts, te, 0.1, B), "walk": S.constant(1e30, B), "hot": S.constant(ts, B)}
            for arm in a.arms:
                spins0 = prob.random_feasible(a.seed, a.replicas, 0)
                st = be.init_state(plan, binding, spins0, replica_id=np.arange(a.replicas, dtype=np.int64),
                                   instance_id=np.zeros(a.replicas, dtype=np.int64), is_feasible=prob.is_feasible)
                t0 = time.perf_counter()
                res = be.run(plan, st, arms[arm], seed=a.seed, stop_on_target=False)
                wall = time.perf_counter() - t0
                c = res.state.counters
                ratio = -np.asarray(res.state.best_energy) / F_ref
                acc_r = np.asarray(c["accepted_changed"], float) / np.asarray(c["attempts"], float)
                if a.per_replica:
                    np.savez(Path(a.per_replica) / f"{name}__{arm}.npz", best_over_ref=ratio, acceptance=acc_r)
                rec = dict(instance=name, n=m["n"], arm=arm, budget_log2=lb, replicas=a.replicas, tau_start=ts, tau_end=te,
                           tau_start_raw=ts_raw, tau_end_raw=te_raw, F_ref=F_ref, p_reach_098=float((ratio >= 0.98).mean()),
                           best_over_ref_mean=float(ratio.mean()), best_over_ref_max=float(ratio.max()),
                           best_over_ref_median=float(np.median(ratio)),
                           acceptance=float(c["accepted_changed"].sum() / c["attempts"].sum()), wall_s=wall)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"{name} {arm:6s} P(0.98)={rec['p_reach_098']:.3f} mean best/ref={rec['best_over_ref_mean']:.3f} "
                      f"acceptance={rec['acceptance']:.3f} ({wall:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

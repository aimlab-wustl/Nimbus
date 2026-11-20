"""Run the tasks of a task table: one task writes one result shard.

A task table (CSV: task_id, fig, kind, backend, est_s, params_json) lists self-contained tasks. Worker ``j`` of ``K``
takes rows ``j, j + K, ...``; finished tasks (existing shards) are skipped, so an interrupted job is simply rerun.

  python worker.py --tasks configs/auto_schedule/tasks_sigma005.csv --out RUNS/auto_schedule/sigma005 --worker 0 --workers 8 --inst INST
  python worker.py --tasks ... --out ... --ids 3 17            # selected tasks only

Task kinds:

``matching_ours``
    The automatically calibrated schedule (``m = 1``, ``kappa`` from the task, default 2, ``lower_end`` rule).
    Params: instances, lib (library spec), budgets (log2 iterations), replicas, seed, kappa, kappa_mode.
    Arrays per (instance, budget): ``<name>_b<k>_bestF``, ``_hit_attempt``, ``_perm``.
``matching_sched``
    Schedules ``(m, kappa)`` applied to one raw calibration per instance and library (training-instance selection
    and test runs of the selected settings). Params: instances, lib, budgets, replicas, seed, role, variants
    ``[{id, kappa, m}]``. Arrays per (instance, variant, budget): ``<alias>|<variant>|b<k>|bestF``, ``|hit_attempt``,
    ``|perm``. A variant with the same temperatures as an earlier variant of the same instance is not rerun
    (``dup_of``).
``matching_dwell``
    Batched libraries (see ``batching.py``).

Environment: ``NIMBUS_MATCHING_INST`` (or ``--inst``) = the staged instance directory.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

import common as C  # noqa: E402


def task_matching_ours(p, ctx):
    from nimbus.annealing.protocol import calibrate_temperatures
    from nimbus.backends import get_backend
    from nimbus.core.model import structural_resources
    from nimbus.problems.matching import describe_library

    ref = C.psi_ref_table() if p.get("use_psi_ref", True) else {}
    records, arrays = [], {}
    be = get_backend(ctx["backend"])
    for name in p["instances"]:
        m, t, prob = C.load_instance(ctx["inst"], name)
        F_t, tkind = C.f_target(m, ref)
        lib_spec = dict(p["lib"])
        if lib_spec.get("extra") == "random":      # one seeded draw per instance (seed = instance index)
            lib_spec["seed"] = int(m["name"].rsplit("_", 1)[-1]) if m["family"] == "synthetic" else 0
        t0 = time.perf_counter()
        model, binding, info = C.compile_instance(prob, t, lib_spec)
        t_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        tau_s, tau_e = calibrate_temperatures(model, binding, prob, p["seed"], min_ratio=p.get("kappa"),
                                              min_ratio_mode=p.get("kappa_mode", "lower_end"))
        t_cal = time.perf_counter() - t0
        plan = C.prepare_plan(be, ctx["backend"], model, ctx["threads"])
        reps = p["replicas"]
        res_ = structural_resources(model)
        for lb in p["budgets"]:
            t1 = time.perf_counter()
            s = C.anneal(be, plan, binding, prob, tau_start=tau_s, tau_end=tau_e, budget_log2=lb, replicas=reps,
                         seed=p["seed"], F_target=F_t)
            wall = time.perf_counter() - t1
            c = s.counters
            bestF = -np.asarray(s.best_energy)
            hit_at = np.asarray(c["target_hit_attempt"])
            key = f"{name}_b{lb}"
            arrays[key + "_bestF"] = bestF.astype(np.float64)
            arrays[key + "_hit_attempt"] = hit_at.astype(np.int64)
            perms = C.best_perms(s, prob.P)
            arrays[key + "_perm"] = perms if prob.P <= 12 else perms[int(np.argmax(bestF))][None, :]
            records.append(dict(instance=name, family=m["family"], role=m["role"], n=int(m["n"]),
                                lib=describe_library(lib_spec), lib_spec=lib_spec, budget_log2=lb, replicas=reps,
                                backend=ctx["backend"], seed=p["seed"], F_target=F_t, target_kind=tkind,
                                F_opt=m.get("F_opt"), tau_start=tau_s, tau_end=tau_e,
                                tau_clipped=bool(tau_e >= tau_s), kappa=p.get("kappa"), wall_s=wall,
                                attempts_total=int(c["attempts"].sum()),
                                accepted_changed_total=int(c["accepted_changed"].sum()),
                                hit_rate=float((hit_at >= 0).mean()) if F_t is not None else None,
                                bestF_max=float(bestF.max()), bestF_median=float(np.median(bestF)),
                                R=int(model.n_groups), R_nodup=info["R_nodup"], copies=info["copies"],
                                select_s=info["select_s"], routes=int(model.n_routes), gates=res_["M_add"],
                                memory_bytes=res_.get("memory_bytes"), t_compile=t_compile, t_calibrate=t_cal))
    return records, arrays


def task_matching_sched(p, ctx):
    from nimbus.annealing.protocol import calibrate_temperatures, scaled_temperatures
    from nimbus.backends import get_backend
    from nimbus.problems.matching import describe_library

    be = get_backend(ctx["backend"])
    records, arrays = [], {}
    for name in p["instances"]:
        m, t, prob = C.load_instance(ctx["inst"], name)
        if p["role"] == "tuning" and m["role"] != "tuning":
            raise ValueError(f"{name}: settings are selected on training instances only")
        lib_spec = dict(p["lib"])
        t0 = time.perf_counter()
        model, binding, info = C.compile_instance(prob, t, lib_spec)
        t_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        ts0, te0 = calibrate_temperatures(model, binding, prob, p["seed"], raw=True)
        t_cal = time.perf_counter() - t0
        plan = C.prepare_plan(be, ctx["backend"], model, ctx["threads"])
        reps = p["replicas"]
        F_opt = m.get("F_opt")
        seen = {}
        for v in p["variants"]:
            ts, te = scaled_temperatures(ts0, te0, v["m"], v["kappa"])
            key_s = (round(ts, 15), round(te, 15))
            dup = seen.get(key_s)
            if dup is None:
                seen[key_s] = v["id"]
            for lb in p["budgets"]:
                rec = dict(instance=m["name"], alias=name, family=m["family"], role=m["role"], n=int(m["n"]),
                           lib=describe_library(lib_spec), variant=v["id"], kappa=v["kappa"], m=v["m"],
                           budget_log2=lb, replicas=reps, seed=p["seed"], backend=ctx["backend"], tau_start_raw=ts0,
                           tau_end_raw=te0, tau_start=ts, tau_end=te, ratio_raw=ts0 / te0, ratio=ts / te, F_opt=F_opt,
                           dup_of=dup, R=int(model.n_groups), t_compile=t_compile, t_calibrate=t_cal)
                if dup is None:
                    t1 = time.perf_counter()
                    s = C.anneal(be, plan, binding, prob, tau_start=ts, tau_end=te, budget_log2=lb, replicas=reps,
                                 seed=p["seed"], F_target=F_opt)
                    wall = time.perf_counter() - t1
                    c = s.counters
                    bestF = -np.asarray(s.best_energy, dtype=np.float64)
                    hit = np.asarray(c["target_hit_attempt"]).astype(np.int64)
                    k = f"{name}|{v['id']}|b{lb}"
                    arrays[k + "|bestF"] = bestF
                    arrays[k + "|hit_attempt"] = hit
                    if prob.P <= 12:
                        arrays[k + "|perm"] = C.best_perms(s, prob.P)
                    rec.update(wall_s=wall, attempts_total=int(c["attempts"].sum()),
                               accepted_changed_total=int(c["accepted_changed"].sum()), bestF_max=float(bestF.max()),
                               hit_rate=float((hit >= 0).mean()) if F_opt is not None else None)
                records.append(rec)
    return records, arrays


#: kind -> function, or "module:function" imported on first use
KINDS = {"matching_ours": task_matching_ours, "matching_sched": task_matching_sched,
         "matching_dwell": "batching:task_matching_dwell"}


def _kind(name):
    f = KINDS[name]
    if isinstance(f, str):
        mod, fn = f.split(":")
        f = getattr(importlib.import_module(mod), fn)
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inst", default=None, help="staged instance directory (default: $NIMBUS_MATCHING_INST)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--backend", default=None, help="override the table's backend (cpu or gpu)")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many tasks")
    a = ap.parse_args(argv)
    tasks = C.read_tasks(a.tasks)
    mine = [t for t in tasks if t["task_id"] in set(a.ids)] if a.ids else tasks[a.worker::a.workers]
    inst = C.inst_dir(a.inst)
    done = 0
    for t in mine:
        sp_ = C.shard_path(a.out, t["task_id"])
        if sp_.exists():
            continue
        if a.limit is not None and done >= a.limit:
            break
        ctx = dict(backend=a.backend or t["backend"], threads=a.threads, inst=inst)
        t0 = time.perf_counter()
        recs, arrays = _kind(t["kind"])(t["params"], ctx)
        for r in recs:
            r["task_id"] = t["task_id"]
            r["task_wall_s"] = time.perf_counter() - t0
        C.write_shard(sp_, recs, arrays)
        done += 1
        print(f"task {t['task_id']} {t['kind']} done in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"worker {a.worker}/{a.workers}: {done} tasks run", flush=True)


if __name__ == "__main__":
    main()

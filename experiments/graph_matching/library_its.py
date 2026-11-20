"""Iterations to solution (ITS) of the move libraries at L = 10 (Fig. 4d, 4e and Supplementary Table S4).

Libraries (classes): path; path + K informed or random shortcuts, K = 4, 9, 18, 27, backbone share 0.5; complete =
backbone + all 36 shortcuts at backbone share 0.5. Test random draws: seeds equal to the instance index and 100-108
(ten draws per instance). Target: the certified optimum Psi*. Schedule: one raw calibration per instance and library
(seed 11, ``nimbus.annealing.protocol.calibrate_temperatures``), then ``tau_start = m tau_start_cal`` and
``tau_end = min(tau_end_cal, tau_start / kappa)``; the automatically calibrated schedule is ``m = 1``, ``kappa = 2``.
Fowler-Nordheim shape eta = 0.1. Budgets: 2^6 .. 2^16 iterations per replica in factor-2 steps (path 2^6 .. 2^20).

  run --role test --settings auto      20 test instances x 46 libraries at (m, kappa) = (1, 2), 256 replicas
                                       (one task per instance: --task 0..19)
  run --role tuning                    6 training instances x 10 classes (--task 0..59: instance task // 10,
                                       class task % 10 in the order of CLASSES), grid m in 1/16 .. 16 x
                                       kappa in 1 .. 32, 32 replicas; random classes 3 draws per instance (seeds i,
                                       200 + i, 300 + i, disjoint from the test draws)
  choose --tune DIR [DIR ...]          per class the setting with the smallest ITS on the training instances (joint
                                       minimum over setting and budget; ties: lower mean log2 iterations to solution
                                       at the selected budget, then closest to (1, 2)); writes choice.json
  run --role test --settings choice.json   test instances at each class's selected setting

    python library_its.py run --role test --settings auto --task 0 --inst INST --out RUNS/library_its/auto
    python library_its.py run --role tuning --task 0 --inst INST --out RUNS/library_its/tune
    python library_its.py run --role tuning --task 6 --variants 64,0.0625 ... 128,16 --out RUNS/library_its/tune_ext
    python library_its.py choose --tune RUNS/library_its/tune RUNS/library_its/tune_ext --out OUT
    python library_its.py run --role test --settings configs/library_its_l10/choice.json --task 0 --inst INST \
        --out RUNS/library_its/tuned

Output: one JSON file per task, ``<role>_task<iii>.json``, with one record per (library, setting, budget): instance,
cls, lib, k, seed, variant, kappa, m, budget_log2, replicas, P_s (fraction of replicas that reach Psi*), temperatures.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

N = 10
KS = (4, 9, 18, 27)
REST = N * (N - 1) // 2 - (N - 1)                       # 36 shortcuts outside the backbone
CLASSES = ["path"] + [f"inf{k}" for k in KS] + [f"rnd{k}" for k in KS] + ["complete"]
BUDGETS = list(range(6, 17))                            # 2^6 .. 2^16, factor-2 steps
BUDGETS_PATH = list(range(6, 21))                       # the path library needs longer runs
MULTS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]
KAPPAS = [1, 2, 4, 8, 16, 32]
AUTO = dict(id="k2_m1", kappa=2.0, m=1.0)


def vid(k, m):
    return dict(id=f"k{k:g}_m{m:g}", kappa=float(k), m=float(m))


def instances(inst: Path, role: str):
    man = json.loads((Path(inst) / "manifest.json").read_text())
    return sorted([m for m in man if m["family"] == "synthetic" and m["n"] == N and "_s0.05_" in m["name"]
                   and m["role"] == role], key=lambda m: m["name"])


def class_specs(cls: str, idx: int, role: str):
    """``[(lib, k, seed, library spec)]`` of one class on instance ``idx``."""
    if cls == "path":
        return [("path", 0, 0, {"base": "path"})]
    if cls == "complete":
        return [("complete", REST, 0, {"base": "path", "extra": "informed", "k": REST, "share": 0.5})]
    k = int(cls[3:])
    if cls.startswith("inf"):
        return [("informed", k, 0, {"base": "path", "extra": "informed", "k": k, "share": 0.5})]
    seeds = [idx] + list(range(100, 109)) if role == "test" else [idx, 200 + idx, 300 + idx]
    return [("random", k, s, {"base": "path", "extra": "random", "k": k, "share": 0.5, "seed": s}) for s in seeds]


def budgets_for(cls, settings=None):
    if settings and settings.get(cls, {}).get("budgets"):
        return list(settings[cls]["budgets"])
    return BUDGETS_PATH if cls == "path" else BUDGETS


# ------------------------------------------------------------------------------------------------ runs
def cmd_run(a):
    from nimbus.annealing import schedule as S
    from nimbus.annealing.protocol import calibrate_temperatures, scaled_temperatures
    from nimbus.backends import get_backend

    inst = C.inst_dir(a.inst)
    ms = instances(inst, a.role)
    if a.role == "tuning":
        cls0 = CLASSES[a.task % len(CLASSES)]
        jobs = [(ms[a.task // len(CLASSES)], cls0)]
        variants = {cls0: [vid(k, m) for m in MULTS for k in KAPPAS]}
        settings = None
        if a.variants:
            variants = {c: [vid(*map(float, x.split(","))) for x in a.variants] for c in variants}
        reps = a.replicas or 32
    else:
        settings = None if a.settings == "auto" else json.loads(Path(a.settings).read_text())
        jobs = [(ms[a.task], c) for c in (a.classes or CLASSES)]
        variants = {c: [AUTO] if settings is None else [vid(settings[c]["kappa"], settings[c]["m"])] for c in CLASSES}
        reps = a.replicas or 256
    out = a.out / f"{a.role}_task{a.task:03d}.json"
    if out.exists() and not a.force:
        print("exists:", out)
        return
    be = get_backend("cpu")
    rows = []
    t00 = time.time()
    for m, cls in jobs:
        _, t, prob = C.load_instance(inst, m["name"])
        F = float(m["F_opt"])
        idx = int(m["name"].rsplit("_", 1)[-1])
        buds = a.budgets or budgets_for(cls, settings)
        for lib, k, seed, spec in class_specs(cls, idx, a.role):
            model, binding, _ = C.compile_instance(prob, t, spec)
            ts0, te0 = calibrate_temperatures(model, binding, prob, a.seed, raw=True)
            plan = C.prepare_plan(be, "cpu", model, a.threads)
            target = np.full(reps, -F + 1e-9 * abs(F))
            done = {}
            for v in variants[cls]:
                ts, te = scaled_temperatures(ts0, te0, v["m"], v["kappa"])
                key = (round(ts, 15), round(te, 15))
                for lb in buds:
                    if (key, lb) in done:                       # identical trajectory: reuse
                        P, dup = done[(key, lb)]
                    else:
                        spins0 = prob.random_feasible(a.seed, reps, 0)
                        kw = dict(replica_id=np.arange(reps, dtype=np.int64),
                                  instance_id=np.zeros(reps, dtype=np.int64), is_feasible=prob.is_feasible)
                        st = be.init_state(plan, binding, spins0, targets=target, **kw)
                        res = be.run(plan, st, S.fn_endpoints(ts, te, 0.1, 2 ** lb), seed=a.seed,
                                     stop_on_target=False)
                        P = float((np.asarray(res.state.counters["target_hit_attempt"]) >= 0).mean())
                        dup = None
                        done[(key, lb)] = (P, v["id"])
                    rows.append(dict(instance=m["name"], role=a.role, cls=cls, lib=lib, k=k, seed=seed,
                                     variant=v["id"], kappa=v["kappa"], m=v["m"], budget_log2=lb, replicas=reps, P_s=P,
                                     dup_of=dup, tau_start_raw=ts0, tau_end_raw=te0, tau_start=ts, tau_end=te,
                                     F_opt=F))
            print(f"{m['name']} {cls} {lib} seed={seed}: {len(variants[cls])} settings x {len(buds)} budgets, "
                  f"{time.time() - t00:.0f} s", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))


# ------------------------------------------------------------------------------------------------ ITS helpers
def load(dirs):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "*.json"))):
            rows += json.loads(Path(f).read_text())
    return rows


def matrix(rows):
    """Rows of one class and setting -> ``(instances, budgets, T[instance, budget])`` of iterations to solution;
    random shortcuts: median over the draws of each instance first."""
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["instance"]][r["budget_log2"]].append(C.tts_at(2.0 ** r["budget_log2"], r["P_s"]))
    insts = sorted(per)
    buds = sorted(set.intersection(*[set(per[i]) for i in insts]))
    T = np.array([[float(np.median(per[i][b])) for b in buds] for i in insts])
    return insts, buds, T


def optimal(T, buds, boot=0, seed=20261023):
    """ITS = minimum over budgets of the median over instances; percentile bootstrap CI over instances."""
    med = np.median(T, axis=0)
    j = int(np.argmin(med))
    o = dict(its=float(med[j]), budget_log2=buds[j], j=j, medians=med, lo=math.nan, hi=math.nan)
    if boot:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, T.shape[0], size=(boot, T.shape[0]))
        mins = np.median(T[idx], axis=1).min(axis=1)
        o["lo"], o["hi"] = (float(x) for x in np.percentile(mins, [2.5, 97.5]))
    return o


# ------------------------------------------------------------------------------------------------ choose
def cmd_choose(a):
    import pandas as pd

    rows = [r for r in load(a.tune) if r["role"] == "tuning"]
    n_inst = len({r["instance"] for r in rows})
    grid, choice = [], {}
    for cls in CLASSES:
        rc = [r for r in rows if r["cls"] == cls]
        if not rc:
            continue
        by_v = defaultdict(list)
        for r in rc:
            by_v[r["variant"]].append(r)
        res = []
        for v, rs in by_v.items():
            insts, buds, T = matrix(rs)
            if len(insts) < n_inst:
                continue
            o = optimal(T, buds, boot=2000)
            col = T[:, o["j"]]
            res.append(dict(cls=cls, setting=v, kappa=rs[0]["kappa"], m=rs[0]["m"], instances=len(insts),
                            its=o["its"], ci_lo=o["lo"], ci_hi=o["hi"], budget_log2=o["budget_log2"],
                            budgets=f"{buds[0]}..{buds[-1]}",
                            mean_log2_at_b=float(np.mean([math.log2(x) if np.isfinite(x) else 64.0 for x in col])),
                            solved_at_b=int(np.isfinite(col).sum())))
        res.sort(key=lambda x: (x["its"], x["mean_log2_at_b"],
                                abs(math.log2(x["kappa"] / 2)) + abs(math.log2(x["m"]))))
        auto = next((x for x in res if x["setting"] == AUTO["id"]), None)
        for i, x in enumerate(res):
            x["rank"] = i + 1
            x["vs_auto"] = x["its"] / auto["its"] if auto and np.isfinite(auto["its"]) else math.nan
        grid += res
        b = res[0]
        ms_ = sorted({x["m"] for x in res})
        ks_ = sorted({x["kappa"] for x in res})
        bmin, bmax = (int(s) for s in b["budgets"].split(".."))
        edge = [e for e, c in (("m min", b["m"] == ms_[0]), ("m max", b["m"] == ms_[-1]),
                               ("kappa max", b["kappa"] == ks_[-1]), ("budget min", b["budget_log2"] == bmin),
                               ("budget max", b["budget_log2"] == bmax)) if c]
        choice[cls] = dict(variant=b["setting"], kappa=b["kappa"], m=b["m"], its=b["its"], ci=[b["ci_lo"], b["ci_hi"]],
                           budget_log2=b["budget_log2"], edge=edge, runner_up=res[1]["setting"] if len(res) > 1 else None,
                           runner_up_ratio=res[1]["its"] / b["its"] if len(res) > 1 else None,
                           settings_ranked=len(res))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid).to_csv(out / "tuning_grid.csv", index=False)
    (out / "choice.json").write_text(json.dumps(choice, indent=1))
    for c, x in choice.items():
        print(f"{c:9s} {x['variant']:12s} ITS {x['its']:.3g} at 2^{x['budget_log2']}  edge {x['edge']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Iterations to solution of the move libraries at L = 10.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--role", choices=["tuning", "test"], required=True)
    r.add_argument("--settings", default="auto", help="auto (m = 1, kappa = 2) or a choice.json (test only)")
    r.add_argument("--task", type=int, default=0)
    r.add_argument("--inst", default=None, help="staged instance directory (default: $NIMBUS_MATCHING_INST)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--budgets", type=int, nargs="+")
    r.add_argument("--variants", nargs="+", help="tuning grid override: kappa,m ...")
    r.add_argument("--classes", nargs="+")
    r.add_argument("--replicas", type=int, default=0)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 6)))
    r.add_argument("--force", action="store_true")
    c = sub.add_parser("choose")
    c.add_argument("--tune", nargs="+", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"run": cmd_run, "choose": cmd_choose}[a.cmd](a)


if __name__ == "__main__":
    main()

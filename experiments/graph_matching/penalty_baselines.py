"""Penalty-machine baselines on graph matching: N-HOIM with penalty and NeuroSA with quadratized penalty.

The machines are :class:`nimbus.baselines.penalty_machines.PenaltyMachine` (``kind`` ``nhoim`` or ``neurosa``). Settings are
given in units of ``d0 = Psi* / L`` (``Psi*`` the certified optimum, the ground-truth objective where no optimum is
certified): penalty strength ``lam = lam_c * d0``, end temperature ``Tend_d0 * d0`` and start temperature
``ratio * Tend_d0 * d0``; ``mult`` is the quadratization strength multiplier (NeuroSA only).
Budgets are in iterations (single-spin update tests).

  # setting selection on training instances (one call per instance; all grid points, 32 replicas)
  python penalty_baselines.py tune --kind nhoim --names syn_n10_s0.05_o0_00 --budgets 14 16 18 20 22 24 \
      --replicas 32 --seed 11 --inst INST --out RUNS/btune/tune_nhoim_synthetic_0.jsonl
  python penalty_baselines.py tune --kind neurosa --names syn_n10_s0.05_o0_00 ... \
      --grid configs/willow_sym/penalty_grid_ratio_high.json --out RUNS/btune/tune_neurosa_synthetic_0_rhi.jsonl
  # test instances with the selected setting
  python penalty_baselines.py test --kind nhoim --choice configs/willow_sym/baseline_choice_willow.json \
      --stage willow_test --names @willow:test:0 @willow:test:1 ... --replicas 256 --seed 11 --inst INST --out OUT.jsonl

One JSON line per (instance, setting, budget) with the per-replica best feasible objective ``bestF`` (null if the replica
never visited a feasible state), the success fractions ``p098`` (``Psi >= 0.98 F_ref``) and ``p1`` (``Psi >= F_ref``), the
valid fractions and the iterations actually spent (``tests``). Instance names may be ``@willow:<role>:<i>`` aliases (the
i-th WILLOW pair of that role); the record keeps the name as given.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

from nimbus.baselines.penalty_machines import PenaltyMachine  # noqa: E402

#: setting grid of the selection (``--grid`` JSON entries override single axes)
GRIDS = {
    "lam_c": [0.5, 1, 2, 4, 8, 16, 32],
    "Tend_d0": [0.1, 0.3, 1.0, 3.0],
    "ratio": [10, 30, 100],
    "mult": [1.0, 2.0, 4.0],
}
PAR = ("lam_c", "mult", "Tend_d0", "ratio", "Tstart_d0")


def f_unit(m):
    return m["F_opt"] if m.get("F_opt") is not None else m["F_gt"]


def records(machine, m, name, F_ref, par, budgets, replicas, seed, threads, extra):
    d0 = f_unit(m) / machine.n
    out = []
    for lb in budgets:
        r = machine.run(2 ** lb, par["Tstart_d0"] * d0, par["Tend_d0"] * d0, replicas, seed, threads)
        F = r["bestF"]
        ok = np.isfinite(F)
        Fz = np.where(ok, F, 0.0)
        out.append(dict(extra, kind=machine.kind, instance=name, n=machine.n, budget_log2=lb, replicas=replicas, F_ref=F_ref,
                        valid_best_frac=float(ok.mean()), valid_final_frac=float(r["final_feasible"].mean()),
                        score_mean=float(np.mean(Fz) / F_ref), score_best=float(np.max(Fz) / F_ref),
                        p098=float(np.mean(ok & (Fz >= 0.98 * F_ref))), p1=float(np.mean(ok & (Fz >= F_ref * (1 - 1e-9)))),
                        n_aux=machine.n_aux, ntot=machine.ntot, ticks=r["ticks"], tests=r["tests"],
                        accepted_mean=float(r["accepted"].mean()), drift=r["drift"], wall_s=r["wall_s"],
                        bestF=[None if not np.isfinite(x) else float(x) for x in F], **par))
    return out


def _ref(m, name, table):
    return f_unit(m) if m.get("F_opt") is not None else table.get(m["name"], float("nan"))


def cmd_tune(a):
    grids = dict(GRIDS)
    if a.grid:
        grids.update(json.loads(a.grid if a.grid.lstrip().startswith("{") else Path(a.grid).read_text()))
    mults = grids["mult"] if a.kind == "neurosa" else [1.0]
    table = C.psi_ref_table(a.psi_ref) if a.psi_ref else C.psi_ref_table()
    inst = C.inst_dir(a.inst)
    with open(a.out, "a") as out:
        for name in a.names:
            m, t, prob = C.load_instance(inst, name)
            if m["role"] != "tuning":
                raise SystemExit(f"{name}: settings are selected on training instances only")
            F_ref = _ref(m, name, table)
            d0 = f_unit(m) / int(prob.P)
            for lam_c, mult in itertools.product(grids["lam_c"], mults):
                machine = PenaltyMachine(prob, a.kind, lam_c * d0, mult)
                for te, rt in itertools.product(grids["Tend_d0"], grids["ratio"]):
                    par = dict(lam_c=lam_c, mult=mult, Tend_d0=te, ratio=rt, Tstart_d0=te * rt)
                    for rec in records(machine, m, name, F_ref, par, a.budgets, a.replicas, a.seed, a.threads,
                                       {"stage": "tuning"}):
                        out.write(json.dumps(rec) + "\n")
                    out.flush()
            print(name, "done", flush=True)


def cmd_test(a):
    choice = json.loads(Path(a.choice).read_text())[a.kind][str(a.n)]
    table = C.psi_ref_table(a.psi_ref) if a.psi_ref else C.psi_ref_table()
    inst = C.inst_dir(a.inst)
    with open(a.out, "a") as out:
        for name in a.names:
            m, t, prob = C.load_instance(inst, name)
            F_ref = _ref(m, name, table)
            d0 = f_unit(m) / int(prob.P)
            machine = PenaltyMachine(prob, a.kind, choice["lam_c"] * d0, choice["mult"])
            par = {k: choice[k] for k in PAR}
            for rec in records(machine, m, name, F_ref, par, a.budgets, a.replicas, a.seed, a.threads, {"stage": a.stage}):
                out.write(json.dumps(rec) + "\n")
            out.flush()
            print(name, "done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Penalty-machine baselines on graph matching.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("tune", "test"):
        p = sub.add_parser(nm)
        p.add_argument("--kind", choices=["nhoim", "neurosa"], required=True)
        p.add_argument("--names", nargs="+", required=True, help="instance names or @willow:<role>:<i> aliases")
        p.add_argument("--seed", type=int, default=11)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--inst", default=os.environ.get("NIMBUS_MATCHING_INST"))
        p.add_argument("--psi-ref", default=None, help="Psi_ref table (default configs/psi_ref.csv); used where no "
                                                       "certified optimum exists")
        p.add_argument("--out", required=True)
    t = sub.choices["tune"]
    t.add_argument("--budgets", type=int, nargs="+", default=[14, 16, 18, 20, 22, 24])
    t.add_argument("--replicas", type=int, default=32)
    t.add_argument("--grid", default=None, help="JSON (file or inline) overriding axes of the setting grid")
    e = sub.choices["test"]
    e.add_argument("--choice", required=True, help="selected settings, e.g. configs/willow_sym/baseline_choice_willow.json")
    e.add_argument("--n", type=int, default=10, help="size key in the choice file")
    e.add_argument("--stage", default="test")
    e.add_argument("--budgets", type=int, nargs="+", default=[14, 16, 18, 20, 22, 24])
    e.add_argument("--replicas", type=int, default=256)
    a = ap.parse_args(argv)
    {"tune": cmd_tune, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()

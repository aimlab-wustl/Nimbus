"""Shared helpers of the graph-matching experiment scripts.

* instance access: :func:`load_instance` reads one instance of the staged set (``manifest.json`` + tensors written by
  ``stage_instances.py``); the directory comes from ``--inst`` or the environment variable ``NIMBUS_MATCHING_INST``.
* result shards: one task writes one ``shard_<task_id>.npz`` (JSON records plus per-replica arrays), atomically, so a
  rerun of an interrupted job skips finished tasks.
* :func:`anneal` runs the replicas of one (instance, library, temperatures, budget) exactly as in the paper runs.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from nimbus.annealing import schedule as S

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
ETA = 0.1


# ------------------------------------------------------------------------------------------------ instances
def inst_dir(path=None) -> Path:
    p = path or os.environ.get("NIMBUS_MATCHING_INST")
    if not p:
        raise SystemExit("set --inst or NIMBUS_MATCHING_INST to the staged instance directory (stage_instances.py)")
    return Path(p)


def manifest(inst) -> list:
    return json.loads((Path(inst) / "manifest.json").read_text())


def resolve_name(mlist, name: str) -> str:
    """``@willow:<role>:<i>`` is the i-th WILLOW pair of that role in manifest order; other names are returned as is."""
    if name.startswith("@willow:"):
        _, role, i = name.split(":")
        return [x for x in mlist if x["family"] == "willow" and x["role"] == role][int(i)]["name"]
    return name


def load_instance(inst, name: str):
    """``(manifest entry, MatchingTensor, MatchingProblem)`` of one staged instance."""
    from nimbus.problems.formats.matching_tensor import load_tensor
    from nimbus.problems.matching import MatchingProblem

    mlist = manifest(inst)
    name = resolve_name(mlist, name)
    m = {x["name"]: x for x in mlist}[name]
    t = load_tensor(Path(inst) / m["tensor"])
    return m, t, MatchingProblem(t, name=name)


def psi_ref_table(path=None) -> dict:
    """Instance name -> Psi_ref (best objective found by the three domain solvers; certified optimum at L = 10)."""
    path = Path(path) if path else CONFIGS / "psi_ref.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["instance"]: float(r["psi_ref"]) for r in csv.DictReader(fh)}


def f_target(m: dict, ref: dict | None):
    """Hit target of an instance: certified optimum if known, else Psi_ref (None if neither)."""
    if m.get("F_opt") is not None:
        return m["F_opt"], "certified"
    if ref and m["name"] in ref:
        return ref[m["name"]], "reference"
    return None, "none"


# ------------------------------------------------------------------------------------------------ shards
def shard_path(out, task_id) -> Path:
    return Path(out) / f"shard_{int(task_id):05d}.npz"


def write_shard(path, records, arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, records=np.array(json.dumps(records)), **arrays)
    os.replace(tmp, path)


def read_shard(path):
    """``(records, arrays)`` of one shard."""
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z["records"])), {k: z[k] for k in z.files if k != "records"}


def read_tasks(path) -> list:
    """Task table (CSV: task_id, fig, kind, backend, est_s, params_json) -> list of dicts."""
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(task_id=int(r["task_id"]), fig=r["fig"], kind=r["kind"], backend=r["backend"],
                     est_s=float(r["est_s"]), params=json.loads(r["params_json"])) for r in csv.DictReader(fh)]


# ------------------------------------------------------------------------------------------------ annealing
def compile_instance(prob, tensor, lib_spec: dict):
    """``(model, binding, info)`` for one instance and library specification (see ``shortcut_library``)."""
    from nimbus.compiler import compile_model
    from nimbus.core.model import bind_weights
    from nimbus.problems.matching import shortcut_library

    library, info = shortcut_library(prob, lib_spec, tensor)
    model = compile_model(prob.structure(), library, identities=tuple(prob.identities()))
    binding = bind_weights(model, prob.term_weights(0), prob.const(0))
    return model, binding, info


def prepare_plan(be, backend: str, model, threads: int, process: str = "select"):
    if backend == "gpu":
        return be.prepare(model, strategy="auto", layout="auto", process=process)
    return be.prepare(model, threads=threads, strategy="auto", layout="auto", process=process)


def anneal(be, plan, binding, prob, *, tau_start, tau_end, budget_log2, replicas, seed, F_target=None):
    """Run ``replicas`` replicas for ``2**budget_log2`` iterations each on the calibrated FN schedule.

    Start states: ``prob.random_feasible(seed, replicas, 0)``; replica ``r`` has replica id ``r``. A short warm-up run
    (at most 64 iterations, discarded) precedes the run, as in the paper runs. Returns the final backend state
    (``best_energy``, ``best_spins``, counters ``target_hit_attempt``, ``attempts``, ``accepted_changed``, ...).
    """
    B = 2 ** int(budget_log2)
    spins0 = prob.random_feasible(seed, replicas, 0)
    target = (np.full(replicas, -F_target + 1e-9 * abs(F_target)) if F_target is not None
              else np.full(replicas, -1e18))
    kw = dict(replica_id=np.arange(replicas, dtype=np.int64), instance_id=np.zeros(replicas, dtype=np.int64),
              is_feasible=prob.is_feasible)
    warm = be.init_state(plan, binding, spins0, **kw)
    be.run(plan, warm, S.fn_endpoints(tau_start, tau_end, ETA, min(64, B)), seed=seed)
    st = be.init_state(plan, binding, spins0, targets=target, **kw)
    res = be.run(plan, st, S.fn_endpoints(tau_start, tau_end, ETA, B), seed=seed, stop_on_target=False)
    return res.state


def best_perms(state, P: int) -> np.ndarray:
    """Best assignment of every replica: ``int8[replicas, P]`` (``perm[i]`` = target of source ``i``)."""
    reps = np.asarray(state.best_spins).shape[0]
    return np.asarray(state.best_spins).reshape(reps, P, P).argmax(axis=2).astype(np.int8)


# ------------------------------------------------------------------------------------------------ statistics
def tts_at(t, p):
    """Iterations to solution at budget ``t`` with per-replica success probability ``p``:
    ``t ln(0.01) / ln(1 - p)`` (``p = 1`` -> ``t``; ``p = 0`` -> inf)."""
    import math

    return float(t) if p >= 1.0 else (float(t) * math.log(0.01) / math.log(1 - p) if p > 0 else np.inf)


def optimal_median_its(per: dict, boot: int = 0, seed: int = 20261015) -> dict:
    """Minimum over budgets of the median over instances of the iterations to solution.

    ``per``: instance -> {budget_log2: (t, p)}. One budget per method and problem set (not a per-instance minimum).
    95% CI: percentile bootstrap over instances (the same resampled set at every budget, then the minimum), ``boot``
    draws. Returns dict(opt, budget_log2_star, medians, lo, hi, solved_at_star, instances).
    """
    import math

    insts = sorted(per)
    buds = sorted(set.intersection(*[set(per[i]) for i in insts]))
    T = np.array([[tts_at(*per[i][b]) for b in buds] for i in insts])
    med = np.median(T, axis=0)
    j = int(np.argmin(med))
    out = dict(opt=float(med[j]), budget_log2_star=buds[j], medians={b: float(x) for b, x in zip(buds, med)},
               solved_at_star=int(np.isfinite(T[:, j]).sum()), instances=T.shape[0], lo=float("nan"), hi=float("nan"))
    if boot:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, T.shape[0], size=(boot, T.shape[0]))
        mins = np.median(T[idx], axis=1).min(axis=1)
        with np.errstate(invalid="ignore"):
            lo, hi = np.percentile(mins, [2.5, 97.5], method="inverted_cdf")
        out["lo"] = float(lo) if math.isfinite(lo) else float("inf")
        out["hi"] = float(hi) if math.isfinite(hi) else float("inf")
    return out


def bootstrap_mean_ci(x, boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap CI of the mean over the first axis (instances)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(boot, len(x)))
    m = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))

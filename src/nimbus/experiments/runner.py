"""Experiment runner: problem -> compile -> bind -> init -> calibrate -> anneal ->
score, with chunked checkpointing and resume.

The pipeline treats "replicas" and "instances" uniformly for one run: each selected instance gets
its own :class:`~nimbus.core.model.WeightBinding` (same compiled topology, different weights) and
its own batch of ``run.replicas`` replicas, run to completion (its attempt budget) before the next
instance starts. This keeps one process's in-memory backend state simple (one native state object
at a time) while still fitting the multi-instance binding model; the checkpoint records
which instances are already finished so a resumed run skips them exactly once.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nimbus.backends import get_backend
from nimbus.compiler import compile_model
from nimbus.core.errors import BackendUnavailable, ConfigError
from nimbus.core.model import bind_weights
from nimbus.experiments import checkpoint as ckpt
from nimbus.experiments import manifest as manifest_mod
from nimbus.experiments.calibrate_driver import calibrate_for_policy
from nimbus.experiments.config import RunConfig, process_spec_of
from nimbus.experiments.results import ReplicaRow, write_shard
from nimbus.problems.factory import make_problem

__all__ = ["STATUS_VALUES", "resolve_backend_name", "resolve_instances", "SolveOutcome", "compile_problem",
           "solve", "resume"]

STATUS_VALUES = ("ok", "target_hit", "budget_exhausted", "cancelled", "resource_limit", "unsupported", "failed")

_HOST_BACKEND_ORDER = ("cpu", "emulation")  # tried in order for backend.name == "auto"


def resolve_backend_name(name: str) -> str:
    """Availability policy for ``backend.name: auto`` (never substitutes a *different explicit* request).

    Tries ``cpu`` (optimized host backend) first, then falls back to ``emulation`` (always
    available); ``oracle`` is validation-only and is never chosen automatically; ``gpu`` is never
    chosen automatically (it must be requested explicitly, since it changes hardware requirements).
    """
    if name != "auto":
        return name
    for candidate in _HOST_BACKEND_ORDER:
        try:
            get_backend(candidate)
            return candidate
        except BackendUnavailable:
            continue
    return "emulation"


def resolve_instances(spec, n_instances: int) -> list[int]:
    """``run.instances`` (already normalized by config.py) -> a concrete, ordered instance-id list."""
    n = int(n_instances)
    if spec == "all":
        return list(range(n))
    if isinstance(spec, dict):
        if "first" in spec:
            return list(range(min(int(spec["first"]), n)))
        a, b = int(spec.get("start", 0)), int(spec["stop"])
        return list(range(a, min(b, n)))
    ids = [int(x) for x in spec]
    bad = [i for i in ids if i < 0 or i >= n]
    if bad:
        raise ConfigError(f"run.instances: id(s) {bad} out of range for {n} instance(s)")
    return ids


def _target_array(targets, reps: int, position: int) -> np.ndarray:
    out = np.full(reps, np.nan)
    if targets is None:
        return out
    if isinstance(targets, dict):
        if "native" in targets:
            raise ConfigError("run.targets: {native: ...} needs a problem-specific energy conversion; "
                               "use {energy: ...}")
        targets = targets["energy"]
    if targets == "reference":
        raise ConfigError("run.targets: 'reference' needs a per-problem reference value (not implemented)")
    if isinstance(targets, (list, tuple)):
        t = targets[position] if position < len(targets) else None
    else:
        t = targets
    if t is not None:
        out[:] = float(t)
    return out


@dataclass
class SolveOutcome:
    status: str
    run_dir: str | None
    manifest: dict
    rows: list
    timings: dict
    message: str | None = None


def compile_problem(cfg: RunConfig):
    """``(problem, model, timings)`` for ``cfg`` (no weights bound, no backend involved)."""
    timings = {}
    t0 = time.perf_counter()
    problem = make_problem(cfg.data["problem"], Path(cfg.base_dir))
    tr = cfg.data["transitions"]
    library = problem.transition_library(tr["library"], **tr["options"])
    timings["problem_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    readout = cfg.data["compile"]["readout"]
    readout = None if readout in (None, "default") else readout
    model = compile_model(problem.structure(), library, readout=readout,
                          identities=tuple(problem.identities()), limits=cfg.compile_limits(),
                          options=cfg.compile_options())
    timings["compile_s"] = time.perf_counter() - t0
    return problem, model, timings


def _run_dir_for(cfg: RunConfig, output_dir=None) -> Path:
    from nimbus.experiments.paths import runs_root

    if output_dir is not None:
        return Path(output_dir)
    out = cfg.data["output"]["path"]
    if out is not None:
        p = Path(out)
        return p if p.is_absolute() else runs_root() / p
    h8 = cfg.full_hash()[:8]
    name = cfg.name
    return runs_root() / f"{time.strftime('%Y-%m-%d')}_{name}_{h8}"


def solve(source, overrides=None, *, output_dir=None, progress=None) -> SolveOutcome:
    """Load+validate ``source`` (path/mapping/RunConfig), then run it end to end.

    ``progress(event: str, info: dict)`` is called at chunk boundaries and instance boundaries
    (compact progress; the CLI uses this for its ``--quiet``-respecting status line).
    """
    from nimbus.experiments.config import load_config

    cfg = load_config(source, overrides)
    run_dir = _run_dir_for(cfg, output_dir)
    out_cfg = cfg.data["output"]
    if run_dir.exists() and not out_cfg["overwrite"] and any(run_dir.iterdir()) and \
            not (ckpt.checkpoint_path(run_dir).exists()):
        raise ConfigError(f"output directory {run_dir} is not empty (set output.overwrite or resume it)")
    run_dir.mkdir(parents=True, exist_ok=True)
    return _solve_into(cfg, run_dir, resume_state=None, progress=progress)


def resume(run_dir, *, progress=None) -> SolveOutcome:
    """Resume an interrupted run from ``run_dir`` (must contain ``checkpoint.npz``)."""
    from nimbus.experiments.config import RunConfig

    run_dir = Path(run_dir)
    cfg = RunConfig.from_resolved(run_dir / "config.resolved.yaml")
    meta, state, recording = ckpt.load_checkpoint(run_dir)
    if meta.trajectory_hash != cfg.trajectory_hash():
        raise ConfigError(f"{run_dir}: checkpoint and config.resolved.yaml disagree on the trajectory hash")
    return _solve_into(cfg, run_dir, resume_state=(meta, state, recording), progress=progress)


def _notify(progress, event, **info):
    if progress is not None:
        progress(event, info)


def _schedule_or_unsupported(cfg: RunConfig, run_dir: Path, run_id: str, kind_label: str):
    """Shared "no calibrated schedules yet for baseline runs" gate (penalty baseline's own check,
    generalized): returns ``(policy, schedule, None)`` on success, or ``(policy, None, outcome)``
    where ``outcome`` is an already-written "unsupported" :class:`SolveOutcome` to return as-is."""
    policy = cfg.policy()
    if policy.needs_calibration:
        manifest = manifest_mod.build_manifest(cfg, status="unsupported", run_id=run_id,
                                               message=f"baseline {kind_label!r} needs an explicit "
                                                       "annealing.params schedule (auto/calibrated schedules "
                                                       "are not wired for baseline runs yet)")
        manifest_mod.write_manifest(run_dir, manifest)
        return policy, None, SolveOutcome("unsupported", str(run_dir), manifest, [], {}, manifest["message"])
    schedule = policy.resolve_schedule(n_total=policy.budget.attempts)
    return policy, schedule, None


def _solve_quadratic_baseline(cfg: RunConfig, run_dir: Path, *, progress) -> SolveOutcome:
    """``baseline.kind: quadratic``: the problem's penalty formulation, degree-reduced to quadratic
    by :func:`nimbus.baselines.quadratize.quadratize_penalty`, then run exactly like the ``penalty``
    baseline. ``baseline.options`` are the problem's own penalty-strength keyword arguments (e.g.
    ``lam``, ``lambda_r``/``lambda_c``), plus an optional ``quad_strength`` (the Rosenberg ``M``;
    omit for the module's own per-substitution default)."""
    from nimbus.baselines.penalty import run_penalty
    from nimbus.baselines.quadratize import quadratize_penalty

    t_total0 = time.perf_counter()
    run_id = cfg.data.get("name") or run_dir.name
    timings: dict[str, Any] = {}
    problem = make_problem(cfg.data["problem"], Path(cfg.base_dir))
    policy, schedule, early = _schedule_or_unsupported(cfg, run_dir, run_id, "quadratic")
    if early is not None:
        return early
    run_cfg, baseline_cfg = cfg.data["run"], cfg.data["baseline"]
    backend_name = resolve_backend_name(cfg.data["backend"]["name"])
    instances = resolve_instances(run_cfg["instances"], problem.n_instances)
    opts = dict(baseline_cfg["options"])
    quad_strength = opts.pop("quad_strength", None)
    rows: list[ReplicaRow] = []
    for pos, iid in enumerate(instances):
        pf = problem.penalty_formulation(iid, **opts)
        qpf = quadratize_penalty(pf, strength=quad_strength)
        result = run_penalty(qpf, backend=backend_name, schedule=schedule, seed=run_cfg["seed"],
                             n_replicas=run_cfg["replicas"], stop_on_target=run_cfg["stop_on_target"])
        counters = {}
        try:
            counters = result.run_result.state.counters
        except AttributeError:
            pass
        for rep in range(len(result.feasible)):
            rows.append(ReplicaRow(
                run_id=run_id, instance_id=int(iid), replica_id=rep, replica_position=rep, seed=run_cfg["seed"],
                best_feasible_objective=(float(result.best_feasible_objective[rep])
                                         if result.feasible[rep] else None),
                best_feasible_attempt=int(result.best_feasible_attempt[rep]) if result.feasible[rep] else None,
                final_energy=float(result.final_penalized_energy[rep]),
                final_feasible=bool(result.final_feasible[rep]),
                attempts=int(counters["attempts"][rep]) if "attempts" in counters else None,
                accepted=int(counters["accepted"][rep]) if "accepted" in counters else None,
                termination=str(result.status[rep]),
                timings_json=json.dumps({"baseline": "quadratic", "n_aux": result.n_aux}),
            ))
        _notify(progress, "instance", instance=iid, position=pos, total=len(instances))
    write_shard(run_dir / "results.parquet", rows)
    timings["total_s"] = time.perf_counter() - t_total0
    manifest = manifest_mod.build_manifest(cfg, problem=problem, policy=policy, timings=timings, status="ok",
                                           run_id=run_id, instances=[{"instance_id": i} for i in instances],
                                           extra={"baseline": "quadratic"})
    manifest_mod.write_manifest(run_dir, manifest)
    return SolveOutcome("ok", str(run_dir), manifest, rows, timings)


_BASELINE_SOLVERS = {
    "quadratic": _solve_quadratic_baseline,
}


def _solve_penalty_baseline(cfg: RunConfig, run_dir: Path, *, progress) -> SolveOutcome:
    """``baseline.kind: penalty`` path: runs through
    :func:`nimbus.baselines.penalty.run_penalty` instead of the ordinary compiled-objective
    pipeline. Needs an explicit schedule (auto/calibrated schedules are not wired for this baseline
    path yet;."""
    from nimbus.baselines.penalty import run_penalty

    t_total0 = time.perf_counter()
    run_id = cfg.data.get("name") or run_dir.name
    timings: dict[str, Any] = {}
    problem = make_problem(cfg.data["problem"], Path(cfg.base_dir))
    policy = cfg.policy()
    if policy.needs_calibration:
        manifest = manifest_mod.build_manifest(cfg, status="unsupported", run_id=run_id,
                                               message="baseline 'penalty' needs an explicit annealing.params "
                                                       "schedule (auto/calibrated schedules are not wired for "
                                                       "baseline runs yet)")
        manifest_mod.write_manifest(run_dir, manifest)
        return SolveOutcome("unsupported", str(run_dir), manifest, [], timings, manifest["message"])
    schedule = policy.resolve_schedule(n_total=policy.budget.attempts)
    run_cfg, baseline_cfg = cfg.data["run"], cfg.data["baseline"]
    backend_name = resolve_backend_name(cfg.data["backend"]["name"])
    instances = resolve_instances(run_cfg["instances"], problem.n_instances)
    rows: list[ReplicaRow] = []
    for pos, iid in enumerate(instances):
        result = run_penalty(problem, instance=iid, strengths=baseline_cfg["options"], backend=backend_name,
                             schedule=schedule, seed=run_cfg["seed"], n_replicas=run_cfg["replicas"],
                             stop_on_target=run_cfg["stop_on_target"])
        counters = {}
        try:
            counters = result.run_result.state.counters
        except AttributeError:
            pass
        for rep in range(len(result.feasible)):
            rows.append(ReplicaRow(
                run_id=run_id, instance_id=int(iid), replica_id=rep, replica_position=rep, seed=run_cfg["seed"],
                best_feasible_objective=(float(result.best_feasible_objective[rep])
                                         if result.feasible[rep] else None),
                best_feasible_attempt=int(result.best_feasible_attempt[rep]) if result.feasible[rep] else None,
                final_energy=float(result.final_penalized_energy[rep]),
                final_feasible=bool(result.final_feasible[rep]),
                attempts=int(counters["attempts"][rep]) if "attempts" in counters else None,
                accepted=int(counters["accepted"][rep]) if "accepted" in counters else None,
                termination=str(result.status[rep]),
                timings_json=json.dumps({"baseline": "penalty"}),
            ))
        _notify(progress, "instance", instance=iid, position=pos, total=len(instances))
    write_shard(run_dir / "results.parquet", rows)
    timings["total_s"] = time.perf_counter() - t_total0
    manifest = manifest_mod.build_manifest(cfg, problem=problem, policy=policy, timings=timings, status="ok",
                                           run_id=run_id, instances=[{"instance_id": i} for i in instances],
                                           extra={"baseline": "penalty"})
    manifest_mod.write_manifest(run_dir, manifest)
    return SolveOutcome("ok", str(run_dir), manifest, rows, timings)


def _solve_into(cfg: RunConfig, run_dir: Path, *, resume_state, progress) -> SolveOutcome:
    t_total0 = time.perf_counter()
    run_id = cfg.data.get("name") or run_dir.name
    cfg.write_resolved(run_dir)
    timings: dict[str, Any] = {}
    baseline_kind = cfg.data["baseline"]["kind"]
    if baseline_kind == "penalty":
        return _solve_penalty_baseline(cfg, run_dir, progress=progress)
    if baseline_kind in _BASELINE_SOLVERS:
        try:
            return _BASELINE_SOLVERS[baseline_kind](cfg, run_dir, progress=progress)
        except ConfigError as exc:  # bad baseline/problem/config combination: a "failed" manifest,
            manifest = manifest_mod.build_manifest(cfg, status="failed", message=str(exc), run_id=run_id)
            manifest_mod.write_manifest(run_dir, manifest)
            return SolveOutcome("failed", str(run_dir), manifest, [], timings, str(exc))
    if baseline_kind not in ("none",):
        manifest = manifest_mod.build_manifest(cfg, status="unsupported", run_id=run_id,
                                               message=f"baseline kind {baseline_kind!r} is not "
                                                       "supported by the runner")
        manifest_mod.write_manifest(run_dir, manifest)
        return SolveOutcome("unsupported", str(run_dir), manifest, [], timings, manifest["message"])
    try:
        problem, model, t_compile = compile_problem(cfg)
        timings.update(t_compile)
    except Exception as exc:  # compile/problem failure: still write a manifest, status failed
        manifest = manifest_mod.build_manifest(cfg, status="failed", message=str(exc), run_id=run_id)
        manifest_mod.write_manifest(run_dir, manifest)
        return SolveOutcome("failed", str(run_dir), manifest, [], timings, str(exc))

    run_cfg = cfg.data["run"]
    backend_cfg = cfg.data["backend"]
    backend_name = resolve_backend_name(backend_cfg["name"])
    backend = get_backend(backend_name)
    proc = process_spec_of(cfg.data["annealing"])
    proc_kw = {} if proc.is_select else {"process": proc.name, "process_options": proc.options()}
    plan = backend.prepare(model, strategy=backend_cfg["strategy"], layout=backend_cfg["layout"],
                           precision=backend_cfg["precision"], device=backend_cfg["device"],
                           **proc_kw, **backend_cfg["options"])

    policy = cfg.policy()
    reps = run_cfg["replicas"]
    seed = run_cfg["seed"]
    instances = resolve_instances(run_cfg["instances"], problem.n_instances)

    finished: list[int] = []
    rows: list[ReplicaRow] = []
    instance_meta: list[dict] = []
    all_bindings: list[tuple] = []
    start_pos = 0
    resume_recording = None
    resume_start_attempt = 0

    if resume_state is not None:
        meta, exported, resume_recording = resume_state
        finished = list(meta.finished_instances)
        start_pos = meta.instance_index
        resume_start_attempt = meta.chunk_attempts_done
        # already-finished instances' rows are on disk (results.parquet); reload if present for the
        # final consolidated write (a resumed run rewrites the shard once, at the end).
        shard = run_dir / "results.parquet"
        if shard.exists():
            from nimbus.experiments.results import read_shard

            rows = [ReplicaRow(**{k: v for k, v in r.items() if k != "schema"}) for r in read_shard(shard)]

    final_status = "ok"
    final_message = None

    for pos in range(start_pos, len(instances)):
        iid = instances[pos]
        weights = problem.term_weights(iid)
        const = problem.const(iid)
        binding = bind_weights(model, weights, const)
        all_bindings.append((iid, binding))

        if pos == start_pos and resume_state is not None:
            native_state = backend.load_state(plan, exported, bindings=(binding,))
        else:
            spins0 = problem.random_feasible(seed, reps, iid)
            targets = _target_array(run_cfg["targets"], reps, pos)
            native_state = backend.init_state(
                plan, binding, spins0,
                replica_id=np.arange(reps, dtype=np.int64) + run_cfg["replica_offset"],
                instance_id=np.full(reps, iid, dtype=np.int64),
                is_feasible=problem.is_feasible, targets=targets)

        calibration_info = None
        if policy.needs_calibration:
            t0 = time.perf_counter()
            pilot = calibrate_for_policy(policy, backend, plan, native_state, model.n_groups, seed,
                                        round_offset=iid, binding=binding,
                                        late_advance=cfg.data["annealing"]["calibration"]["late_advance"],
                                        late_advance_attempts=cfg.data["annealing"]["calibration"]["late_advance_attempts"])
            schedule = policy.resolve_schedule(pilot.result, n_total=policy.budget.attempts)
            timings[f"calibration_s.instance{iid}"] = time.perf_counter() - t0
            calibration_info = pilot.result.to_dict()
        else:
            schedule = policy.resolve_schedule(n_total=policy.budget.attempts)

        B = schedule.n_total
        chunk = run_cfg["chunk_attempts"]
        chunk = max(1, min(B, 4096)) if chunk == "auto" else min(chunk, B)
        a = resume_start_attempt if pos == start_pos else 0
        resume_start_attempt = 0
        recording_spec = cfg.recording_spec()
        termination = None
        exported_state = exported if (pos == start_pos and resume_state is not None) else None
        t_search0 = time.perf_counter()
        while a < B:
            b = min(a + chunk, B)
            result = backend.run(plan, native_state, schedule, n_attempts_total=B, start=a, stop=b,
                                 seed=seed, recording=recording_spec,
                                 targets=None, stop_on_target=run_cfg["stop_on_target"])
            exported_state = result.state
            native_state = backend.load_state(plan, exported_state, bindings=(binding,))
            a = b
            termination = result.termination
            ckpt.save_checkpoint(
                run_dir,
                ckpt.CheckpointMeta(run_id=run_id, trajectory_hash=cfg.trajectory_hash(),
                                   topology_hash=model.topology_hash, weight_hashes=(binding.weight_hash,),
                                   schedule={"kind": schedule.name, "p": list(schedule.p), "n_total": B},
                                   seed=seed, instance_order=tuple(instances), instance_index=pos,
                                   finished_instances=tuple(finished), chunk_attempts_done=a,
                                   n_attempts_total=B, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                   updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
                exported_state)
            _notify(progress, "chunk", instance=iid, attempts=a, total=B)
            if all(t != "budget" and t != "chunk" for t in termination):
                break
        timings[f"search_s.instance{iid}"] = time.perf_counter() - t_search0

        s = exported_state
        for rep in range(s.n_replicas):
            feasible_energy = s.feasible_best_energy[rep]
            feasible = np.isfinite(feasible_energy)
            native_val = None
            if feasible:
                try:
                    native_val = float(problem.native_objective(s.feasible_best_spins[rep], iid))
                except Exception:
                    native_val = None
            term = termination[rep] if termination is not None else None
            has_target = bool(np.isfinite(s.targets[rep]))
            hit_attempt = int(s.counters["target_hit_attempt"][rep]) if has_target else -1
            if term == "target":
                final_status = "target_hit" if final_status == "ok" else final_status
            rows.append(ReplicaRow(
                run_id=run_id, instance_id=int(iid), replica_id=int(s.replica_id[rep]), replica_position=rep,
                seed=seed,
                best_feasible_objective=float(feasible_energy) if feasible else None,
                best_feasible_native=native_val,
                best_feasible_attempt=int(s.feasible_best_attempt[rep]) if feasible else None,
                best_energy=float(s.best_energy[rep]), best_attempt=int(s.best_attempt[rep]),
                final_energy=float(s.energies[rep, 0]),
                final_feasible=bool(problem.is_feasible(s.spins[rep])),
                attempts=int(s.counters["attempts"][rep]), accepted=int(s.counters["accepted"][rep]),
                accepted_changed=int(s.counters["accepted_changed"][rep]),
                self_loops=int(s.counters["self_loops"][rep]), rejected=int(s.counters["rejected"][rep]),
                uphill_accepted=int(s.counters["uphill_accepted"][rep]),
                termination=str(term) if term is not None else None,
                target_hit=(hit_attempt >= 0) if has_target else None,
                target_hit_attempt=hit_attempt if hit_attempt >= 0 else None,
                process=proc.name,
                steps=int(s.counters["steps"][rep]) if "steps" in s.counters else None,
                candidate_evaluations=(int(s.counters["candidate_evaluations"][rep])
                                       if "candidate_evaluations" in s.counters else None),
                timings_json=json.dumps({"calibration": calibration_info is not None}),
            ))
        instance_meta.append({"instance_id": iid, "n_attempts_total": B, "calibration": calibration_info,
                              "schedule": {"kind": schedule.name, "p": list(schedule.p)}})
        finished.append(iid)
        write_shard(run_dir / "results.parquet", rows)
        if pos + 1 < len(instances) or cfg.data["output"]["keep_checkpoint"]:
            # boundary checkpoint: finished_instances/instance_index now agree with results.parquet,
            # so a resume after this point never reprocesses (and re-appends rows for) this instance.
            ckpt.save_checkpoint(
                run_dir,
                ckpt.CheckpointMeta(run_id=run_id, trajectory_hash=cfg.trajectory_hash(),
                                   topology_hash=model.topology_hash, weight_hashes=(binding.weight_hash,),
                                   schedule={"kind": schedule.name, "p": list(schedule.p), "n_total": B},
                                   seed=seed, instance_order=tuple(instances), instance_index=pos + 1,
                                   finished_instances=tuple(finished), chunk_attempts_done=0,
                                   n_attempts_total=B, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                   updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
                exported_state)
        _notify(progress, "instance", instance=iid, position=pos, total=len(instances))

    ckpt.remove_checkpoint(run_dir) if not cfg.data["output"]["keep_checkpoint"] else None
    timings["total_s"] = time.perf_counter() - t_total0
    manifest = manifest_mod.build_manifest(cfg, problem=problem, model=model, bindings=all_bindings,
                                           backend_plan=plan, policy=policy, calibration=None,
                                           timings=timings, status=final_status, run_id=run_id,
                                           instances=instance_meta)
    manifest_mod.write_manifest(run_dir, manifest)
    return SolveOutcome(final_status, str(run_dir), manifest, rows, timings, final_message)

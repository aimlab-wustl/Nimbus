"""Versioned run configuration ``nimbus.run/1``.

A run configuration is a YAML/JSON mapping with these sections (all strictly validated: unknown
fields and mutually inconsistent options raise :class:`ConfigError` naming the offending path):

``schema`` (required)
    ``nimbus.run/1``.
``name``, ``description``, ``tags``
    labels recorded in the manifest (``tags`` is a free mapping of strings/numbers).
``problem`` (required)
    ``{kind: <registered kind>, ...adapter fields}``; the adapter (``nimbus.problems.factory``)
    validates its own fields. Relative input paths resolve against the configuration file.
``transitions`` (required)
    ``{library: <name from problem.library_names()>, options: {...}}``.
``compile``
    ``{readout: default|none|all|[spins], limits: {max_clauses, max_routes,
    max_memory_bytes, max_closure_rounds, time_limit_s}, options: {use_identities,
    gate_representative, allow_non_involutive, table_max_states, run_preflight}}``.
``backend``
    ``{name: auto|oracle|emulation|cpu|gpu, strategy: auto, layout: auto, precision: float64,
    threads: null, device: null, options: {...backend prepare options}}``.
``annealing`` (required)
    ``{schedule: fn_endpoints|fn_legacy|geometric|constant|zero, params: {...}|auto,
    calibration: {...} (only with params: auto), attempts: B, batching:
    {capacity: C}, process: select|shared|parallel_trial|nfold|sweep, process_options: {cohort_size,
    pt_exclude_self_loops, nfold_clock}}``; see :data:`CALIBRATION_FIELDS`. ``process`` is the proposal
    process (default ``select``, the reference process); ``attempts`` is the
    budget B in the process's clock units (steps for parallel_trial and nfold_clock: steps). A default
    process (select, no options) is not written into the normalized config, so the hashes of existing
    configurations are unchanged.
``run``
    ``{seed, replicas, replica_offset, instances: all|n|[ids], targets: null|number|[per instance]|
    reference|{native: value}, target_tol, stop_on_target, chunk_attempts: auto|n}``.
``recording``
    :class:`nimbus.experiments.recording.RecordingSpec` fields (``profile`` none|summary|anytime|
    events|trace, ...).
``output``
    ``{path: null (default <runs root>/<date>_<name>_<hash8>), persist: true, overwrite: false,
    keep_checkpoint: false}``. A relative ``path`` is under the runs root (``$NIMBUS_RUNS_ROOT`` or
    ``<repo>/local_runs``); the CLI ``--output`` is relative to the working directory.
``baseline``
    ``{kind: none|penalty|quadratic, options: {...}}``.
``split``
    ``{id: <experiment split id>, role: tuning|validation|test|null, ...}`` recorded in the manifest.

Overrides (CLI ``--seed`` etc. or ``overrides={"run.seed": 3}``) are applied to the raw mapping and
the result is validated again. :meth:`RunConfig.trajectory_hash` covers every field that affects the
stochastic process or the stored results; resume requires it to match.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from nimbus.core.errors import ConfigError

__all__ = ["SCHEMA", "RunConfig", "load_config", "load_yaml", "dump_yaml", "apply_overrides", "parse_set_value",
           "BACKEND_NAMES", "BASELINE_KINDS", "SCHEDULE_KINDS", "CALIBRATION_FIELDS", "OVERRIDE_KEYS",
           "canonical_json", "config_digest", "process_spec_of"]

SCHEMA = "nimbus.run/1"
BACKEND_NAMES = ("auto", "oracle", "emulation", "cpu", "gpu")
_BACKEND_ALIASES = {"emulator": "emulation", "event": "emulation", "direct": "oracle"}
BASELINE_KINDS = ("none", "penalty", "quadratic")
SCHEDULE_KINDS = ("fn_endpoints", "fn_legacy", "geometric", "constant", "zero")
_EXPLICIT = {"fn_endpoints": ("tau_start", "tau_end", "eta"), "fn_legacy": ("tau0", "c_fn", "t_start", "dt"),
             "geometric": ("tau_start", "tau_end"), "constant": ("tau",), "zero": ()}
#: CalibrationSpec fields (nimbus.annealing.policy) accepted in annealing.calibration
_CAL_SPEC = ("rule", "p_start", "p_end", "n_pilot", "late_fraction", "budget_cap_fraction", "min_pilot",
             "min_positive", "max_rounds", "quantile", "legacy_factor", "class_weights", "coefficient_scale",
             "fallback", "min_ratio", "min_ratio_mode")
#: schedule-shape fields of an auto schedule (fn_endpoints/fn_legacy: eta; fn_legacy: t_start; constant: constant_at)
_CAL_SHAPE = ("eta", "t_start", "constant_at")
#: runner-side pilot fields
_CAL_RUNNER = ("scope", "pilot_states", "late_advance", "late_advance_attempts", "cache")
CALIBRATION_FIELDS = _CAL_SPEC + _CAL_SHAPE + _CAL_RUNNER

#: documented override keys (CLI flags) -> config path
OVERRIDE_KEYS = {
    "seed": "run.seed", "attempts": "annealing.attempts",
    "replicas": "run.replicas", "threads": "backend.threads", "backend": "backend.name",
    "record": "recording.profile", "output": "output.path",
    "device": "backend.device", "strategy": "backend.strategy", "instances": "run.instances",
    "process": "annealing.process",
}

_TOP = ("schema", "name", "description", "tags", "problem", "transitions", "compile", "backend", "annealing", "run",
        "recording", "output", "baseline", "split")

# fields that do not change the stochastic process or the stored per-replica results
_NON_TRAJECTORY = {("name",), ("description",), ("tags",), ("output",), ("run", "chunk_attempts"),
                   ("backend", "threads"), ("recording", "persist"), ("split",)}


# ============================================================================================
# YAML with duplicate-key detection
# ============================================================================================
class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    out = {}
    for k_node, v_node in node.value:
        key = loader.construct_object(k_node, deep=deep)
        if key in out:
            raise ConfigError(f"line {k_node.start_mark.line + 1}: duplicate key {key!r}")
        out[key] = loader.construct_object(v_node, deep=deep)
    return out


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(path_or_text, *, is_text: bool = False):
    """Parse YAML (or JSON) with duplicate-key detection; errors name the file and line."""
    if is_text:
        text, where = str(path_or_text), "<text>"
    else:
        p = Path(path_or_text)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read configuration {p}: {exc}") from None
        where = str(p)
    try:
        return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
    except ConfigError as exc:
        raise ConfigError(f"{where}: {exc}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{where}: invalid YAML: {exc}") from None


def dump_yaml(data) -> str:
    return yaml.safe_dump(_plain(data), sort_keys=False, default_flow_style=None, width=110)


def _plain(x):
    """Plain Python containers (tuples -> lists, numpy scalars -> Python) for YAML/JSON output."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if np is not None:
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.ndarray):
            return _plain(x.tolist())
    if isinstance(x, Path):
        return str(x)
    return x


def canonical_json(x) -> str:
    """Deterministic JSON (sorted keys, no whitespace) used for hashing."""
    return json.dumps(_plain(x), sort_keys=True, separators=(",", ":"), allow_nan=True)


def config_digest(x) -> str:
    return hashlib.sha256(canonical_json(x).encode("utf-8")).hexdigest()


# ============================================================================================
# validation helpers (every error names its config path)
# ============================================================================================
def _err(path: str, msg: str) -> ConfigError:
    return ConfigError(f"{path}: {msg}")


def _mapping(v, path: str, allowed=None, required=()) -> dict:
    if v is None:
        v = {}
    if not isinstance(v, Mapping):
        raise _err(path, f"must be a mapping, got {type(v).__name__}")
    d = dict(v)
    for k in d:
        if not isinstance(k, str):
            raise _err(path, f"keys must be strings, got {k!r}")
    if allowed is not None:
        unknown = sorted(set(d) - set(allowed))
        if unknown:
            raise _err(path, f"unknown field(s) {unknown}; allowed: {sorted(allowed)}")
    missing = [k for k in required if k not in d or d[k] is None]
    if missing:
        raise _err(path, f"missing required field(s) {missing}")
    return d


def _int(v, path, lo=None, hi=None, allow_none=False):
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        else:
            raise _err(path, f"must be an integer, got {v!r}")
    if lo is not None and v < lo:
        raise _err(path, f"must be >= {lo}, got {v}")
    if hi is not None and v > hi:
        raise _err(path, f"must be <= {hi}, got {v}")
    return int(v)


def _num(v, path, *, positive=False, nonneg=False, allow_none=False):
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _err(path, f"must be a number, got {v!r}")
    x = float(v)
    if not math.isfinite(x):
        raise _err(path, f"must be finite, got {v!r}")
    if positive and not x > 0:
        raise _err(path, f"must be > 0, got {x}")
    if nonneg and not x >= 0:
        raise _err(path, f"must be >= 0, got {x}")
    return x


def _bool(v, path):
    if not isinstance(v, bool):
        raise _err(path, f"must be true/false, got {v!r}")
    return v


def _str(v, path, choices=None, allow_none=False):
    if v is None and allow_none:
        return None
    if not isinstance(v, str) or not v:
        raise _err(path, f"must be a non-empty string, got {v!r}")
    if choices is not None and v not in choices:
        raise _err(path, f"must be one of {list(choices)}, got {v!r}")
    return v


# ============================================================================================
# section normalizers
# ============================================================================================
def _norm_problem(v):
    d = _mapping(v, "problem", required=("kind",))
    _str(d["kind"], "problem.kind")
    return d


def _norm_transitions(v):
    d = _mapping(v, "transitions", allowed=("library", "options"), required=("library",))
    return {"library": _str(d["library"], "transitions.library"),
            "options": _mapping(d.get("options"), "transitions.options")}


_LIMITS = ("max_clauses", "max_routes", "max_memory_bytes", "max_closure_rounds", "time_limit_s")
_COPTS = ("use_identities", "gate_representative", "allow_non_involutive", "table_max_states", "run_preflight")


def _norm_compile(v, base_dir: Path):
    d = _mapping(v, "compile", allowed=("readout", "limits", "options"))
    ro = d.get("readout", "default")
    if ro is None or ro == "none":
        ro = None
    elif isinstance(ro, str):
        _str(ro, "compile.readout", ("default", "all", "none"))
    elif isinstance(ro, (list, tuple)):
        ro = [_int(x, f"compile.readout[{i}]", lo=0) for i, x in enumerate(ro)]
    else:
        raise _err("compile.readout", f"must be default, none, all or a list of spin indices, got {ro!r}")
    lim = _mapping(d.get("limits"), "compile.limits", allowed=_LIMITS)
    limits = {}
    for k in _LIMITS:
        if k in lim and lim[k] is not None:
            limits[k] = (_num(lim[k], f"compile.limits.{k}", positive=True) if k == "time_limit_s"
                         else _int(lim[k], f"compile.limits.{k}", lo=1))
        elif k in lim:
            limits[k] = None
    opts = _mapping(d.get("options"), "compile.options", allowed=_COPTS)
    for k in ("use_identities", "allow_non_involutive", "run_preflight"):
        if k in opts:
            _bool(opts[k], f"compile.options.{k}")
    if "gate_representative" in opts:
        _str(opts["gate_representative"], "compile.options.gate_representative", ("best", "requested"))
    if "table_max_states" in opts:
        _int(opts["table_max_states"], "compile.options.table_max_states", lo=1)
    return {"readout": ro, "limits": limits, "options": opts}


def _norm_backend(v):
    d = _mapping(v, "backend", allowed=("name", "strategy", "layout", "precision", "threads", "device", "options"))
    name = d.get("name", "auto")
    name = _BACKEND_ALIASES.get(name, name)
    _str(name, "backend.name", BACKEND_NAMES)
    prec = _str(d.get("precision", "float64"), "backend.precision", ("float64", "float32"))
    if prec != "float64":
        raise _err("backend.precision", f"{prec} is not supported: no backend computes in {prec} (all backends use float64)")
    dev = d.get("device")
    if dev is not None and not isinstance(dev, (str, int)):
        raise _err("backend.device", f"must be a string or device index, got {dev!r}")
    return {"name": name, "strategy": _str(d.get("strategy", "auto"), "backend.strategy"),
            "layout": _str(d.get("layout", "auto"), "backend.layout"), "precision": prec,
            "threads": _int(d.get("threads"), "backend.threads", lo=1, allow_none=True),
            "device": dev, "options": _mapping(d.get("options"), "backend.options")}


def _norm_calibration(c, kind):
    c = _mapping(c, "annealing.calibration", allowed=CALIBRATION_FIELDS)
    # shape parameters required per kind (see nimbus.annealing.policy._AUTO_PARAMS)
    if kind == "zero":
        raise _err("annealing.calibration", "the zero schedule has no temperatures to calibrate")
    if kind in ("fn_endpoints", "fn_legacy") and "eta" not in c:
        raise _err("annealing.calibration.eta", f"required for an auto {kind} schedule (FN shape t_end/C)")
    if kind == "constant" and c.get("constant_at") not in ("start", "end"):
        raise _err("annealing.calibration.constant_at", "an auto constant schedule needs 'start' or 'end'")
    if "t_start" in c and kind != "fn_legacy":
        raise _err("annealing.calibration.t_start", "only applies to fn_legacy")
    if "eta" in c and kind not in ("fn_endpoints", "fn_legacy"):
        raise _err("annealing.calibration.eta", f"does not apply to {kind}")
    if "constant_at" in c and kind != "constant":
        raise _err("annealing.calibration.constant_at", f"does not apply to {kind}")
    out = dict(c)
    if "eta" in out:
        out["eta"] = _num(out["eta"], "annealing.calibration.eta", positive=True)
    if "t_start" in out:
        out["t_start"] = _num(out["t_start"], "annealing.calibration.t_start", positive=True)
    out["scope"] = _str(c.get("scope", "run"), "annealing.calibration.scope", ("run", "instance"))
    if "pilot_states" in c and c["pilot_states"] is not None:
        out["pilot_states"] = _int(c["pilot_states"], "annealing.calibration.pilot_states", lo=1)
    out["late_advance"] = _str(c.get("late_advance", "descent"), "annealing.calibration.late_advance",
                               ("descent", "none"))
    la = c.get("late_advance_attempts", "auto")
    if la != "auto":
        la = _int(la, "annealing.calibration.late_advance_attempts", lo=0)
    out["late_advance_attempts"] = la
    if c.get("cache") is not None:
        out["cache"] = _str(c["cache"], "annealing.calibration.cache")
    # validate the CalibrationSpec part through the policy module (same rules as RunPolicy)
    from nimbus.annealing.policy import CalibrationSpec

    spec = {k: out[k] for k in _CAL_SPEC if k in out}
    try:
        CalibrationSpec.from_dict(spec)
    except ConfigError as exc:
        raise _err("annealing.calibration", str(exc)) from None
    return out


def _norm_annealing(v):
    d = _mapping(v, "annealing", allowed=("schedule", "params", "calibration", "attempts", "batching",
                                          "process", "process_options"),
                 required=("schedule",))
    kind = _str(d["schedule"], "annealing.schedule", SCHEDULE_KINDS)
    params = d.get("params", {} if kind == "zero" else None)
    cal = d.get("calibration")
    if params is None:
        raise _err("annealing.params", "required: explicit schedule parameters or 'auto'")
    if params == "auto":
        cal = _norm_calibration(cal, kind)
    else:
        if cal is not None:
            raise _err("annealing.calibration", "only allowed with params: auto (explicit temperatures given)")
        p = _mapping(params, "annealing.params", allowed=_EXPLICIT[kind])
        missing = [n for n in _EXPLICIT[kind] if n not in p]
        if missing:
            raise _err("annealing.params", f"schedule {kind} needs {missing} (or params: auto)")
        params = {n: _num(p[n], f"annealing.params.{n}") for n in _EXPLICIT[kind]}
        from nimbus.annealing.policy import _build_explicit

        try:
            _build_explicit(kind, params, 1)
        except ValueError as exc:
            raise _err("annealing.params", str(exc)) from None
    attempts = _int(d.get("attempts"), "annealing.attempts", lo=1, allow_none=True)
    if attempts is None:
        raise _err("annealing", "needs 'attempts' (iterations per replica)")
    batching = d.get("batching")
    if batching is not None:
        b = _mapping(batching, "annealing.batching", allowed=("capacity", "rule"), required=("capacity",))
        batching = {"capacity": _int(b["capacity"], "annealing.batching.capacity", lo=1),
                    "rule": _str(b.get("rule", "proposal_preserving"), "annealing.batching.rule",
                                 ("proposal_preserving",))}
    out = {"schedule": kind, "params": params, "calibration": cal, "attempts": attempts,
           "batching": batching}
    spec = process_spec_of(d)
    if not spec.is_select or spec.options():
        out["process"] = spec.name
        out["process_options"] = spec.options()
    return out


def process_spec_of(annealing: Mapping):
    """The validated :class:`nimbus.backends.base.ProcessSpec` of an (raw or normalized) annealing section."""
    from nimbus.backends.base import ProcessSpec

    proc = annealing.get("process", "select")
    opts = annealing.get("process_options")
    if opts is not None and not isinstance(opts, Mapping):
        raise _err("annealing.process_options", f"must be a mapping, got {opts!r}")
    try:
        return ProcessSpec.coerce(proc, opts)
    except ConfigError as exc:
        raise _err("annealing.process", str(exc)) from None


def _norm_run(v):
    d = _mapping(v, "run", allowed=("seed", "replicas", "replica_offset", "instances", "targets", "target_tol",
                                    "stop_on_target", "chunk_attempts"))
    seed = _int(d.get("seed", 0), "run.seed", lo=0, hi=0xFFFFFFFF)
    reps = _int(d.get("replicas", 1), "run.replicas", lo=1)
    off = _int(d.get("replica_offset", 0), "run.replica_offset", lo=0)
    if off + reps > 0xFFFFFFFF:
        raise _err("run.replica_offset", "replica ids must stay below 2**32 (Philox counter word)")
    inst = d.get("instances", "all")
    if isinstance(inst, str):
        _str(inst, "run.instances", ("all",))
    elif isinstance(inst, bool):
        raise _err("run.instances", "must be 'all', a count or a list of instance ids")
    elif isinstance(inst, int):
        inst = {"first": _int(inst, "run.instances", lo=1)}
    elif isinstance(inst, Mapping):
        m = _mapping(inst, "run.instances", allowed=("first", "start", "stop"))
        if "first" in m:
            if len(m) != 1:
                raise _err("run.instances", "use either {first: n} or {start, stop}")
            inst = {"first": _int(m["first"], "run.instances.first", lo=1)}
        else:
            a = _int(m.get("start", 0), "run.instances.start", lo=0)
            b = _int(m.get("stop"), "run.instances.stop", lo=a + 1)
            inst = {"start": a, "stop": b}
    elif isinstance(inst, (list, tuple)):
        ids = [_int(x, f"run.instances[{i}]", lo=0) for i, x in enumerate(inst)]
        if not ids:
            raise _err("run.instances", "empty instance list")
        if len(set(ids)) != len(ids):
            raise _err("run.instances", "duplicate instance ids")
        inst = ids
    else:
        raise _err("run.instances", f"must be 'all', a count, a range or a list, got {inst!r}")
    tg = d.get("targets")
    if tg is None or tg == "reference":
        pass
    elif isinstance(tg, (int, float)) and not isinstance(tg, bool):
        tg = _num(tg, "run.targets")
    elif isinstance(tg, (list, tuple)):
        tg = [None if x is None else _num(x, f"run.targets[{i}]") for i, x in enumerate(tg)]
    elif isinstance(tg, Mapping):
        m = _mapping(tg, "run.targets", allowed=("native", "energy"))
        if len(m) != 1:
            raise _err("run.targets", "give exactly one of {native: value} or {energy: value}")
        k = next(iter(m))
        val = m[k]
        if isinstance(val, (list, tuple)):
            val = [None if x is None else _num(x, f"run.targets.{k}[{i}]") for i, x in enumerate(val)]
        else:
            val = _num(val, f"run.targets.{k}")
        tg = {k: val}
    else:
        raise _err("run.targets", f"must be null, a number, a list, 'reference' or {{native: value}}, got {tg!r}")
    stop = _bool(d.get("stop_on_target", False), "run.stop_on_target")
    if stop and tg is None:
        raise _err("run.stop_on_target", "needs run.targets")
    ch = d.get("chunk_attempts", "auto")
    if ch != "auto":
        ch = _int(ch, "run.chunk_attempts", lo=1)
    return {"seed": seed, "replicas": reps, "replica_offset": off, "instances": inst, "targets": tg,
            "target_tol": _num(d.get("target_tol", 0.0), "run.target_tol", nonneg=True), "stop_on_target": stop,
            "chunk_attempts": ch}


def _norm_recording(v):
    from nimbus.experiments.recording import RecordingSpec

    if isinstance(v, str):
        v = {"profile": v}
    d = _mapping(v, "recording")
    try:
        spec = RecordingSpec.from_dict(d)
    except ConfigError as exc:
        raise _err("recording", str(exc)) from None
    except TypeError as exc:
        raise _err("recording", str(exc)) from None
    out = spec.to_dict()
    out.pop("schema", None)
    return out


def _norm_output(v):
    d = _mapping(v, "output", allowed=("path", "persist", "overwrite", "keep_checkpoint"))
    path = d.get("path")
    if path is not None:
        path = _str(str(path) if isinstance(path, Path) else path, "output.path")
    return {"path": path, "persist": _bool(d.get("persist", True), "output.persist"),
            "overwrite": _bool(d.get("overwrite", False), "output.overwrite"),
            "keep_checkpoint": _bool(d.get("keep_checkpoint", False), "output.keep_checkpoint")}


def _norm_baseline(v):
    d = _mapping(v, "baseline", allowed=("kind", "options", "label"))
    kind = _str(d.get("kind", "none"), "baseline.kind", BASELINE_KINDS)
    opts = _mapping(d.get("options"), "baseline.options")
    if kind == "none" and opts:
        raise _err("baseline.options", "given without a baseline kind")
    return {"kind": kind, "options": opts, "label": _str(d.get("label", kind), "baseline.label")}


def _norm_split(v):
    d = _mapping(v, "split")
    if "role" in d and d["role"] is not None:
        _str(d["role"], "split.role", ("tuning", "validation", "test", "smoke", "development"))
    return d


def _norm_tags(v):
    d = _mapping(v, "tags")
    for k, x in d.items():
        if not isinstance(x, (str, int, float, bool)) and x is not None:
            raise _err(f"tags.{k}", "tag values must be scalars")
    return d


def _resolve_path(p: str, base_dir: Path) -> Path:
    from nimbus.experiments.paths import data_root, expand_path

    if p.startswith("data:"):
        return data_root() / p[5:].lstrip("/\\")
    q = expand_path(p)
    return q if q.is_absolute() else (base_dir / q)


def _normalize(raw: Mapping, base_dir: Path) -> dict:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"configuration must be a mapping, got {type(raw).__name__}")
    d = _mapping(raw, "<config>", allowed=_TOP)
    if d.get("schema") != SCHEMA:
        raise _err("schema", f"must be {SCHEMA!r}, got {d.get('schema')!r}")
    for sec in ("problem", "transitions", "annealing"):
        if sec not in d:
            raise _err(sec, "section is required")
    name = d.get("name")
    if name is not None:
        _str(name, "name")
        if any(ch in name for ch in '/\\:*?"<>|'):
            raise _err("name", "must not contain path separators or reserved characters")
    out = {
        "schema": SCHEMA,
        "name": name,
        "description": d.get("description"),
        "tags": _norm_tags(d.get("tags")),
        "problem": _norm_problem(d["problem"]),
        "transitions": _norm_transitions(d["transitions"]),
        "compile": _norm_compile(d.get("compile"), base_dir),
        "backend": _norm_backend(d.get("backend")),
        "annealing": _norm_annealing(d["annealing"]),
        "run": _norm_run(d.get("run")),
        "recording": _norm_recording(d.get("recording")),
        "output": _norm_output(d.get("output")),
        "baseline": _norm_baseline(d.get("baseline")),
        "split": _norm_split(d.get("split")),
    }
    if out["description"] is not None and not isinstance(out["description"], str):
        raise _err("description", "must be a string")
    # cross-section consistency
    a, r, b = out["annealing"], out["run"], out["baseline"]
    proc = process_spec_of(a)
    if not proc.is_select:
        if b["kind"] != "none":
            raise _err("annealing.process", f"process {proc.name!r} applies to architecture runs; the {b['kind']} "
                                            "baseline uses select")
        if out["backend"]["name"] == "oracle":
            raise _err("annealing.process", "the oracle implements the select process only")
    if proc.full_library and (a["batching"] is not None or "capacity" in out["backend"]["options"]
                          or "batches" in out["backend"]["options"]):
        raise _err("annealing.process", f"process {proc.name!r} needs every input settled every step and runs on "
                                        "the full library: capacity batching cannot be combined with it")
    if out["backend"]["name"] == "oracle" and a["batching"] is not None:
        raise _err("annealing.batching", "the oracle has no transition-neuron capacity; batching applies to "
                                         "the emulation (backend.options.capacity) or CPU/GPU plans")
    if r["stop_on_target"] and out["recording"]["profile"] == "none" and False:  # pragma: no cover
        pass
    return out


# ============================================================================================
# overrides
# ============================================================================================
def parse_set_value(text: str):
    """Value of a ``--set path=value`` override, parsed as YAML (numbers, booleans, lists, null)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse override value {text!r}: {exc}") from None


def apply_overrides(raw: Mapping, overrides: Mapping | Sequence | None) -> tuple[dict, list]:
    """Return (new raw mapping, applied list). ``overrides``: {dotted.path or short key: value}."""
    d = copy.deepcopy(dict(raw))
    applied = []
    if not overrides:
        return d, applied
    items = overrides.items() if isinstance(overrides, Mapping) else overrides
    for key, value in items:
        path = OVERRIDE_KEYS.get(str(key), str(key))
        parts = path.split(".")
        if parts[0] not in _TOP:
            raise ConfigError(f"override {key!r}: unknown section {parts[0]!r}")
        if path == "recording.profile" and isinstance(d.get("recording"), str):
            d["recording"] = {"profile": d["recording"]}
        cur = d
        for p in parts[:-1]:
            nxt = cur.get(p)
            if nxt is None:
                nxt = {}
                cur[p] = nxt
            if not isinstance(nxt, dict):
                raise ConfigError(f"override {key!r}: {p!r} is not a mapping")
            cur = nxt
        if path == "annealing.attempts" and value is not None:
            value = _int(value, f"override {key}", lo=1)
        cur[parts[-1]] = value
        applied.append({"key": str(key), "path": path, "value": _plain(value)})
    return d, applied


# ============================================================================================
# the config object
# ============================================================================================
@dataclass
class RunConfig:
    """A validated, normalized run configuration.

    ``data``: normalized mapping (defaults filled in); ``raw``: the mapping after overrides, as given;
    ``source``: configuration file (None for in-memory dicts); ``base_dir``: absolute directory that
    relative input paths resolve against; ``overrides``: the applied overrides.
    """

    data: dict
    raw: dict
    source: str | None
    base_dir: str
    overrides: list = field(default_factory=list)

    # -- accessors ---------------------------------------------------------------------------
    def __getitem__(self, key):
        return self.data[key]

    @property
    def name(self) -> str:
        n = self.data.get("name")
        if n:
            return n
        if self.source:
            return Path(self.source).stem
        return f"{self.data['problem']['kind']}_run"

    def get(self, dotted: str, default=None):
        cur: Any = self.data
        for p in dotted.split("."):
            if not isinstance(cur, Mapping) or p not in cur:
                return default
            cur = cur[p]
        return cur

    # -- hashing -----------------------------------------------------------------------------
    def trajectory_view(self) -> dict:
        """The normalized config without fields that cannot change trajectories or result rows."""
        d = copy.deepcopy(self.data)
        for path in _NON_TRAJECTORY:
            cur = d
            for p in path[:-1]:
                cur = cur.get(p, {})
            if isinstance(cur, dict):
                cur.pop(path[-1], None)
        return d

    def trajectory_hash(self) -> str:
        return config_digest(self.trajectory_view())

    def full_hash(self) -> str:
        return config_digest(self.data)

    # -- derived objects ---------------------------------------------------------------------
    def policy(self):
        """nimbus.annealing.policy.RunPolicy for this configuration."""
        from nimbus.annealing.policy import RunPolicy

        a, r = self.data["annealing"], self.data["run"]
        sched = {"kind": a["schedule"]}
        if a["params"] == "auto":
            cal = a["calibration"]
            sched.update({k: cal[k] for k in _CAL_SHAPE if k in cal})
            sched["auto"] = {k: cal[k] for k in _CAL_SPEC if k in cal} or True
        else:
            sched.update(a["params"])
        stop = {"stop_on_target": False}
        d = {"schedule": sched, "budget": {"attempts": a["attempts"]},
             "batching": a["batching"], "stopping": stop, "replicas": r["replicas"]}
        try:
            return RunPolicy.from_dict(d)
        except ConfigError as exc:
            raise _err("annealing", str(exc)) from None

    def recording_spec(self):
        from nimbus.experiments.recording import RecordingSpec

        return RecordingSpec.from_dict(self.data["recording"])

    def compile_limits(self):
        from nimbus.compiler import CompileLimits

        return CompileLimits(**self.data["compile"]["limits"])

    def compile_options(self, family=None):
        from nimbus.compiler import CompileOptions

        o = dict(self.data["compile"]["options"])
        if family is not None:
            o["family"] = family
        return CompileOptions(**o)

    def to_dict(self) -> dict:
        return {"resolved": _plain(self.data), "raw": _plain(self.raw), "source": self.source,
                "base_dir": self.base_dir, "overrides": _plain(self.overrides),
                "trajectory_hash": self.trajectory_hash(), "full_hash": self.full_hash()}

    def with_overrides(self, overrides) -> "RunConfig":
        """A new validated config with further overrides applied on top of this one's raw mapping."""
        raw, applied = apply_overrides(self.raw, overrides)
        data = _normalize(raw, Path(self.base_dir))
        return RunConfig(data=data, raw=raw, source=self.source, base_dir=self.base_dir,
                         overrides=list(self.overrides) + applied)

    def write_resolved(self, run_dir) -> Path:
        """Write ``config.resolved.yaml`` (normalized + raw + provenance) into ``run_dir`` atomically."""
        from nimbus.experiments.results import atomic_write_text

        p = Path(run_dir) / "config.resolved.yaml"
        atomic_write_text(p, dump_yaml(self.to_dict()))
        return p

    @classmethod
    def from_resolved(cls, path) -> "RunConfig":
        """Reload a ``config.resolved.yaml``; the raw mapping is re-validated (same base_dir)."""
        d = load_yaml(path)
        if not isinstance(d, Mapping) or "raw" not in d or "base_dir" not in d:
            raise ConfigError(f"{path}: not a resolved run configuration")
        base = Path(d["base_dir"])
        data = _normalize(d["raw"], base)
        cfg = cls(data=data, raw=dict(d["raw"]), source=d.get("source"), base_dir=str(base),
                  overrides=list(d.get("overrides") or []))
        if d.get("trajectory_hash") and d["trajectory_hash"] != cfg.trajectory_hash():
            raise ConfigError(f"{path}: the stored configuration no longer validates to the same trajectory "
                              "hash (schema or default change); refusing to reuse it")
        return cfg


def load_config(source, overrides=None, *, base_dir=None) -> RunConfig:
    """Load and validate a run configuration.

    ``source``: path to a YAML/JSON file, a mapping, or a RunConfig (overrides re-applied).
    ``overrides``: {dotted path or short key (see OVERRIDE_KEYS): value}.
    ``base_dir``: directory for relative paths of an in-memory mapping (default: working directory).
    """
    if isinstance(source, RunConfig):
        return source.with_overrides(overrides) if overrides else source
    if isinstance(source, (str, os.PathLike)):
        p = Path(source).resolve()
        raw = load_yaml(p)
        base = p.parent if base_dir is None else Path(base_dir).resolve()
        src = str(p)
    elif isinstance(source, Mapping):
        raw = dict(source)
        base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        src = None
    else:
        raise ConfigError(f"cannot load a configuration from {type(source).__name__}")
    raw2, applied = apply_overrides(raw, overrides)
    try:
        data = _normalize(raw2, base)
    except ConfigError as exc:
        raise ConfigError(f"{src}: {exc}" if src else str(exc)) from None
    return RunConfig(data=data, raw=_plain(raw2), source=src, base_dir=str(base), overrides=applied)

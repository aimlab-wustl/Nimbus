"""Recording profiles and in-memory containers for optional run observations.

A :class:`RecordingSpec` decides, *before* a run starts, which optional
observations a backend computes. Disabled diagnostics are not computed: a backend reads the
``want_*`` flags once and dispatches to a code path without the corresponding work.

Profiles (each level includes the ones before it):

=========  ===================================================================================
none       minimal in-memory return value (final/best state and the basic counters, which are
           solver work anyway); nothing is meant to be persisted
summary    default: as ``none``, intended for a persisted manifest + summary row
anytime    + log-spaced checkpoints of best/current energy and a capped buffer of best improvements
events     + architecture event counters and bounded per-accepted-transition histograms
trace      + per-attempt records for selected attempts/replicas, with a byte cap and truncation
           metadata
=========  ===================================================================================

``collect`` and ``persist`` are separate switches: ``collect=False`` turns every optional
observation off (whatever the profile); ``persist=False`` keeps the in-memory observations but
tells the runner not to write them. The explicit ``anytime`` / ``events`` / ``trace`` overrides
switch a single observation on or off independently of the profile level.

All containers serialize to a JSON-able dict (``to_dict``) and to flat numpy arrays
(``to_arrays``, suitable for ``np.savez``); ``from_dict`` / ``from_arrays`` invert them.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np

from nimbus.core.errors import ConfigError

__all__ = [
    "PROFILES", "TRACE_SNAPSHOTS", "RecordingSpec", "AnytimeLog", "EventLog", "TraceLog", "RecordingData",
    "log_checkpoints",
]

PROFILES = ("none", "summary", "anytime", "events", "trace")
_LEVEL = {p: i for i, p in enumerate(PROFILES)}
TRACE_SNAPSHOTS = ("spins", "T", "v")
SCHEMA = "nimbus.recording/1"


def _jsonable(x):
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        f = float(x)
        return f if math.isfinite(f) else repr(f)
    if isinstance(x, float) and not math.isfinite(x):
        return repr(x)  # 'inf', '-inf', 'nan' (JSON has no non-finite numbers)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _unjson_float(x):
    if isinstance(x, str):
        return float(x)
    return float(x)


def log_checkpoints(n_total: int, points: int) -> np.ndarray:
    """Log-spaced attempt counts in [1, n_total] (unique, ascending, always containing n_total)."""
    n_total = int(n_total)
    if n_total < 1:
        return np.zeros(0, dtype=np.int64)
    points = max(1, int(points))
    c = np.unique(np.rint(np.geomspace(1.0, float(n_total), num=points)).astype(np.int64))
    c = c[(c >= 1) & (c <= n_total)]
    if c.size == 0 or c[-1] != n_total:
        c = np.append(c, n_total)
    return c.astype(np.int64)


# ============================================================================================
# specification
# ============================================================================================
@dataclass(frozen=True)
class RecordingSpec:
    """What a run observes beyond the basic solver state.

    Parameters
    ----------
    profile: one of ``PROFILES``.
    collect: compute optional observations at all (False = none of anytime/events/trace).
    persist: whether the runner should write the observations (backends ignore it).
    anytime_points: number of log-spaced checkpoints over the run budget ``[1, B]`` (attempt counts).
    anytime_checkpoints: explicit attempt counts overriding ``anytime_points``.
    improvement_cap: maximum stored best improvements per replica; later ones are counted as dropped.
    event_hist_bins: bins of the per-accepted-transition histograms; values >= bins-1 go to the last
        (overflow) bin.
    trace_max_bytes: byte cap of the trace (records beyond it are dropped and counted).
    trace_every: trace only attempts with ``n % trace_every == 0`` (global attempt index n).
    trace_first: trace only attempts ``n < trace_first`` (None = no limit).
    trace_replicas: replica positions (row indices of the state) to trace (None = all).
    trace_accepted_only: trace only accepted attempts.
    trace_snapshots: subset of ``("spins", "T", "v")`` copied into every trace record (post-attempt).
    anytime / events / trace: explicit on/off overrides of the profile level (None = by profile).
    """

    profile: str = "summary"
    collect: bool = True
    persist: bool = True
    anytime_points: int = 64
    anytime_checkpoints: tuple | None = None
    improvement_cap: int = 4096
    event_hist_bins: int = 65
    trace_max_bytes: int = 8 * 2 ** 20
    trace_every: int = 1
    trace_first: int | None = None
    trace_replicas: tuple | None = None
    trace_accepted_only: bool = False
    trace_snapshots: tuple = ()
    anytime: bool | None = None
    events: bool | None = None
    trace: bool | None = None

    def __post_init__(self):
        if self.profile not in _LEVEL:
            raise ConfigError(f"unknown recording profile {self.profile!r}; expected one of {PROFILES}")
        for name in ("anytime_points", "improvement_cap", "trace_max_bytes", "trace_every"):
            if int(getattr(self, name)) < (1 if name in ("anytime_points", "trace_every") else 0):
                raise ConfigError(f"RecordingSpec.{name} must be positive")
        if int(self.event_hist_bins) < 2:
            raise ConfigError("RecordingSpec.event_hist_bins must be >= 2 (one exact bin plus overflow)")
        if self.trace_first is not None and int(self.trace_first) < 0:
            raise ConfigError("RecordingSpec.trace_first must be >= 0")
        snaps = tuple(str(x) for x in self.trace_snapshots)
        bad = [x for x in snaps if x not in TRACE_SNAPSHOTS]
        if bad:
            raise ConfigError(f"unknown trace snapshots {bad}; allowed {TRACE_SNAPSHOTS}")
        object.__setattr__(self, "trace_snapshots", snaps)
        if self.anytime_checkpoints is not None:
            cps = tuple(sorted({int(c) for c in self.anytime_checkpoints}))
            if cps and cps[0] < 1:
                raise ConfigError("anytime checkpoints are attempt counts >= 1")
            object.__setattr__(self, "anytime_checkpoints", cps)
        if self.trace_replicas is not None:
            object.__setattr__(self, "trace_replicas", tuple(sorted({int(i) for i in self.trace_replicas})))

    # ------------------------------------------------------------------------------------------
    @classmethod
    def coerce(cls, x) -> "RecordingSpec":
        """None -> default (summary); str -> that profile; Mapping -> fields; RecordingSpec -> itself."""
        if x is None:
            return cls()
        if isinstance(x, RecordingSpec):
            return x
        if isinstance(x, str):
            return cls(profile=x)
        if isinstance(x, Mapping):
            return cls.from_dict(x)
        raise ConfigError(f"cannot interpret {x!r} as a RecordingSpec")

    @property
    def level(self) -> int:
        return _LEVEL[self.profile]

    def _want(self, override, lvl) -> bool:
        if not self.collect:
            return False
        if override is not None:
            return bool(override)
        return self.level >= lvl

    @property
    def want_anytime(self) -> bool:
        return self._want(self.anytime, _LEVEL["anytime"])

    @property
    def want_events(self) -> bool:
        return self._want(self.events, _LEVEL["events"])

    @property
    def want_trace(self) -> bool:
        return self._want(self.trace, _LEVEL["trace"])

    @property
    def should_persist(self) -> bool:
        return bool(self.persist) and self.profile != "none"

    def checkpoints(self, n_total: int) -> np.ndarray:
        """Anytime checkpoints (attempt counts in [1, n_total])."""
        if self.anytime_checkpoints is not None:
            c = np.asarray(self.anytime_checkpoints, dtype=np.int64)
            return c[(c >= 1) & (c <= int(n_total))]
        return log_checkpoints(n_total, self.anytime_points)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("anytime_checkpoints", "trace_replicas", "trace_snapshots"):
            if d[k] is not None:
                d[k] = list(d[k])
        d["schema"] = SCHEMA
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "RecordingSpec":
        d = dict(d)
        d.pop("schema", None)
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(d) - known)
        if unknown:
            raise ConfigError(f"unknown RecordingSpec fields {unknown}")
        for k in ("anytime_checkpoints", "trace_replicas", "trace_snapshots"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return cls(**d)


# ============================================================================================
# anytime
# ============================================================================================
class AnytimeLog:
    """Best/current energy at log-spaced checkpoints plus a capped buffer of best improvements.

    ``checkpoints``: attempt counts c (state after c attempts). ``best``/``current``/
    ``feasible_best``: float64[n_rep, n_cp], NaN where the checkpoint was not reached by this run
    (chunk). Improvements: per replica (attempt count, new best slot-0 energy), at most
    ``improvement_cap`` each; ``improvements_dropped`` counts the rest (truncation metadata).
    """

    def __init__(self, n_replicas: int, checkpoints, improvement_cap: int):
        self.checkpoints = np.asarray(checkpoints, dtype=np.int64).copy()
        n = int(n_replicas)
        k = self.checkpoints.size
        self.best = np.full((n, k), np.nan)
        self.current = np.full((n, k), np.nan)
        self.feasible_best = np.full((n, k), np.nan)
        self.improvement_cap = int(improvement_cap)
        self._imp = [[] for _ in range(n)]
        self.improvements_dropped = np.zeros(n, dtype=np.int64)

    @property
    def n_replicas(self) -> int:
        return self.best.shape[0]

    def record_checkpoint(self, rep: int, j: int, best: float, current: float, feasible_best: float):
        self.best[rep, j] = best
        self.current[rep, j] = current
        self.feasible_best[rep, j] = feasible_best

    def record_improvement(self, rep: int, count: int, energy: float):
        buf = self._imp[rep]
        if len(buf) < self.improvement_cap:
            buf.append((int(count), float(energy)))
        else:
            self.improvements_dropped[rep] += 1

    def improvements(self, rep: int) -> tuple[np.ndarray, np.ndarray]:
        buf = self._imp[rep]
        return (np.asarray([a for a, _ in buf], dtype=np.int64), np.asarray([e for _, e in buf], dtype=np.float64))

    @property
    def truncated(self) -> bool:
        return bool(np.any(self.improvements_dropped > 0))

    def to_arrays(self, prefix: str = "anytime.") -> dict:
        ptr = np.zeros(self.n_replicas + 1, dtype=np.int64)
        ptr[1:] = np.cumsum([len(b) for b in self._imp])
        att = np.asarray([a for b in self._imp for a, _ in b], dtype=np.int64)
        en = np.asarray([e for b in self._imp for _, e in b], dtype=np.float64)
        return {prefix + "checkpoints": self.checkpoints, prefix + "best": self.best, prefix + "current": self.current,
                prefix + "feasible_best": self.feasible_best, prefix + "improvement_ptr": ptr,
                prefix + "improvement_attempt": att, prefix + "improvement_energy": en,
                prefix + "improvements_dropped": self.improvements_dropped,
                prefix + "improvement_cap": np.asarray(self.improvement_cap, dtype=np.int64)}

    @classmethod
    def from_arrays(cls, a: Mapping, prefix: str = "anytime.") -> "AnytimeLog":
        best = np.asarray(a[prefix + "best"])
        log = cls(best.shape[0], a[prefix + "checkpoints"], int(np.asarray(a[prefix + "improvement_cap"])))
        log.best[...] = best
        log.current[...] = a[prefix + "current"]
        log.feasible_best[...] = a[prefix + "feasible_best"]
        ptr = np.asarray(a[prefix + "improvement_ptr"])
        att = np.asarray(a[prefix + "improvement_attempt"])
        en = np.asarray(a[prefix + "improvement_energy"])
        for i in range(best.shape[0]):
            log._imp[i] = list(zip(att[ptr[i]:ptr[i + 1]].tolist(), en[ptr[i]:ptr[i + 1]].tolist()))
        log.improvements_dropped[...] = a[prefix + "improvements_dropped"]
        return log

    def to_dict(self) -> dict:
        return _jsonable({"checkpoints": self.checkpoints, "best": self.best, "current": self.current,
                          "feasible_best": self.feasible_best, "improvement_cap": self.improvement_cap,
                          "improvements": [[list(p) for p in b] for b in self._imp],
                          "improvements_dropped": self.improvements_dropped, "truncated": self.truncated})

    @classmethod
    def from_dict(cls, d: Mapping) -> "AnytimeLog":
        def arr(x):
            return np.asarray([[_unjson_float(v) for v in row] for row in x], dtype=np.float64).reshape(len(x), -1)

        best = arr(d["best"])
        log = cls(best.shape[0], d["checkpoints"], d["improvement_cap"])
        log.best[...] = best
        log.current[...] = arr(d["current"])
        log.feasible_best[...] = arr(d["feasible_best"])
        log._imp = [[(int(a), _unjson_float(e)) for a, e in b] for b in d["improvements"]]
        log.improvements_dropped[...] = d["improvements_dropped"]
        return log


# ============================================================================================
# events
# ============================================================================================
class EventLog:
    """Architecture event counters (int64[n_rep] per name) and bounded histograms.

    Counters are stored as one int64[n_rep, n_counters] array (``index[name]`` gives the column),
    so a backend can update them with plain integer indexing in its hot path. Histograms hold
    int64[n_rep, bins] each: value x goes to bin ``min(x, bins - 1)`` (the last bin is overflow).
    ``meta`` carries scalar context (e.g. full-scan route count per transition).
    """

    def __init__(self, n_replicas: int, counter_names: Sequence[str], hist_names: Sequence[str] = (),
                 bins: int = 65, meta: Mapping | None = None):
        self.counter_names = tuple(str(x) for x in counter_names)
        self.hist_names = tuple(str(x) for x in hist_names)
        self.index = {n: i for i, n in enumerate(self.counter_names)}
        self.hindex = {n: i for i, n in enumerate(self.hist_names)}
        self.bins = int(bins)
        self.counts = np.zeros((int(n_replicas), len(self.counter_names)), dtype=np.int64)
        self.hist = np.zeros((len(self.hist_names), int(n_replicas), self.bins), dtype=np.int64)
        self.meta = dict(meta or {})

    @property
    def n_replicas(self) -> int:
        return self.counts.shape[0]

    def counter(self, name: str) -> np.ndarray:
        return self.counts[:, self.index[name]]

    def histogram(self, name: str) -> np.ndarray:
        return self.hist[self.hindex[name]]

    def add_hist(self, rep: int, h: int, value: int):
        self.hist[h, rep, value if value < self.bins - 1 else self.bins - 1] += 1

    def totals(self) -> dict:
        return {n: int(self.counts[:, i].sum()) for i, n in enumerate(self.counter_names)}

    def merge(self, other: "EventLog") -> "EventLog":
        """Sum of two logs over the same counters/replicas (e.g. consecutive chunks)."""
        if other.counter_names != self.counter_names or other.hist_names != self.hist_names \
                or other.bins != self.bins or other.counts.shape != self.counts.shape:
            raise ConfigError("cannot merge event logs with different layouts")
        out = EventLog(self.n_replicas, self.counter_names, self.hist_names, self.bins, self.meta)
        out.counts[...] = self.counts + other.counts
        out.hist[...] = self.hist + other.hist
        return out

    def to_arrays(self, prefix: str = "events.") -> dict:
        out = {prefix + "counts": self.counts, prefix + "hist": self.hist,
               prefix + "counter_names": np.asarray(self.counter_names, dtype=str),
               prefix + "hist_names": np.asarray(self.hist_names, dtype=str),
               prefix + "meta_json": np.asarray(json.dumps(_jsonable(self.meta)))}
        return out

    @classmethod
    def from_arrays(cls, a: Mapping, prefix: str = "events.") -> "EventLog":
        counts = np.asarray(a[prefix + "counts"])
        hist = np.asarray(a[prefix + "hist"])
        log = cls(counts.shape[0], [str(x) for x in np.asarray(a[prefix + "counter_names"]).tolist()],
                  [str(x) for x in np.asarray(a[prefix + "hist_names"]).tolist()],
                  hist.shape[2] if hist.ndim == 3 else 2, json.loads(str(np.asarray(a[prefix + "meta_json"]))))
        log.counts[...] = counts
        if hist.size:
            log.hist[...] = hist
        return log

    def to_dict(self) -> dict:
        return _jsonable({"counters": {n: self.counts[:, i] for i, n in enumerate(self.counter_names)},
                          "histograms": {n: self.hist[i] for i, n in enumerate(self.hist_names)},
                          "bins": self.bins, "overflow_bin": self.bins - 1, "meta": self.meta})

    @classmethod
    def from_dict(cls, d: Mapping) -> "EventLog":
        names = list(d["counters"].keys())
        hnames = list(d["histograms"].keys())
        n_rep = len(next(iter(d["counters"].values()))) if names else 0
        log = cls(n_rep, names, hnames, d["bins"], d.get("meta"))
        for i, n in enumerate(names):
            log.counts[:, i] = d["counters"][n]
        for i, n in enumerate(hnames):
            log.hist[i] = np.asarray(d["histograms"][n], dtype=np.int64)
        return log


# ============================================================================================
# trace
# ============================================================================================
_TRACE_FIELDS = (
    ("attempt", np.int64), ("replica", np.int64), ("replica_id", np.int64), ("group", np.int64),
    ("u", np.float64), ("tau", np.float64), ("xi", np.float64), ("v", np.float64),
    ("accepted", np.bool_), ("changed", np.bool_), ("energy", np.float64),
    ("n_toggles", np.int64), ("n_switch_changes", np.int64), ("batch", np.int64),
)
TRACE_FIELD_NAMES = tuple(n for n, _ in _TRACE_FIELDS)


class TraceLog:
    """Per-attempt trace with sampling, a byte cap and truncation metadata.

    Every record has the scalar fields ``TRACE_FIELD_NAMES`` (``v`` = the selected input read for
    the decision, ``energy`` = slot-0 energy after the attempt, ``n_toggles`` = latched clause
    toggles, ``n_switch_changes`` = switch changes of the transaction; -1 where a backend has no
    such quantity) plus optional post-attempt snapshots (``spins``, ``T``, ``v`` arrays). The byte
    size of one record is fixed; once the next record would exceed ``max_bytes`` it is dropped and
    counted (``n_dropped``, ``first_dropped_attempt``, ``truncated``).
    """

    def __init__(self, spec: RecordingSpec, n_spins: int, n_clauses: int, n_groups: int):
        self.max_bytes = int(spec.trace_max_bytes)
        self.every = int(spec.trace_every)
        self.first = None if spec.trace_first is None else int(spec.trace_first)
        self.replicas = None if spec.trace_replicas is None else frozenset(spec.trace_replicas)
        self.accepted_only = bool(spec.trace_accepted_only)
        self.snapshots = tuple(spec.trace_snapshots)
        self.snap_sizes = {"spins": int(n_spins), "T": int(n_clauses), "v": int(n_groups)}
        self.record_bytes = sum(np.dtype(t).itemsize for _, t in _TRACE_FIELDS) + sum(
            self.snap_sizes[s] * (8 if s == "v" else 1) for s in self.snapshots)
        self._rows: list[tuple] = []
        self._snap: dict[str, list] = {s: [] for s in self.snapshots}
        self.n_dropped = 0
        self.first_dropped_attempt = -1
        self.bytes_used = 0

    @property
    def truncated(self) -> bool:
        return self.n_dropped > 0

    @property
    def n_records(self) -> int:
        return len(self._rows)

    def wants(self, n: int, rep: int) -> bool:
        """Sampling policy for attempt n of replica position rep (before acceptance is known)."""
        if self.replicas is not None and rep not in self.replicas:
            return False
        if self.first is not None and n >= self.first:
            return False
        return n % self.every == 0

    def append(self, row: tuple, spins=None, T=None, v=None) -> bool:
        """Append one record (``row`` in ``TRACE_FIELD_NAMES`` order). Returns False if dropped."""
        if self.accepted_only and not row[8]:
            return False
        if self.bytes_used + self.record_bytes > self.max_bytes:
            if self.n_dropped == 0:
                self.first_dropped_attempt = int(row[0])
            self.n_dropped += 1
            return False
        self._rows.append(row)
        snaps = {"spins": spins, "T": T, "v": v}
        for s in self.snapshots:
            x = snaps[s]
            if x is None:
                x = np.full(self.snap_sizes[s], np.nan if s == "v" else 0, dtype=np.float64 if s == "v" else np.int8)
            self._snap[s].append(np.array(x, dtype=np.float64 if s == "v" else np.int8, copy=True))
        self.bytes_used += self.record_bytes
        return True

    def columns(self) -> dict:
        cols = {}
        for j, (name, dt) in enumerate(_TRACE_FIELDS):
            cols[name] = np.asarray([row[j] for row in self._rows], dtype=dt)
        for s in self.snapshots:
            n = self.snap_sizes[s]
            dt = np.float64 if s == "v" else np.int8
            cols["snap_" + s] = np.stack(self._snap[s]).astype(dt) if self._snap[s] else np.zeros((0, n), dtype=dt)
        return cols

    def metadata(self) -> dict:
        return {"n_records": self.n_records, "bytes_used": self.bytes_used, "max_bytes": self.max_bytes,
                "record_bytes": self.record_bytes, "truncated": self.truncated, "n_dropped": self.n_dropped,
                "first_dropped_attempt": self.first_dropped_attempt, "every": self.every, "first": self.first,
                "replicas": None if self.replicas is None else sorted(self.replicas),
                "accepted_only": self.accepted_only, "snapshots": list(self.snapshots)}

    def to_arrays(self, prefix: str = "trace.") -> dict:
        out = {prefix + k: v for k, v in self.columns().items()}
        out[prefix + "meta_json"] = np.asarray(json.dumps(_jsonable(self.metadata())))
        return out

    def to_dict(self) -> dict:
        return _jsonable({"columns": self.columns(), "meta": self.metadata()})

    @classmethod
    def from_arrays(cls, a: Mapping, prefix: str = "trace.") -> "TraceLog":
        meta = json.loads(str(np.asarray(a[prefix + "meta_json"])))
        return cls._rebuild({k[len(prefix):]: np.asarray(v) for k, v in a.items()
                             if k.startswith(prefix) and k != prefix + "meta_json"}, meta)

    @classmethod
    def from_dict(cls, d: Mapping) -> "TraceLog":
        return cls._rebuild({k: np.asarray(v) for k, v in d["columns"].items()}, d["meta"])

    @classmethod
    def _rebuild(cls, cols: Mapping, meta: Mapping) -> "TraceLog":
        snaps = tuple(meta.get("snapshots", ()))
        sizes = {s: int(np.asarray(cols["snap_" + s]).shape[1]) for s in snaps}
        spec = RecordingSpec(profile="trace", trace_max_bytes=int(meta["max_bytes"]), trace_every=int(meta["every"]),
                             trace_first=meta.get("first"), trace_replicas=meta.get("replicas"),
                             trace_accepted_only=bool(meta.get("accepted_only", False)), trace_snapshots=snaps)
        log = cls(spec, sizes.get("spins", 0), sizes.get("T", 0), sizes.get("v", 0))
        n = len(cols["attempt"]) if "attempt" in cols else 0
        fields = [np.asarray(cols[name]).astype(dt) for name, dt in _TRACE_FIELDS]
        log._rows = [tuple(f[i].item() for f in fields) for i in range(n)]
        for s in snaps:
            log._snap[s] = [np.asarray(x) for x in np.asarray(cols["snap_" + s])]
        log.n_dropped = int(meta["n_dropped"])
        log.first_dropped_attempt = int(meta["first_dropped_attempt"])
        log.bytes_used = int(meta["bytes_used"])
        return log


# ============================================================================================
# bundle
# ============================================================================================
@dataclass
class RecordingData:
    """Observations collected by one ``run`` call (one chunk). Absent observation = None.

    ``notes`` records why an observation is absent when the profile asked for it (e.g.
    ``{"events": "not applicable: the oracle has no architecture events"}``): unsupported metrics
    are reported as not applicable, never as zero.
    """

    spec: RecordingSpec
    anytime: AnytimeLog | None = None
    events: EventLog | None = None
    trace: TraceLog | None = None
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "spec": self.spec.to_dict(),
                "anytime": self.anytime.to_dict() if self.anytime is not None else None,
                "events": self.events.to_dict() if self.events is not None else None,
                "trace": self.trace.to_dict() if self.trace is not None else None,
                "notes": _jsonable(self.notes)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "RecordingData":
        return cls(spec=RecordingSpec.from_dict(d["spec"]),
                   anytime=AnytimeLog.from_dict(d["anytime"]) if d.get("anytime") is not None else None,
                   events=EventLog.from_dict(d["events"]) if d.get("events") is not None else None,
                   trace=TraceLog.from_dict(d["trace"]) if d.get("trace") is not None else None,
                   notes=dict(d.get("notes") or {}))

    def to_arrays(self) -> dict:
        out = {"recording.spec_json": np.asarray(json.dumps(self.spec.to_dict())),
               "recording.notes_json": np.asarray(json.dumps(_jsonable(self.notes)))}
        if self.anytime is not None:
            out.update(self.anytime.to_arrays())
        if self.events is not None:
            out.update(self.events.to_arrays())
        if self.trace is not None:
            out.update(self.trace.to_arrays())
        return out

    @classmethod
    def from_arrays(cls, a: Mapping) -> "RecordingData":
        spec = RecordingSpec.from_dict(json.loads(str(np.asarray(a["recording.spec_json"]))))
        notes = json.loads(str(np.asarray(a["recording.notes_json"])))
        return cls(spec=spec,
                   anytime=AnytimeLog.from_arrays(a) if "anytime.checkpoints" in a else None,
                   events=EventLog.from_arrays(a) if "events.counts" in a else None,
                   trace=TraceLog.from_arrays(a) if "trace.meta_json" in a else None,
                   notes=notes)

"""Run policy: proposal rule, batching, acceptance, schedule, budget, stopping and replica count.

A :class:`RunPolicy` is the immutable, validated description of the stochastic process a run executes
(Methods, Stochastic transition dynamics and FN annealing and Supplementary Information, Algorithmic implementation):

proposal = 'uniform'
    At attempt n one group r_n is drawn uniformly from the fixed group list 0..R-1 by the solver
    stream. Self-loops and rejections advance the clock; there is never any
    resampling until a state change (that would add a state-dependent normalization,
    Supplementary Information, Proposal symmetry and state-dependent proposal normalization).
batching (optional) = {capacity C_max, rule 'proposal_preserving'}
    When the library does not fit in C_max transition neurons, the fixed group list is cut into
    contiguous batches of at most C_max groups (``BatchPlan``). The exact proposal-preserving rule of
    Supplementary Information, Algorithmic implementation selects batch b with probability L_b / R and then one of its L_b groups
    uniformly, so every group keeps probability 1/R. We realize it with the identical solver draw
    r ~ U{0..R-1} and the batch b = batch_of_group[r]: the joint law of (b, r) equals the two-stage
    draw, and the proposal stream is bit-identical to the fully loaded run, so batching changes only
    the reload (re-initialization) count, never the trajectory. Dwell-time batching (keeping a batch
    for many attempts) is a different stochastic process and is rejected here.
acceptance = 'metropolis_threshold'
    Accept iff v_r <= xi_n, xi_n = -(tau_n / 2) log u_n (tau_n > 0), and iff v_r <= 0 at tau_n = 0
    (Methods, Stochastic transition dynamics and FN annealing). ``metropolis_threshold`` and
    ``metropolis_accept`` below are the pure-Python reference that Numba/CUDA reproduce.
schedule
    Either explicit (kind + parameters) or ``auto`` (pilot calibration spec); resolved
    into a concrete :class:`nimbus.annealing.schedule.Schedule` by :meth:`RunPolicy.resolve_schedule`.
budget
    ``attempts`` = attempts per replica (the schedule length B) and/or ``wall_time_s`` (whole run).
stopping
    optional ``target_energy`` (hit iff E <= target + target_tol) and ``stop_on_target``.
replicas
    independent restarts per instance/binding (RNG counter word c2 = replica index).

Dict form (YAML/JSON), all keys optional except ``schedule`` and ``budget``::

    {"schema": "nimbus.policy/1", "proposal": "uniform", "acceptance": "metropolis_threshold",
     "batching": {"capacity": 64, "rule": "proposal_preserving"},
     "schedule": {"kind": "fn_endpoints", "eta": 0.1, "auto": {"rule": "acceptance", "p_start": 0.8, "p_end": 0.01}},
     "budget": {"attempts": 100000, "wall_time_s": 60.0},
     "stopping": {"target_energy": -12.0, "stop_on_target": true, "target_tol": 0.0},
     "replicas": 16}

Explicit schedule parameters per kind: fn_endpoints {tau_start, tau_end, eta}; fn_legacy {tau0, c_fn,
t_start, dt}; geometric {tau_start, tau_end}; constant {tau}; zero {}. With ``auto`` the temperatures
come from calibration and only shape parameters are given: fn_endpoints {eta}; fn_legacy {eta, t_start
(default 1)}; geometric {}; constant {constant_at: 'start'|'end'}. ``auto: true`` uses the default
:class:`CalibrationSpec`. Unknown or inconsistent fields raise :class:`ConfigError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from nimbus.annealing import schedule as _schedule
from nimbus.annealing.calibration import RULES as CALIBRATION_RULES
from nimbus.annealing.calibration import CalibrationResult, make_schedule
from nimbus.core.errors import ConfigError

__all__ = [
    "SCHEMA",
    "metropolis_threshold",
    "metropolis_accept",
    "BatchPlan",
    "BatchingSpec",
    "CalibrationSpec",
    "ScheduleSpec",
    "BudgetSpec",
    "StoppingSpec",
    "RunPolicy",
]

SCHEMA = "nimbus.policy/1"
PROPOSALS = ("uniform",)
ACCEPTANCES = ("metropolis_threshold",)
BATCH_RULES = ("proposal_preserving",)
SCHEDULE_KINDS = ("fn_endpoints", "fn_legacy", "geometric", "constant", "zero")

_EXPLICIT_PARAMS = {
    "fn_endpoints": ("tau_start", "tau_end", "eta"),
    "fn_legacy": ("tau0", "c_fn", "t_start", "dt"),
    "geometric": ("tau_start", "tau_end"),
    "constant": ("tau",),
    "zero": (),
}
# auto mode: (required, optional-with-default)
_AUTO_PARAMS = {
    "fn_endpoints": (("eta",), {}),
    "fn_legacy": (("eta",), {"t_start": 1.0}),
    "geometric": ((), {}),
    "constant": (("constant_at",), {}),
}


# --------------------------------------------------------------------------------------------
# Acceptance reference (pure Python)
# --------------------------------------------------------------------------------------------

def metropolis_threshold(tau: float, u: float) -> float:
    """xi = -(tau / 2) * log(u) (Methods, Stochastic transition dynamics and FN annealing): exponential threshold with mean tau / 2."""
    return -(tau / 2.0) * math.log(u)


def metropolis_accept(v: float, tau: float, u: float) -> bool:
    """Accept the selected transition with neuron input v (Delta E = 2 v) at temperature tau, uniform u."""
    if tau == 0.0:
        return v <= 0.0
    return v <= -(tau / 2.0) * math.log(u)


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------

def _mapping(d, what) -> dict:
    if not isinstance(d, Mapping):
        raise ConfigError(f"{what} must be a mapping, got {type(d).__name__}")
    return dict(d)


def _reject_unknown(d: Mapping, allowed, what):
    unknown = sorted(set(d) - set(allowed))
    if unknown:
        raise ConfigError(f"unknown {what} field(s) {unknown}; allowed: {sorted(allowed)}")


def _int(v, what, lo=None) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        else:
            raise ConfigError(f"{what} must be an integer, got {v!r}")
    v = int(v)
    if lo is not None and v < lo:
        raise ConfigError(f"{what} must be >= {lo}, got {v}")
    return v


def _float(v, what, *, positive=False, nonneg=False) -> float:
    if isinstance(v, bool):
        raise ConfigError(f"{what} must be a number, got {v!r}")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ConfigError(f"{what} must be a number, got {v!r}") from None
    if not math.isfinite(x):
        raise ConfigError(f"{what} must be finite, got {v!r}")
    if positive and not x > 0:
        raise ConfigError(f"{what} must be > 0, got {x}")
    if nonneg and not x >= 0:
        raise ConfigError(f"{what} must be >= 0, got {x}")
    return x


def _bool(v, what) -> bool:
    if not isinstance(v, (bool, np.bool_)):
        raise ConfigError(f"{what} must be true/false, got {v!r}")
    return bool(v)


# --------------------------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchPlan:
    """Contiguous partition of the fixed group list 0..R-1 into batches of at most ``capacity`` groups."""

    capacity: int
    n_groups: int
    batch_ptr: np.ndarray  # int64[n_batches + 1]; batch b holds groups [batch_ptr[b], batch_ptr[b+1])
    batch_of_group: np.ndarray  # int32[R]

    @property
    def n_batches(self) -> int:
        return int(self.batch_ptr.shape[0] - 1)

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.batch_ptr)

    def batch_probabilities(self) -> np.ndarray:
        """P(batch b) = L_b / R under the proposal-preserving rule."""
        return self.sizes.astype(np.float64) / float(self.n_groups)

    def batches_of(self, groups) -> np.ndarray:
        """Batch index of each drawn group (b = batch_of_group[r])."""
        return self.batch_of_group[np.asarray(groups, dtype=np.int64)]

    def count_reloads(self, groups, loaded: int = -1) -> int:
        """Number of batch changes along a proposal sequence (the first load counts if loaded == -1)."""
        b = self.batches_of(groups)
        if b.size == 0:
            return 0
        prev = np.concatenate([[loaded], b[:-1]])
        return int(np.sum(b != prev))

    def to_dict(self) -> dict:
        return {"capacity": int(self.capacity), "n_groups": int(self.n_groups), "n_batches": self.n_batches,
                "sizes": [int(x) for x in self.sizes]}


@dataclass(frozen=True)
class BatchingSpec:
    capacity: int
    rule: str = "proposal_preserving"

    def __post_init__(self):
        object.__setattr__(self, "capacity", _int(self.capacity, "batching.capacity", 1))
        if self.rule not in BATCH_RULES:
            raise ConfigError(f"batching.rule {self.rule!r} not supported; only {BATCH_RULES} preserves the "
                              "uniform full-library proposal (dwell-time batching is a different process)")

    def plan(self, n_groups: int) -> BatchPlan:
        R = _int(n_groups, "n_groups", 1)
        C = self.capacity
        ptr = np.minimum(np.arange(0, R + C, C, dtype=np.int64), R)
        ptr = np.unique(ptr)
        if ptr[-1] != R:  # pragma: no cover - arange covers R
            ptr = np.append(ptr, R)
        bog = np.repeat(np.arange(ptr.shape[0] - 1, dtype=np.int32), np.diff(ptr))
        return BatchPlan(C, R, ptr, bog)

    def to_dict(self) -> dict:
        return {"capacity": int(self.capacity), "rule": self.rule}

    @staticmethod
    def from_dict(d) -> "BatchingSpec":
        d = _mapping(d, "batching")
        _reject_unknown(d, ("capacity", "rule"), "batching")
        if "capacity" not in d:
            raise ConfigError("batching needs 'capacity'")
        return BatchingSpec(d["capacity"], d.get("rule", "proposal_preserving"))


# --------------------------------------------------------------------------------------------
# Calibration and schedule specs
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationSpec:
    """Immutable arguments of :func:`nimbus.annealing.calibration.calibrate`."""

    rule: str = "acceptance"
    p_start: float = 0.8
    p_end: float = 0.01
    n_pilot: int = 2048
    late_fraction: float = 0.5
    budget_cap_fraction: float = 0.05
    min_pilot: int = 64
    min_positive: int = 8
    max_rounds: int = 2
    quantile: float = 0.5
    legacy_factor: float = 1.1
    class_weights: Optional[Mapping[str, float]] = None
    coefficient_scale: Optional[Any] = None  # float or [start, end]
    fallback: str = "coefficients"
    min_ratio: Optional[float] = None
    min_ratio_mode: str = "raise_start"

    _FIELDS = ("rule", "p_start", "p_end", "n_pilot", "late_fraction", "budget_cap_fraction", "min_pilot",
               "min_positive", "max_rounds", "quantile", "legacy_factor", "class_weights", "coefficient_scale",
               "fallback", "min_ratio", "min_ratio_mode")

    def __post_init__(self):
        if self.rule not in CALIBRATION_RULES:
            raise ConfigError(f"calibration rule {self.rule!r} unknown; expected one of {CALIBRATION_RULES}")
        ps = _float(self.p_start, "auto.p_start")
        pe = _float(self.p_end, "auto.p_end")
        if not (0 < pe < ps < 1):
            raise ConfigError(f"need 0 < p_end < p_start < 1, got p_start={ps}, p_end={pe}")
        object.__setattr__(self, "p_start", ps)
        object.__setattr__(self, "p_end", pe)
        object.__setattr__(self, "n_pilot", _int(self.n_pilot, "auto.n_pilot", 1))
        lf = _float(self.late_fraction, "auto.late_fraction")
        if not 0 <= lf < 1:
            raise ConfigError(f"auto.late_fraction must be in [0, 1), got {lf}")
        object.__setattr__(self, "late_fraction", lf)
        bc = _float(self.budget_cap_fraction, "auto.budget_cap_fraction")
        if not 0 < bc <= 1:
            raise ConfigError(f"auto.budget_cap_fraction must be in (0, 1], got {bc}")
        object.__setattr__(self, "budget_cap_fraction", bc)
        for name in ("min_pilot", "min_positive", "max_rounds"):
            object.__setattr__(self, name, _int(getattr(self, name), f"auto.{name}", 1))
        q = _float(self.quantile, "auto.quantile")
        if not 0 <= q <= 1:
            raise ConfigError(f"auto.quantile must be in [0, 1], got {q}")
        object.__setattr__(self, "quantile", q)
        lfac = _float(self.legacy_factor, "auto.legacy_factor")
        if lfac < 1:
            raise ConfigError(f"auto.legacy_factor must be >= 1, got {lfac}")
        object.__setattr__(self, "legacy_factor", lfac)
        if self.class_weights is not None:
            cw = _mapping(self.class_weights, "auto.class_weights")
            if not cw:
                raise ConfigError("auto.class_weights must not be empty")
            cw = {str(k): _float(v, f"auto.class_weights[{k!r}]", nonneg=True) for k, v in cw.items()}
            if not any(v > 0 for v in cw.values()):
                raise ConfigError("auto.class_weights must have a positive entry")
            object.__setattr__(self, "class_weights", cw)
        if self.coefficient_scale is not None:
            cs = self.coefficient_scale
            if isinstance(cs, (list, tuple)):
                if len(cs) != 2:
                    raise ConfigError("auto.coefficient_scale must be a number or [start, end]")
                cs = (_float(cs[0], "auto.coefficient_scale[0]", positive=True),
                      _float(cs[1], "auto.coefficient_scale[1]", positive=True))
            else:
                cs = _float(cs, "auto.coefficient_scale", positive=True)
            object.__setattr__(self, "coefficient_scale", cs)
        if self.fallback not in ("coefficients", "error"):
            raise ConfigError(f"auto.fallback must be 'coefficients' or 'error', got {self.fallback!r}")
        if self.min_ratio is not None:
            mr = _float(self.min_ratio, "auto.min_ratio")
            if mr < 1:
                raise ConfigError(f"auto.min_ratio (kappa) must be >= 1, got {mr}")
            object.__setattr__(self, "min_ratio", mr)
        if self.min_ratio_mode not in ("raise_start", "lower_end"):
            raise ConfigError(f"auto.min_ratio_mode must be 'raise_start' or 'lower_end', got {self.min_ratio_mode!r}")

    def to_kwargs(self) -> dict:
        """Keyword arguments for ``calibrate(sample_fn, **kwargs, run_attempts=..., weights=..., eta=...)``."""
        kw = {name: getattr(self, name) for name in self._FIELDS}
        if kw["class_weights"] is not None:
            kw["class_weights"] = dict(kw["class_weights"])
        return kw

    def to_dict(self) -> dict:
        d = self.to_kwargs()
        if isinstance(d["coefficient_scale"], tuple):
            d["coefficient_scale"] = list(d["coefficient_scale"])
        return d

    @staticmethod
    def from_dict(d) -> "CalibrationSpec":
        if d is True:
            return CalibrationSpec()
        d = _mapping(d, "schedule.auto")
        _reject_unknown(d, CalibrationSpec._FIELDS, "schedule.auto")
        return CalibrationSpec(**d)


@dataclass(frozen=True)
class ScheduleSpec:
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)
    auto: Optional[CalibrationSpec] = None

    def __post_init__(self):
        if self.kind not in SCHEDULE_KINDS:
            raise ConfigError(f"schedule.kind {self.kind!r} unknown; expected one of {SCHEDULE_KINDS}")
        params = _mapping(self.params, "schedule params")
        if self.auto is None:
            names = _EXPLICIT_PARAMS[self.kind]
            _reject_unknown(params, names, f"schedule ({self.kind}, explicit)")
            missing = [n for n in names if n not in params]
            if missing:
                raise ConfigError(f"schedule {self.kind} needs {missing} (or an 'auto' calibration spec)")
            params = {n: _float(params[n], f"schedule.{n}") for n in names}
            # Validate by building a one-attempt schedule (the schedule module checks the ranges).
            try:
                _build_explicit(self.kind, params, 1)
            except ValueError as exc:
                raise ConfigError(f"schedule {self.kind}: {exc}") from None
        else:
            if not isinstance(self.auto, CalibrationSpec):
                raise ConfigError("schedule.auto must be a CalibrationSpec")
            if self.kind not in _AUTO_PARAMS:
                raise ConfigError(f"schedule kind {self.kind!r} takes no calibration")
            required, optional = _AUTO_PARAMS[self.kind]
            temps = set(params) & {"tau_start", "tau_end", "tau0", "tau", "c_fn", "dt"}
            if temps:
                raise ConfigError(f"schedule has 'auto' calibration and explicit temperature fields {sorted(temps)}")
            _reject_unknown(params, tuple(required) + tuple(optional), f"schedule ({self.kind}, auto)")
            missing = [n for n in required if n not in params]
            if missing:
                raise ConfigError(f"auto schedule {self.kind} needs {missing}")
            out = dict(optional)
            out.update(params)
            for n in list(out):
                if n == "constant_at":
                    if out[n] not in ("start", "end"):
                        raise ConfigError("schedule.constant_at must be 'start' or 'end'")
                else:
                    out[n] = _float(out[n], f"schedule.{n}", positive=True)
            params = out
        object.__setattr__(self, "params", params)

    @property
    def is_auto(self) -> bool:
        return self.auto is not None

    def to_dict(self) -> dict:
        d = {"kind": self.kind}
        d.update(self.params)
        if self.auto is not None:
            d["auto"] = self.auto.to_dict()
        return d

    @staticmethod
    def from_dict(d) -> "ScheduleSpec":
        d = _mapping(d, "schedule")
        if "kind" not in d:
            raise ConfigError("schedule needs 'kind'")
        kind = d.pop("kind")
        auto = d.pop("auto", None)
        if auto is False:
            auto = None
        return ScheduleSpec(kind, d, None if auto is None else CalibrationSpec.from_dict(auto))


def _build_explicit(kind: str, p: Mapping[str, float], n_total: int) -> _schedule.Schedule:
    if kind == "fn_endpoints":
        return _schedule.fn_endpoints(p["tau_start"], p["tau_end"], p["eta"], n_total)
    if kind == "fn_legacy":
        return _schedule.fn_legacy(p["tau0"], p["c_fn"], p["t_start"], p["dt"], n_total)
    if kind == "geometric":
        return _schedule.geometric(p["tau_start"], p["tau_end"], n_total)
    if kind == "constant":
        return _schedule.constant(p["tau"], n_total)
    return _schedule.zero(n_total)


# --------------------------------------------------------------------------------------------
# Budget, stopping, policy
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetSpec:
    attempts: Optional[int] = None  # per replica (schedule length B)
    wall_time_s: Optional[float] = None  # whole run

    def __post_init__(self):
        if self.attempts is None and self.wall_time_s is None:
            raise ConfigError("budget needs 'attempts' and/or 'wall_time_s'")
        if self.attempts is not None:
            object.__setattr__(self, "attempts", _int(self.attempts, "budget.attempts", 1))
        if self.wall_time_s is not None:
            object.__setattr__(self, "wall_time_s", _float(self.wall_time_s, "budget.wall_time_s", positive=True))

    def to_dict(self) -> dict:
        return {"attempts": self.attempts, "wall_time_s": self.wall_time_s}

    @staticmethod
    def from_dict(d) -> "BudgetSpec":
        d = _mapping(d, "budget")
        _reject_unknown(d, ("attempts", "wall_time_s"), "budget")
        return BudgetSpec(d.get("attempts"), d.get("wall_time_s"))


@dataclass(frozen=True)
class StoppingSpec:
    target_energy: Optional[float] = None
    stop_on_target: bool = False
    target_tol: float = 0.0

    def __post_init__(self):
        if self.target_energy is not None:
            object.__setattr__(self, "target_energy", _float(self.target_energy, "stopping.target_energy"))
        object.__setattr__(self, "stop_on_target", _bool(self.stop_on_target, "stopping.stop_on_target"))
        object.__setattr__(self, "target_tol", _float(self.target_tol, "stopping.target_tol", nonneg=True))
        if self.stop_on_target and self.target_energy is None:
            raise ConfigError("stopping.stop_on_target needs stopping.target_energy")

    def hit(self, energy: float) -> bool:
        """E <= target + tol (False without a target)."""
        return self.target_energy is not None and float(energy) <= self.target_energy + self.target_tol

    def to_dict(self) -> dict:
        return {"target_energy": self.target_energy, "stop_on_target": self.stop_on_target,
                "target_tol": self.target_tol}

    @staticmethod
    def from_dict(d) -> "StoppingSpec":
        d = _mapping(d, "stopping")
        _reject_unknown(d, ("target_energy", "stop_on_target", "target_tol"), "stopping")
        return StoppingSpec(**d)


@dataclass(frozen=True)
class RunPolicy:
    schedule: ScheduleSpec
    budget: BudgetSpec
    proposal: str = "uniform"
    acceptance: str = "metropolis_threshold"
    batching: Optional[BatchingSpec] = None
    stopping: StoppingSpec = field(default_factory=StoppingSpec)
    replicas: int = 1

    _KEYS = ("schema", "proposal", "acceptance", "batching", "schedule", "budget", "stopping", "replicas")

    def __post_init__(self):
        if self.proposal not in PROPOSALS:
            raise ConfigError(f"proposal {self.proposal!r} not supported; expected one of {PROPOSALS} "
                              "(uniform over the fixed group list, self-loops kept)")
        if self.acceptance not in ACCEPTANCES:
            raise ConfigError(f"acceptance {self.acceptance!r} not supported; expected one of {ACCEPTANCES}")
        if not isinstance(self.schedule, ScheduleSpec):
            raise ConfigError("schedule must be a ScheduleSpec")
        if not isinstance(self.budget, BudgetSpec):
            raise ConfigError("budget must be a BudgetSpec")
        if self.batching is not None and not isinstance(self.batching, BatchingSpec):
            raise ConfigError("batching must be a BatchingSpec or None")
        if not isinstance(self.stopping, StoppingSpec):
            raise ConfigError("stopping must be a StoppingSpec")
        object.__setattr__(self, "replicas", _int(self.replicas, "replicas", 1))
        if self.schedule.kind in ("fn_endpoints", "geometric") and self.budget.attempts is None:
            raise ConfigError(f"schedule {self.schedule.kind} depends on the attempt budget B; set budget.attempts "
                              "(a wall-time-only budget needs constant, zero or fn_legacy)")

    # -- schedule resolution ---------------------------------------------------------------------
    @property
    def needs_calibration(self) -> bool:
        return self.schedule.is_auto

    def calibration_kwargs(self) -> dict:
        """kwargs for ``calibrate(sample_fn, **kw)``: the auto spec plus run_attempts and eta."""
        if not self.schedule.is_auto:
            raise ConfigError("schedule is explicit; no calibration requested")
        kw = self.schedule.auto.to_kwargs()
        if self.budget.attempts is not None:
            kw["run_attempts"] = int(self.budget.attempts) * int(self.replicas)
        if "eta" in self.schedule.params:
            kw["eta"] = float(self.schedule.params["eta"])
        return kw

    def resolve_schedule(self, calibration: Optional[CalibrationResult] = None,
                         n_total: Optional[int] = None) -> _schedule.Schedule:
        """Concrete attempt-indexed Schedule (B = n_total, default budget.attempts)."""
        if n_total is None:
            n_total = self.budget.attempts
        if n_total is None:
            raise ConfigError("schedule length unknown: pass n_total or set budget.attempts")
        n_total = _int(n_total, "n_total", 1)
        spec = self.schedule
        if not spec.is_auto:
            if calibration is not None:
                raise ConfigError("schedule is explicit; a calibration result would be ignored")
            return _build_explicit(spec.kind, spec.params, n_total)
        if calibration is None:
            raise ConfigError("schedule is 'auto': run calibrate() and pass its CalibrationResult")
        p = spec.params
        return make_schedule(calibration, spec.kind, n_total, p.get("eta"), t_start=p.get("t_start", 1.0),
                             constant_at=p.get("constant_at"))

    def batch_plan(self, n_groups: int) -> Optional[BatchPlan]:
        return None if self.batching is None else self.batching.plan(n_groups)

    def label(self) -> str:
        """Short human-readable algorithm label for result metadata."""
        s = self.schedule
        if s.is_auto:
            a = s.auto
            sched = f"{s.kind}[auto:{a.rule} p={a.p_start:g}->{a.p_end:g}" + (
                f" eta={s.params['eta']:g}" if "eta" in s.params else "") + "]"
        else:
            sched = s.kind + "[" + ",".join(f"{k}={v:g}" for k, v in s.params.items()) + "]"
        batch = "none" if self.batching is None else f"{self.batching.rule}(C={self.batching.capacity})"
        return f"proposal={self.proposal}; batching={batch}; acceptance={self.acceptance}; schedule={sched}"

    # -- (de)serialization -------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "proposal": self.proposal, "acceptance": self.acceptance,
                "batching": None if self.batching is None else self.batching.to_dict(),
                "schedule": self.schedule.to_dict(), "budget": self.budget.to_dict(),
                "stopping": self.stopping.to_dict(), "replicas": int(self.replicas)}

    @staticmethod
    def from_dict(d) -> "RunPolicy":
        d = _mapping(d, "policy")
        _reject_unknown(d, RunPolicy._KEYS, "policy")
        if d.get("schema", SCHEMA) != SCHEMA:
            raise ConfigError(f"unsupported policy schema {d.get('schema')!r}; expected {SCHEMA!r}")
        if "schedule" not in d or "budget" not in d:
            raise ConfigError("policy needs 'schedule' and 'budget'")
        batching = d.get("batching")
        return RunPolicy(
            schedule=ScheduleSpec.from_dict(d["schedule"]),
            budget=BudgetSpec.from_dict(d["budget"]),
            proposal=d.get("proposal", "uniform"),
            acceptance=d.get("acceptance", "metropolis_threshold"),
            batching=None if batching is None else BatchingSpec.from_dict(batching),
            stopping=StoppingSpec.from_dict(d.get("stopping", {})),
            replicas=d.get("replicas", 1),
        )

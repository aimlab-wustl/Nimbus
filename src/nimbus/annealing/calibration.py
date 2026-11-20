"""Pilot-based temperature calibration from the actual proposed moves.

The energy changes come from the ACTUAL configured transition library, objective and proposal
mixture. The runner supplies ``sample_fn(phase, n)``: it draws ``n`` proposals with the production
proposal rule (uniform over the fixed group list) from the PILOT RNG stream, on cloned pilot states,
and returns Delta E = 2 v_r for each selected group, where v_r is read through an architecture
backend's ``evaluate_inputs`` (the weighted encoder over stored clauses). It is never a
problem-native delta and never a single-spin surrogate for multi-spin production moves. Each call
must use fresh pilot states/proposals (advance the PILOT counters), so that a retry round sees new
states. Self-loops (v = 0) and zero deltas are kept in the sample for the bookkeeping but do not
determine the uphill energy scale: every rule below uses only the positive changes.

Rules (temperatures in the units of Delta E, i.e. an uphill change dE is accepted w.p. exp(-dE/tau)):

``acceptance`` (default, the proposed pilot rule)
    Solve (1/L) sum_j exp(-dE_j / tau) = p over the L positive dE_j, by bisection in log tau
    (``acceptance_target_temperature``). Start: p_start (0.8), end: p_end (0.01).
``quantile``
    tau = Q_q(dE+) / log(1/p): the q-quantile uphill change is accepted with probability p
    (``quantile_rule``), robust to outliers.
``legacy_max``
    N-HOIM supplement: the FN amplitude A is chosen slightly above an observed maximum energy change,
    A = factor * max(dE+) (``legacy_max_rule``). Because tau(t) = A / (C log(1 + t/C)) ~ A / t for
    t/C << 1 and the legacy clock starts at t ~ 1, the start temperature is tau_start = A. The legacy
    rule does not define an end temperature; the end uses the observed maximum at the end target,
    tau_end = max(dE+) / log(1/p_end) (= ``quantile_rule`` with q = 1).

Phases. With ``late_fraction`` f > 0, the pilot budget is split into a start phase (``sample_fn('start',
n_start)``, initial/random pilot states) and an end phase (``sample_fn('end', n_end)``). The phase
argument is a :class:`PilotPhase` (a ``str`` subclass equal to ``'start'``/``'end'``) whose attribute
``tau_start`` carries the calibrated start temperature in the end phase, so the runner can advance its
cloned pilot states by a short anneal before sampling late-pilot deltas (late-pilot deltas may give
a better end-temperature scale). With f = 0 both temperatures come from the start sample.

Budget. ``n_pilot`` (default 2048) attempted proposals in total. If ``run_attempts`` (the intended total
search attempts) is given, the pilot is capped at ``floor(budget_cap_fraction * run_attempts)``. If the
cap leaves fewer than ``min_pilot`` attempts, a supplied ``cached`` calibration (family-level, truly
reusable) is returned, otherwise the coefficient-scale fallback is used (with a warning), otherwise a
:class:`CalibrationError` is raised. Pilot attempts and wall time are reported so they can be charged to
end-to-end results.

Flat / one-sided samples. A phase is *flat* if it has no nonzero changes, *one-sided* if it has no
positive changes, and *thin* if it has fewer than ``min_positive`` positive changes. Then up to
``max_rounds - 1`` retry rounds of the same size are requested from ``sample_fn`` (fresh states) while the
pilot limit allows. If the end phase is still insufficient, the pooled start + end positives are used.
Only then does the documented coefficient-scale fallback apply:

    dE_hi = coefficient scale for the start (default: median over groups of the group bound
            2 * sum_{weighted routes of r} |J_k| when routes are supplied, else 2 * max_k |J_k|)
    dE_lo = coefficient scale for the end (default: 2 * min_{J_k != 0} |J_k|, floored at 1e-12 dE_hi)
    tau_start = dE_hi / log(1/p_start),  tau_end = dE_lo / log(1/p_end)

i.e. a change equal to the scale is accepted with the target probability. An explicit
``coefficient_scale`` (one float or a (start, end) pair) overrides the derived values. Every fallback
emits a :class:`CalibrationWarning` and sets ``fallback_used``. No path divides by an empty sample or
returns a zero, NaN or infinite temperature: if no valid temperature can be formed a
:class:`CalibrationError` is raised. If the end temperature exceeds the start temperature it is
clipped to the start value (warning), as every endpoint schedule needs tau_start >= tau_end.

Class weights. If the pilot samples move classes disproportionately (e.g. a stratified pilot), pass the
per-delta class label in the sample and ``class_weights`` = production proposal probability of each
class. Each delta of class c then carries weight pi_c / n_c (n_c = all pilot deltas of class c, zeros
included), so that the weighted positive-delta statistics estimate the production-mixture
expectation E[f(dE) | dE > 0]. When the pilot already uses the production proposal rule, no weights
are needed (the default).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from nimbus.annealing import schedule as _schedule
from nimbus.core.errors import ConfigError, NimbusError

__all__ = [
    "CalibrationError",
    "CalibrationWarning",
    "PilotPhase",
    "PilotSample",
    "CalibrationResult",
    "RULES",
    "acceptance_target_temperature",
    "quantile_rule",
    "legacy_max_rule",
    "coefficient_scale_from_weights",
    "coefficient_scale_for_model",
    "fallback_temperatures",
    "sample_statistics",
    "calibrate",
    "make_schedule",
]

RULES = ("acceptance", "quantile", "legacy_max")
SCHEMA = "nimbus.calibration/1"


class CalibrationError(NimbusError, ValueError):
    """No valid temperature can be formed (empty/invalid sample and no usable fallback)."""


class CalibrationWarning(UserWarning):
    """A calibration fallback or adjustment was applied (also listed in ``CalibrationResult.warnings``)."""


class PilotPhase(str):
    """Phase name passed to ``sample_fn``: compares equal to ``'start'`` or ``'end'``.

    ``tau_start`` is None in the start phase and the calibrated start temperature in the end phase;
    ``round`` is the 0-based retry round of this phase.
    """

    tau_start: Optional[float]
    round: int

    def __new__(cls, name: str, tau_start: Optional[float] = None, round_index: int = 0):
        if name not in ("start", "end"):
            raise ValueError(f"pilot phase must be 'start' or 'end', got {name!r}")
        obj = super().__new__(cls, name)
        obj.tau_start = None if tau_start is None else float(tau_start)
        obj.round = int(round_index)
        return obj


@dataclass
class PilotSample:
    """What ``sample_fn`` returns (a bare array of deltas or a mapping with these keys is also accepted).

    deltas   : float64[n], Delta E = 2 v_r of each attempted pilot proposal (all signs, zeros kept).
    classes  : optional int/str labels per delta (move class), used only with ``class_weights``.
    attempts : pilot attempts consumed (default len(deltas)); charged to the pilot budget.
    info     : free-form runner metadata (e.g. pilot RNG coordinates, number of pilot states).
    """

    deltas: np.ndarray
    classes: Optional[np.ndarray] = None
    attempts: Optional[int] = None
    info: dict = field(default_factory=dict)


@dataclass
class CalibrationResult:
    tau_start: float
    tau_end: float
    rule: str  # "acceptance" | "quantile" | "legacy_max" | "coefficient_fallback" | "cached"
    details: dict
    n_samples: int  # pilot deltas received (all signs)
    n_positive: int  # positive deltas among them
    pilot_attempts: int  # attempted pilot proposals charged to the budget
    seconds: float  # wall time of the calibration, including sample_fn
    warnings: list = field(default_factory=list)
    fallback_used: bool = False
    eta: Optional[float] = None  # FN shape passthrough (t_end / C) for make_schedule
    p_start: Optional[float] = None
    p_end: Optional[float] = None

    def to_dict(self) -> dict:
        return _jsonable({
            "schema": SCHEMA,
            "tau_start": float(self.tau_start),
            "tau_end": float(self.tau_end),
            "rule": self.rule,
            "details": self.details,
            "n_samples": int(self.n_samples),
            "n_positive": int(self.n_positive),
            "pilot_attempts": int(self.pilot_attempts),
            "seconds": float(self.seconds),
            "warnings": list(self.warnings),
            "fallback_used": bool(self.fallback_used),
            "eta": None if self.eta is None else float(self.eta),
            "p_start": None if self.p_start is None else float(self.p_start),
            "p_end": None if self.p_end is None else float(self.p_end),
        })

    @staticmethod
    def from_dict(d: Mapping) -> "CalibrationResult":
        if d.get("schema", SCHEMA) != SCHEMA:
            raise ConfigError(f"unsupported calibration schema {d.get('schema')!r}")
        res = CalibrationResult(
            tau_start=float(d["tau_start"]), tau_end=float(d["tau_end"]), rule=str(d["rule"]),
            details=dict(d.get("details", {})), n_samples=int(d.get("n_samples", 0)),
            n_positive=int(d.get("n_positive", 0)), pilot_attempts=int(d.get("pilot_attempts", 0)),
            seconds=float(d.get("seconds", 0.0)), warnings=list(d.get("warnings", [])),
            fallback_used=bool(d.get("fallback_used", False)),
            eta=None if d.get("eta") is None else float(d["eta"]),
            p_start=None if d.get("p_start") is None else float(d["p_start"]),
            p_end=None if d.get("p_end") is None else float(d["p_end"]))
        _check_temperature(res.tau_start, "tau_start")
        _check_temperature(res.tau_end, "tau_end")
        return res


# --------------------------------------------------------------------------------------------
# Elementary rules
# --------------------------------------------------------------------------------------------

def _check_probability(p, name="p_target") -> float:
    try:
        p = float(p)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a number in (0, 1), got {p!r}") from None
    if not (0.0 < p < 1.0):
        raise ConfigError(f"{name} must lie strictly inside (0, 1), got {p}")
    return p


def _check_temperature(tau, name) -> float:
    tau = float(tau)
    if not (math.isfinite(tau) and tau > 0.0):
        raise CalibrationError(f"{name} must be finite and > 0, got {tau}")
    return tau


def _as_weights(weights, n) -> Optional[np.ndarray]:
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != n:
        raise CalibrationError(f"weights have length {w.shape[0]}, deltas have length {n}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise CalibrationError("weights must be finite and >= 0")
    if not np.any(w > 0):
        raise CalibrationError("all weights are zero (empty effective sample)")
    return w


def _positive_sample(deltas_positive, weights, what):
    d = np.asarray(deltas_positive, dtype=np.float64).ravel()
    if d.size == 0:
        raise CalibrationError(f"{what}: empty sample of positive energy changes")
    if not np.all(np.isfinite(d)):
        raise CalibrationError(f"{what}: non-finite energy changes in the sample")
    if np.any(d <= 0):
        raise CalibrationError(f"{what}: expects strictly positive energy changes")
    w = _as_weights(weights, d.shape[0])
    if w is not None:
        keep = w > 0
        d, w = d[keep], w[keep] / np.sum(w[keep])
    return d, w


def acceptance_target_temperature(deltas_positive, p_target, *, weights=None, rtol: float = 1e-12,
                                  max_iter: int = 200, full_output: bool = False):
    """Solve mean_j exp(-dE_j / tau) = p_target over positive dE_j (optionally weighted).

    The mean m(tau) is strictly increasing from 0 to 1, so the root is unique. Bracket:
    exp(-dE_max/tau) <= m(tau) <= exp(-dE_mean/tau) (Jensen) and m(tau) <= exp(-dE_min/tau) give
    tau in [dE_min / L, dE_mean / L] with L = log(1 / p). Bisection runs on log tau until the
    bracket ratio is within ``rtol`` or ``max_iter`` halvings (a non-converged result raises).
    Returns tau (and an info dict if ``full_output``).
    """
    p = _check_probability(p_target)
    d, w = _positive_sample(deltas_positive, weights, "acceptance rule")
    if not (rtol > 0) or int(max_iter) < 1:
        raise ConfigError("need rtol > 0 and max_iter >= 1")
    L = -math.log(p)

    def mean_acc(tau):
        e = np.exp(-d / tau)
        return float(np.mean(e)) if w is None else float(np.dot(w, e))

    d_min, d_max = float(np.min(d)), float(np.max(d))
    d_mean = float(np.mean(d)) if w is None else float(np.dot(w, d))
    info = {"bracket_initial": None, "iterations": 0, "converged": True, "n": int(d.size)}
    if d_max <= d_min * (1.0 + 1e-15):  # (numerically) constant sample: exact closed form
        tau = d_min / L
        info.update(bracket_initial=[tau, tau], residual=mean_acc(tau) - p, method="closed_form")
        return (tau, info) if full_output else tau
    lo, hi = d_min / L, max(d_mean, d_min) / L
    # Guard the theoretical bracket against rounding; expand geometrically if needed.
    for _ in range(64):
        if mean_acc(lo) <= p:
            break
        lo *= 0.5
    for _ in range(64):
        if mean_acc(hi) >= p:
            break
        hi *= 2.0
    if not (mean_acc(lo) <= p <= mean_acc(hi)):
        raise CalibrationError("acceptance rule: could not bracket the target temperature")
    info["bracket_initial"] = [lo, hi]
    log_lo, log_hi = math.log(lo), math.log(hi)
    # Stop at the requested ratio, but never below what doubles can resolve in log space.
    tol = max(math.log1p(rtol), 8.0 * np.finfo(np.float64).eps * max(1.0, abs(log_lo), abs(log_hi)))
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        mid = 0.5 * (log_lo + log_hi)
        if mean_acc(math.exp(mid)) < p:
            log_lo = mid
        else:
            log_hi = mid
        if log_hi - log_lo <= tol:
            converged = True
            break
    tau = math.exp(0.5 * (log_lo + log_hi))
    info.update(iterations=it, converged=converged, residual=mean_acc(tau) - p, method="bisection_log_tau")
    if not converged:
        raise CalibrationError(f"acceptance rule: bisection did not converge in {max_iter} iterations")
    _check_temperature(tau, "acceptance-rule temperature")
    return (tau, info) if full_output else tau


def _positive_part(deltas, weights, zero_tol):
    d = np.asarray(deltas, dtype=np.float64).ravel()
    if not np.all(np.isfinite(d)):
        raise CalibrationError("non-finite energy changes in the sample")
    w = _as_weights(weights, d.shape[0]) if weights is not None else None
    keep = d > float(zero_tol)
    if w is not None:
        keep &= w > 0
    if not np.any(keep):
        raise CalibrationError("no positive energy changes in the sample (flat or one-sided)")
    return d[keep], (None if w is None else w[keep])


def _weighted_quantile(d, w, q):
    """Inverted-CDF weighted quantile: smallest d with cumulative weight >= q (q = 0: min, q = 1: max)."""
    if q <= 0.0:
        return float(np.min(d))
    if q >= 1.0:
        return float(np.max(d))
    order = np.argsort(d, kind="stable")
    d, w = d[order], w[order]
    c = np.cumsum(w)
    c /= c[-1]
    idx = int(np.searchsorted(c, q, side="left"))
    return float(d[min(idx, d.size - 1)])


def quantile_rule(deltas, q, p, *, weights=None, zero_tol: float = 0.0) -> float:
    """tau = Q_q(dE+) / log(1/p) over the positive changes (dE > zero_tol).

    Unweighted: numpy linear-interpolation quantile; weighted: inverted-CDF quantile. q in [0, 1]
    (q = 1 is the observed maximum).
    """
    p = _check_probability(p, "p")
    q = float(q)
    if not (0.0 <= q <= 1.0):
        raise ConfigError(f"quantile q must be in [0, 1], got {q}")
    d, w = _positive_part(deltas, weights, zero_tol)
    Q = float(np.quantile(d, q)) if w is None else _weighted_quantile(d, w, q)
    return _check_temperature(Q / -math.log(p), "quantile-rule temperature")


def legacy_max_rule(deltas, factor: float = 1.1, *, weights=None, zero_tol: float = 0.0) -> float:
    """FN amplitude A = factor * max(dE+) ("A slightly above an observed maximum energy change").

    Zero-weight entries (classes absent from the production mixture) are ignored. factor >= 1.
    """
    factor = float(factor)
    if not (math.isfinite(factor) and factor >= 1.0):
        raise ConfigError(f"legacy factor must be finite and >= 1 (slightly above the maximum), got {factor}")
    d, _ = _positive_part(deltas, weights, zero_tol)
    return _check_temperature(factor * float(np.max(d)), "legacy amplitude A")


# --------------------------------------------------------------------------------------------
# Coefficient-scale fallback
# --------------------------------------------------------------------------------------------

def coefficient_scale_from_weights(J, *, route_ptr=None, route_wend=None, route_clause=None):
    """(dE_hi, dE_lo) energy-change scales from the run-slot clause weights J (fallback only).

    dE_hi: median over groups with a nonzero bound of 2 * sum_{weighted routes of r} |J_k| when the
    route CSR (route_ptr, route_wend, route_clause) is given, else 2 * max_k |J_k|.
    dE_lo: 2 * min over nonzero |J_k| (floored at 1e-12 * dE_hi), the smallest single-clause step.
    Raises CalibrationError if all weights vanish.
    """
    a = np.abs(np.asarray(J, dtype=np.float64).ravel())
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise CalibrationError("coefficient scale needs finite clause weights")
    nz = a[a > 0]
    if nz.size == 0:
        raise CalibrationError("all clause weights are zero: no coefficient scale")
    hi = 2.0 * float(np.max(nz))
    if route_ptr is not None:
        if route_wend is None or route_clause is None:
            raise ConfigError("route_ptr, route_wend and route_clause must be given together")
        ptr = np.asarray(route_ptr, dtype=np.int64)
        wend = np.asarray(route_wend, dtype=np.int64)
        rc = np.asarray(route_clause, dtype=np.int64)
        contrib = a[rc]
        csum = np.concatenate([[0.0], np.cumsum(contrib)])
        bounds = 2.0 * (csum[wend] - csum[ptr[:-1]])
        bounds = bounds[bounds > 0]
        if bounds.size:
            hi = float(np.median(bounds))
    lo = max(2.0 * float(np.min(nz)), 1e-12 * hi)
    return hi, min(lo, hi)


def coefficient_scale_for_model(model, binding, slot: int = 0):
    """Duck-typed wrapper: uses ``model.route_ptr/route_wend/route_clause`` and ``binding.J[slot]``."""
    return coefficient_scale_from_weights(np.asarray(binding.J)[slot], route_ptr=model.route_ptr,
                                          route_wend=model.route_wend, route_clause=model.route_clause)


def fallback_temperatures(scale_start: float, scale_end: float, p_start: float, p_end: float):
    """tau = scale / log(1/p) for each endpoint (a change equal to the scale accepted w.p. p)."""
    p_start = _check_probability(p_start, "p_start")
    p_end = _check_probability(p_end, "p_end")
    s0, s1 = float(scale_start), float(scale_end)
    if not (math.isfinite(s0) and s0 > 0 and math.isfinite(s1) and s1 > 0):
        raise CalibrationError(f"coefficient scales must be finite and > 0, got {s0}, {s1}")
    return (_check_temperature(s0 / -math.log(p_start), "fallback tau_start"),
            _check_temperature(s1 / -math.log(p_end), "fallback tau_end"))


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

def sample_statistics(deltas, zero_tol: float = 0.0) -> dict:
    d = np.asarray(deltas, dtype=np.float64).ravel()
    pos = d[d > zero_tol]
    out = {"n": int(d.size), "n_positive": int(pos.size), "n_zero": int(np.sum(np.abs(d) <= zero_tol)),
           "n_negative": int(np.sum(d < -zero_tol)), "max_abs": float(np.max(np.abs(d))) if d.size else 0.0}
    if pos.size:
        qs = np.quantile(pos, [0.0, 0.1, 0.5, 0.9, 1.0])
        out.update(pos_min=float(qs[0]), pos_q10=float(qs[1]), pos_median=float(qs[2]), pos_q90=float(qs[3]),
                   pos_max=float(qs[4]), pos_mean=float(np.mean(pos)))
    out["flat"] = bool(d.size == 0 or out["max_abs"] <= zero_tol)
    out["one_sided"] = bool(pos.size == 0)
    return out


def _coerce_sample(obj) -> PilotSample:
    if isinstance(obj, PilotSample):
        s = obj
    elif isinstance(obj, Mapping):
        unknown = set(obj) - {"deltas", "classes", "attempts", "info"}
        if unknown or "deltas" not in obj:
            raise ConfigError(f"sample_fn mapping needs 'deltas' (+ optional classes/attempts/info); got {sorted(obj)}")
        s = PilotSample(obj["deltas"], obj.get("classes"), obj.get("attempts"), dict(obj.get("info", {})))
    else:
        s = PilotSample(np.asarray(obj))
    d = np.asarray(s.deltas, dtype=np.float64).ravel()
    if not np.all(np.isfinite(d)):
        raise CalibrationError("sample_fn returned non-finite energy changes (architecture inputs must be finite)")
    classes = None if s.classes is None else np.asarray(s.classes).ravel()
    if classes is not None and classes.shape[0] != d.shape[0]:
        raise CalibrationError("sample_fn: classes and deltas differ in length")
    attempts = int(d.shape[0] if s.attempts is None else s.attempts)
    if attempts < 0:
        raise CalibrationError("sample_fn reported a negative number of attempts")
    return PilotSample(d, classes, attempts, dict(s.info))


def _class_weights(classes, class_weights, warn_list):
    """Per-delta weights pi_c / n_c (n_c counts all deltas of class c), or None."""
    if class_weights is None:
        return None
    if classes is None:
        raise ConfigError("class_weights given but sample_fn returned no per-delta classes")
    labels, counts = np.unique(classes, return_counts=True)
    w = np.empty(classes.shape[0], dtype=np.float64)
    keys = {str(k): v for k, v in class_weights.items()}
    for lab, cnt in zip(labels, counts):
        key = str(lab.item() if hasattr(lab, "item") else lab)
        if key not in keys:
            raise ConfigError(f"pilot class {key!r} has no production weight in class_weights")
        pi = float(keys[key])
        if not (math.isfinite(pi) and pi >= 0):
            raise ConfigError(f"class weight of {key!r} must be finite and >= 0")
        w[classes == lab] = pi / cnt
    present = {str(lab.item() if hasattr(lab, "item") else lab) for lab in labels}
    for key, pi in keys.items():
        if key not in present and float(pi) > 0:
            warn_list.append(f"class {key!r} has production weight {pi} but no pilot samples")
    return w


def _n_effective_positive(d, w, zero_tol) -> int:
    """Positive changes that carry weight (classes with zero production weight do not count)."""
    keep = d > zero_tol
    if w is not None:
        keep &= w > 0
    return int(np.sum(keep))


def _phase_temperature(rule, d, w, p, zero_tol, quantile, legacy_factor, rtol, max_iter, which):
    """Temperature of one phase from deltas d (all signs) and optional weights w."""
    keep = d > zero_tol
    if w is not None:
        keep &= w > 0
    pos = d[keep]
    pw = None if w is None else w[keep]
    info: dict = {"n_used": int(pos.size)}
    if rule == "acceptance":
        tau, sinfo = acceptance_target_temperature(pos, p, weights=pw, rtol=rtol, max_iter=max_iter, full_output=True)
        info["solver"] = sinfo
    elif rule == "quantile":
        tau = quantile_rule(pos, quantile, p, weights=pw)
        info["quantile"] = float(quantile)
    elif rule == "legacy_max":
        if which == "start":
            tau = legacy_max_rule(pos, legacy_factor, weights=pw)
            info["A"] = tau
            info["factor"] = float(legacy_factor)
            info["observed_max"] = float(np.max(pos))
        else:
            tau = quantile_rule(pos, 1.0, p, weights=pw)
            info["observed_max"] = float(np.max(pos))
    else:  # pragma: no cover - validated by the caller
        raise ConfigError(f"unknown calibration rule {rule!r}")
    return _check_temperature(tau, f"{which} temperature"), info


def calibrate(sample_fn: Callable[[PilotPhase, int], Any], rule: str = "acceptance", p_start: float = 0.8,
              p_end: float = 0.01, n_pilot: int = 2048, late_fraction: float = 0.5,
              budget_cap_fraction: float = 0.05, *, run_attempts: Optional[int] = None, min_pilot: int = 64,
              min_positive: int = 8, max_rounds: int = 2, quantile: float = 0.5, legacy_factor: float = 1.1,
              class_weights: Optional[Mapping[Any, float]] = None, coefficient_scale=None, weights=None,
              routes: Optional[tuple] = None, fallback: str = "coefficients",
              cached: Optional[CalibrationResult] = None, eta: Optional[float] = None,
              rel_zero_tol: float = 1e-12, rtol: float = 1e-12, max_iter: int = 200,
              min_ratio: Optional[float] = None, min_ratio_mode: str = "raise_start") -> CalibrationResult:
    """Choose (tau_start, tau_end) from pilot energy changes of the production proposal process.

    Parameters (see the module docstring for the rules and fallbacks):
      sample_fn(phase, n)  runner callback; phase is a PilotPhase ('start'/'end'), returns deltas
                           (Delta E = 2 v) as an array, a PilotSample or a mapping with 'deltas'.
      rule                 'acceptance' | 'quantile' | 'legacy_max'.
      p_start, p_end       acceptance targets in (0, 1), p_end < p_start.
      n_pilot              total pilot attempts (default 2048); late_fraction in [0, 1) goes to the end phase.
      budget_cap_fraction  cap relative to run_attempts (ignored when run_attempts is None).
      min_pilot            smallest pilot worth running; below it use cached / fallback.
      min_positive         positive changes needed per phase before falling back.
      max_rounds           sampling rounds per phase (>= 1), each of the phase size, within the pilot limit.
      class_weights        production proposal probability per move class (needs per-delta classes).
      coefficient_scale    float or (start, end) energy-change scale for the fallback (overrides weights).
      weights, routes      run-slot clause weights J and optional (route_ptr, route_wend, route_clause)
                           used by ``coefficient_scale_from_weights`` for the fallback.
      fallback             'coefficients' (documented scale fallback) or 'error' (raise instead).
      cached               reusable CalibrationResult used when the budget is too small.
      eta                  FN shape t_end / C, passed through to the result for ``make_schedule``.
      rel_zero_tol         |dE| <= rel_zero_tol * max|dE| counts as zero (round-off of cancelling clauses).
      min_ratio_mode       'raise_start' (default): tau_start = max(tau_start, kappa tau_end) as below. 'lower_end': keep
                           tau_start and set tau_end = min(tau_end, tau_start / kappa): an FN anneal over a factor kappa
                           from the calibrated start (on graph matching this works far better than raising the start;
                           see nimbus.annealing.protocol).
      min_ratio            kappa >= 1 or None. tau_start = max(tau_start, kappa * tau_end) (applied before the
                           tau_end <= tau_start clip), so a calibration whose end pilot is hotter than its start
                           pilot still anneals over a factor kappa instead of degenerating to a constant
                           temperature.
    """
    t0 = time.perf_counter()
    if rule not in RULES:
        raise ConfigError(f"unknown calibration rule {rule!r}; expected one of {RULES}")
    p_start = _check_probability(p_start, "p_start")
    p_end = _check_probability(p_end, "p_end")
    if not p_end < p_start:
        raise ConfigError(f"need p_end < p_start, got p_start={p_start}, p_end={p_end}")
    n_pilot = int(n_pilot)
    if n_pilot < 1:
        raise ConfigError("n_pilot must be >= 1")
    late_fraction = float(late_fraction)
    if not (0.0 <= late_fraction < 1.0):
        raise ConfigError(f"late_fraction must be in [0, 1), got {late_fraction}")
    budget_cap_fraction = float(budget_cap_fraction)
    if not (0.0 < budget_cap_fraction <= 1.0):
        raise ConfigError(f"budget_cap_fraction must be in (0, 1], got {budget_cap_fraction}")
    if int(min_pilot) < 1 or int(min_positive) < 1 or int(max_rounds) < 1:
        raise ConfigError("min_pilot, min_positive and max_rounds must be >= 1")
    if fallback not in ("coefficients", "error"):
        raise ConfigError(f"fallback must be 'coefficients' or 'error', got {fallback!r}")
    if eta is not None and not (math.isfinite(float(eta)) and float(eta) > 0):
        raise ConfigError(f"eta must be finite and > 0, got {eta}")
    if not (0.0 <= float(rel_zero_tol) < 1.0):
        raise ConfigError("rel_zero_tol must be in [0, 1)")
    eta = None if eta is None else float(eta)
    min_pilot, min_positive, max_rounds = int(min_pilot), int(min_positive), int(max_rounds)
    explicit_scale = _parse_scale(coefficient_scale)
    if class_weights is not None and (not isinstance(class_weights, Mapping) or not class_weights):
        raise ConfigError("class_weights must be a non-empty mapping {class label: production probability}")
    if routes is not None and (not isinstance(routes, (tuple, list)) or len(routes) != 3):
        raise ConfigError("routes must be (route_ptr, route_wend, route_clause)")

    warn_list: list = []
    details: dict = {"rule": rule, "p_start": p_start, "p_end": p_end, "n_pilot_requested": n_pilot,
                     "late_fraction": late_fraction, "budget_cap_fraction": budget_cap_fraction,
                     "run_attempts": None if run_attempts is None else int(run_attempts),
                     "min_pilot": min_pilot, "min_positive": min_positive, "max_rounds": max_rounds,
                     "class_weights": None if class_weights is None else {str(k): float(v) for k, v in class_weights.items()},
                     "fallback": {"start": None, "end": None, "scale": None}, "phases": {}}

    # ---- budget -----------------------------------------------------------------------------
    limit = n_pilot * max_rounds
    n_eff = n_pilot
    if run_attempts is not None:
        if int(run_attempts) < 1:
            raise ConfigError("run_attempts must be >= 1")
        cap = int(math.floor(budget_cap_fraction * int(run_attempts)))
        details["budget_cap"] = cap
        n_eff = min(n_pilot, cap)
        limit = min(limit, cap)
    details["n_pilot_effective"] = n_eff

    def scales():
        if explicit_scale is not None:
            s0, s1 = explicit_scale
            src = "explicit"
        elif weights is not None:
            if routes is not None:
                rp, rw, rc = routes
                s0, s1 = coefficient_scale_from_weights(weights, route_ptr=rp, route_wend=rw, route_clause=rc)
                src = "weights+routes"
            else:
                s0, s1 = coefficient_scale_from_weights(weights)
                src = "weights"
        else:
            return None
        details["fallback"]["scale"] = {"start": s0, "end": s1, "source": src}
        return s0, s1

    def finish(tau_s, tau_e, used_rule, n_samples, n_positive, attempts, fb):
        if min_ratio is not None and min_ratio_mode == "lower_end" and tau_e * float(min_ratio) > tau_s:
            details["tau_end_before_min_ratio"] = tau_e
            tau_e = tau_s / float(min_ratio)
        elif min_ratio is not None and tau_s < float(min_ratio) * tau_e:
            details["tau_start_before_min_ratio"] = tau_s
            tau_s = float(min_ratio) * tau_e
        if tau_e > tau_s:
            warn_list.append(f"tau_end {tau_e:.6g} > tau_start {tau_s:.6g}; clipped tau_end to tau_start")
            details["tau_end_unclipped"] = tau_e
            tau_e = tau_s
        tau_s = _check_temperature(tau_s, "tau_start")
        tau_e = _check_temperature(tau_e, "tau_end")
        for msg in warn_list:
            warnings.warn(msg, CalibrationWarning, stacklevel=3)
        return CalibrationResult(tau_start=tau_s, tau_end=tau_e, rule=used_rule, details=_jsonable(details),
                                 n_samples=int(n_samples), n_positive=int(n_positive), pilot_attempts=int(attempts),
                                 seconds=time.perf_counter() - t0, warnings=list(warn_list), fallback_used=bool(fb),
                                 eta=eta, p_start=p_start, p_end=p_end)

    if n_eff < min_pilot:
        reason = f"pilot budget {n_eff} < min_pilot {min_pilot}"
        if cached is not None:
            warn_list.append(f"{reason}; using the cached calibration")
            details["cached"] = cached.to_dict()
            details["fallback"]["start"] = details["fallback"]["end"] = "cached"
            return finish(cached.tau_start, cached.tau_end, "cached", 0, 0, 0, True)
        sc = scales() if fallback == "coefficients" else None
        if sc is None:
            raise CalibrationError(f"{reason}, and neither a cached calibration nor a coefficient scale is available")
        warn_list.append(f"{reason}; using the coefficient-scale fallback")
        details["fallback"]["start"] = details["fallback"]["end"] = reason
        ts, te = fallback_temperatures(sc[0], sc[1], p_start, p_end)
        return finish(ts, te, "coefficient_fallback", 0, 0, 0, True)

    n_end = int(round(late_fraction * n_eff)) if late_fraction > 0 else 0
    n_end = min(n_end, n_eff - 1)
    n_start = n_eff - n_end
    details["n_start"], details["n_end"] = n_start, n_end
    used = {"attempts": 0}

    def collect(phase_name, n, tau_hint):
        """Rounds of sample_fn for one phase until min_positive positives or the limit is reached."""
        ds, cs, infos = [], [], []
        any_classes = None
        rnd = 0
        while True:
            obj = sample_fn(PilotPhase(phase_name, tau_hint, rnd), n)
            s = _coerce_sample(obj)
            used["attempts"] += s.attempts
            ds.append(s.deltas)
            has_cls = s.classes is not None
            if any_classes is None:
                any_classes = has_cls
            elif any_classes != has_cls:
                raise CalibrationError("sample_fn returned classes in some rounds but not others")
            if has_cls:
                cs.append(s.classes)
            infos.append(s.info)
            d = np.concatenate(ds)
            zt = float(rel_zero_tol) * (float(np.max(np.abs(d))) if d.size else 0.0)
            npos = int(np.sum(d > zt))
            rnd += 1
            if npos >= min_positive or rnd >= max_rounds or used["attempts"] + n > limit:
                break
            warn_list.append(f"{phase_name} pilot round {rnd}: only {npos} positive changes; sampling fresh pilot states")
        classes = np.concatenate(cs) if any_classes else None
        return d, classes, zt, rnd, infos

    def phase_record(name, d, zt, rounds, infos):
        st = sample_statistics(d, zt)
        st.update(rounds=rounds, zero_tol=zt, runner_info=infos)
        details["phases"][name] = st
        return st

    # ---- start phase --------------------------------------------------------------------------
    d0, c0, zt0, r0, inf0 = collect("start", n_start, None)
    st0 = phase_record("start", d0, zt0, r0, inf0)
    w0 = _class_weights(c0, class_weights, warn_list)
    fb_used = False
    if _n_effective_positive(d0, w0, zt0) >= min_positive:
        tau_s, info = _phase_temperature(rule, d0, w0, p_start, zt0, quantile, legacy_factor, rtol, max_iter, "start")
        st0.update(source="pilot", tau=tau_s, **info)
    else:
        kind = "flat" if st0["flat"] else ("one-sided" if st0["one_sided"] else "too few positive changes")
        reason = f"start pilot {kind} ({st0['n_positive']} positive of {st0['n']})"
        sc = scales() if fallback == "coefficients" else None
        if sc is None:
            raise CalibrationError(f"{reason} and no coefficient scale available (pass weights or coefficient_scale)")
        tau_s = fallback_temperatures(sc[0], sc[1], p_start, p_end)[0]
        warn_list.append(f"{reason}; using the coefficient-scale fallback for tau_start")
        details["fallback"]["start"] = reason
        st0.update(source="coefficient_fallback", tau=tau_s)
        fb_used = True

    # ---- end phase ------------------------------------------------------------------------------
    if n_end > 0 and used["attempts"] + n_end <= limit:
        d1, c1, zt1, r1, inf1 = collect("end", n_end, tau_s)
        st1 = phase_record("end", d1, zt1, r1, inf1)
        w1 = _class_weights(c1, class_weights, warn_list)
    else:
        if n_end > 0:
            warn_list.append("pilot limit reached before the end phase; end temperature from the start sample")
        d1, w1, zt1, st1 = None, None, zt0, None

    tau_e = None
    if st1 is not None and _n_effective_positive(d1, w1, zt1) >= min_positive:
        tau_e, info = _phase_temperature(rule, d1, w1, p_end, zt1, quantile, legacy_factor, rtol, max_iter, "end")
        st1.update(source="pilot", tau=tau_e, **info)
    else:
        if st1 is None:
            dp, wp, ztp, src = d0, w0, zt0, "start_sample"
        else:
            dp = np.concatenate([d0, d1])
            wp = None if (w0 is None or w1 is None) else np.concatenate([w0 * 0.5, w1 * 0.5])
            ztp = float(rel_zero_tol) * float(np.max(np.abs(dp))) if dp.size else 0.0
            src = "pooled_start_end"
            warn_list.append(f"end pilot had only {st1['n_positive']} positive changes; using the pooled start+end sample")
        npos = _n_effective_positive(dp, wp, ztp)
        rec = {"source": src, "n_positive_pooled": npos}
        if npos >= min_positive:
            tau_e, info = _phase_temperature(rule, dp, wp, p_end, ztp, quantile, legacy_factor, rtol, max_iter, "end")
            rec.update(tau=tau_e, **info)
        else:
            reason = f"end temperature: {npos} positive changes in the available pilot sample"
            sc = scales() if fallback == "coefficients" else None
            if sc is None:
                raise CalibrationError(f"{reason} and no coefficient scale available (pass weights or coefficient_scale)")
            tau_e = fallback_temperatures(sc[0], sc[1], p_start, p_end)[1]
            warn_list.append(f"{reason}; using the coefficient-scale fallback for tau_end")
            details["fallback"]["end"] = reason
            rec.update(source="coefficient_fallback", tau=tau_e)
            fb_used = True
        if st1 is not None:
            st1.update(rec)
        else:
            details["phases"]["end"] = rec

    n_samples = st0["n"] + (st1["n"] if st1 is not None else 0)
    n_positive = st0["n_positive"] + (st1["n_positive"] if st1 is not None else 0)
    used_rule = rule if not (details["fallback"]["start"] and details["fallback"]["end"]) else "coefficient_fallback"
    return finish(tau_s, tau_e, used_rule, n_samples, n_positive, used["attempts"], fb_used)


# --------------------------------------------------------------------------------------------
# Schedules from a calibration
# --------------------------------------------------------------------------------------------

def make_schedule(cal: CalibrationResult, kind: str, n_total: int, eta: Optional[float] = None, *,
                  t_start: float = 1.0, constant_at: Optional[str] = None) -> _schedule.Schedule:
    """Concrete schedule through the calibrated endpoints.

    fn_endpoints : the FN endpoint rule with eta (argument, else ``cal.eta``).
    fn_legacy    : the same curve in legacy parameters (Methods, Stochastic transition dynamics and FN annealing) with clock start ``t_start``:
                   tau0 = tau_end log1p(eta), q_start = expm1(tau0 / tau_start), c_FN = t_start / q_start,
                   dt = (eta - q_start) c_FN / (B - 1); A = tau0 c_FN is recorded (needs tau_start > tau_end).
    geometric    : log-linear between the endpoints.
    constant     : tau_start or tau_end, chosen explicitly by ``constant_at`` ('start' | 'end').
    """
    if not isinstance(cal, CalibrationResult):
        raise ConfigError("make_schedule needs a CalibrationResult")
    n_total = int(n_total)
    if n_total < 1:
        raise ConfigError("n_total must be >= 1")
    ts, te = float(cal.tau_start), float(cal.tau_end)
    if eta is None:
        eta = cal.eta
    elif cal.eta is not None and float(cal.eta) != float(eta):
        raise ConfigError(f"eta {eta} differs from the calibration's eta {cal.eta}")
    if kind == "fn_endpoints":
        if eta is None:
            raise ConfigError("fn_endpoints needs eta")
        s = _schedule.fn_endpoints(ts, te, float(eta), n_total)
    elif kind == "fn_legacy":
        if eta is None:
            raise ConfigError("fn_legacy needs eta")
        eta = float(eta)
        t_start = float(t_start)
        if not (math.isfinite(t_start) and t_start > 0):
            raise ConfigError("t_start must be finite and > 0")
        tau0 = te * math.log1p(eta)
        q_start = math.expm1(tau0 / ts)
        if not (eta - q_start > 0):
            raise ConfigError("fn_legacy needs tau_start > tau_end (the legacy clock must advance)")
        c_fn = t_start / q_start
        dt = (eta - q_start) * c_fn / (n_total - 1) if n_total > 1 else (eta - q_start) * c_fn
        s = _schedule.fn_legacy(tau0, c_fn, t_start, dt, n_total)
    elif kind == "geometric":
        s = _schedule.geometric(ts, te, n_total)
    elif kind == "constant":
        if constant_at not in ("start", "end"):
            raise ConfigError("constant schedule from a calibration needs constant_at='start' or 'end'")
        s = _schedule.constant(ts if constant_at == "start" else te, n_total)
    else:
        raise ConfigError(f"cannot build a calibrated schedule of kind {kind!r}")
    src = dict(s.source)
    src["calibration"] = {"rule": cal.rule, "tau_start": ts, "tau_end": te, "fallback_used": bool(cal.fallback_used),
                          "p_start": cal.p_start, "p_end": cal.p_end, "pilot_attempts": int(cal.pilot_attempts)}
    return _schedule.Schedule(s.kind, s.p, s.n_total, src)


def _parse_scale(cs):
    """None, a positive float, or a (start, end) pair of positive floats."""
    if cs is None:
        return None
    if isinstance(cs, (tuple, list, np.ndarray)):
        if len(cs) != 2:
            raise ConfigError("coefficient_scale must be a float or a (start, end) pair")
        s0, s1 = float(cs[0]), float(cs[1])
    else:
        s0 = s1 = float(cs)
    if not (math.isfinite(s0) and s0 > 0 and math.isfinite(s1) and s1 > 0):
        raise ConfigError(f"coefficient_scale entries must be finite and > 0, got {cs!r}")
    return s0, s1


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays and tuples into JSON-serializable Python objects."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

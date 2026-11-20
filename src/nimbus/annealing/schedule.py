"""Attempt-indexed temperature schedules (Methods, Stochastic transition dynamics and FN annealing).

Every backend evaluates tau_n in closed form from (kind code, p0..p3, n, B) using exactly
these expressions, so a checkpoint needs only the attempt index n:

    frac = 0.0 if B == 1 else n / (B - 1)                     (double division)
    ZERO        (0): tau = 0
    CONSTANT    (1): tau = p0
    GEOMETRIC   (2): tau = exp(p0 + (p1 - p0) * frac)          p0 = log tau_start, p1 = log tau_end
    FN_ENDPOINT (3): tau = p0 / log1p(p1 + (p2 - p1) * frac)   p0 = tau0, p1 = q_start, p2 = eta
    FN_LEGACY   (4): tau = p0 / log1p((p2 + n * p3) / p1)      p0 = tau0, p1 = c_FN, p2 = t_start, p3 = dt

FN_ENDPOINT: tau0 = tau_end * log1p(eta), q_start = expm1(tau0 / tau_start), so tau_0 = tau_start and
tau_{B-1} = tau_end (up to rounding). For q << 1, tau ~ A/t with A = tau0 * C (inverse-time amplitude).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ZERO, CONSTANT, GEOMETRIC, FN_ENDPOINT, FN_LEGACY = 0, 1, 2, 3, 4
_KIND_NAMES = {ZERO: "zero", CONSTANT: "constant", GEOMETRIC: "geometric", FN_ENDPOINT: "fn_endpoints",
               FN_LEGACY: "fn_legacy"}


@dataclass(frozen=True)
class Schedule:
    kind: int
    p: tuple  # (p0, p1, p2, p3) floats
    n_total: int  # B, number of attempts in the run
    source: dict = field(default_factory=dict, compare=False)  # human-readable construction parameters

    def __post_init__(self):
        if self.kind not in _KIND_NAMES:
            raise ValueError(f"unknown schedule kind {self.kind}")
        if int(self.n_total) < 1:
            raise ValueError("schedule needs n_total >= 1 attempts")
        if len(self.p) != 4:
            raise ValueError("schedule parameters must be a 4-tuple")

    @property
    def name(self) -> str:
        return _KIND_NAMES[self.kind]

    def tau(self, n: int) -> float:
        p0, p1, p2, p3 = (float(x) for x in self.p)
        B = int(self.n_total)
        frac = 0.0 if B == 1 else n / (B - 1)
        if self.kind == ZERO:
            return 0.0
        if self.kind == CONSTANT:
            return p0
        if self.kind == GEOMETRIC:
            return math.exp(p0 + (p1 - p0) * frac)
        if self.kind == FN_ENDPOINT:
            return p0 / math.log1p(p1 + (p2 - p1) * frac)
        return p0 / math.log1p((p2 + n * p3) / p1)

    def taus(self, n) -> np.ndarray:
        n = np.asarray(n, dtype=np.float64)
        p0, p1, p2, p3 = (float(x) for x in self.p)
        B = int(self.n_total)
        frac = np.zeros_like(n) if B == 1 else n / (B - 1)
        if self.kind == ZERO:
            return np.zeros_like(n)
        if self.kind == CONSTANT:
            return np.full_like(n, p0)
        if self.kind == GEOMETRIC:
            return np.exp(p0 + (p1 - p0) * frac)
        if self.kind == FN_ENDPOINT:
            return p0 / np.log1p(p1 + (p2 - p1) * frac)
        return p0 / np.log1p((p2 + n * p3) / p1)

    def kernel_params(self):
        """(kind code, p0, p1, p2, p3, B) for compiled kernels."""
        return (int(self.kind),) + tuple(float(x) for x in self.p) + (int(self.n_total),)

    def with_total(self, n_total: int) -> "Schedule":
        return Schedule(self.kind, self.p, int(n_total), dict(self.source))

    def to_dict(self) -> dict:
        return {"kind": self.name, "kind_code": self.kind, "p": [float(x) for x in self.p],
                "n_total": int(self.n_total), "source": dict(self.source)}

    @staticmethod
    def from_dict(d: dict) -> "Schedule":
        return Schedule(int(d["kind_code"]), tuple(float(x) for x in d["p"]), int(d["n_total"]), dict(d.get("source", {})))


def zero(n_total: int) -> Schedule:
    return Schedule(ZERO, (0.0, 0.0, 0.0, 0.0), n_total, {"kind": "zero"})


def constant(tau: float, n_total: int) -> Schedule:
    if not (tau >= 0 and math.isfinite(tau)):
        raise ValueError("constant temperature must be finite and >= 0")
    return Schedule(CONSTANT, (float(tau), 0.0, 0.0, 0.0), n_total, {"kind": "constant", "tau": float(tau)})


def geometric(tau_start: float, tau_end: float, n_total: int) -> Schedule:
    _check_endpoints(tau_start, tau_end)
    return Schedule(GEOMETRIC, (math.log(tau_start), math.log(tau_end), 0.0, 0.0), n_total,
                    {"kind": "geometric", "tau_start": float(tau_start), "tau_end": float(tau_end)})


def fn_endpoints(tau_start: float, tau_end: float, eta: float, n_total: int) -> Schedule:
    """FN inverse-log schedule through both endpoints; eta = t_end / C (dimensionless final ratio)."""
    _check_endpoints(tau_start, tau_end)
    if not (eta > 0 and math.isfinite(eta)):
        raise ValueError("eta must be finite and > 0")
    tau0 = tau_end * math.log1p(eta)
    q_start = math.expm1(tau0 / tau_start)
    if q_start > eta:  # cannot happen when tau_start >= tau_end; guard rounding
        q_start = eta
    return Schedule(FN_ENDPOINT, (tau0, q_start, float(eta), 0.0), n_total,
                    {"kind": "fn_endpoints", "tau_start": float(tau_start), "tau_end": float(tau_end),
                     "eta": float(eta), "tau0": tau0, "q_start": q_start})


def fn_legacy(tau0: float, c_fn: float, t_start: float, dt: float, n_total: int) -> Schedule:
    """Legacy NeuroSA/N-HOIM parameterization tau_n = tau0 / log(1 + (t_start + n dt)/c_FN) (Methods, Stochastic transition dynamics and FN annealing)."""
    for name, val in (("tau0", tau0), ("c_fn", c_fn), ("t_start", t_start), ("dt", dt)):
        if not (val > 0 and math.isfinite(val)):
            raise ValueError(f"{name} must be finite and > 0")
    return Schedule(FN_LEGACY, (float(tau0), float(c_fn), float(t_start), float(dt)), n_total,
                    {"kind": "fn_legacy", "tau0": float(tau0), "c_fn": float(c_fn), "t_start": float(t_start),
                     "dt": float(dt), "A": float(tau0) * float(c_fn)})


def _check_endpoints(tau_start, tau_end):
    if not (math.isfinite(tau_start) and math.isfinite(tau_end) and tau_end > 0 and tau_start >= tau_end):
        raise ValueError(f"need finite tau_start >= tau_end > 0, got {tau_start}, {tau_end}")

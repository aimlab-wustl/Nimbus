"""Pilot temperature calibration."""

import json
import math

import numpy as np
import pytest

from nimbus.annealing import calibration as C
from nimbus.annealing import schedule as S
from nimbus.core.errors import ConfigError

pytestmark = pytest.mark.quick


def _mean_acc(d, tau, w=None):
    e = np.exp(-np.asarray(d) / tau)
    return float(np.mean(e)) if w is None else float(np.dot(w, e) / np.sum(w))


# ---------------------------------------------------------------- acceptance-target bisection

def test_single_value_closed_form():
    for d, p in ((3.0, 0.8), (1e-6, 0.01), (2e5, 0.5)):
        tau = C.acceptance_target_temperature([d] * 5, p)
        assert tau == pytest.approx(d / -math.log(p), rel=1e-15)


def test_two_point_analytic():
    # deltas {a, 2a}: (x + x^2)/2 = p with x = exp(-a/tau)
    a, p = 1.7, 0.3
    x = (-1 + math.sqrt(1 + 8 * p)) / 2
    tau = C.acceptance_target_temperature([a, 2 * a], p)
    assert tau == pytest.approx(-a / math.log(x), rel=1e-11)


def test_exponential_deltas_match_closed_form():
    # dE ~ Exp(mean m): E exp(-dE/tau) = 1 / (1 + m/tau) = p  ->  tau = m p / (1 - p)
    m, L = 2.5, 200_000
    d = -m * np.log1p(-(np.arange(L) + 0.5) / L)  # deterministic quantile sample
    for p in (0.8, 0.5, 0.01):
        tau, info = C.acceptance_target_temperature(d, p, full_output=True)
        assert tau == pytest.approx(m * p / (1 - p), rel=2e-3)
        assert abs(info["residual"]) < 1e-10
        assert info["converged"] and info["iterations"] < 200


def test_residual_and_monotonic_in_target():
    g = np.random.default_rng(0)
    d = g.lognormal(0.0, 2.0, 5000)
    taus = [C.acceptance_target_temperature(d, p) for p in (0.01, 0.1, 0.5, 0.8, 0.99)]
    assert all(t1 < t2 for t1, t2 in zip(taus, taus[1:]))
    for p, t in zip((0.01, 0.1, 0.5, 0.8, 0.99), taus):
        assert _mean_acc(d, t) == pytest.approx(p, abs=1e-10)


def test_extreme_spread_is_finite_and_bracketed():
    d = np.logspace(-8, 8, 1001)
    for p in (0.8, 1e-3, 1 - 1e-9, 1e-12):
        tau = C.acceptance_target_temperature(d, p)
        assert math.isfinite(tau) and tau > 0
        assert d.min() / -math.log(p) <= tau * (1 + 1e-12)
        assert tau <= np.mean(d) / -math.log(p) * (1 + 1e-12)
        assert _mean_acc(d, tau) == pytest.approx(p, rel=1e-8, abs=1e-14)


def test_weights_equal_replication():
    d = np.array([0.5, 1.0, 4.0])
    w = np.array([1.0, 3.0, 2.0])
    rep = np.repeat(d, [1, 3, 2])
    assert C.acceptance_target_temperature(d, 0.6, weights=w) == pytest.approx(
        C.acceptance_target_temperature(rep, 0.6), rel=1e-11)
    # zero-weight entries are ignored
    assert C.acceptance_target_temperature([1.0, 50.0], 0.5, weights=[1.0, 0.0]) == pytest.approx(
        1.0 / math.log(2.0), rel=1e-14)


@pytest.mark.parametrize("deltas,p,exc", [
    ([], 0.5, C.CalibrationError),
    ([1.0, 0.0], 0.5, C.CalibrationError),
    ([1.0, -2.0], 0.5, C.CalibrationError),
    ([1.0, math.nan], 0.5, C.CalibrationError),
    ([1.0, math.inf], 0.5, C.CalibrationError),
    ([1.0], 0.0, ConfigError),
    ([1.0], 1.0, ConfigError),
    ([1.0], math.nan, ConfigError),
])
def test_acceptance_rule_invalid_inputs(deltas, p, exc):
    with pytest.raises(exc):
        C.acceptance_target_temperature(deltas, p)


def test_acceptance_rule_bad_weights():
    with pytest.raises(C.CalibrationError):
        C.acceptance_target_temperature([1.0, 2.0], 0.5, weights=[0.0, 0.0])
    with pytest.raises(C.CalibrationError):
        C.acceptance_target_temperature([1.0, 2.0], 0.5, weights=[1.0])
    with pytest.raises(C.CalibrationError):
        C.acceptance_target_temperature([1.0, 2.0], 0.5, weights=[1.0, -1.0])


# ---------------------------------------------------------------- quantile and legacy rules

def test_quantile_rule():
    d = np.array([-3.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 100.0])
    assert C.quantile_rule(d, 0.5, 0.5) == pytest.approx(3.0 / math.log(2.0))
    assert C.quantile_rule(d, 1.0, 0.01) == pytest.approx(100.0 / math.log(100.0))
    assert C.quantile_rule(d, 0.0, 0.8) == pytest.approx(1.0 / -math.log(0.8))
    # weighted inverted CDF
    tau = C.quantile_rule([1.0, 2.0, 10.0], 0.5, 0.5, weights=[1.0, 0.1, 0.1])
    assert tau == pytest.approx(1.0 / math.log(2.0))
    assert C.quantile_rule([1.0, 2.0, 10.0], 1.0, 0.5, weights=[1.0, 1.0, 0.0]) == pytest.approx(2.0 / math.log(2.0))
    with pytest.raises(C.CalibrationError):
        C.quantile_rule([0.0, -1.0], 0.5, 0.5)
    with pytest.raises(ConfigError):
        C.quantile_rule([1.0], 1.5, 0.5)


def test_legacy_max_rule():
    d = [-5.0, 0.0, 1.0, 7.0, 3.0]
    assert C.legacy_max_rule(d, 1.1) == pytest.approx(7.7)
    assert C.legacy_max_rule(d, 1.0) == 7.0
    with pytest.raises(ConfigError):
        C.legacy_max_rule(d, 0.9)
    with pytest.raises(C.CalibrationError):
        C.legacy_max_rule([0.0, 0.0], 1.1)


# ---------------------------------------------------------------- coefficient-scale fallback

def test_coefficient_scale_from_weights_and_routes():
    J = np.array([0.0, 2.0, -0.5, 1.0])
    hi, lo = C.coefficient_scale_from_weights(J)
    assert (hi, lo) == (4.0, 1.0)
    # groups: r0 weighted routes {1, 2}; r1 weighted {3} + an unweighted route; r2 none
    route_ptr = np.array([0, 2, 4, 4])
    route_wend = np.array([2, 3, 4])
    route_clause = np.array([1, 2, 3, 0])
    hi, lo = C.coefficient_scale_from_weights(J, route_ptr=route_ptr, route_wend=route_wend, route_clause=route_clause)
    assert hi == pytest.approx(np.median([2 * 2.5, 2 * 1.0]))
    assert lo == 1.0
    with pytest.raises(C.CalibrationError):
        C.coefficient_scale_from_weights(np.zeros(4))


def test_fallback_temperatures():
    ts, te = C.fallback_temperatures(4.0, 1.0, 0.8, 0.01)
    assert ts == pytest.approx(4.0 / -math.log(0.8)) and te == pytest.approx(1.0 / math.log(100.0))
    with pytest.raises(C.CalibrationError):
        C.fallback_temperatures(0.0, 1.0, 0.8, 0.01)


# ---------------------------------------------------------------- driver

class Sampler:
    """Deterministic pilot sampler that records its calls."""

    def __init__(self, start, end=None, seed=0):
        self.start, self.end = start, end if end is not None else start
        self.calls = []
        self.g = np.random.default_rng(seed)

    def __call__(self, phase, n):
        self.calls.append((str(phase), int(n), phase.tau_start, phase.round))
        f = self.start if phase == "start" else self.end
        return f(self.g, n, phase.round)


def _mixed(scale):
    def f(g, n, rnd):
        d = g.exponential(scale, n) * np.where(g.random(n) < 0.5, 1.0, -1.0)
        d[: n // 8] = 0.0  # self-loops
        return d
    return f


def test_calibrate_default_acceptance_rule():
    smp = Sampler(_mixed(3.0), _mixed(1.0))
    cal = C.calibrate(smp, eta=0.1)
    assert [c[:2] for c in smp.calls] == [("start", 1024), ("end", 1024)]
    assert smp.calls[0][2] is None and smp.calls[1][2] == cal.tau_start  # end phase sees tau_start
    assert cal.rule == "acceptance" and not cal.fallback_used and cal.warnings == []
    assert cal.tau_start > cal.tau_end > 0
    assert cal.n_samples == 2048 and cal.pilot_attempts == 2048 and 0 < cal.n_positive < 2048
    assert cal.eta == 0.1 and cal.p_start == 0.8 and cal.p_end == 0.01
    st = cal.details["phases"]["start"]
    assert st["n_positive"] + st["n_zero"] + st["n_negative"] == st["n"] == 1024
    # the start temperature solves the target equation on the start positives
    pos = cal.details["phases"]["start"]
    assert pos["source"] == "pilot" and abs(pos["solver"]["residual"]) < 1e-10
    d = json.loads(json.dumps(cal.to_dict()))
    back = C.CalibrationResult.from_dict(d)
    assert back.tau_start == cal.tau_start and back.tau_end == cal.tau_end and back.rule == cal.rule


def test_calibrate_matches_direct_rule_on_same_sample():
    data = np.random.default_rng(4).exponential(2.0, 4096)

    def fn(phase, n):
        return data[:n] if phase == "start" else data[n:2 * n]

    cal = C.calibrate(fn, late_fraction=0.0, n_pilot=1000)
    assert cal.tau_start == C.acceptance_target_temperature(data[:1000], 0.8)
    assert cal.tau_end == C.acceptance_target_temperature(data[:1000], 0.01)
    calq = C.calibrate(fn, rule="quantile", quantile=0.9, late_fraction=0.0, n_pilot=1000)
    assert calq.tau_start == C.quantile_rule(data[:1000], 0.9, 0.8)
    call = C.calibrate(fn, rule="legacy_max", legacy_factor=1.05, late_fraction=0.0, n_pilot=1000)
    assert call.tau_start == pytest.approx(1.05 * data[:1000].max())
    assert call.tau_end == pytest.approx(data[:1000].max() / math.log(100.0))
    assert call.details["phases"]["start"]["A"] == call.tau_start


@pytest.mark.parametrize("kind,maker", [
    ("flat", lambda g, n, r: np.zeros(n)),
    ("one-sided", lambda g, n, r: -g.exponential(1.0, n)),
    ("too few", lambda g, n, r: np.concatenate([[1.0], -np.ones(n - 1)])),
])
def test_flat_or_one_sided_uses_coefficient_fallback_with_warning(kind, maker):
    smp = Sampler(maker)
    J = np.array([0.0, 1.5, -0.25])
    with pytest.warns(C.CalibrationWarning):
        cal = C.calibrate(smp, weights=J)
    assert cal.fallback_used and cal.rule == "coefficient_fallback"
    assert cal.tau_start == pytest.approx(3.0 / -math.log(0.8))
    assert cal.tau_end == pytest.approx(0.5 / math.log(100.0))
    assert cal.details["fallback"]["scale"]["source"] == "weights"
    # one retry round per phase (max_rounds = 2) with fresh states before falling back
    assert [c[3] for c in smp.calls] == [0, 1, 0, 1]
    assert cal.pilot_attempts == 2 * 1024 + 2 * 1024
    assert any("fallback" in w for w in cal.warnings)


def test_flat_without_scale_raises():
    with pytest.raises(C.CalibrationError):
        C.calibrate(lambda ph, n: np.zeros(n))
    with pytest.raises(C.CalibrationError):
        C.calibrate(lambda ph, n: np.zeros(n), weights=[1.0], fallback="error")


def test_explicit_coefficient_scale_pair():
    with pytest.warns(C.CalibrationWarning):
        cal = C.calibrate(lambda ph, n: np.zeros(n), coefficient_scale=(10.0, 0.1), max_rounds=1)
    assert cal.tau_start == pytest.approx(10.0 / -math.log(0.8))
    assert cal.tau_end == pytest.approx(0.1 / math.log(100.0))
    with pytest.raises(ConfigError):
        C.calibrate(lambda ph, n: np.ones(n), coefficient_scale=-1.0)


def test_retry_round_rescues_thin_sample():
    def start(g, n, rnd):
        return np.zeros(n) if rnd == 0 else g.exponential(1.0, n)

    smp = Sampler(start, _mixed(0.5))
    with pytest.warns(C.CalibrationWarning, match="sampling fresh pilot states"):
        cal = C.calibrate(smp)
    assert not cal.fallback_used and cal.rule == "acceptance"
    assert [c[:2] + (c[3],) for c in smp.calls] == [("start", 1024, 0), ("start", 1024, 1), ("end", 1024, 0)]


def test_end_phase_pools_with_start_when_late_sample_is_thin():
    smp = Sampler(_mixed(2.0), lambda g, n, r: -np.ones(n))
    with pytest.warns(C.CalibrationWarning, match="pooled"):
        cal = C.calibrate(smp, max_rounds=1)
    assert not cal.fallback_used
    assert cal.details["phases"]["end"]["source"] == "pooled_start_end"
    assert cal.tau_end < cal.tau_start


def test_tau_end_clipped_to_tau_start():
    # late deltas much larger than early ones: end target would exceed the start temperature
    smp = Sampler(lambda g, n, r: np.full(n, 0.01), lambda g, n, r: np.full(n, 1e4))
    with pytest.warns(C.CalibrationWarning, match="clipped"):
        cal = C.calibrate(smp)
    assert cal.tau_end == cal.tau_start
    assert cal.details["tau_end_unclipped"] > cal.tau_start


def test_min_ratio_keeps_an_anneal_when_end_is_hotter_than_start():
    smp = Sampler(lambda g, n, r: np.full(n, 0.01), lambda g, n, r: np.full(n, 1e4))
    cal = C.calibrate(smp, min_ratio=4.0)
    assert cal.tau_start == pytest.approx(4.0 * cal.tau_end)
    assert cal.details["tau_start_before_min_ratio"] < cal.tau_start
    assert "tau_end_unclipped" not in cal.details


def test_min_ratio_inactive_when_ratio_already_larger():
    smp = Sampler(_mixed(1.0))
    a, b = C.calibrate(smp), C.calibrate(Sampler(_mixed(1.0)), min_ratio=2.0)
    assert (a.tau_start, a.tau_end) == (b.tau_start, b.tau_end) or b.tau_start >= 2.0 * b.tau_end


def test_min_ratio_lower_end_keeps_start_and_cools_the_end():
    smp = Sampler(lambda g, n, r: np.full(n, 0.01), lambda g, n, r: np.full(n, 1e4))
    base = C.calibrate(Sampler(lambda g, n, r: np.full(n, 0.01), lambda g, n, r: np.full(n, 1e4)))
    cal = C.calibrate(smp, min_ratio=2.0, min_ratio_mode="lower_end")
    assert cal.tau_start == pytest.approx(base.tau_start)
    assert cal.tau_end == pytest.approx(base.tau_start / 2.0)


def test_budget_cap_reduces_pilot():
    smp = Sampler(_mixed(1.0))
    cal = C.calibrate(smp, run_attempts=10_000, budget_cap_fraction=0.05)  # cap 500
    assert sum(c[1] for c in smp.calls) == 500 == cal.pilot_attempts
    assert cal.details["n_pilot_effective"] == 500


def test_tiny_budget_uses_cached_or_fallback():
    cached = C.CalibrationResult(5.0, 0.05, "acceptance", {}, 2048, 900, 2048, 0.1)
    calls = []

    def fn(ph, n):
        calls.append(n)
        return np.ones(n)

    with pytest.warns(C.CalibrationWarning, match="cached"):
        cal = C.calibrate(fn, run_attempts=100, cached=cached)
    assert calls == [] and cal.rule == "cached" and cal.tau_start == 5.0 and cal.pilot_attempts == 0
    with pytest.warns(C.CalibrationWarning, match="coefficient-scale"):
        cal = C.calibrate(fn, run_attempts=100, weights=[2.0])
    assert cal.rule == "coefficient_fallback" and calls == []
    with pytest.raises(C.CalibrationError):
        C.calibrate(fn, run_attempts=100)


def test_class_weights_match_production_mixture():
    # class 0 (small moves, dE ~ 1) and class 1 (large moves, dE ~ 20); production mixture 90% / 10%.
    rng = np.random.default_rng(9)
    n = 20_000
    prod_cls = (rng.random(n) < 0.1).astype(int)
    prod = np.where(prod_cls == 0, rng.exponential(1.0, n), rng.exponential(20.0, n))
    # stratified pilot: 50% / 50%
    pil_cls = np.repeat([0, 1], n // 2)
    pil = np.where(pil_cls == 0, rng.exponential(1.0, n), rng.exponential(20.0, n))

    def fn_strat(ph, k):
        idx = np.r_[0:k // 2, n // 2:n // 2 + (k - k // 2)]
        return C.PilotSample(pil[idx], classes=pil_cls[idx], attempts=k)

    weighted = C.calibrate(fn_strat, n_pilot=n, late_fraction=0.0, class_weights={0: 0.9, 1: 0.1})
    unweighted = C.calibrate(fn_strat, n_pilot=n, late_fraction=0.0)
    ref = C.acceptance_target_temperature(prod, 0.8)
    assert weighted.tau_start == pytest.approx(ref, rel=0.08)
    assert abs(unweighted.tau_start - ref) > 3 * abs(weighted.tau_start - ref)
    with pytest.raises(ConfigError):
        C.calibrate(lambda ph, k: np.ones(k), class_weights={0: 1.0})  # classes missing
    with pytest.raises(ConfigError):
        C.calibrate(fn_strat, n_pilot=n, late_fraction=0.0, class_weights={0: 1.0})  # class 1 unweighted


def test_sample_fn_return_forms_and_errors():
    cal = C.calibrate(lambda ph, n: {"deltas": np.linspace(-1, 3, n), "attempts": n, "info": {"states": 4}})
    assert cal.details["phases"]["start"]["runner_info"] == [{"states": 4}]
    with pytest.raises(ConfigError):
        C.calibrate(lambda ph, n: {"delta": np.ones(n)})
    with pytest.raises(C.CalibrationError):
        C.calibrate(lambda ph, n: np.full(n, np.nan))


@pytest.mark.parametrize("kw", [
    dict(rule="max"), dict(p_start=0.01, p_end=0.8), dict(p_start=1.0), dict(n_pilot=0), dict(late_fraction=1.0),
    dict(budget_cap_fraction=0.0), dict(min_positive=0), dict(max_rounds=0), dict(fallback="silent"),
    dict(eta=-1.0), dict(run_attempts=0),
])
def test_calibrate_validation(kw):
    with pytest.raises(ConfigError):
        C.calibrate(lambda ph, n: np.ones(n), **kw)


def test_pilot_phase_is_a_string():
    ph = C.PilotPhase("end", 2.5, 1)
    assert ph == "end" and isinstance(ph, str) and ph.tau_start == 2.5 and ph.round == 1
    with pytest.raises(ValueError):
        C.PilotPhase("middle")


# ---------------------------------------------------------------- make_schedule

def _cal(ts=8.0, te=0.08, eta=None):
    return C.CalibrationResult(ts, te, "acceptance", {}, 100, 50, 100, 0.01, eta=eta, p_start=0.8, p_end=0.01)


def test_make_schedule_kinds():
    cal = _cal(eta=0.1)
    s = C.make_schedule(cal, "fn_endpoints", 1000)
    assert s.kind == S.FN_ENDPOINT and s.source["calibration"]["rule"] == "acceptance"
    assert s.tau(0) == pytest.approx(8.0, rel=1e-12) and s.tau(999) == pytest.approx(0.08, rel=1e-12)
    g = C.make_schedule(cal, "geometric", 50)
    assert g.tau(0) == pytest.approx(8.0) and g.tau(49) == pytest.approx(0.08)
    assert C.make_schedule(cal, "constant", 10, constant_at="end").tau(3) == 0.08
    assert C.make_schedule(cal, "constant", 10, constant_at="start").tau(3) == 8.0
    with pytest.raises(ConfigError):
        C.make_schedule(cal, "constant", 10)
    with pytest.raises(ConfigError):
        C.make_schedule(_cal(), "fn_endpoints", 10)  # no eta anywhere
    with pytest.raises(ConfigError):
        C.make_schedule(cal, "fn_endpoints", 10, eta=0.2)  # differs from calibration eta
    with pytest.raises(ConfigError):
        C.make_schedule(cal, "zero", 10)


def test_make_schedule_legacy_is_the_same_curve():
    cal = _cal(eta=0.3)
    B = 5000
    fe = C.make_schedule(cal, "fn_endpoints", B)
    fl = C.make_schedule(cal, "fn_legacy", B, t_start=1.0)
    assert fl.kind == S.FN_LEGACY
    tau0, c_fn, t_start, dt = fl.p
    assert fl.source["A"] == tau0 * c_fn and t_start == 1.0
    n = np.arange(0, B, 7)
    np.testing.assert_allclose(fl.taus(n), fe.taus(n), rtol=1e-11)
    assert fl.tau(0) == pytest.approx(8.0, rel=1e-12) and fl.tau(B - 1) == pytest.approx(0.08, rel=1e-11)
    one = C.make_schedule(cal, "fn_legacy", 1)
    assert one.tau(0) == pytest.approx(8.0, rel=1e-12)
    with pytest.raises(ConfigError):
        C.make_schedule(_cal(2.0, 2.0, eta=0.1), "fn_legacy", 10)


def test_positives_only_in_zero_weight_class_fall_back():
    def fn(ph, k):
        cls = np.array([0, 1] * (k // 2))
        d = np.where(cls == 1, 5.0, -1.0)  # positives only in class 1, which has production weight 0
        return C.PilotSample(d, classes=cls)

    with pytest.warns(C.CalibrationWarning):
        cal = C.calibrate(fn, class_weights={0: 1.0, 1: 0.0}, coefficient_scale=2.0, max_rounds=1)
    assert cal.fallback_used and cal.rule == "coefficient_fallback"
    with pytest.raises(ConfigError):
        C.calibrate(fn, class_weights=[0.5, 0.5])

"""Attempt-indexed schedules (Methods, Stochastic transition dynamics and FN annealing)."""

import math

import numpy as np
import pytest

from nimbus.annealing import schedule as S
from nimbus.backends import _numba_common as nc

pytestmark = pytest.mark.quick


def _all_taus(s):
    return np.array([s.tau(n) for n in range(s.n_total)])


@pytest.mark.parametrize("tau_start,tau_end,eta,B", [
    (10.0, 0.05, 0.1, 1000), (3.0, 0.3, 1.0, 17), (50.0, 1e-4, 10.0, 5000), (1.0, 0.999, 0.1, 2),
    (7.5, 7.5, 0.1, 100),
])
def test_fn_endpoints_hits_both_endpoints_and_is_monotone(tau_start, tau_end, eta, B):
    s = S.fn_endpoints(tau_start, tau_end, eta, B)
    t = _all_taus(s)
    assert t[0] == pytest.approx(tau_start, rel=1e-12)
    assert t[-1] == pytest.approx(tau_end, rel=1e-12)
    assert np.all(np.diff(t) <= 1e-12 * tau_start)  # non-increasing (up to rounding)
    assert np.all(t > 0) and np.all(np.isfinite(t))
    # parameters as documented: tau0 = tau_end log1p(eta), q_start = expm1(tau0 / tau_start)
    tau0, q_start, eta_p, _ = s.p
    assert tau0 == tau_end * math.log1p(eta)
    assert q_start == min(math.expm1(tau0 / tau_start), eta)
    assert eta_p == eta


def test_fn_endpoints_closed_form_matches_spec_formula():
    s = S.fn_endpoints(4.0, 0.2, 0.3, 101)
    tau0 = 0.2 * math.log1p(0.3)
    q_start = math.expm1(tau0 / 4.0)
    for n in (0, 1, 50, 99, 100):
        q_n = q_start + n / 100 * (0.3 - q_start)
        assert s.tau(n) == tau0 / math.log1p(q_n)


def test_eta_small_is_close_to_inverse_time():
    # eta = 0.1 keeps q <= 0.1: tau within about 5% of the inverse-time form tau0 / q (= A / t)
    s = S.fn_endpoints(10.0, 0.05, 0.1, 1000)
    tau0, q_start, eta, _ = s.p
    q = q_start + np.arange(1000) / 999 * (eta - q_start)
    ratio = s.taus(np.arange(1000)) / (tau0 / q)
    assert np.all(ratio >= 1.0 - 1e-12) and np.all(ratio <= 1.05)


@pytest.mark.parametrize("make", [
    lambda B: S.fn_endpoints(5.0, 0.5, 0.1, B), lambda B: S.geometric(5.0, 0.5, B),
    lambda B: S.constant(2.0, B), lambda B: S.zero(B), lambda B: S.fn_legacy(1.0, 10.0, 1.0, 0.5, B),
])
def test_single_attempt_schedule(make):
    s = make(1)
    assert s.n_total == 1
    if s.kind in (S.FN_ENDPOINT, S.GEOMETRIC):
        assert s.tau(0) == pytest.approx(5.0, rel=1e-12)  # B = 1 -> tau_start
    k = s.kernel_params()
    assert nc.schedule_tau(*k, 0) == s.tau(0)


def test_geometric_endpoints_and_log_linearity():
    s = S.geometric(8.0, 0.125, 7)
    t = _all_taus(s)
    assert t[0] == pytest.approx(8.0, rel=1e-15) and t[-1] == pytest.approx(0.125, rel=1e-15)
    np.testing.assert_allclose(np.diff(np.log(t)), math.log(0.125 / 8.0) / 6, rtol=1e-12)


def test_legacy_amplitude_and_inverse_time_limit():
    tau0, c_fn, t_start, dt = 0.02, 500.0, 1.0, 0.05
    s = S.fn_legacy(tau0, c_fn, t_start, dt, 2000)
    assert s.source["A"] == tau0 * c_fn  # A = tau0 * C sets the inverse-time amplitude
    A = s.source["A"]
    for n in (0, 10, 1999):
        t = t_start + n * dt
        assert s.tau(n) == tau0 / math.log1p(t / c_fn)
        # t / C <= 0.2 here: tau ~ A / t within ~ t / (2C)
        assert s.tau(n) == pytest.approx(A / t, rel=t / c_fn)
    t = _all_taus(s)
    assert np.all(np.diff(t) < 0)
    # tau0 alone is not the start temperature
    assert s.tau(0) > 100 * tau0


def test_constant_and_zero():
    assert np.all(_all_taus(S.constant(0.7, 20)) == 0.7)
    assert np.all(_all_taus(S.zero(20)) == 0.0)
    assert np.all(S.zero(5).taus(np.arange(5)) == 0.0)


@pytest.mark.parametrize("bad", [
    lambda: S.fn_endpoints(1.0, 2.0, 0.1, 10),  # start < end
    lambda: S.fn_endpoints(1.0, 0.0, 0.1, 10),
    lambda: S.fn_endpoints(1.0, 0.5, 0.0, 10),
    lambda: S.fn_endpoints(math.inf, 0.5, 0.1, 10),
    lambda: S.geometric(1.0, -1.0, 10),
    lambda: S.constant(-1.0, 10),
    lambda: S.constant(math.nan, 10),
    lambda: S.fn_legacy(1.0, 0.0, 1.0, 1.0, 10),
    lambda: S.fn_legacy(1.0, 1.0, 1.0, 0.0, 10),
    lambda: S.zero(0),
    lambda: S.Schedule(7, (0.0, 0.0, 0.0, 0.0), 10),
    lambda: S.Schedule(S.CONSTANT, (1.0, 0.0, 0.0), 10),
])
def test_invalid_schedules_raise(bad):
    with pytest.raises(ValueError):
        bad()


def test_roundtrip_with_total_and_kernel_params():
    s = S.fn_endpoints(6.0, 0.06, 0.5, 300)
    d = s.to_dict()
    s2 = S.Schedule.from_dict(d)
    assert s2 == s and s2.source == s.source
    assert s.kernel_params() == (S.FN_ENDPOINT,) + tuple(s.p) + (300,)
    s3 = s.with_total(600)
    assert s3.n_total == 600 and s3.p == s.p
    assert d["kind"] == "fn_endpoints"


def test_vectorized_taus_match_scalar():
    for s in (S.fn_endpoints(9.0, 0.01, 2.0, 10_000), S.geometric(4.0, 0.04, 10_000),
              S.fn_legacy(0.1, 30.0, 1.0, 0.3, 10_000)):
        n = np.arange(10_000)
        ref = _all_taus(s)
        ulp = np.abs(s.taus(n).view(np.int64) - ref.view(np.int64))
        assert int(ulp.max()) <= 2

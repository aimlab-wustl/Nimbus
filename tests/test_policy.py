"""RunPolicy validation, (de)serialization, schedule resolution, batching and the acceptance reference."""

import json
import math

import numpy as np
import pytest

from nimbus.annealing import calibration as C
from nimbus.annealing import policy as P
from nimbus.annealing import schedule as S
from nimbus.core import rng
from nimbus.core.errors import ConfigError

pytestmark = pytest.mark.quick


def _base(**over):
    d = {"schedule": {"kind": "fn_endpoints", "tau_start": 5.0, "tau_end": 0.05, "eta": 0.1},
         "budget": {"attempts": 1000}}
    d.update(over)
    return d


def _cal(ts=6.0, te=0.06, eta=None):
    return C.CalibrationResult(ts, te, "acceptance", {}, 10, 5, 10, 0.0, eta=eta, p_start=0.8, p_end=0.01)


def test_minimal_policy_defaults_and_roundtrip():
    pol = P.RunPolicy.from_dict(_base())
    assert pol.proposal == "uniform" and pol.acceptance == "metropolis_threshold"
    assert pol.batching is None and pol.replicas == 1 and not pol.stopping.stop_on_target
    assert not pol.needs_calibration
    d = json.loads(json.dumps(pol.to_dict()))
    assert d["schema"] == P.SCHEMA
    assert P.RunPolicy.from_dict(d) == pol


def test_full_policy_roundtrip():
    d = {"schema": "nimbus.policy/1", "proposal": "uniform", "acceptance": "metropolis_threshold",
         "batching": {"capacity": 4, "rule": "proposal_preserving"},
         "schedule": {"kind": "fn_legacy", "eta": 0.2, "t_start": 2.0,
                      "auto": {"rule": "quantile", "quantile": 0.75, "p_start": 0.7, "p_end": 0.02,
                               "class_weights": {"swap": 0.75, "reverse": 0.25}, "coefficient_scale": [4.0, 1.0]}},
         "budget": {"attempts": 500, "wall_time_s": 30.0},
         "stopping": {"target_energy": -10.0, "stop_on_target": True, "target_tol": 1e-9},
         "replicas": 8}
    pol = P.RunPolicy.from_dict(d)
    assert pol.needs_calibration and pol.schedule.auto.rule == "quantile"
    back = P.RunPolicy.from_dict(json.loads(json.dumps(pol.to_dict())))
    assert back == pol
    kw = pol.calibration_kwargs()
    assert kw["run_attempts"] == 500 * 8 and kw["eta"] == 0.2 and kw["quantile"] == 0.75
    assert kw["coefficient_scale"] == (4.0, 1.0) and kw["class_weights"] == {"swap": 0.75, "reverse": 0.25}
    assert "fn_legacy[auto:quantile" in pol.label() and "proposal_preserving(C=4)" in pol.label()


def test_auto_true_uses_default_calibration_spec():
    pol = P.RunPolicy.from_dict(_base(schedule={"kind": "geometric", "auto": True}))
    assert pol.schedule.auto == P.CalibrationSpec()
    kw = pol.calibration_kwargs()
    assert kw["rule"] == "acceptance" and kw["p_start"] == 0.8 and kw["p_end"] == 0.01 and kw["n_pilot"] == 2048


@pytest.mark.parametrize("d", [
    _base(unknown=1),
    _base(proposal="resample_until_change"),
    _base(proposal="weighted"),
    _base(acceptance="glauber"),
    _base(replicas=0),
    _base(replicas=1.5),
    _base(replicas=True),
    _base(schema="nimbus.policy/0"),
    _base(budget={}),
    _base(budget={"attempts": 0}),
    _base(budget={"attempts": 10, "wall_time_s": -1}),
    _base(budget={"attempts": 10, "cycles": 3}),
    _base(stopping={"stop_on_target": True}),
    _base(stopping={"target_energy": 1.0, "stop_on_target": "yes"}),
    _base(stopping={"target_energy": 1.0, "target_tol": -1.0}),
    _base(stopping={"target": 1.0}),
    _base(batching={"capacity": 0}),
    _base(batching={"capacity": 8, "rule": "dwell"}),
    _base(batching={"rule": "proposal_preserving"}),
    _base(batching={"capacity": 8, "dwell": 100}),
    {"budget": {"attempts": 10}},
    {"schedule": {"kind": "constant", "tau": 1.0}},
    _base(schedule={"tau": 1.0}),
    _base(schedule={"kind": "logarithmic", "tau": 1.0}),
    _base(schedule={"kind": "fn_endpoints", "tau_start": 5.0, "tau_end": 0.05}),  # eta missing
    _base(schedule={"kind": "fn_endpoints", "tau_start": 0.05, "tau_end": 5.0, "eta": 0.1}),  # start < end
    _base(schedule={"kind": "fn_endpoints", "tau_start": 5.0, "tau_end": 0.05, "eta": 0.1, "C": 3}),
    _base(schedule={"kind": "constant", "tau": "hot"}),
    _base(schedule={"kind": "constant", "tau": math.inf}),
    _base(schedule={"kind": "zero", "tau": 0.0}),
    _base(schedule={"kind": "zero", "auto": True}),
    _base(schedule={"kind": "fn_endpoints", "tau_start": 5.0, "eta": 0.1, "auto": True}),  # auto + explicit
    _base(schedule={"kind": "fn_endpoints", "auto": True}),  # auto needs eta
    _base(schedule={"kind": "constant", "auto": True}),  # needs constant_at
    _base(schedule={"kind": "constant", "constant_at": "middle", "auto": True}),
    _base(schedule={"kind": "geometric", "auto": {"rule": "maximum"}}),
    _base(schedule={"kind": "geometric", "auto": {"p_start": 0.01, "p_end": 0.5}}),
    _base(schedule={"kind": "geometric", "auto": {"n_pilot": 0}}),
    _base(schedule={"kind": "geometric", "auto": {"late_fraction": 1.0}}),
    _base(schedule={"kind": "geometric", "auto": {"legacy_factor": 0.5}}),
    _base(schedule={"kind": "geometric", "auto": {"pilot": 100}}),
    _base(schedule={"kind": "geometric", "auto": {"class_weights": {}}}),
    _base(schedule={"kind": "geometric", "auto": {"class_weights": {"a": -1.0}}}),
    _base(schedule={"kind": "geometric", "auto": {"coefficient_scale": [1.0, 2.0, 3.0]}}),
    _base(schedule={"kind": "geometric", "auto": {"fallback": "zero"}}),
    _base(schedule={"kind": "geometric", "auto": "yes"}),
    # B-dependent schedule with a wall-time-only budget
    {"schedule": {"kind": "fn_endpoints", "tau_start": 5.0, "tau_end": 0.05, "eta": 0.1}, "budget": {"wall_time_s": 5}},
    "not a mapping",
])
def test_invalid_policies_raise_config_error(d):
    with pytest.raises(ConfigError):
        P.RunPolicy.from_dict(d)


def test_resolve_explicit_schedules():
    pol = P.RunPolicy.from_dict(_base())
    s = pol.resolve_schedule()
    assert s == S.fn_endpoints(5.0, 0.05, 0.1, 1000)
    assert pol.resolve_schedule(n_total=10).n_total == 10
    with pytest.raises(ConfigError):
        pol.resolve_schedule(_cal())  # explicit schedule + calibration is inconsistent
    with pytest.raises(ConfigError):
        pol.calibration_kwargs()
    leg = P.RunPolicy.from_dict(_base(schedule={"kind": "fn_legacy", "tau0": 0.1, "c_fn": 50.0, "t_start": 1.0,
                                                "dt": 0.01}))
    assert leg.resolve_schedule() == S.fn_legacy(0.1, 50.0, 1.0, 0.01, 1000)
    z = P.RunPolicy.from_dict({"schedule": {"kind": "zero"}, "budget": {"wall_time_s": 1.0}})
    with pytest.raises(ConfigError):
        z.resolve_schedule()  # no attempts budget and no n_total
    assert z.resolve_schedule(n_total=5).tau(4) == 0.0


def test_resolve_auto_schedules():
    pol = P.RunPolicy.from_dict(_base(schedule={"kind": "fn_endpoints", "eta": 0.1, "auto": True}))
    with pytest.raises(ConfigError):
        pol.resolve_schedule()  # calibration required
    s = pol.resolve_schedule(_cal())
    assert s.kind == S.FN_ENDPOINT and s.n_total == 1000
    assert s.tau(0) == pytest.approx(6.0, rel=1e-12) and s.tau(999) == pytest.approx(0.06, rel=1e-12)
    with pytest.raises(ConfigError):
        pol.resolve_schedule(_cal(eta=0.5))  # eta of calibration contradicts the policy
    assert pol.resolve_schedule(_cal(eta=0.1)).p == s.p
    c = P.RunPolicy.from_dict(_base(schedule={"kind": "constant", "constant_at": "end", "auto": True}))
    assert c.resolve_schedule(_cal()).tau(0) == 0.06
    lg = P.RunPolicy.from_dict(_base(schedule={"kind": "fn_legacy", "eta": 0.1, "auto": True}))
    sl = lg.resolve_schedule(_cal())
    assert sl.kind == S.FN_LEGACY and sl.p[2] == 1.0  # default t_start
    np.testing.assert_allclose(sl.taus(np.arange(1000)), s.taus(np.arange(1000)), rtol=1e-11)


def test_batch_plan_partition():
    plan = P.BatchingSpec(4).plan(10)
    assert plan.n_batches == 3
    np.testing.assert_array_equal(plan.sizes, [4, 4, 2])
    np.testing.assert_array_equal(plan.batch_of_group, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    np.testing.assert_allclose(plan.batch_probabilities(), [0.4, 0.4, 0.2])
    assert plan.count_reloads([0, 1, 5, 6, 0, 9, 9]) == 4
    assert plan.count_reloads([0, 1], loaded=0) == 0
    assert P.BatchingSpec(64).plan(10).n_batches == 1
    assert plan.to_dict()["sizes"] == [4, 4, 2]
    with pytest.raises(ConfigError):
        P.BatchingSpec(4).plan(0)


def test_proposal_preserving_batching_keeps_uniform_group_law():
    # The batch of the full-library solver draw has law L_b / R and the group is uniform within it.
    R = 10
    plan = P.BatchingSpec(4).plan(R)
    n = 200_000
    r, _ = rng.solver_draws(2026, np.arange(n, dtype=np.uint64), 0, 0, R)
    b = plan.batches_of(r)
    freq_b = np.bincount(b, minlength=plan.n_batches) / n
    np.testing.assert_allclose(freq_b, plan.batch_probabilities(), atol=4 * math.sqrt(0.25 / n))
    for k in range(plan.n_batches):
        within = np.bincount(r[b == k] - plan.batch_ptr[k], minlength=plan.sizes[k]) / np.sum(b == k)
        np.testing.assert_allclose(within, 1.0 / plan.sizes[k], atol=0.01)


def test_stopping_hit():
    st = P.StoppingSpec(target_energy=-3.0, stop_on_target=True, target_tol=1e-9)
    assert st.hit(-3.0) and st.hit(-3.0 + 5e-10) and not st.hit(-2.9)
    assert not P.StoppingSpec().hit(-1e9)


def test_metropolis_reference():
    assert P.metropolis_accept(0.0, 0.0, 0.5) and P.metropolis_accept(-1.0, 0.0, 0.5)
    assert not P.metropolis_accept(1e-300, 0.0, 0.999)
    assert P.metropolis_accept(0.0, 1.0, 1.0 - 2.0 ** -53)  # v = 0 always accepted (self-loops)
    assert P.metropolis_threshold(2.0, math.exp(-3.0)) == pytest.approx(3.0)
    # uphill acceptance probability exp(-dE / tau) with dE = 2 v, on the solver stream uniforms
    n = 400_000
    _, u = rng.solver_draws(77, np.arange(n, dtype=np.uint64), 1, 2, 5)
    tau, v = 1.3, 0.4
    acc = np.mean(np.array([P.metropolis_accept(v, tau, x) for x in u[:n]]))
    p = math.exp(-2 * v / tau)
    assert acc == pytest.approx(p, abs=4 * math.sqrt(p * (1 - p) / n))

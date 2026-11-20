"""The calibrated Fowler-Nordheim (FN) schedule used for the graph-matching results.

Temperatures (Methods, Stochastic transition dynamics and FN annealing)::

    tau_start = m * tau_start_cal
    tau_end   = min(tau_end_cal, tau_start / kappa)

``tau_start_cal`` and ``tau_end_cal`` come from one acceptance-rule pilot calibration (:func:`calibrate_temperatures`):

* start phase: proposals of the production move library at 32 random feasible states; ``tau_start_cal`` is the
  temperature at which the mean acceptance probability of the uphill proposals is ``p_start = 0.8``;
* end phase (late pilot): the same states first advanced by a short FN descent of ``late_attempts = 8192`` attempts
  from ``tau_start_cal`` down to ``late_end_ratio * tau_start_cal = 0.02 tau_start_cal``; ``tau_end_cal`` is the
  temperature at which the mean acceptance of the uphill proposals there is ``p_end = 0.01``.

``m`` is the start factor and ``kappa`` the minimum ratio of start to end temperature. With ``m = 1`` the rule is the
calibration option ``min_ratio = kappa`` with ``min_ratio_mode = "lower_end"`` (the automatically calibrated schedule
of the paper uses ``m = 1``, ``kappa = 2``). The schedule shape parameter is ``eta = 0.1``
(:func:`nimbus.annealing.schedule.fn_endpoints`).

In a run configuration the same calibration is written as::

    annealing:
      schedule: fn_endpoints
      params: auto
      calibration: {rule: acceptance, p_start: 0.8, p_end: 0.01, eta: 0.1, min_ratio: 2, min_ratio_mode: lower_end}

(the config runner draws its pilot states with its own seeds, so its temperatures are statistically equivalent but not
bit-identical to :func:`calibrate_temperatures`).
"""
from __future__ import annotations

from nimbus.annealing import schedule as _sched
from nimbus.annealing.calibration import calibrate

__all__ = ["P_START", "P_END", "ETA", "LATE_ATTEMPTS", "LATE_END_RATIO", "PILOT_STATES",
           "calibrate_temperatures", "scaled_temperatures", "schedule_temperatures"]

P_START = 0.8
P_END = 0.01
ETA = 0.1
LATE_ATTEMPTS = 8192
LATE_END_RATIO = 0.02
PILOT_STATES = 32


def calibrate_temperatures(model, binding, problem, seed: int, *, p_start: float = P_START, p_end: float = P_END,
                           eta: float = ETA, late: bool = True, late_attempts: int = LATE_ATTEMPTS,
                           late_end_ratio: float = LATE_END_RATIO, pilot_states: int = PILOT_STATES,
                           min_ratio: float | None = None, min_ratio_mode: str = "lower_end", raw: bool = False):
    """Acceptance-rule pilot calibration of ``(tau_start, tau_end)`` for one compiled problem.

    ``model``/``binding``: the compiled machine and its weights; ``problem``: supplies ``random_feasible`` for the
    pilot states (``problem.random_feasible(seed, pilot_states, 0)``). The pilot runs on the CPU backend with one
    thread. ``late=False`` samples the end phase at the random start states too (end temperature far too cold on
    matching; for comparison only).

    ``min_ratio`` (kappa) and ``min_ratio_mode`` are applied by :func:`nimbus.annealing.calibration.calibrate`.
    ``raw=True`` returns the temperatures before the kappa rule and before the ``tau_end <= tau_start`` clip, for use
    with :func:`scaled_temperatures`. Returns ``(tau_start, tau_end)`` as floats.
    """
    from nimbus.backends import get_backend
    from nimbus.experiments.calibrate_driver import pilot_sample_fn

    be = get_backend("cpu")
    plan = be.prepare(model, threads=1, process="select")
    spins0 = problem.random_feasible(seed, int(pilot_states), 0)
    state = be.init_state(plan, binding, spins0)
    base = pilot_sample_fn(be, plan, state, model.n_groups, seed, replica=0, round_offset=0)
    adv = {}

    def fn(phase, n):
        if late and str(phase) == "end":
            ts = float(getattr(phase, "tau_start", 1.0))
            if "st" not in adv:
                st2 = be.init_state(plan, binding, spins0)
                be.run(plan, st2, _sched.fn_endpoints(ts, ts * late_end_ratio, eta, int(late_attempts)), seed=seed + 7)
                adv["st"] = st2
                adv["fn"] = pilot_sample_fn(be, plan, st2, model.n_groups, seed, replica=0, round_offset=0)
            return adv["fn"](phase, n)
        return base(phase, n)

    r = calibrate(fn, rule="acceptance", p_start=p_start, p_end=p_end, eta=eta, weights=binding.J[0],
                  routes=(model.route_ptr, model.route_wend, model.route_clause), min_ratio=min_ratio,
                  min_ratio_mode=min_ratio_mode)
    if raw:
        return (float(r.details.get("tau_start_before_min_ratio", r.tau_start)),
                float(r.details.get("tau_end_unclipped", r.tau_end)))
    return float(r.tau_start), float(r.tau_end)


def scaled_temperatures(tau_start_cal: float, tau_end_cal: float, m: float, kappa: float):
    """``(m * tau_start_cal, min(tau_end_cal, m * tau_start_cal / kappa))`` from the raw calibration."""
    ts = float(m) * float(tau_start_cal)
    return ts, min(float(tau_end_cal), ts / float(kappa))


def schedule_temperatures(model, binding, problem, seed: int, *, m: float = 1.0, kappa: float = 2.0):
    """Calibrate once and apply the start factor ``m`` and minimum ratio ``kappa``: ``(tau_start, tau_end)``."""
    ts0, te0 = calibrate_temperatures(model, binding, problem, seed, raw=True)
    return scaled_temperatures(ts0, te0, m, kappa)

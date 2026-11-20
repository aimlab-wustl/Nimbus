"""Calibration pilots driven through the architecture.

:func:`pilot_sample_fn` builds the ``sample_fn`` that :func:`nimbus.annealing.calibration.calibrate`
needs: it draws pilot group indices from the dedicated ``PILOT`` RNG stream and reads the resulting
``Delta E = 2v`` straight from the architecture via ``backend.evaluate_inputs`` on a *clone* of the
state (never committed, never advances the solver's own counters). :func:`calibrate_for_policy` ties
this to a :class:`~nimbus.annealing.policy.RunPolicy` and reports the pilot cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from nimbus.annealing import CalibrationResult, calibrate
from nimbus.annealing import schedule as _sched
from nimbus.core import rng

__all__ = ["pilot_sample_fn", "late_pilot_sample_fn", "calibrate_for_policy"]

_PHASE_CODE = {"start": 0, "end": 1}


def pilot_sample_fn(backend, plan, state, n_groups: int, seed: int, *, replica: int = 0, round_offset: int = 0):
    """``sample_fn(phase, n) -> float64[n]`` of ``Delta E = 2v`` for ``n`` pilot proposals.

    Group indices for the pilot are independent of the solver's own ``SOLVER``-stream draws: drawn from the ``PILOT`` stream, keyed by (phase, round) so repeated rounds within
    one calibration call are reproducible but distinct.
    """

    def sample_fn(phase, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        phase_code = _PHASE_CODE.get(getattr(phase, "value", phase) if not isinstance(phase, str) else phase,
                                     0 if str(phase) == "start" else 1)
        round_no = round_offset + int(getattr(phase, "round", 0))
        groups = rng.stream_integers(seed, rng.PILOT, n, n_groups, c2=round_no, c1=phase_code)
        groups = np.asarray(groups, dtype=np.int64)
        v = backend.evaluate_inputs(state, groups=groups)
        return 2.0 * np.asarray(v)[replica, :]

    return sample_fn


#: default length (attempts) of the short descent that advances the pilot state before the end-phase sample
LATE_ADVANCE_DEFAULT = 8192
#: the descent ends at this fraction of the calibrated tau_start
LATE_ADVANCE_END_RATIO = 0.02


def late_pilot_sample_fn(backend, plan, state, n_groups: int, seed: int, *, binding, eta: float = 0.1,
                         advance_attempts: int = LATE_ADVANCE_DEFAULT, replica: int = 0, round_offset: int = 0):
    """Like :func:`pilot_sample_fn`, but the *end* phase samples a state first advanced by a short FN descent.

    Sampling uphill moves of a random state gives a tau_end far too cold (matching: about 50x), because the typical uphill move at a random state is much cheaper than at an annealed state. The
    end phase therefore clones replica ``replica`` of ``state`` into a one-replica state, runs a descent
    ``tau_start -> LATE_ADVANCE_END_RATIO * tau_start`` over ``advance_attempts`` attempts (``tau_start`` is
    the calibrated start temperature, carried by the PilotPhase), and draws the end-phase deltas there. The
    input ``state`` is never modified. The advanced state is built once per call (cached across rounds).
    """
    base = pilot_sample_fn(backend, plan, state, n_groups, seed, replica=replica, round_offset=round_offset)
    cache: dict = {}

    def sample_fn(phase, n: int) -> np.ndarray:
        if str(phase) != "end" or advance_attempts <= 0:
            return base(phase, n)
        if "fn" not in cache:
            ts = float(getattr(phase, "tau_start", None) or 1.0)
            spins = np.asarray(backend.export_state(state).spins_of(replica))[None, :]
            st2 = backend.init_state(plan, binding, spins)
            backend.run(plan, st2, _sched.fn_endpoints(ts, ts * LATE_ADVANCE_END_RATIO, float(eta),
                                                       int(advance_attempts)), seed=seed + 7)
            cache["fn"] = pilot_sample_fn(backend, plan, st2, n_groups, seed, replica=0,
                                          round_offset=round_offset)
        return cache["fn"](phase, n)

    return sample_fn


@dataclass
class PilotReport:
    result: CalibrationResult
    seconds: float
    n_samples: int


def calibrate_for_policy(policy, backend, plan, state, n_groups: int, seed: int, *, eta=None,
                          round_offset: int = 0, binding=None, late_advance: str = "descent",
                          late_advance_attempts="auto") -> PilotReport:
    """Run the pilot calibration for ``policy`` (``RunPolicy.needs_calibration`` must be true).

    ``binding`` (the instance's ``WeightBinding``), if given, supplies the ``weights``/``routes``
    fallback so a one-sided pilot (e.g. a feasible start with no uphill proposal in this family of
    moves) still gets a usable ``coefficient_scale`` instead of raising.

    ``late_advance`` ('descent', the default, or 'none'): with 'descent' (and ``binding`` given) the
    end-phase pilot samples states advanced by a short FN descent (:func:`late_pilot_sample_fn`);
    'none' reproduces the old behaviour (end phase sampled at the random start state). When
    ``late_fraction`` is 0 there is no end phase and the setting has no effect.
    """
    kw = policy.calibration_kwargs()
    if eta is not None:
        kw.setdefault("eta", eta)
    if binding is not None and "weights" not in kw:
        model = plan.model
        kw["weights"] = binding.J[0]
        kw["routes"] = (model.route_ptr, model.route_wend, model.route_clause)
    if late_advance == "descent" and binding is not None:
        n_adv = LATE_ADVANCE_DEFAULT if late_advance_attempts == "auto" else int(late_advance_attempts)
        sample_fn = late_pilot_sample_fn(backend, plan, state, n_groups, seed, binding=binding,
                                         eta=float(kw.get("eta", 0.1)), advance_attempts=n_adv,
                                         round_offset=round_offset)
    else:
        sample_fn = pilot_sample_fn(backend, plan, state, n_groups, seed, round_offset=round_offset)
    t0 = time.perf_counter()
    result = calibrate(sample_fn, **kw)
    dt = time.perf_counter() - t0
    return PilotReport(result=result, seconds=dt, n_samples=int(result.n_samples))

"""Annealing services: attempt-indexed schedules, pilot temperature calibration and the run policy.

* :mod:`nimbus.annealing.schedule`    closed-form tau_n (Fowler-Nordheim, geometric, constant)
* :mod:`nimbus.annealing.calibration` pilot calibration from the actual proposed moves
* :mod:`nimbus.annealing.protocol`    the calibrated FN schedule of the graph-matching results (start factor m,
  minimum ratio kappa, late pilot)
* :mod:`nimbus.annealing.policy`      RunPolicy (proposal, batching, Metropolis threshold acceptance,
  schedule spec, budget, stopping, replicas) and the pure-Python acceptance reference
"""

from nimbus.annealing import calibration, policy, protocol, schedule
from nimbus.annealing.calibration import (
    CalibrationError,
    CalibrationResult,
    CalibrationWarning,
    PilotPhase,
    PilotSample,
    acceptance_target_temperature,
    calibrate,
    legacy_max_rule,
    make_schedule,
    quantile_rule,
)
from nimbus.annealing.policy import (
    BatchingSpec,
    BatchPlan,
    BudgetSpec,
    CalibrationSpec,
    RunPolicy,
    ScheduleSpec,
    StoppingSpec,
    metropolis_accept,
    metropolis_threshold,
)
from nimbus.annealing.schedule import Schedule

__all__ = [
    "schedule", "calibration", "policy", "protocol", "Schedule",
    "CalibrationError", "CalibrationResult", "CalibrationWarning", "PilotPhase", "PilotSample",
    "acceptance_target_temperature", "calibrate", "legacy_max_rule", "make_schedule", "quantile_rule",
    "BatchingSpec", "BatchPlan", "BudgetSpec", "CalibrationSpec", "RunPolicy", "ScheduleSpec", "StoppingSpec",
    "metropolis_accept", "metropolis_threshold",
]

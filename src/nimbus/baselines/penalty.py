"""Generic penalty baseline built on the shared architecture infrastructure.

Given a :class:`nimbus.problems.base.PenaltyFormulation` (``run``/``objective``/``violation``
``SpinPolynomial`` objects over ``n_physical`` + auxiliary spins, auxiliaries at the end), this
module compiles an *unconstrained* architecture model with a single-spin-flip ``MASK`` library and
``score_structures`` for ``"objective"`` and ``"violation"``, binds the three slots, and runs it on
any architecture backend (``oracle``/``emulation``/``cpu``/``gpu``) with the ordinary transaction
semantics. Slot order is always ``("run", "objective", "violation")``, which
makes ``nimbus.backends.base.SlotLayout.from_bindings`` recognise the feasible-best bookkeeping
automatically: ``feasible_best_*`` tracks slot ``"objective"`` restricted to states with
``|violation| <= 1e-9 * scale``, so this module never has to reimplement that
logic -- it only has to compile and bind the model the same way every time and read the result back
honestly (never reporting a penalized energy as a feasible objective value).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from nimbus.compiler import compile_model
from nimbus.core.model import CompiledModel, ObjectiveStructure, WeightBinding, bind_weights
from nimbus.core.rng import INIT, stream_bits
from nimbus.problems.base import PenaltyFormulation, Problem, SpinPolynomial
from nimbus.transitions import builders as B
from nimbus.transitions.library import TransitionLibrary

__all__ = [
    "SLOT_NAMES", "build_penalty_model", "bind_penalty", "PenaltyRunResult", "run_penalty",
    "conservative_strength_bound", "strength_sweep", "random_physical_start",
]

SLOT_NAMES = ("run", "objective", "violation")


# ============================================================================================
# model construction
# ============================================================================================
def _structure(poly: SpinPolynomial, n_spins: int) -> ObjectiveStructure:
    return ObjectiveStructure(n_spins=n_spins, terms=list(poly.supports))


def build_penalty_model(pf: PenaltyFormulation, *, identities: Sequence[tuple] = (),
                        library: TransitionLibrary | None = None, **compile_kwargs) -> tuple[CompiledModel, TransitionLibrary]:
    """Compile the unconstrained architecture for a penalty formulation.

    ``library`` defaults to single-spin ``MASK`` flips over every physical+auxiliary spin
    (``nimbus.transitions.builders.single_flips``); a caller may pass a richer verified library
    (e.g. to exercise other move sets) as long as it covers ``pf.run.n_spins`` spins.
    """
    n = int(pf.run.n_spins)
    lib = B.single_flips(n) if library is None else library
    structure = _structure(pf.run, n)
    score_structures = {"objective": _structure(pf.objective, n), "violation": _structure(pf.violation, n)}
    model = compile_model(structure, lib, identities=identities, score_structures=score_structures, **compile_kwargs)
    return model, lib


def bind_penalty(model: CompiledModel, pf: PenaltyFormulation) -> WeightBinding:
    """Bind the ``run``/``objective``/``violation`` slots from a penalty formulation's polynomials."""
    extra = {
        "objective": (pf.objective.clause_weights(), pf.objective.const),
        "violation": (pf.violation.clause_weights(), pf.violation.const),
    }
    return bind_weights(model, pf.run.clause_weights(), pf.run.const, extra_slots=extra)


# ============================================================================================
# running
# ============================================================================================
@dataclass
class PenaltyRunResult:
    """Honest readback of a penalty run: feasible-best ORIGINAL objective is reported separately
    from the penalized energy, and is only ever populated from states that were actually feasible.
    """

    backend: str
    n_physical: int
    n_aux: int
    feasible: np.ndarray            # bool[n_rep]: was the run ever feasible (feasible_best_attempt >= 0)
    best_feasible_objective: np.ndarray  # float64[n_rep]; NaN where never feasible
    best_feasible_spins: np.ndarray      # int8[n_rep, n_spins]; last feasible-best known (meaningless where not feasible)
    best_feasible_attempt: np.ndarray    # int64[n_rep]; -1 if never feasible
    final_penalized_energy: np.ndarray   # float64[n_rep]; slot "run", for diagnostics only
    final_feasible: np.ndarray           # bool[n_rep]
    status: list                          # "ok" | "never_feasible" per replica
    run_result: Any = None
    ledger: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "backend": self.backend, "n_physical": self.n_physical, "n_aux": self.n_aux,
            "feasible": self.feasible.tolist(), "status": list(self.status),
            "best_feasible_objective": [float(x) if np.isfinite(x) else None for x in self.best_feasible_objective],
            "best_feasible_attempt": self.best_feasible_attempt.tolist(),
            "final_penalized_energy": self.final_penalized_energy.tolist(),
            "final_feasible": self.final_feasible.tolist(), "ledger": dict(self.ledger),
        }


def random_physical_start(n_physical: int, seed: int, n: int, *, replica0: int = 0, instance: int = 0) -> np.ndarray:
    """Native (non-feasible-aware) random +-1 start from the INIT stream, one draw per replica."""
    out = np.empty((n, n_physical), dtype=np.int8)
    for i in range(n):
        bits = stream_bits(seed, INIT, n_physical, c2=replica0 + i, c3=instance)
        out[i] = 1 - 2 * np.asarray(bits, dtype=np.int8)
    return out


def run_penalty(problem: Problem | PenaltyFormulation, *, instance: int = 0, strengths: Mapping | None = None,
                backend: str = "emulation", schedule, seed: int, n_replicas: int = 1,
                feasible_start: bool = True, aux_init: str = "zero", library: TransitionLibrary | None = None,
                identities: Sequence[tuple] = (), backend_options: Mapping | None = None,
                stop_on_target: bool = False, targets=None) -> PenaltyRunResult:
    """Run a problem's penalty formulation on an architecture backend.

    ``feasible_start``: if True and ``problem`` is a :class:`Problem` (not a bare
    ``PenaltyFormulation``), start replicas from ``problem.random_feasible`` (same feasible states
    the feasible method would use); otherwise use a native random +-1 start over the physical spins
    (``random_physical_start``), recorded in the ledger either way. Auxiliary spins (if any) are
    appended as +1 (``aux_init="zero"`` i.e. the all-true/"no AND yet" boolean point) unless
    ``aux_init="random"``.
    """
    from nimbus.backends.base import get_backend

    if isinstance(problem, PenaltyFormulation):
        pf = problem
        prob_obj = None
    else:
        pf = problem.penalty_formulation(instance, **dict(strengths or {}))
        prob_obj = problem

    model, lib = build_penalty_model(pf, identities=identities, library=library)
    binding = bind_penalty(model, pf)
    be = get_backend(backend)
    plan = be.prepare(model, library=lib, **dict(backend_options or {}))

    n_phys = int(pf.n_physical)
    n_aux = int(pf.run.n_spins) - n_phys
    if feasible_start and prob_obj is not None:
        phys0 = np.asarray(prob_obj.random_feasible(seed, n_replicas, instance), dtype=np.int8)
        start_kind = "feasible (problem.random_feasible)"
    else:
        phys0 = random_physical_start(n_phys, seed, n_replicas)
        start_kind = "native random (+-1 per physical spin, INIT stream)"
    if n_aux:
        if aux_init == "random":
            aux0 = random_physical_start(n_aux, seed + 1, n_replicas)
        else:
            aux0 = np.ones((n_replicas, n_aux), dtype=np.int8)
        spins0 = np.concatenate([phys0, aux0], axis=1)
    else:
        spins0 = phys0

    rb = np.zeros(n_replicas, dtype=np.int64)
    rid = np.arange(n_replicas, dtype=np.int64)
    iid = np.zeros(n_replicas, dtype=np.int64)
    state = be.init_state(plan, [binding], spins0, rb, rid, iid)
    if targets is not None or stop_on_target:
        from nimbus.backends.base import apply_targets, SlotLayout
        apply_targets(state, SlotLayout.from_bindings([binding]), targets, stop_on_target=stop_on_target)
    res = be.run(plan, state, schedule, seed=seed, stop_on_target=stop_on_target)

    s = res.state
    feasible = s.feasible_best_attempt >= 0
    best_obj = np.where(feasible, s.feasible_best_energy, np.nan)
    status = ["ok" if f else "never_feasible" for f in feasible.tolist()]
    return PenaltyRunResult(
        backend=backend, n_physical=n_phys, n_aux=n_aux, feasible=feasible, best_feasible_objective=best_obj,
        best_feasible_spins=s.feasible_best_spins, best_feasible_attempt=s.feasible_best_attempt,
        final_penalized_energy=s.energies[:, 0], final_feasible=feasible, status=status, run_result=res,
        ledger={"start": start_kind, "aux_init": aux_init if n_aux else None, **dict(pf.ledger)},
    )


# ============================================================================================
# penalty-strength helpers
# ============================================================================================
def conservative_strength_bound(objective: SpinPolynomial, min_positive_violation: float, *, margin: float = 1.5) -> float:
    """Sufficient penalty strength: ``margin * 2 * sum_k |J_k(objective)| / min_positive_violation``.

    Valid whenever the violation polynomial takes only the value 0 on feasible states and at least
    ``min_positive_violation`` on every infeasible state reachable by the library (true for the
    integer cardinality/parity penalties used here, where the minimum nonzero violation is exactly
    1 for a single broken unit constraint); it is NOT valid if some infeasible state's violation can
    be smaller than ``min_positive_violation``, or if the objective's clause weights are unbounded
    across instances (recompute per instance).
    """
    j_sum = float(np.sum(np.abs(objective.clause_weights())))
    if min_positive_violation <= 0:
        raise ValueError("min_positive_violation must be > 0")
    return float(margin) * 2.0 * j_sum / float(min_positive_violation)


def strength_sweep(problem: Problem, strength_values: Sequence[float], *, strength_key: str = "lam", instance: int = 0,
                   backend: str = "emulation", schedule, seed: int, n_replicas: int = 8, **run_kwargs) -> list[dict]:
    """Run the penalty baseline at each strength; report feasible fraction and best feasible objective.

    One row per strength value: ``{"strength": lam, "feasible_fraction": ..., "best_feasible_objective": ...}``.
    ``best_feasible_objective`` is the best (min) over replicas that were ever feasible (NaN if none were).
    """
    out = []
    for lam in strength_values:
        res = run_penalty(problem, instance=instance, strengths={strength_key: lam}, backend=backend,
                          schedule=schedule, seed=seed, n_replicas=n_replicas, **run_kwargs)
        feas_frac = float(np.mean(res.feasible))
        best = float(np.nanmin(res.best_feasible_objective)) if np.any(res.feasible) else math.nan
        out.append({"strength": float(lam), "feasible_fraction": feas_frac, "best_feasible_objective": best})
    return out

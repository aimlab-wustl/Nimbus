"""Optimized CPU architecture backend (``nimbus.backends.cpu``), Numba-accelerated.

Mirrors ``nimbus.backends.emulator`` exactly (same acceptance law, same
transaction order, same counters/best/target conventions) with three selectable *strategies* for
maintaining the transition-neuron input ``v`` and two *layouts* for traversing a group's routes;
see ``nimbus.backends._numba_kernels`` for the kernels.

Plan (``CPUPlan``, built once per compiled model in :meth:`CPUBackend.prepare`):

* constant routes (``route_ctrl == -1``) are mapped, once, to a dummy controller clause index
  ``M`` (``T`` arrays are sized ``M + 1``, the last entry pinned to ``+1`` and never toggled), so
  every route's switch is the same branch-free expression ``(1 + pol * T[ctrl]) >> 1`` (no
  ``ctrl == -1`` special case in the hot loop). This is a plan-level convenience, not a model
  change: the model's own ``route_ctrl``/``route_pol`` arrays are untouched.
* layout ``'gate_blocked'`` additionally groups, within each group, the routes by their
  ``(ctrl, pol)`` pair (weighted sub-range first, stable order, so summation order is deterministic
  and recorded in :meth:`CPUPlan.describe`); a closed block is skipped after reading its
  controller's sign once, instead of reading every route's controller individually (plans may reorder
  routes within a group as long as ``v_r``, sigma and the toggles are
  unchanged).

Strategy ``'incremental'`` additionally maintains an explicit per-route switch cache (``alpha``,
mirroring ``EmulatorState.alpha``) so that stage 1/2 propagation never
re-derives a switch from its controller's sign; strategies ``'recompute'``/``'lazy'`` do not use it.

``run`` has two code paths sharing the same per-attempt kernel
(``_numba_kernels.attempt_kernel``):

* the parallel bulk path (``_numba_kernels.run_block_prange``, used when the recording profile asks
  for nothing beyond counters/best/feasible-best/targets) runs every replica's whole attempt window
  on its own ``prange`` lane, with **no Python objects under nogil** - the counters/best bookkeeping
  is array arithmetic equivalent to ``base.RunBook.on_change``;
* the sequential path (used whenever anytime/events/trace recording is requested, and by
  :meth:`CPUBackend.step`) calls the same kernel one attempt at a time from Python and reuses
  ``base.RunBook`` for bookkeeping, so the "diagnostics cost extra, summary pays nothing" rule
 holds by construction: the fast path simply never visits that code.

Proposal processes (``prepare(..., process=, process_options=)``): select
runs the code above unchanged. shared/sweep use the same attempt (``_numba_kernels.attempt_given``)
with their own group draw, one replica per prange lane (or, ``shared_impl='lockstep'``, the
replica-minor lock-step kernel). parallel_trial/nfold evaluate every input every step, either
recomputed per step (``'recompute'``, gate-blocked) or maintained (``'incremental'``, flat latch
order plus open spin-singleton route counts); ``'auto'`` follows ``choose_stepwise``. Kernels live in
``_numba_process``.
"""

from __future__ import annotations

import copy
import platform
import time

import numpy as np

try:
    import numba
except ImportError as exc:  # pragma: no cover - numba is a hard dependency of this backend
    raise ImportError(f"the cpu backend needs numba: {exc}") from exc

from nimbus.annealing.schedule import Schedule
from nimbus.backends import _numba_kernels as K
from nimbus.backends import _numba_process as PK
from nimbus.backends.coloring import color_groups
from nimbus.backends.base import (
    BEST_RTOL,
    COUNTER_NAMES,
    FEASIBLE_RTOL,
    TARGET_RTOL,
    UPHILL_RTOL,
    ArchitectureValidationError,
    Backend,
    BackendPlan,
    ProcessSpec,
    ReplicaStates,
    RunBook,
    RunResult,
    SlotLayout,
    StepRecord,
    apply_targets,
    check_host_device,
    check_initial_feasibility,
    check_process_budget,
    clause_signs,
    coerce_bindings,
    coerce_replicas,
    cohort_ids,
    encoder_inputs,
    ensure_process_counters,
    ensure_event_counters,
    host_versions,
    initial_replica_states,
    register_backend,
    resolve_window,
    route_switches,
    singleton_routes,
    termination_labels,
)
from nimbus.core.errors import ConfigError, IncompatibleCheckpoint
from nimbus.core.model import CompiledModel, structural_resources
from nimbus.experiments.recording import AnytimeLog, EventLog, RecordingData, RecordingSpec

__all__ = ["CPUBackend", "CPUPlan", "CPUState", "choose_auto", "choose_adaptive_strategy",
          "adaptive_subwindow_size", "STRATEGIES", "LAYOUTS", "ADAPTIVE",
          "ADAPTIVE_MIN_SUBWINDOW", "ADAPTIVE_SUBWINDOW_DIVISOR", "ADAPTIVE_HYSTERESIS", "ADAPTIVE_C0",
          "SCHEDULINGS", "QUEUE_CHUNKSIZE", "resolve_cpu_scheduling"]

STRATEGIES = ("recompute", "incremental", "lazy")
LAYOUTS = ("flat", "gate_blocked")
SHARED_IMPLS = ("auto", "lockstep", "per_replica")
SCHEDULINGS = ("static", "queue", "auto")
#: numba's dynamic-scheduling chunk size for scheduling='queue' (numba.set_parallel_chunksize): small
#: enough that an early-stopped replica's thread re-joins the pool at single-replica granularity.
#: Measured with numba 0.68.0: this version's default prange scheduler (chunksize 0) already dynamically
#: load-balances (no naive one-contiguous-chunk-per-thread static partition), so explicit chunking
#: measured no further improvement in every imbalanced-workload construction tried (one huge outlier
#: replica; half-of-all-replicas heavy scattered; one thread's whole static chunk heavy). The option is implemented and
#: plumbed through regardless (forward/portability: a different numba version or threading layer could
#: behave differently), with 'static' = chunksize left at numba's default (0).
QUEUE_CHUNKSIZE = 1
#: lock-step lanes are at least this many replicas wide (SIMD over replicas)
LOCKSTEP_MIN_WIDTH = 8
ADAPTIVE = "adaptive"
_STRAT_CODE = {"recompute": K.STRAT_RECOMPUTE, "incremental": K.STRAT_INCREMENTAL, "lazy": K.STRAT_LAZY}
_LAYOUT_CODE = {"flat": K.LAYOUT_FLAT, "gate_blocked": K.LAYOUT_BLOCKED}

# ============================================================================================
# adaptive strategy: switches strategy_code between sub-windows of a run; layout stays fixed for the whole plan.
# ============================================================================================
ADAPTIVE_MIN_SUBWINDOW = 4096  # attempts
ADAPTIVE_SUBWINDOW_DIVISOR = 64  # sub-window size = max(ADAPTIVE_MIN_SUBWINDOW, B / this)
ADAPTIVE_HYSTERESIS = 0.20  # only switch when the predicted cost drops by more than this fraction
ADAPTIVE_C0 = 1.0  # incremental's O(1) cached-read unit cost (same units as the recompute read cost)


def adaptive_subwindow_size(n_total: int) -> int:
    """Sub-window length in attempts: ``max(ADAPTIVE_MIN_SUBWINDOW, B / ADAPTIVE_SUBWINDOW_DIVISOR)``,
    capped at the run length ``B`` itself (short runs get a single, forced-``recompute`` window)."""
    B = max(1, int(n_total))
    return max(1, min(max(ADAPTIVE_MIN_SUBWINDOW, B // ADAPTIVE_SUBWINDOW_DIVISOR), B))


def _adaptive_costs(consts: dict, p_acc: float) -> tuple:
    """(cost_recompute, cost_incremental) for one sub-window at acceptance fraction ``p_acc``:
    cost_recompute ~ mean weighted-route read per attempt plus the
    shared accept-time latch/toggle cost (``accept_work``) paid with probability ``p_acc``;
    cost_incremental ~ an O(1) cached read (``ADAPTIVE_C0``) plus ``accept_work`` *and* the stage-1/2
    propagation upper bound (``propagation_work``), both paid with probability ``p_acc``."""
    cr = consts["read_cost"] + p_acc * consts["accept_work"]
    ci = ADAPTIVE_C0 + p_acc * (consts["accept_work"] + consts["propagation_work"])
    return cr, ci


def choose_adaptive_strategy(consts: dict, p_acc: float, prev_code: int,
                             hysteresis: float = ADAPTIVE_HYSTERESIS) -> tuple:
    """(strategy_code, cost_recompute, cost_incremental) for the next sub-window: the cheaper of
    the two predicted costs, unless switching away from ``prev_code`` would not cut the predicted
    cost by more than ``hysteresis`` (avoids thrashing between strategies every sub-window)."""
    cr, ci = _adaptive_costs(consts, p_acc)
    best_code, best_cost = (K.STRAT_RECOMPUTE, cr) if cr <= ci else (K.STRAT_INCREMENTAL, ci)
    if best_code == prev_code:
        return best_code, cr, ci
    prev_cost = cr if prev_code == K.STRAT_RECOMPUTE else ci
    if prev_cost <= 0.0 or (prev_cost - best_cost) / prev_cost <= hysteresis:
        return prev_code, cr, ci
    return best_code, cr, ci


def resolve_cpu_scheduling(scheduling: str, stop_on_target: bool, process_name: str) -> str:
    """``scheduling="auto"`` -> ``"static"`` or ``"queue"`` (same rule as the gpu backend's
    ``nimbus.backends.gpu.resolve_scheduling``): dynamic (work-queue) scheduling is only worth
    considering when replicas in this run can end before the full attempt budget --
    ``stop_on_target`` (every process), or ``process_name == "nfold"`` (can freeze early even without
    a target). Otherwise every replica's prange lane runs to the same
    attempt_stop regardless of scheduling, so 'static' (numba's own default chunking) is used."""
    if scheduling not in SCHEDULINGS:
        raise ConfigError(f"cpu scheduling must be one of {SCHEDULINGS}, got {scheduling!r}")
    if scheduling != "auto":
        return scheduling
    return "queue" if (stop_on_target or process_name == "nfold") else "static"


class _chunksize_scope:
    """Context manager: ``numba.set_parallel_chunksize(n)`` for the block, restoring the previous value
    on exit (even on error). ``n = 0`` means "numba's own default" (what 'static' uses)."""

    def __init__(self, n: int):
        self.n = int(n)
        self._prev = None

    def __enter__(self):
        self._prev = numba.get_parallel_chunksize()
        numba.set_parallel_chunksize(self.n)
        return self

    def __exit__(self, *exc):
        numba.set_parallel_chunksize(self._prev)
        return False


def _scheduling_chunksize(scheduling: str) -> int:
    return QUEUE_CHUNKSIZE if scheduling == "queue" else 0


def _merge_anytime_logs(a, b):
    if a is None or b is None:
        return a if a is not None else b
    out = AnytimeLog(a.n_replicas, a.checkpoints, a.improvement_cap)
    out.best = np.where(np.isnan(a.best), b.best, a.best)
    out.current = np.where(np.isnan(a.current), b.current, a.current)
    out.feasible_best = np.where(np.isnan(a.feasible_best), b.feasible_best, a.feasible_best)
    out._imp = [ia + ib for ia, ib in zip(a._imp, b._imp)]
    out.improvements_dropped = a.improvements_dropped + b.improvements_dropped
    return out


def _merge_trace_logs(a, b):
    if a is None or b is None:
        return a if a is not None else b
    out = copy.copy(a)
    out._rows = list(a._rows) + list(b._rows)
    out._snap = {s: list(a._snap[s]) + list(b._snap[s]) for s in a.snapshots}
    out.n_dropped = a.n_dropped + b.n_dropped
    out.first_dropped_attempt = a.first_dropped_attempt if a.first_dropped_attempt >= 0 else b.first_dropped_attempt
    out.bytes_used = a.bytes_used + b.bytes_used
    return out


def _merge_recording(acc: RecordingData | None, new: RecordingData) -> RecordingData:
    """Concatenate two ``RecordingData`` chunks covering disjoint, adjacent attempt ranges (one per
    adaptive sub-window) into one logical observation of the whole run() call, the same way a
    caller would merge consecutive chunked ``run()`` calls by hand."""
    if acc is None:
        return new
    a_events, b_events = acc.events or {}, new.events or {}
    events = {k: a_events.get(k, 0) + b_events.get(k, 0) for k in set(a_events) | set(b_events)} or None
    return RecordingData(spec=new.spec, anytime=_merge_anytime_logs(acc.anytime, new.anytime), events=events,
                         trace=_merge_trace_logs(acc.trace, new.trace), notes={**acc.notes, **new.notes})


# ============================================================================================
# auto policy
# ============================================================================================
def choose_auto(model: CompiledModel) -> tuple:
    """Structural ``(strategy, layout)`` policy, chosen from timings on compiled models of several sizes
    (dense and sparse objectives, complete and path/tree move libraries), keyed only on compiled-model structure (reproducible from the model alone, before any
    run and before any schedule/acceptance regime is known):

    * ``layout``: ``'gate_blocked'`` whenever groups average more than 8 routes (``fan_in_mean``):
      even with zero gating, grouping routes by their ``(ctrl, pol)`` lets the switch of a whole
      block be read once instead of on every route (measured 1.4-2.1x faster than ``'flat'`` on
      every model tried, including the gate-free ones - the win is in avoiding the
      per-route branch, not only in skipping closed blocks). Below that, the extra block indirection
      is not worth it.
    * ``strategy``: a single accepted transaction's stage-1/2 propagation cost (``'incremental'``)
      scales with ``kt_size * obj_fanout_mean`` (how many stored clauses typically toggle, times how
      many groups/weighted-routes reference each toggled objective clause) and with
      ``control_fan_out_mean`` (switch re-evaluation fan-out). ``kt_size`` is not known before a run,
      but ``max_clause_degree`` is a strong proxy (a degree-d clause can be among the ones toggled by
      any of its d member spins, so dense high-degree objectives toggle many clauses per
      move): ``recompute`` when ``max_clause_degree * obj_fanout_mean > 40`` (measured: values from
      72 to 109 on dense objectives and complete assignment libraries - here ``incremental``
      was 14-100x *slower* than ``recompute`` because one accepted move updates most of the model;
      ``recompute``'s cost is bounded by one group's fan-in regardless). Otherwise, when switch
      fan-out is still non-trivial (``control_fan_out_mean > 30``, e.g. an assignment problem with a sparse/tree
      library: ``obj_fanout_mean`` stays low but routes are still gated) choose ``'lazy'`` (marking
      beats eager correction when many of the marked groups are not read again before they change
      once more); else ``'incremental'`` (``control_fan_out_mean == 0``, degree-1
      objective - maintaining every ``v`` exactly is free and every strategy performed within 20% of
      each other, so the cached O(1) read is the safe default).
    """
    res = structural_resources(model)
    from nimbus.core.model import ROLE_OBJECTIVE

    obj = np.flatnonzero(model.clause_role & ROLE_OBJECTIVE)
    obj_fanout_mean = float(np.diff(model.clause_wroute_ptr)[obj].mean()) if obj.size else 0.0
    max_clause_degree = float(res["max_clause_degree"])
    control_fan_out_mean = float(res["control_fan_out_mean"])
    fan_in_mean = float(res["fan_in_mean"])
    layout = "gate_blocked" if fan_in_mean > 8.0 else "flat"
    propagation_cost = max_clause_degree * obj_fanout_mean
    if propagation_cost > 40.0:
        strategy = "recompute"
    elif control_fan_out_mean > 30.0:
        strategy = "lazy"
    else:
        strategy = "incremental"
    return strategy, layout


#: relative cost of one route read in the recompute scan vs one propagation update (sequential,
#: cache-friendly reads vs scattered read-modify-writes)
STEPWISE_RECOMPUTE_FACTOR = 0.45


def stepwise_costs(model: CompiledModel) -> dict:
    """Structural per-step costs (route reads) of the two parallel_trial / nfold input strategies:
    ``recompute`` = one switch read per (group, ctrl, pol) block + every weighted route (upper bound: all
    blocks open); ``incremental`` = mean over groups of one commit's stage-1/2 fan-out (weighted routes
    of the reached objective clauses + routes controlled by the reached clauses) + R for the scan."""
    from nimbus.core.model import ROLE_OBJECTIVE

    R = model.n_groups
    if R == 0:
        return {"recompute": 0.0, "incremental": 0.0}
    rp = model.route_ptr.astype(np.int64)
    we = model.route_wend.astype(np.int64)
    grp = model.route_group().astype(np.int64)
    key = grp * (2 * (model.n_clauses + 1)) + (model.route_ctrl.astype(np.int64) + 1) * 2 + (
        model.route_pol.astype(np.int64) > 0)
    nblk = int(np.unique(key).size)
    is_obj = (model.clause_role & ROLE_OBJECTIVE) != 0
    cw = np.diff(model.clause_wroute_ptr).astype(np.float64)
    cr = np.diff(model.ctrl_route_ptr).astype(np.float64)
    rc = model.route_clause.astype(np.int64)
    contrib = np.where(is_obj[rc], cw[rc], 0.0) + cr[rc]
    return {"recompute": float(nblk + int((we - rp[:-1]).sum())), "incremental": float(contrib.sum()) / R + R}


def choose_stepwise(model: CompiledModel) -> str:
    """Input strategy of parallel_trial / nfold ('auto'): both processes need every v_r at every step and
    commit at almost every step, so the cheaper of the two structural costs of ``stepwise_costs`` wins
    (recompute scaled by ``STEPWISE_RECOMPUTE_FACTOR``)."""
    c = stepwise_costs(model)
    return "recompute" if c["recompute"] * STEPWISE_RECOMPUTE_FACTOR < c["incremental"] else "incremental"


# ============================================================================================
# plan
# ============================================================================================
class CPUPlan(BackendPlan):
    """Numba-ready arrays for one compiled model: dummy-controller-mapped routes, dependency CSRs
    and (layout ``'gate_blocked'``) the (ctrl, pol)-blocked route reordering."""

    def __init__(self, model: CompiledModel, *, strategy="auto", layout="auto", precision="float64",
                device=None, threads=None, process="select", process_options=None, shared_impl="auto",
                lockstep_width=None, scheduling="auto", event_counters=False, event_windows=None, **options):
        if options:
            raise ConfigError(f"unknown cpu backend options {sorted(options)}")
        if scheduling not in SCHEDULINGS:
            raise ConfigError(f"cpu scheduling must be one of {SCHEDULINGS}, got {scheduling!r}")
        self.scheduling = scheduling
        if precision != "float64":
            raise ConfigError("the cpu backend computes in float64 only (float32 is not implemented)")
        proc = ProcessSpec.coerce(process, process_options)
        self.process = proc
        self.event_counters = bool(event_counters) or event_windows is not None
        # per-window counters: global attempt edges
        self.event_windows = None
        if event_windows is not None:
            ew = np.ascontiguousarray(np.asarray(event_windows, dtype=np.int64))
            if ew.ndim != 1 or ew.size < 2 or np.any(np.diff(ew) <= 0) or ew[0] < 0:
                raise ConfigError("event_windows must be strictly increasing non-negative attempt edges (>= 2)")
            if strategy == ADAPTIVE:
                raise ConfigError("event_windows are not implemented for the adaptive strategy")
            self.event_windows = ew
        if self.event_counters and not proc.is_select:
            raise ConfigError("event_counters are implemented for the select process only "
                              f"(got process {proc.name!r})")
        if shared_impl not in SHARED_IMPLS:
            raise ConfigError(f"shared_impl must be one of {SHARED_IMPLS}, got {shared_impl!r}")
        if lockstep_width is not None and int(lockstep_width) < 1:
            raise ConfigError("lockstep_width must be >= 1")
        self.lockstep_width = None if lockstep_width is None else int(lockstep_width)
        self.process_impl = "select"
        if not proc.is_select and strategy == ADAPTIVE:
            raise ConfigError(f"the adaptive strategy is implemented for the select process only, not {proc.name!r}")
        if proc.stepwise:
            # every v_r settled every step: 'incremental' = maintained inputs, latch in
            # model route order (flat); 'recompute' = every input recomputed per step (gate-blocked sums)
            if strategy == "auto":
                strategy = choose_stepwise(model)
            if strategy == "incremental":
                if layout not in ("auto", "flat"):
                    raise ConfigError(f"process {proc.name!r} with maintained inputs uses layout 'flat'")
                layout = "flat"
                self.process_impl = "stepwise_maintained"
            elif strategy == "recompute":
                if layout not in ("auto", "gate_blocked"):
                    raise ConfigError(f"process {proc.name!r} with recomputed inputs uses layout 'gate_blocked'")
                layout = "gate_blocked"
                self.process_impl = "stepwise_recompute"
            else:
                raise ConfigError(f"process {proc.name!r}: strategy must be 'auto', 'incremental' or 'recompute', "
                                  f"got {strategy!r}")
        elif proc.name == "shared":
            if shared_impl == "lockstep" and strategy not in ("auto", "recompute"):
                raise ConfigError("the lock-step shared implementation recomputes v_r (strategy 'recompute')")
            if shared_impl == "lockstep":
                strategy, layout = "recompute", "gate_blocked"
                self.process_impl = "shared_lockstep"
            else:
                self.process_impl = "per_replica"
        elif proc.name == "sweep":
            self.process_impl = "per_replica"
        elif proc.colored:
            self.process_impl = "colored"  # per-replica select-style attempts; strategy/layout as for select
        elif shared_impl != "auto":
            raise ConfigError("shared_impl applies to the shared process only")
        auto_strategy, auto_layout = choose_auto(model)
        is_adaptive = strategy == ADAPTIVE
        if not is_adaptive:
            strategy = auto_strategy if strategy == "auto" else strategy
            if strategy not in STRATEGIES:
                raise ConfigError(f"cpu strategy must be one of {STRATEGIES}, 'auto' or {ADAPTIVE!r}, "
                                  f"got {strategy!r}")
        # adaptive: layout is fixed for the whole plan (only strategy_code switches between
        # sub-windows); 'auto' defaults it to 'gate_blocked' rather than the model-dependent
        # choose_auto layout, per the adaptive design.
        layout = (("gate_blocked" if is_adaptive else auto_layout) if layout == "auto" else layout)
        if layout not in LAYOUTS:
            raise ConfigError(f"cpu layout must be one of {LAYOUTS} or 'auto', got {layout!r}")
        if threads is not None and int(threads) < 1:
            raise ConfigError("threads must be >= 1")
        N, M, R, nnz = model.n_spins, model.n_clauses, model.n_groups, model.n_routes
        self.N, self.M, self.R, self.nnz, self.DUMMY = N, M, R, nnz, M
        self.strategy, self.layout = strategy, layout
        self.is_adaptive = is_adaptive
        self.strategy_code = K.STRAT_RECOMPUTE if is_adaptive else _STRAT_CODE[strategy]
        self.layout_code = _LAYOUT_CODE[layout]
        self.threads = None if threads is None else int(threads)

        def intp(a):
            return np.ascontiguousarray(a, dtype=np.intp)

        route_ctrl_raw = model.route_ctrl.astype(np.int64)
        self.route_ptr = intp(model.route_ptr)
        self.route_wend = intp(model.route_wend)
        self.route_clause = intp(model.route_clause)
        self.route_ctrl = intp(np.where(route_ctrl_raw < 0, self.DUMMY, route_ctrl_raw))
        self.route_pol = np.ascontiguousarray(np.where(route_ctrl_raw < 0, 1, model.route_pol), dtype=np.int8)
        self.route_group = intp(model.route_group())
        self.route_weighted = np.ascontiguousarray(model.weighted_route_mask(), dtype=np.bool_)
        self.clause_wroute_ptr = intp(model.clause_wroute_ptr)
        self.clause_wroute = intp(model.clause_wroute)
        self.ctrl_route_ptr = intp(model.ctrl_route_ptr)
        self.ctrl_route = intp(model.ctrl_route)
        from nimbus.core.model import ROLE_GATE, ROLE_OBJECTIVE

        self.is_obj = np.ascontiguousarray((model.clause_role & ROLE_OBJECTIVE) != 0, dtype=np.bool_)
        # gate-only clauses (GATE and not OBJECTIVE; the emulator's CLASS_GATE), + the dummy controller slot
        gate_only = np.zeros(M + 1, dtype=np.bool_)
        gate_only[:M] = ((model.clause_role & ROLE_GATE) != 0) & ((model.clause_role & ROLE_OBJECTIVE) == 0)
        self.is_gate_only = np.ascontiguousarray(gate_only)
        self.has_ctrl = np.ascontiguousarray(np.diff(self.ctrl_route_ptr) > 0, dtype=np.bool_)
        # spins are stored singleton clauses updated by the decoder: no spin-action
        # arrays reach the kernels; the state changed iff a toggled clause is a spin singleton
        self.spin_clause = intp(model.readout_clause_of_spin)
        is_spin = np.zeros(M + 1, dtype=np.bool_)  # + the pinned dummy controller (never a spin)
        is_spin[self.spin_clause] = True
        self.is_spin_clause = np.ascontiguousarray(is_spin)
        self.max_group_routes = int(np.diff(self.route_ptr).max()) if R else 0
        self.coloring = color_groups(model) if proc.colored else None
        if self.coloring is not None:
            self.col_order, self.col_ptr, self.col_class_of_pos = (
                intp(self.coloring.order), intp(self.coloring.class_ptr), intp(self.coloring.class_of_pos))
        sr_ptr, sr_pos = singleton_routes(model)
        self.sr_ptr, self.sr_pos = intp(sr_ptr), intp(sr_pos)
        self._build_blocks(model)
        self.adaptive_consts = self._compute_adaptive_consts() if is_adaptive else None
        super().__init__(backend="cpu", model=model, strategy=strategy, layout=layout, precision=precision,
                         device=check_host_device("cpu", device),
                         options={"threads": self.threads, "numba": numba.__version__, "scheduling": self.scheduling,
                                  "summation_order": "weighted-then-unweighted per (ctrl,pol) block, "
                                                      "stable within block" if layout == "gate_blocked"
                                                      else "model route order (weighted-first per group)",
                                  "process": proc.name, "process_impl": self.process_impl,
                                  "event_counters": self.event_counters,
                                  "event_windows": (None if self.event_windows is None else int(self.event_windows.size - 1)),
                                  **({"coloring_" + k: v for k, v in self.coloring.summary().items() if v is not None and not isinstance(v, list)}
                                     if self.coloring is not None else {}),
                                  **{"process_" + k: v for k, v in proc.describe().items() if k != "process"},
                                  "lockstep_width": self.lockstep_width,
                                  **({"adaptive_subwindow_min": ADAPTIVE_MIN_SUBWINDOW,
                                      "adaptive_subwindow_divisor": ADAPTIVE_SUBWINDOW_DIVISOR,
                                      "adaptive_hysteresis": ADAPTIVE_HYSTERESIS} if is_adaptive else {})})

    def _compute_adaptive_consts(self) -> dict:
        """Structural constants for the adaptive strategy's per-sub-window cost model, computed once
        from plan topology alone:

        * ``read_cost``: mean over groups of the recompute-strategy read cost for ``self.layout`` -
          weighted routes per group (``route_wend - route_ptr``) for ``'flat'``, or its
          ``'gate_blocked'`` equivalent (one controller read per ``(ctrl, pol)`` block plus its
          weighted routes, summed per group) - an upper bound (it does not know which blocks are
          closed at run time, same convention as ``choose_auto``).
        * ``accept_work``: mean over groups of ALL routes (weighted and not) latched into ``kt_buf``
          on acceptance (``route_ptr`` fan-out) - the kt-latch-and-toggle cost every strategy pays
          identically on an accepted attempt.
        * ``propagation_work``: mean over groups of sum over that group's routes of
          (``len(clause_wroute[k])`` for an objective clause ``k`` + ``len(ctrl_route[k])``) - an
          upper bound on the extra stage-1/2 propagation ``'incremental'`` pays on an accepted
          attempt, on top of ``accept_work``.
        """
        R = self.R
        if R == 0:
            return {"read_cost": 0.0, "accept_work": 0.0, "propagation_work": 0.0}
        if self.layout == "gate_blocked":
            widths = (self.blk_wend - self.blk_start).astype(np.float64)
            ptr = self.blk_group_ptr
            per_group = np.empty(R, dtype=np.float64)
            for r in range(R):
                a, b = int(ptr[r]), int(ptr[r + 1])
                per_group[r] = (b - a) + (float(widths[a:b].sum()) if b > a else 0.0)
            read_cost = float(per_group.mean())
        else:
            fan_in = (self.route_wend - self.route_ptr[:-1]).astype(np.float64)
            read_cost = float(fan_in.mean()) if fan_in.size else 0.0
        accept_work = float(np.diff(self.route_ptr).astype(np.float64).mean()) if R else 0.0
        cw_len = np.diff(self.clause_wroute_ptr).astype(np.float64)
        cr_len = np.diff(self.ctrl_route_ptr).astype(np.float64)
        rc = self.route_clause
        contrib = np.where(self.is_obj[rc], cw_len[rc], 0.0) + cr_len[rc] if rc.size else np.zeros(0)
        prop_per_group = np.empty(R, dtype=np.float64)
        for r in range(R):
            a, b = int(self.route_ptr[r]), int(self.route_ptr[r + 1])
            prop_per_group[r] = float(contrib[a:b].sum()) if b > a else 0.0
        propagation_work = float(prop_per_group.mean())
        return {"read_cost": read_cost, "accept_work": accept_work, "propagation_work": propagation_work}

    def _build_blocks(self, model: CompiledModel):
        """(ctrl, pol)-grouped route reordering for layout ``'gate_blocked'`` (pure numpy, plan-build
        time only). No-op arrays (size 0) when the layout is ``'flat'``."""
        R, nnz = self.R, self.nnz
        if self.layout != "gate_blocked" or nnz == 0:
            self.blk_group_ptr = np.zeros(R + 1, dtype=np.intp)
            self.blk_ctrl = np.zeros(0, dtype=np.intp)
            self.blk_pol = np.zeros(0, dtype=np.int8)
            self.blk_start = np.zeros(0, dtype=np.intp)
            self.blk_wend = np.zeros(0, dtype=np.intp)
            self.blk_end = np.zeros(0, dtype=np.intp)
            self.plan_route_clause = np.zeros(0, dtype=np.intp)
            return
        group_of = self.route_group
        key = self.route_ctrl.astype(np.int64) * 2 + (self.route_pol.astype(np.int64) + 1) // 2
        # weighted-first within each (group, key): sort by (group, not-weighted, key), stable
        order = np.lexsort((key, (~self.route_weighted).astype(np.int64), group_of))
        pos = order  # original nnz position for each new slot
        key_sorted = key[pos]
        w_sorted = self.route_weighted[pos]
        plan_route_clause = self.route_clause[pos]
        plan_ctrl = self.route_ctrl[pos]
        plan_pol = self.route_pol[pos]
        blk_group_ptr = np.zeros(R + 1, dtype=np.intp)
        blk_ctrl, blk_pol, blk_start, blk_wend, blk_end = [], [], [], [], []
        i = 0
        for r in range(R):
            blk_group_ptr[r] = len(blk_ctrl)
            a, b = int(self.route_ptr[r]), int(self.route_ptr[r + 1])
            i = a
            while i < b:
                j = i
                k0 = key_sorted[i]
                while j < b and key_sorted[j] == k0:
                    j += 1
                wend = i
                while wend < j and w_sorted[wend]:
                    wend += 1
                blk_ctrl.append(int(plan_ctrl[i]))
                blk_pol.append(int(plan_pol[i]))
                blk_start.append(i)
                blk_wend.append(wend)
                blk_end.append(j)
                i = j
        blk_group_ptr[R] = len(blk_ctrl)
        self.blk_group_ptr = blk_group_ptr
        self.blk_ctrl = np.asarray(blk_ctrl, dtype=np.intp)
        self.blk_pol = np.asarray(blk_pol, dtype=np.int8)
        self.blk_start = np.asarray(blk_start, dtype=np.intp)
        self.blk_wend = np.asarray(blk_wend, dtype=np.intp)
        self.blk_end = np.asarray(blk_end, dtype=np.intp)
        self.plan_route_clause = np.ascontiguousarray(plan_route_clause, dtype=np.intp)

    def resources(self) -> dict:
        return structural_resources(self.model)


# ============================================================================================
# state
# ============================================================================================
class CPUState:
    """Replica states plus the per-replica scratch buffers and (strategy ``'incremental'``) the
    persistent switch cache. ``rs.v`` is ``None`` for ``'recompute'`` (nothing maintained); for
    ``'incremental'``/``'lazy'`` it is always present with ``NaN`` marking a stale (unheld) group
    ("NaN for groups whose input a backend does not hold")."""

    def __init__(self, plan: CPUPlan, rs: ReplicaStates, bindings, alpha, T_full=None):
        self.plan = plan
        self.rs = rs
        if T_full is None:
            # rebuild the dummy-extended array (last column pinned to +1) from rs.T (model.n_clauses columns)
            T_full = np.empty((rs.n_replicas, plan.M + 1), dtype=np.int8)
            T_full[:, : plan.M] = rs.T
            T_full[:, plan.M] = 1
        self.T_full = np.ascontiguousarray(T_full, dtype=np.int8)
        rs.T = self.T_full[:, : plan.M]  # a view: kernels write T_full in place, rs.T stays in sync
        self.bindings = tuple(bindings)
        self.layout = SlotLayout.from_bindings(self.bindings)
        n_slots = self.bindings[0].n_slots
        self.J_stack = np.ascontiguousarray(np.stack([np.asarray(b.J, dtype=np.float64) for b in bindings]))
        self.J0_stack = np.ascontiguousarray(self.J_stack[:, 0, :])
        self.alpha = np.ascontiguousarray(alpha, dtype=np.int8)
        n = rs.n_replicas
        self.kt_buf = np.zeros((n, max(plan.max_group_routes, 1)), dtype=np.intp)
        self.told_buf = np.zeros((n, max(plan.max_group_routes, 1)), dtype=np.int8)
        self.J0_of_rep = self.J0_stack[rs.replica_binding.astype(np.int64)]
        self.J_of_rep = self.J_stack[rs.replica_binding.astype(np.int64)]
        self.n_slots = n_slots
        self.v_scratch = np.full((n, max(plan.R, 1)), np.nan)  # used only when rs.v is None ('recompute')
        proc = plan.process
        self.cohort = cohort_ids(proc, rs.replica_id)
        self.perm_buf = np.zeros((n, plan.R if proc.name == "sweep" else 1), dtype=np.int64)
        self.perm_sweep = np.full(n, -1, dtype=np.int64)
        if proc.stepwise:
            self.open_cnt = self.open_counts(plan, self.alpha)
            self.cand_buf = np.zeros((n, max(plan.R, 1)), dtype=np.int64)
            self.pref_buf = np.zeros((n, max(plan.R, 1)), dtype=np.float64)
        else:
            self.open_cnt = np.zeros((n, 1), dtype=np.int64)

    @staticmethod
    def open_counts(plan, alpha) -> np.ndarray:
        """int64[n_rep, R]: open spin-singleton routes per group (group r changes the state iff > 0)."""
        out = np.zeros((alpha.shape[0], plan.R), dtype=np.int64)
        if plan.sr_pos.size:
            grp = plan.route_group[plan.sr_pos]
            for i in range(alpha.shape[0]):
                out[i] = np.bincount(grp, weights=alpha[i, plan.sr_pos], minlength=plan.R).astype(np.int64)
        return out

    @property
    def n_replicas(self) -> int:
        return self.rs.n_replicas

    @property
    def vrows(self) -> np.ndarray:
        """The input rows a kernel reads/writes: the maintained ``rs.v`` or, without one, the scratch."""
        return self.rs.v if self.rs.v is not None else self.v_scratch


# ============================================================================================
# backend
# ============================================================================================
class CPUBackend(Backend):
    """Optimized CPU architecture backend (Numba); backend API plus ``step``."""

    name = "cpu"

    def prepare(self, model: CompiledModel, *, strategy="auto", layout="auto", precision="float64", device=None,
               threads=None, process="select", process_options=None, scheduling="auto", **options) -> CPUPlan:
        """``process`` / ``process_options``: proposal process; options
        ``shared_impl`` ('auto' | 'lockstep' | 'per_replica') and ``lockstep_width`` select the shared
        implementation. ``scheduling`` ('static' | 'queue' | 'auto'): dynamic
        work-queue scheduling of replicas over the prange threads; the lock-step shared implementation always uses 'static' (its replicas are
        processed in SIMD-like lanes, not independent prange lanes)."""
        return CPUPlan(model, strategy=strategy, layout=layout, precision=precision, device=device,
                      threads=threads, process=process, process_options=process_options, scheduling=scheduling,
                      **options)

    # ------------------------------------------------------------------------------------------
    def init_state(self, plan: CPUPlan, bindings, spins0, replica_binding=None, replica_id=None,
                   instance_id=None, *, is_feasible=None, targets=None) -> CPUState:
        model = plan.model
        bindings = coerce_bindings(model, bindings)
        spins, rb, rid, iid = coerce_replicas(model, len(bindings), spins0, replica_binding, replica_id, instance_id)
        check_initial_feasibility(spins, is_feasible)
        T0 = clause_signs(model, spins)
        n = spins.shape[0]
        T_full = np.empty((n, plan.M + 1), dtype=np.int8)
        T_full[:, : plan.M] = T0
        T_full[:, plan.M] = 1
        E = np.stack([bindings[int(rb[i])].const - bindings[int(rb[i])].J @ T0[i].astype(np.float64)
                     for i in range(n)])
        alpha = route_switches(model, T0)  # every switch from its controller sign, at init
        v = None
        if plan.strategy != "recompute":
            J0rows = np.stack([bindings[int(rb[i])].J[0] for i in range(n)])
            v = encoder_inputs(model, J0rows, T0)
        rs = initial_replica_states("cpu", model, bindings, spins, rb, rid, iid, T0, E, v=v)
        ensure_process_counters(rs)
        if plan.event_counters:
            ensure_event_counters(rs)
        st = CPUState(plan, rs, bindings, alpha, T_full=T_full)
        apply_targets(rs, st.layout, targets)
        return st

    # ------------------------------------------------------------------------------------------
    def _kernel_args(self, plan: CPUPlan):
        return (plan.route_ptr, plan.route_wend, plan.route_clause, plan.route_ctrl, plan.route_pol,
               plan.route_group, plan.route_weighted, plan.blk_group_ptr, plan.blk_ctrl, plan.blk_pol,
               plan.blk_start, plan.blk_wend, plan.blk_end, plan.plan_route_clause, plan.is_obj,
               plan.has_ctrl, plan.clause_wroute_ptr, plan.clause_wroute, plan.ctrl_route_ptr, plan.ctrl_route,
               plan.is_spin_clause)

    def _set_threads(self, plan: CPUPlan):
        if plan.threads is not None:
            cap = int(numba.config.NUMBA_DEFAULT_NUM_THREADS)
            numba.set_num_threads(max(1, min(plan.threads, cap)))

    # ------------------------------------------------------------------------------------------
    def run(self, plan: CPUPlan, state: CPUState, schedule: Schedule, *, n_attempts_total=None, start=None,
           stop=None, seed: int, recording=None, targets=None, stop_on_target: bool = False) -> RunResult:
        if state.plan is not plan:
            raise ConfigError("state was initialized with a different plan")
        spec = RecordingSpec.coerce(recording)
        rs = state.rs
        apply_targets(rs, state.layout, targets, stop_on_target)
        proc = plan.process
        B, a0, a1 = resolve_window(rs, schedule, n_attempts_total, start, stop, ragged=proc.ragged)
        check_process_budget(proc, B, plan.R)
        self._set_threads(plan)
        was_stopped = rs.stopped.copy()
        att0 = rs.counters["attempts"].copy()
        t0 = time.perf_counter()
        want_diag = bool(spec.want_anytime or spec.want_events or spec.want_trace)
        strategy_schedule = None
        # scheduling: only the fast
        # (prange) paths below are affected; the sequential (diagnostics) paths process one replica at
        # a time already and the lock-step shared path always stays 'static' (resolved inside
        # _run_fast_process). Irrelevant when nothing can end early: resolve_cpu_scheduling('auto', ...)
        # then picks 'static' and this is a no-op (chunksize 0, numba's own default).
        effective_scheduling = resolve_cpu_scheduling(plan.scheduling, stop_on_target, proc.name)
        if not proc.is_select:
            if want_diag:
                hit_now = self._run_sequential_process(plan, state, schedule, a0, a1, B, seed, spec, stop_on_target)
                rec = self._last_recording
            else:
                hit_now = self._run_fast_process(plan, state, schedule, a0, a1, B, seed, stop_on_target,
                                                 effective_scheduling)
                rec = RecordingData(spec=spec, anytime=None, events=None, trace=None)
        elif plan.is_adaptive:
            hit_now, strategy_schedule = self._run_adaptive(plan, state, schedule, a0, a1, B, seed, spec,
                                                             stop_on_target, want_diag, effective_scheduling)
            rec = self._last_recording if want_diag else RecordingData(spec=spec, anytime=None, events=None,
                                                                       trace=None)
        elif not want_diag:
            with _chunksize_scope(_scheduling_chunksize(effective_scheduling)):
                hit_now = self._run_fast(plan, state, schedule, a0, a1, seed, stop_on_target)
            rec = RecordingData(spec=spec, anytime=None, events=None, trace=None)
        else:
            hit_now = self._run_sequential(plan, state, schedule, a0, a1, B, seed, spec, stop_on_target)
            rec = self._last_recording
        elapsed = time.perf_counter() - t0
        if proc.is_select:  # one round and one acceptance test per attempt
            d_att = rs.counters["attempts"] - att0
            rs.counters["steps"] += d_att
            rs.counters["candidate_evaluations"] += d_att
        term = termination_labels(rs, was_stopped, hit_now, a1, B)
        metadata = {"plan": plan.describe(), "schedule": schedule.to_dict(), "process": {**proc.describe(), **({"coloring": plan.coloring.summary()} if proc.colored else {})},
                   "scheduling": effective_scheduling}
        if strategy_schedule is not None:
            metadata["strategy_schedule"] = strategy_schedule
        return RunResult(backend=self.name, start=a0, stop=a1, n_attempts_total=B, seed=int(seed),
                         state=self.export_state(state), attempts_run=rs.counters["attempts"] - att0,
                         termination=term, recording=rec, timings={"run_s": elapsed}, metadata=metadata)

    # -- adaptive path: switches plan.strategy_code between sub-windows, re-using _run_fast /
    # _run_sequential unchanged over each sub-window's attempt range ------------------------------
    def _run_adaptive(self, plan: CPUPlan, state: CPUState, schedule: Schedule, a0: int, a1: int, B: int,
                      seed: int, spec: RecordingSpec, stop_on_target: bool, want_diag: bool,
                      scheduling: str = "static"):
        rs = state.rs
        consts = plan.adaptive_consts
        window = adaptive_subwindow_size(B)
        prev_code = K.STRAT_RECOMPUTE
        plan.strategy_code = prev_code
        if rs.v is not None:
            rs.v[:] = np.nan  # adaptive always starts its first sub-window 'recompute' (unheld)
        p_acc_next = None
        rec_acc = None
        hit_any = np.zeros(rs.n_replicas, dtype=bool)
        sched_log = []
        cur, first = a0, True
        while cur < a1:
            nxt = min(cur + window, a1)
            if first:
                code, cr, ci, p_in = K.STRAT_RECOMPUTE, None, None, None
            else:
                p_in = p_acc_next
                code, cr, ci = choose_adaptive_strategy(consts, p_in, prev_code)
            if code != prev_code:
                if code == K.STRAT_INCREMENTAL:
                    self._reinit_v_all(plan, state)
                else:
                    rs.v[:] = np.nan
            plan.strategy_code = code
            att_before = rs.counters["attempts"].copy()
            acc_before = rs.counters["accepted_changed"].copy()
            if want_diag:
                hit_sub = self._run_sequential(plan, state, schedule, cur, nxt, B, seed, spec, stop_on_target)
                rec_acc = _merge_recording(rec_acc, self._last_recording)
            else:
                with _chunksize_scope(_scheduling_chunksize(scheduling)):
                    hit_sub = self._run_fast(plan, state, schedule, cur, nxt, seed, stop_on_target)
            hit_any |= hit_sub
            d_att = int((rs.counters["attempts"] - att_before).sum())
            d_acc = int((rs.counters["accepted_changed"] - acc_before).sum())
            p_acc_next = (d_acc / d_att) if d_att > 0 else 0.0
            sched_log.append({"start": int(cur), "stop": int(nxt),
                              "strategy": "recompute" if code == K.STRAT_RECOMPUTE else "incremental",
                              "p_acc_in": p_in, "p_acc_observed": p_acc_next, "cost_recompute": cr,
                              "cost_incremental": ci})
            prev_code, cur, first = code, nxt, False
        if want_diag:
            self._last_recording = rec_acc
        return hit_any, sched_log

    def _reinit_v_all(self, plan: CPUPlan, state: CPUState):
        """Full re-initialization of ``v`` *and* ``alpha`` for every replica from the CURRENT clause
        state (same path as ``init_state``/``load_state``): needed when switching recompute ->
        incremental mid-run. Both are left untouched (stale) while ``strategy_code`` is
        ``recompute`` - ``commit_update``'s ``STRAT_RECOMPUTE`` branch maintains neither, so
        ``alpha`` (the per-route switch cache ``incremental``'s stage 1 reads via ``if
        alpha_row[p]``) must be rebuilt from the controllers' current signs exactly like ``v``,
        or stage 1 would apply corrections using switches that no longer match the true state."""
        rs = state.rs
        state.alpha[...] = route_switches(plan.model, rs.T)
        v_full = encoder_inputs(plan.model, state.J0_of_rep, rs.T)
        if rs.v is None:
            rs.v = v_full
        else:
            rs.v[...] = v_full

    # -- fast path: prange over replicas, array-only bookkeeping, no diagnostics -----------------
    def _run_fast(self, plan: CPUPlan, state: CPUState, schedule: Schedule, a0: int, a1: int, seed: int,
                 stop_on_target: bool) -> np.ndarray:
        rs = state.rs
        layout = state.layout
        b = rs.replica_binding.astype(np.int64)
        has_viol = layout.violation_slot is not None
        viol_slot = layout.violation_slot if has_viol else 0
        scale0 = layout.scale[b, 0]
        scale_s = layout.scale[b, layout.score_slot]
        scale_v = layout.scale[b, viol_slot] if has_viol else np.ones_like(scale0)
        from nimbus.backends.base import BEST_RTOL, FEASIBLE_RTOL, TARGET_RTOL, UPHILL_RTOL

        hit_before = rs.counters["target_hit_attempt"].copy()
        next_attempt = rs.next_attempt.copy()
        v_arr = rs.v if rs.v is not None else state.v_scratch
        ev_on = bool(plan.event_counters)
        ev_counts = (np.ascontiguousarray(np.stack([rs.counters["ev_" + k] for k in K.EVENT_NAMES], axis=1))
                     if ev_on else np.zeros((rs.n_replicas, 1), dtype=np.int64))
        if plan.event_windows is not None:
            self._ensure_windows(plan, state)
            K.run_block_prange_windowed(
                int(seed), int(a0), int(a1), rs.replica_id.astype(np.int64), rs.instance_id.astype(np.int64),
                plan.R, plan.strategy_code, plan.layout_code, state.T_full, state.alpha, v_arr,
                state.kt_buf, state.told_buf, state.J0_of_rep, state.J_of_rep,
                schedule.kind, schedule.p[0], schedule.p[1], schedule.p[2], schedule.p[3], schedule.n_total,
                *self._kernel_args(plan), plan.spin_clause,
                rs.energies, rs.best_energy, rs.best_spins, rs.best_attempt, rs.feasible_best_energy,
                rs.feasible_best_spins, rs.feasible_best_attempt, rs.targets, rs.stopped,
                rs.counters["target_hit_attempt"],
                layout.score_slot, viol_slot, has_viol, BEST_RTOL * scale0, FEASIBLE_RTOL * scale_v,
                BEST_RTOL * scale_s, TARGET_RTOL * scale_s, UPHILL_RTOL * scale0, rs.counters["attempts"],
                rs.counters["accepted"], rs.counters["accepted_changed"], rs.counters["self_loops"],
                rs.counters["rejected"], rs.counters["uphill_accepted"], bool(stop_on_target), next_attempt,
                ev_counts, plan.is_gate_only, plan.event_windows, state.win, state.win_best, state.win_fbest)
        else:
            K.run_block_prange(
                int(seed), int(a0), int(a1), rs.replica_id.astype(np.int64), rs.instance_id.astype(np.int64),
                plan.R, plan.strategy_code, plan.layout_code, state.T_full, state.alpha, v_arr,
                state.kt_buf, state.told_buf, state.J0_of_rep, state.J_of_rep,
                schedule.kind, schedule.p[0], schedule.p[1], schedule.p[2], schedule.p[3], schedule.n_total,
                *self._kernel_args(plan), plan.spin_clause,
                rs.energies, rs.best_energy, rs.best_spins, rs.best_attempt, rs.feasible_best_energy,
                rs.feasible_best_spins, rs.feasible_best_attempt, rs.targets, rs.stopped, rs.counters["target_hit_attempt"],
                layout.score_slot, viol_slot, has_viol, BEST_RTOL * scale0, FEASIBLE_RTOL * scale_v,
                BEST_RTOL * scale_s, TARGET_RTOL * scale_s, UPHILL_RTOL * scale0, rs.counters["attempts"],
                rs.counters["accepted"], rs.counters["accepted_changed"], rs.counters["self_loops"],
                rs.counters["rejected"], rs.counters["uphill_accepted"], bool(stop_on_target), next_attempt,
                ev_on, ev_counts, plan.is_gate_only)
        if ev_on:
            for i, k in enumerate(K.EVENT_NAMES):
                rs.counters["ev_" + k][:] = ev_counts[:, i]
        rs.next_attempt[:] = next_attempt
        after = rs.counters["target_hit_attempt"]
        return (hit_before < 0) & (after >= 0)

    @staticmethod
    def _ensure_windows(plan: CPUPlan, state: CPUState):
        if getattr(state, "win", None) is None:
            n, nw = state.rs.n_replicas, plan.event_windows.size - 1
            state.win = np.zeros((n, nw, len(K.WIN_FIELDS)), dtype=np.int64)
            state.win_best = np.full((n, nw), np.nan)
            state.win_fbest = np.full((n, nw), np.nan)

    def window_counters(self, state: CPUState) -> dict:
        """Per-window counters of a plan prepared with ``event_windows``: ``edges`` int64[K+1]; one int64[n_rep, K] array per name in ``K.WIN_FIELDS``;
        ``best_energy`` / ``feasible_best`` float[n_rep, K] (NaN for a window not reached yet)."""
        plan = state.plan
        if plan.event_windows is None:
            raise ConfigError("the plan was prepared without event_windows")
        self._ensure_windows(plan, state)
        out = {"edges": plan.event_windows.copy(), "best_energy": state.win_best.copy(),
               "feasible_best": state.win_fbest.copy()}
        for i, k in enumerate(K.WIN_FIELDS):
            out[k] = state.win[:, :, i].copy()
        return out

    # -- sequential path: RunBook bookkeeping, anytime/events/trace --------------------------------
    def _run_sequential(self, plan: CPUPlan, state: CPUState, schedule: Schedule, a0: int, a1: int, B: int,
                       seed: int, spec: RecordingSpec, stop_on_target: bool) -> np.ndarray:
        rs = state.rs
        if plan.event_windows is not None:
            raise ConfigError("event_windows are implemented on the fast path only (recording 'none'/'summary')")
        book = RunBook(rs, state.layout, spec, B, a0, plan.N, plan.M, plan.R)
        ev = EventLog(rs.n_replicas, ("transactions",), (), spec.event_hist_bins) if spec.want_events else None
        hit_now = np.zeros(rs.n_replicas, dtype=bool)
        for rep in range(rs.n_replicas):
            if rs.stopped[rep]:
                continue
            nxt_cp = book.next_checkpoint(rep) if book.anytime is not None else -1
            J0 = state.J0_of_rep[rep]
            att = acc = chg = slf = rej = up = 0
            ev_row = np.zeros(len(K.EVENT_NAMES), dtype=np.int64)
            n = a0
            while n < a1:
                tau = float(schedule.tau(n))
                v_row = rs.v[rep] if rs.v is not None else state.v_scratch[rep]
                r, u, xi, v, accepted, changed, kt_n = K.attempt_kernel(
                    int(seed), int(n), int(rs.replica_id[rep]), int(rs.instance_id[rep]), plan.R,
                    plan.strategy_code, plan.layout_code, state.T_full[rep], state.alpha[rep], v_row,
                    state.kt_buf[rep], state.told_buf[rep], J0, tau,
                    *self._kernel_args(plan))
                att += 1
                hit = False
                if accepted:
                    acc += 1
                    if kt_n > 0:  # energy from the decoder toggles alone
                        K.apply_energy_update(state.J_of_rep[rep], state.kt_buf[rep], state.told_buf[rep], kt_n,
                                              rs.energies[rep])
                        if plan.event_counters:
                            K.event_counts_commit(state.kt_buf[rep], kt_n, state.T_full[rep], plan.is_obj,
                                                  plan.is_gate_only, plan.has_ctrl, plan.clause_wroute_ptr,
                                                  plan.clause_wroute, plan.ctrl_route_ptr, plan.ctrl_route,
                                                  plan.route_ctrl, plan.route_pol, plan.route_weighted, ev_row)
                    if changed:
                        chg += 1
                        if v > book.uphill_tol[rep]:
                            up += 1
                        hit = book.on_change(rep, n + 1, rs.energies[rep], rs.T[rep])
                    else:
                        slf += 1
                else:
                    rej += 1
                if ev is not None:
                    ev.counts[rep][0] += 1
                if book.trace is not None and book.trace.wants(n, rep):
                    book.trace.append((n, rep, int(rs.replica_id[rep]), r, u, tau, xi, v, accepted, changed,
                                      float(rs.energies[rep, 0]), kt_n, -1, -1),
                                     spins=rs.spins_of(rep), T=rs.T[rep],
                                     v=rs.v[rep] if rs.v is not None else None)
                if nxt_cp == n + 1:
                    nxt_cp = book.checkpoint(rep, float(rs.energies[rep, 0]))
                n += 1
                if hit and stop_on_target:
                    rs.stopped[rep] = True
                    hit_now[rep] = True
                    break
            cnt = rs.counters
            cnt["attempts"][rep] += att
            cnt["accepted"][rep] += acc
            cnt["accepted_changed"][rep] += chg
            cnt["self_loops"][rep] += slf
            cnt["rejected"][rep] += rej
            cnt["uphill_accepted"][rep] += up
            if plan.event_counters:
                for i, k in enumerate(K.EVENT_NAMES):
                    cnt["ev_" + k][rep] += ev_row[i]
            rs.next_attempt[rep] = n
        self._last_recording = RecordingData(spec=spec, anytime=book.anytime,
                                             events={"transactions": ev.counter("transactions")} if ev else None,
                                             trace=book.trace)
        return hit_now

    # ==========================================================================================
    # proposal processes other than select (kernels in _numba_process)
    # ==========================================================================================
    @staticmethod
    def _book_arrays(state: CPUState):
        rs = state.rs
        layout = state.layout
        b = rs.replica_binding.astype(np.int64)
        has_viol = layout.violation_slot is not None
        viol_slot = layout.violation_slot if has_viol else 0
        scale0 = layout.scale[b, 0]
        scale_s = layout.scale[b, layout.score_slot]
        scale_v = layout.scale[b, viol_slot] if has_viol else np.ones_like(scale0)
        tol = np.ascontiguousarray(np.stack([BEST_RTOL * scale0, FEASIBLE_RTOL * scale_v, BEST_RTOL * scale_s,
                                             TARGET_RTOL * scale_s, UPHILL_RTOL * scale0]))
        cnt = np.ascontiguousarray(np.stack([rs.counters[k] for k in PK.CNT_NAMES]).astype(np.int64))
        return tol, cnt, int(layout.score_slot), int(viol_slot), bool(has_viol)

    @staticmethod
    def _unstack_counters(rs: ReplicaStates, cnt: np.ndarray):
        for i, k in enumerate(PK.CNT_NAMES):
            rs.counters[k][:] = cnt[i]

    def _run_fast_process(self, plan: CPUPlan, state: CPUState, schedule: Schedule, a0: int, a1: int, B: int,
                          seed: int, stop_on_target: bool, scheduling: str = "static") -> np.ndarray:
        rs = state.rs
        proc = plan.process
        tol, cnt, score_slot, viol_slot, has_viol = self._book_arrays(state)
        hit_before = cnt[PK.C_HIT].copy()
        next_attempt = rs.next_attempt.copy()
        sched = (schedule.kind, schedule.p[0], schedule.p[1], schedule.p[2], schedule.p[3], schedule.n_total)
        book = (plan.spin_clause, rs.energies, rs.best_energy, rs.best_spins, rs.best_attempt,
                rs.feasible_best_energy, rs.feasible_best_spins, rs.feasible_best_attempt, rs.targets, rs.stopped,
                cnt, tol, score_slot, viol_slot, has_viol, bool(stop_on_target), next_attempt)
        if proc.colored:
            with _chunksize_scope(_scheduling_chunksize(scheduling)):
                PK.run_colored_prange(
                    int(seed), int(a0), int(a1), plan.R, plan.coloring.n_colors, rs.replica_id.astype(np.int64),
                    rs.instance_id.astype(np.int64), plan.col_order, plan.col_ptr, plan.col_class_of_pos,
                    plan.strategy_code, plan.layout_code, state.T_full, state.alpha, state.vrows, state.kt_buf,
                    state.told_buf, state.J0_of_rep, state.J_of_rep, *sched, *self._kernel_args(plan), *book)
        elif proc.stepwise:
            with _chunksize_scope(_scheduling_chunksize(scheduling)):
                PK.run_steps_prange(
                    PK.PROC_PT if proc.name == "parallel_trial" else PK.PROC_NFOLD, bool(proc.pt_exclude_self_loops),
                    proc.clock == "attempts", plan.process_impl == "stepwise_maintained", int(seed), int(a1), int(B), plan.R, rs.replica_id.astype(np.int64),
                    rs.instance_id.astype(np.int64), state.T_full, state.alpha, state.vrows, state.open_cnt, state.kt_buf,
                    state.told_buf, state.cand_buf, state.pref_buf, state.J0_of_rep, state.J_of_rep, *sched,
                    plan.route_ptr, plan.route_clause, plan.route_ctrl, plan.route_pol, plan.route_group,
                    plan.route_weighted, plan.is_obj, plan.has_ctrl, plan.clause_wroute_ptr, plan.clause_wroute,
                    plan.ctrl_route_ptr, plan.ctrl_route, plan.is_spin_clause, plan.sr_ptr, plan.sr_pos,
                    plan.blk_group_ptr, plan.blk_ctrl, plan.blk_pol, plan.blk_start, plan.blk_wend, plan.blk_end,
                    plan.plan_route_clause, *book)
        elif plan.process_impl == "shared_lockstep":
            # lock-step lanes process several replicas together in SIMD-like fashion;
            # not independent prange-over-replicas work items, so queue scheduling does not apply here --
            # always static, matching the gpu backend's equivalent exception for lockstep cohorts.
            self._run_lockstep(plan, state, sched, book, a0, a1, seed)
        else:
            v_arr = rs.v if rs.v is not None else state.v_scratch
            with _chunksize_scope(_scheduling_chunksize(scheduling)):
                PK.run_attempts_prange(
                    PK.MODE_SHARED if proc.name == "shared" else PK.MODE_SWEEP, int(seed), int(a0), int(a1),
                    rs.replica_id.astype(np.int64), rs.instance_id.astype(np.int64), state.cohort, plan.R,
                    plan.strategy_code, plan.layout_code, state.T_full, state.alpha, v_arr, state.kt_buf,
                    state.told_buf, state.perm_buf, state.perm_sweep, state.J0_of_rep, state.J_of_rep, *sched,
                    *self._kernel_args(plan), *book)
        rs.next_attempt[:] = next_attempt
        self._unstack_counters(rs, cnt)
        return (hit_before < 0) & (cnt[PK.C_HIT] >= 0)

    def _lockstep_lanes(self, plan: CPUPlan, state: CPUState):
        """Lanes of the shared lock-step path: replicas grouped by (instance, cohort, binding) - equal group
        sequences and weights - and split into blocks of ``width`` replicas, width = lockstep_width or
        max(LOCKSTEP_MIN_WIDTH, ceil(cohort / threads)) so that small cohorts still use every thread."""
        rs = state.rs
        keys = np.stack([rs.instance_id.astype(np.int64), state.cohort, rs.replica_binding.astype(np.int64)], 1)
        order = np.lexsort((np.arange(rs.n_replicas), keys[:, 2], keys[:, 1], keys[:, 0]))
        threads = int(numba.get_num_threads())
        ptr, reps, cid, iid = [0], [], [], []
        i = 0
        while i < order.size:
            j = i
            while j < order.size and np.array_equal(keys[order[j]], keys[order[i]]):
                j += 1
            size = j - i
            width = plan.lockstep_width or max(LOCKSTEP_MIN_WIDTH, -(-size // threads))
            for a in range(i, j, width):
                b = min(a + width, j)
                reps.extend(order[a:b].tolist())
                ptr.append(len(reps))
                cid.append(int(keys[order[i], 1]))
                iid.append(int(keys[order[i], 0]))
            i = j
        return (np.asarray(ptr, np.int64), np.asarray(reps, np.int64), np.asarray(cid, np.int64),
                np.asarray(iid, np.int64))

    def _run_lockstep(self, plan: CPUPlan, state: CPUState, sched, book, a0: int, a1: int, seed: int):
        lane_ptr, lane_reps, lane_cid, lane_iid = self._lockstep_lanes(plan, state)
        n_lanes = lane_ptr.size - 1
        wmax = int(np.diff(lane_ptr).max())
        T3 = np.empty((n_lanes, plan.M + 1, wmax), dtype=np.int8)
        for l in range(n_lanes):
            rr = lane_reps[lane_ptr[l]:lane_ptr[l + 1]]
            T3[l, :, : rr.size] = state.T_full[rr].T
            T3[l, :, rr.size:] = 1
        kt = np.zeros((n_lanes, max(plan.max_group_routes, 1)), dtype=np.intp)
        told = np.zeros((n_lanes, max(plan.max_group_routes, 1)), dtype=np.int8)
        vbuf = np.zeros((n_lanes, wmax), dtype=np.float64)
        albuf = np.zeros((n_lanes, wmax), dtype=np.int64)
        rs = state.rs
        PK.run_shared_lockstep(int(seed), int(a0), int(a1), plan.R, lane_ptr, lane_reps, lane_cid, lane_iid, T3,
                               vbuf, albuf, kt, told, rs.replica_id.astype(np.int64), state.J0_of_rep,
                               state.J_of_rep, *sched, plan.blk_group_ptr, plan.blk_ctrl, plan.blk_pol,
                               plan.blk_start, plan.blk_wend, plan.blk_end, plan.plan_route_clause,
                               plan.is_spin_clause, *book)
        for l in range(n_lanes):
            rr = lane_reps[lane_ptr[l]:lane_ptr[l + 1]]
            state.T_full[rr] = T3[l, :, : rr.size].T

    def _one_attempt_process(self, plan: CPUPlan, state: CPUState, rep: int, n: int, tau: float, seed: int):
        """One shared/sweep attempt of one replica (sequential path and ``step``): (r, u, xi, v, acc, chg, kt_n)."""
        rs = state.rs
        mode = PK.MODE_SHARED if plan.process.name == "shared" else PK.MODE_SWEEP
        r, u = PK.process_draw(mode, int(seed), int(n), int(rs.replica_id[rep]), int(rs.instance_id[rep]),
                               int(state.cohort[rep]), plan.R, state.perm_buf[rep], state.perm_sweep, rep)
        v_row = rs.v[rep] if rs.v is not None else state.v_scratch[rep]
        xi, v, accepted, changed, kt_n = K.attempt_given(
            int(r), u, plan.strategy_code, plan.layout_code, state.T_full[rep], state.alpha[rep], v_row,
            state.kt_buf[rep], state.told_buf[rep], state.J0_of_rep[rep], tau, *self._kernel_args(plan))
        return int(r), u, xi, v, accepted, changed, kt_n

    def _decide_stepwise(self, plan: CPUPlan, state: CPUState, rep: int, k: int, n: int, tau: float, seed: int,
                         B: int):
        """(outcome, r, W, n_eval) of one parallel_trial / nfold step (no state change)."""
        rs = state.rs
        proc = plan.process
        rid, iid = int(rs.replica_id[rep]), int(rs.instance_id[rep])
        if plan.process_impl == "stepwise_recompute":
            PK.refresh_inputs(plan.R, state.T_full[rep], state.J0_of_rep[rep], state.vrows[rep], state.open_cnt[rep],
                              bool(proc.pt_exclude_self_loops) or proc.name != "parallel_trial", plan.sr_ptr,
                              plan.sr_pos, plan.route_ctrl, plan.route_pol, plan.blk_group_ptr, plan.blk_ctrl,
                              plan.blk_pol, plan.blk_start, plan.blk_wend, plan.plan_route_clause)
        if proc.name == "parallel_trial":
            r, n_eval, _ = PK.pt_decide(int(seed), int(n), rid, iid, plan.R, tau, state.vrows[rep], state.open_cnt[rep],
                                        bool(proc.pt_exclude_self_loops), state.cand_buf[rep])
            return (PK.OUT_COMMIT if r >= 0 else PK.OUT_NOFIRE), int(r), 1, int(n_eval)
        out, r, W, n_eval = PK.nfold_decide(int(seed), int(k), int(n), rid, iid, plan.R, tau, state.vrows[rep],
                                            state.open_cnt[rep], state.cand_buf[rep], state.pref_buf[rep], int(B),
                                            proc.clock == "attempts")
        return int(out), int(r), int(W), int(n_eval)

    def _commit_stepwise(self, plan: CPUPlan, state: CPUState, rep: int, r: int):
        rs = state.rs
        if plan.process_impl == "stepwise_recompute":
            changed, kt_n = PK.commit_toggle_blocked(r, state.T_full[rep], state.kt_buf[rep], state.told_buf[rep],
                                                     plan.blk_group_ptr, plan.blk_ctrl, plan.blk_pol, plan.blk_start,
                                                     plan.blk_end, plan.plan_route_clause, plan.is_spin_clause)
            if kt_n > 0:
                K.apply_energy_update(state.J_of_rep[rep], state.kt_buf[rep], state.told_buf[rep], kt_n,
                                      rs.energies[rep])
            return bool(changed), int(kt_n)
        changed, kt_n = PK.commit_stepwise(
            r, state.T_full[rep], state.alpha[rep], rs.v[rep], state.open_cnt[rep], state.kt_buf[rep],
            state.told_buf[rep], state.J0_of_rep[rep], plan.route_ptr, plan.route_clause, plan.route_ctrl,
            plan.route_pol, plan.route_group, plan.route_weighted, plan.is_obj, plan.has_ctrl,
            plan.clause_wroute_ptr, plan.clause_wroute, plan.ctrl_route_ptr, plan.ctrl_route, plan.is_spin_clause)
        if kt_n > 0:
            K.apply_energy_update(state.J_of_rep[rep], state.kt_buf[rep], state.told_buf[rep], kt_n,
                                  rs.energies[rep])
        return bool(changed), int(kt_n)

    def _run_sequential_process(self, plan: CPUPlan, state: CPUState, schedule: Schedule, a0: int, a1: int,
                                B: int, seed: int, spec: RecordingSpec, stop_on_target: bool) -> np.ndarray:
        """Recording path (anytime / events / trace) of the non-select processes: the same numeric kernels
        one attempt/step at a time, bookkeeping by ``base.RunBook``."""
        rs = state.rs
        proc = plan.process
        if proc.colored:
            return self._run_sequential_colored(plan, state, schedule, a0, a1, B, seed, spec, stop_on_target)
        book = RunBook(rs, state.layout, spec, B, a0, plan.N, plan.M, plan.R)
        ev = EventLog(rs.n_replicas, ("transactions",), (), spec.event_hist_bins) if spec.want_events else None
        hit_now = np.zeros(rs.n_replicas, dtype=bool)
        c = rs.counters
        for rep in range(rs.n_replicas):
            if rs.stopped[rep]:
                continue
            nxt_cp = book.next_checkpoint(rep) if book.anytime is not None else -1
            rid = int(rs.replica_id[rep])
            n = int(rs.next_attempt[rep])
            k = int(c["steps"][rep])
            while n < a1:
                tau = float(schedule.tau(n))
                E0 = float(rs.energies[rep, 0])
                if proc.stepwise:
                    out, r, W, n_eval = self._decide_stepwise(plan, state, rep, k, n, tau, seed, B)
                    u = xi = float("nan")
                else:
                    out, W, n_eval = PK.OUT_COMMIT, 1, 1
                k += 1
                c["steps"][rep] += 1
                c["candidate_evaluations"][rep] += n_eval
                if out in (PK.OUT_FROZEN, PK.OUT_OVERSHOOT):
                    while nxt_cp != -1 and nxt_cp <= B:
                        nxt_cp = book.checkpoint(rep, E0)
                    c["attempts"][rep] += B - n
                    c["rejected"][rep] += B - n
                    if out == PK.OUT_FROZEN:
                        c["frozen_attempt"][rep] = n
                    n = B
                    break
                while nxt_cp != -1 and nxt_cp < n + W:
                    nxt_cp = book.checkpoint(rep, E0)
                c["attempts"][rep] += W
                if proc.stepwise:
                    accepted = out == PK.OUT_COMMIT
                    v = float(state.vrows[rep, r]) if accepted else float("nan")
                    changed, kt_n = self._commit_stepwise(plan, state, rep, r) if accepted else (False, 0)
                else:
                    r, u, xi, v, accepted, changed, kt_n = self._one_attempt_process(plan, state, rep, n, tau, seed)
                    if accepted and kt_n > 0:
                        K.apply_energy_update(state.J_of_rep[rep], state.kt_buf[rep], state.told_buf[rep], kt_n,
                                              rs.energies[rep])
                hit = False
                if accepted:
                    c["accepted"][rep] += 1
                    c["rejected"][rep] += W - 1
                    if changed:
                        c["accepted_changed"][rep] += 1
                        if v > book.uphill_tol[rep]:
                            c["uphill_accepted"][rep] += 1
                        hit = book.on_change(rep, n + W, rs.energies[rep], rs.T[rep])
                    else:
                        c["self_loops"][rep] += 1
                else:
                    c["rejected"][rep] += 1
                if ev is not None:
                    ev.counts[rep][0] += 1
                if book.trace is not None and book.trace.wants(n, rep):
                    book.trace.append((n, rep, rid, r, u, tau, xi, v, accepted, changed, float(rs.energies[rep, 0]),
                                       kt_n, -1, -1), spins=rs.spins_of(rep), T=rs.T[rep],
                                      v=rs.v[rep] if rs.v is not None else None)
                n += W
                if nxt_cp == n:
                    nxt_cp = book.checkpoint(rep, float(rs.energies[rep, 0]))
                if hit and stop_on_target:
                    rs.stopped[rep] = True
                    hit_now[rep] = True
                    break
            rs.next_attempt[rep] = n
        self._last_recording = RecordingData(spec=spec, anytime=book.anytime,
                                             events={"transactions": ev.counter("transactions")} if ev else None,
                                             trace=book.trace)
        return hit_now

    def _run_sequential_colored(self, plan: CPUPlan, state: CPUState, schedule: Schedule, a0: int, a1: int,
                                B: int, seed: int, spec: RecordingSpec, stop_on_target: bool) -> np.ndarray:
        """Recording path (anytime / events) of ``colored``: the same ``color_segment`` kernel per class,
        bookkeeping by ``base.RunBook`` once per class. Checkpoints inside a class see the pre-class state."""
        if spec.want_trace:
            raise ConfigError("trace recording is not implemented for the colored process on the cpu backend "
                              "(use the emulation)")
        rs = state.rs
        book = RunBook(rs, state.layout, spec, B, a0, plan.N, plan.M, plan.R)
        ev = EventLog(rs.n_replicas, ("transactions",), (), spec.event_hist_bins) if spec.want_events else None
        hit_now = np.zeros(rs.n_replicas, dtype=bool)
        c = rs.counters
        sched = (schedule.kind, schedule.p[0], schedule.p[1], schedule.p[2], schedule.p[3], schedule.n_total)
        kargs = self._kernel_args(plan)
        C = plan.coloring.n_colors
        for rep in range(rs.n_replicas):
            if rs.stopped[rep]:
                continue
            nxt_cp = book.next_checkpoint(rep) if book.anytime is not None else -1
            a = int(rs.next_attempt[rep])
            while a < a1:
                E0 = float(rs.energies[rep, 0])
                n_do, acc_n, chg_n, slf_n, up_n, st_n = PK.color_segment(
                    a, a1, plan.R, C, int(seed), int(rs.replica_id[rep]), int(rs.instance_id[rep]), plan.col_order,
                    plan.col_ptr, plan.col_class_of_pos, plan.strategy_code, plan.layout_code, state.T_full[rep],
                    state.alpha[rep], state.vrows[rep], state.kt_buf[rep], state.told_buf[rep],
                    state.J0_of_rep[rep], state.J_of_rep[rep], rs.energies[rep], *sched, *kargs,
                    float(book.uphill_tol[rep]))
                a_new = a + n_do
                c["attempts"][rep] += n_do
                c["steps"][rep] += st_n
                c["candidate_evaluations"][rep] += n_do
                c["accepted"][rep] += acc_n
                c["rejected"][rep] += n_do - acc_n
                c["accepted_changed"][rep] += chg_n
                c["self_loops"][rep] += slf_n
                c["uphill_accepted"][rep] += up_n
                while nxt_cp != -1 and nxt_cp < a_new:
                    nxt_cp = book.checkpoint(rep, E0)
                hit = book.on_change(rep, a_new, rs.energies[rep], rs.T[rep]) if chg_n else False
                if ev is not None:
                    ev.counts[rep][0] += n_do
                if nxt_cp == a_new:
                    nxt_cp = book.checkpoint(rep, float(rs.energies[rep, 0]))
                a = a_new
                rs.next_attempt[rep] = a
                if hit and stop_on_target:
                    rs.stopped[rep] = True
                    hit_now[rep] = True
                    break
        self._last_recording = RecordingData(spec=spec, anytime=book.anytime,
                                             events={"transactions": ev.counter("transactions")} if ev else None,
                                             trace=None)
        return hit_now

    # ------------------------------------------------------------------------------------------
    def step(self, state: CPUState, n: int, schedule: Schedule, seed: int) -> StepRecord:
        """Clock index n for every active replica (lock-step API; ``tests/test_replay_core.py``).
        select/shared/sweep: attempt n; parallel_trial: step n; nfold has no lock-step form (use run)."""
        plan, rs = state.plan, state.rs
        n = int(n)
        if plan.process.name == "nfold":
            raise ConfigError("nfold replicas advance by different numbers of attempts; use run() (chunks)")
        if plan.process.colored:
            raise ConfigError("colored commits a whole color class per step; use run() (chunks)")
        if not plan.process.is_select:
            return self._step_process(state, n, schedule, seed)
        tau = float(schedule.tau(n))
        rec = StepRecord.empty(n, tau, rs.n_replicas)
        book = RunBook(rs, state.layout, RecordingSpec(profile="none"), int(schedule.n_total), n, plan.N, plan.M,
                      plan.R)
        c = rs.counters
        for rep in range(rs.n_replicas):
            if rs.stopped[rep]:
                continue
            if int(rs.next_attempt[rep]) != n:
                raise ConfigError(f"replica {rep}: next attempt is {int(rs.next_attempt[rep])}, not {n}")
            v_row = rs.v[rep] if rs.v is not None else state.v_scratch[rep]
            r, u, xi, v, accepted, changed, kt_n = K.attempt_kernel(
                int(seed), n, int(rs.replica_id[rep]), int(rs.instance_id[rep]), plan.R, plan.strategy_code,
                plan.layout_code, state.T_full[rep], state.alpha[rep], v_row, state.kt_buf[rep],
                state.told_buf[rep], state.J0_of_rep[rep], tau, *self._kernel_args(plan))
            c["attempts"][rep] += 1
            if accepted:
                c["accepted"][rep] += 1
                if kt_n > 0:  # energy from the decoder toggles alone
                    K.apply_energy_update(state.J_of_rep[rep], state.kt_buf[rep], state.told_buf[rep], kt_n,
                                          rs.energies[rep])
                if changed:
                    c["accepted_changed"][rep] += 1
                    c["uphill_accepted"][rep] += v > book.uphill_tol[rep]
                    book.on_change(rep, n + 1, rs.energies[rep], rs.T[rep])
                else:
                    c["self_loops"][rep] += 1
            else:
                c["rejected"][rep] += 1
            rs.next_attempt[rep] = n + 1
            c["steps"][rep] += 1
            c["candidate_evaluations"][rep] += 1
            rec.group[rep], rec.u[rep], rec.xi[rep], rec.v[rep] = r, u, xi, v
            rec.accepted[rep], rec.changed[rep], rec.active[rep] = accepted, changed, True
            rec.n_toggles[rep] = kt_n
        return rec

    def _step_process(self, state: CPUState, n: int, schedule: Schedule, seed: int) -> StepRecord:
        plan, rs = state.plan, state.rs
        self._set_threads(plan)
        tau = float(schedule.tau(n))
        rec = StepRecord.empty(n, tau, rs.n_replicas)
        book = RunBook(rs, state.layout, RecordingSpec(profile="none"), int(schedule.n_total), n, plan.N, plan.M,
                       plan.R)
        c = rs.counters
        for rep in range(rs.n_replicas):
            if rs.stopped[rep]:
                continue
            if int(rs.next_attempt[rep]) != n:
                raise ConfigError(f"replica {rep}: next attempt is {int(rs.next_attempt[rep])}, not {n}")
            if plan.process.stepwise:
                out, r, _, n_eval = self._decide_stepwise(plan, state, rep, int(c["steps"][rep]), n, tau, seed,
                                                          int(schedule.n_total))
                accepted = out == PK.OUT_COMMIT
                v = float(state.vrows[rep, r]) if accepted else float("nan")
                u = xi = float("nan")
                changed, kt_n = self._commit_stepwise(plan, state, rep, r) if accepted else (False, 0)
            else:
                n_eval = 1
                r, u, xi, v, accepted, changed, kt_n = self._one_attempt_process(plan, state, rep, n, tau, seed)
                if accepted and kt_n > 0:
                    K.apply_energy_update(state.J_of_rep[rep], state.kt_buf[rep], state.told_buf[rep], kt_n,
                                          rs.energies[rep])
            c["attempts"][rep] += 1
            c["steps"][rep] += 1
            c["candidate_evaluations"][rep] += n_eval
            if accepted:
                c["accepted"][rep] += 1
                if changed:
                    c["accepted_changed"][rep] += 1
                    c["uphill_accepted"][rep] += v > book.uphill_tol[rep]
                    book.on_change(rep, n + 1, rs.energies[rep], rs.T[rep])
                else:
                    c["self_loops"][rep] += 1
            else:
                c["rejected"][rep] += 1
            rs.next_attempt[rep] = n + 1
            rec.group[rep], rec.u[rep], rec.xi[rep], rec.v[rep] = r, u, xi, v
            rec.accepted[rep], rec.changed[rep], rec.active[rep] = accepted, changed, True
            rec.n_toggles[rep] = kt_n
        return rec

    # ------------------------------------------------------------------------------------------
    def export_state(self, state: CPUState) -> ReplicaStates:
        rs = state.rs.copy()
        rs.bindings = state.bindings
        rs.extras = {"alpha": state.alpha.copy()}
        return rs

    def load_state(self, plan: CPUPlan, states: ReplicaStates, bindings=None) -> CPUState:
        model = plan.model
        bindings = coerce_bindings(model, bindings if bindings is not None else states.bindings)
        states.check_compatible(model, bindings)
        missing = [k for k in COUNTER_NAMES if k not in states.counters]
        if missing:
            raise IncompatibleCheckpoint(f"snapshot lacks counters {missing}")
        rs = states.copy()
        if not np.array_equal(clause_signs(model, rs.spins), rs.T):
            raise IncompatibleCheckpoint("snapshot clause state is inconsistent with its singleton readout")
        ensure_process_counters(rs)
        if plan.event_counters:
            ensure_event_counters(rs)
        # switches are a function of the stored clause signs; rebuild rather than trust a snapshot
        alpha = route_switches(model, rs.T)
        if plan.strategy != "recompute" and rs.v is None:
            J0rows = np.stack([bindings[int(rs.replica_binding[i])].J[0] for i in range(rs.n_replicas)])
            rs.v = encoder_inputs(model, J0rows, rs.T)
        elif plan.process_impl == "stepwise_maintained" and (rs.v.shape != (rs.n_replicas, plan.R)
                                                             or np.isnan(rs.v).any()):
            # maintained parallel_trial/nfold inputs must hold every group (a lazy/batched snapshot may not)
            J0rows = np.stack([bindings[int(rs.replica_binding[i])].J[0] for i in range(rs.n_replicas)])
            rs.v = encoder_inputs(model, J0rows, rs.T)
        rs.backend = self.name
        rs.extras = {}
        st = CPUState(plan, rs, bindings, alpha)  # rebuilds the dummy-extended T_full from rs.T
        rs.bindings = st.bindings
        return st

    def evaluate_inputs(self, state: CPUState, groups=None) -> np.ndarray:
        """v_r freshly recomputed from the current clause signs (no cache, no commit)."""
        model = state.plan.model
        gs = list(range(state.plan.R)) if groups is None else [int(g) for g in groups]
        out = np.empty((state.n_replicas, len(gs)))
        if not gs:
            return out
        T0 = state.rs.T
        for rep in range(state.n_replicas):
            b = int(state.rs.replica_binding[rep])
            J0 = state.bindings[b].J[0][None, :]
            out[rep] = encoder_inputs(model, J0, T0[rep:rep + 1], groups=gs)[0]
        return out

    # ------------------------------------------------------------------------------------------
    def validate_state(self, state: CPUState):
        """Audit helper (corruption detection): clause state vs its singleton readout, and (incremental/lazy) v vs a fresh
        encoder sum, for every replica. Raises ArchitectureValidationError on the first mismatch."""
        plan, rs = state.plan, state.rs
        Td = clause_signs(plan.model, rs.spins)
        bad = np.flatnonzero(np.any(Td != rs.T, axis=1))
        if bad.size:
            rep = int(bad[0])
            raise ArchitectureValidationError(
                "T_k == prod_i s_i^H_ki violated: stored clause signs differ from the product of their singleton readouts",
                {"identity": "T_k == prod_i s_i^H_ki", "replica": rep})
        if not np.all(state.T_full[:, plan.M] == 1):
            raise ArchitectureValidationError("the dummy controller slot T[M] drifted away from +1", {})
        if rs.v is not None:
            for rep in range(rs.n_replicas):
                held = ~np.isnan(rs.v[rep])
                if not held.any():
                    continue
                b = int(rs.replica_binding[rep])
                J0 = state.bindings[b].J[0][None, :]
                fresh = encoder_inputs(plan.model, J0, Td[rep:rep + 1], groups=np.flatnonzero(held).tolist())[0]
                scale = float(state.layout.scale[b, 0])
                dev = np.abs(rs.v[rep, held] - fresh)
                if np.any(dev > 1e-9 * scale):
                    raise ArchitectureValidationError(
                        "2 v_r == E(mu_r(s)) - E(s) violated: maintained input disagrees with a fresh encoder sum",
                        {"identity": "v_r == encoder_inputs", "replica": rep,
                         "max_deviation": float(dev.max()), "tolerance": 1e-9 * scale})

    def describe(self) -> dict:
        return {"backend": self.name, "role": "optimized CPU architecture backend (Numba)",
               "numba": numba.__version__, "cpu": platform.processor() or platform.machine(),
               **host_versions()}


register_backend("cpu", CPUBackend, replace=True)

"""Backend API, registry, replica-state container and helpers shared by every backend.

Manuscript quantities handled here:

* stored clause signs T_k = prod_i s_i^{H_ki} (Methods, Derivation of the transition-programmable architecture), computed from the initial spins
  through the clause CSR (:func:`clause_signs`) at initialization only. After that the clause state
  is authoritative: every spin is a stored singleton clause updated by the decoder like every other
  clause, and spins are read out from those singletons (``ReplicaStates.spins``);
* switches alpha of every retained connection (Methods, Derivation of the transition-programmable architecture), read from the controlling
  stored sign (:func:`route_switches`);
* transition-neuron inputs v_r = sum over weighted routes J_k alpha T_k
  (Supplementary Information, Exact local maintenance of move-neuron inputs, :func:`encoder_inputs`);
* slot energies E[slot] = E_const - sum_k J[slot, k] T_k (Methods, Derivation of the transition-programmable architecture, :func:`slot_energies`).

Conventions fixed by this module (documented for all backends):

* Attempt counts. ``best_attempt``, ``feasible_best_attempt`` and ``target_hit_attempt`` store the
  number of attempts executed when the state was first reached: ``n + 1`` for the transaction of
  the 0-based attempt index n, ``0`` for the initial state, ``-1`` = never (targets / no feasible
  state yet). Anytime checkpoints use the same attempt counts.
* Best tracking. ``best_*`` tracks slot 0 ("run"). ``feasible_best_*`` tracks the *score slot*
  (``"objective"`` if the binding has one, else slot 0) over feasible states: states with
  |E["violation"]| <= 1e-9 * scale if a ``"violation"`` slot exists (penalty runs), every state
  otherwise. Improvements must exceed 1e-12 * scale (so rounding-level drift of an incremental
  energy does not move the recorded attempt). scale = max(1, sum_k |J[slot,k]| + |const[slot]|).
* Targets. A per-replica target on the score slot; hit iff the state is feasible and
  E[score] <= target + 1e-9 * scale. ``stop_on_target`` stops that replica at its hitting attempt.
* Uphill. ``uphill_accepted`` counts accepted transitions that changed the state with
  v_r > 1e-12 * scale (slot 0): a self-loop is never uphill, and rounding-level residues of a
  maintained input are not uphill moves.
"""

from __future__ import annotations

import abc
import importlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from nimbus.core.errors import (
    BackendUnavailable,
    ConfigError,
    IncompatibleCheckpoint,
    InfeasibleState,
    NimbusError,
)
from nimbus.core.model import CompiledModel, WeightBinding
from nimbus.experiments.recording import AnytimeLog, RecordingSpec, TraceLog

__all__ = [
    "STATE_SCHEMA", "COUNTER_NAMES", "UPHILL_RTOL", "ArchitectureValidationError", "ReplicaStates", "BackendPlan", "StepRecord",
    "RunResult", "ChunkResult", "Backend", "register_backend", "get_backend", "available_backends",
    "SlotLayout", "coerce_bindings", "coerce_replicas", "clause_signs", "route_switches", "encoder_inputs",
    "slot_energies", "check_initial_feasibility", "initial_replica_states", "apply_targets", "resolve_window",
    "metropolis_threshold", "metropolis_accept", "csr_gather", "csr_rows", "RunBook", "check_host_device", "host_versions",
    "PROCESSES", "PROCESS_COUNTER_NAMES", "ProcessSpec", "ensure_process_counters", "EVENT_COUNTER_NAMES",
    "ensure_event_counters", "singleton_routes",
    "cohort_ids", "termination_labels", "check_process_budget",
]

STATE_SCHEMA = "nimbus.replica_states/2"  # /2: clause state authoritative, spins read out
COUNTER_NAMES = ("attempts", "accepted", "accepted_changed", "self_loops", "rejected", "uphill_accepted",
                 "target_hit_attempt")
#: Counters added with the proposal processes. Kept out of ``COUNTER_NAMES`` so that
#: older snapshots and backends that do not maintain them stay compatible: ``steps`` = parallel rounds
#: (= attempts for the attempt-clock processes), ``candidate_evaluations`` = acceptance tests/weights
#: evaluated (1 per attempt for select/shared/sweep), ``frozen_attempt`` = attempt count at which an
#: ``nfold`` replica found no allowed move (-1 = never).
PROCESS_COUNTER_NAMES = ("steps", "candidate_evaluations", "frozen_attempt")
#: Proposal processes. ``select`` is the reference process.
PROCESSES = ("select", "shared", "parallel_trial", "nfold", "sweep", "colored")

BEST_RTOL = 1e-12
UPHILL_RTOL = 1e-12
FEASIBLE_RTOL = 1e-9
TARGET_RTOL = 1e-9


class ArchitectureValidationError(NimbusError, AssertionError):
    """A settled architecture state violates Methods, Derivation of the transition-programmable architecture (or a derived invariant).

    ``context`` carries the attempt, replica, group, the failing identity and the offending values.
    """

    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = dict(context or {})


# ============================================================================================
# replica states (portable numpy snapshot)
# ============================================================================================
_STATE_ARRAYS = ("replica_binding", "replica_id", "instance_id", "T", "energies", "best_energy",
                 "best_spins", "best_attempt", "feasible_best_energy", "feasible_best_spins",
                 "feasible_best_attempt", "next_attempt", "targets", "stopped")


@dataclass
class ReplicaStates:
    """Exported per-replica state. All arrays are numpy, first axis = replica.

    T int8[n_rep, M~] is the authoritative state. ``spins`` is a read-only readout
    T[:, spin_clause] of the singleton clauses (spin_clause int32[N] = the model's
    readout_clause_of_spin); no backend holds a separately updated spin array.
    v float64[n_rep, R] or None (cached/maintained inputs;
    NaN for groups whose input a backend does not hold, e.g. groups outside the loaded batch);
    energies float64[n_rep, n_slots]; best_energy/best_spins/best_attempt (slot 0);
    feasible_best_energy/feasible_best_spins/feasible_best_attempt (score slot over feasible states);
    counters {name: int64[n_rep]} with ``COUNTER_NAMES``; next_attempt int64[n_rep] (index n of the next
    attempt); targets float64[n_rep] (NaN = none); stopped bool[n_rep] (stopped on target);
    extras: backend caches needed for exact continuation (e.g. emulator switches); bindings: the
    in-memory WeightBinding objects (not persisted; ``weight_hashes`` are).
    """

    backend: str
    topology_hash: str
    weight_hashes: tuple
    slot_names: tuple
    replica_binding: np.ndarray
    replica_id: np.ndarray
    instance_id: np.ndarray
    spin_clause: np.ndarray
    T: np.ndarray
    v: np.ndarray | None
    energies: np.ndarray
    best_energy: np.ndarray
    best_spins: np.ndarray
    best_attempt: np.ndarray
    feasible_best_energy: np.ndarray
    feasible_best_spins: np.ndarray
    feasible_best_attempt: np.ndarray
    counters: dict
    next_attempt: np.ndarray
    targets: np.ndarray
    stopped: np.ndarray
    extras: dict = field(default_factory=dict)
    bindings: tuple = ()
    schema_version: str = STATE_SCHEMA

    @property
    def n_replicas(self) -> int:
        return int(self.T.shape[0])

    @property
    def spins(self) -> np.ndarray:
        """Read-only spin readout from the singleton clauses, int8[n_rep, N] (a fresh array)."""
        s = self.T[:, self.spin_clause.astype(np.int64)].astype(np.int8)
        s.flags.writeable = False
        return s

    def spins_of(self, rep: int) -> np.ndarray:
        """Spin readout of one replica from its singleton clauses (a fresh int8[N] array)."""
        return self.T[rep, self.spin_clause.astype(np.int64)].astype(np.int8)

    def copy(self) -> "ReplicaStates":
        def c(x):
            return None if x is None else np.array(x, copy=True)

        return ReplicaStates(
            backend=self.backend, topology_hash=self.topology_hash, weight_hashes=tuple(self.weight_hashes),
            slot_names=tuple(self.slot_names), v=c(self.v), counters={k: c(v) for k, v in self.counters.items()},
            extras={k: c(v) for k, v in self.extras.items()}, bindings=tuple(self.bindings),
            spin_clause=c(self.spin_clause), schema_version=self.schema_version,
            **{n: c(getattr(self, n)) for n in _STATE_ARRAYS})

    def counter(self, name: str) -> np.ndarray:
        return self.counters[name]

    # -- persistence ---------------------------------------------------------------------------
    def to_arrays(self) -> dict:
        """Flat dict of numpy arrays (``np.savez``-ready, no pickling); bindings are not included."""
        out = {"state." + n: np.asarray(getattr(self, n)) for n in _STATE_ARRAYS}
        out["state.spin_clause"] = np.asarray(self.spin_clause)
        if self.v is not None:
            out["state.v"] = np.asarray(self.v)
        for k, v in self.counters.items():
            out["counter." + k] = np.asarray(v)
        for k, v in self.extras.items():
            out["extra." + k] = np.asarray(v)
        header = {"schema_version": self.schema_version, "backend": self.backend,
                  "topology_hash": self.topology_hash, "weight_hashes": list(self.weight_hashes),
                  "slot_names": list(self.slot_names), "counters": list(self.counters.keys()),
                  "extras": list(self.extras.keys()), "has_v": self.v is not None}
        out["state.header_json"] = np.asarray(json.dumps(header))
        return out

    @classmethod
    def from_arrays(cls, a: Mapping) -> "ReplicaStates":
        try:
            header = json.loads(str(np.asarray(a["state.header_json"])))
        except (KeyError, ValueError) as exc:
            raise IncompatibleCheckpoint(f"not a replica-state snapshot ({exc})") from exc
        if header.get("schema_version") != STATE_SCHEMA:
            raise IncompatibleCheckpoint(f"unsupported replica-state schema {header.get('schema_version')!r}")
        kw = {n: np.array(a["state." + n], copy=True) for n in _STATE_ARRAYS}
        return cls(backend=header["backend"], topology_hash=header["topology_hash"],
                   weight_hashes=tuple(header["weight_hashes"]), slot_names=tuple(header["slot_names"]),
                   v=np.array(a["state.v"], copy=True) if header["has_v"] else None,
                   counters={k: np.array(a["counter." + k], copy=True) for k in header["counters"]},
                   extras={k: np.array(a["extra." + k], copy=True) for k in header["extras"]},
                   spin_clause=np.array(a["state.spin_clause"], copy=True), **kw)

    def save(self, path) -> str:
        p = os.fspath(path)
        if not p.endswith(".npz"):
            p += ".npz"
        np.savez_compressed(p, **self.to_arrays())
        return p

    @classmethod
    def load(cls, path) -> "ReplicaStates":
        p = os.fspath(path)
        if not p.endswith(".npz"):
            p += ".npz"
        with np.load(p, allow_pickle=False) as z:
            return cls.from_arrays({k: z[k] for k in z.files})

    def check_compatible(self, model: CompiledModel, bindings: Sequence[WeightBinding]):
        """Raise IncompatibleCheckpoint unless topology/weights/shapes match."""
        if self.topology_hash != model.topology_hash:
            raise IncompatibleCheckpoint("replica states were made for a different model topology")
        hashes = tuple(b.weight_hash for b in bindings)
        if hashes != tuple(self.weight_hashes):
            raise IncompatibleCheckpoint("replica states were made with different weight bindings")
        if self.spin_clause.shape != (model.n_spins,) or self.T.shape[1] != model.n_clauses:
            raise IncompatibleCheckpoint("replica-state array shapes do not match the model")
        if not np.array_equal(self.spin_clause, model.readout_clause_of_spin):
            raise IncompatibleCheckpoint("replica-state singleton map differs from the model's")
        if tuple(bindings[0].slot_names) != tuple(self.slot_names):
            raise IncompatibleCheckpoint("slot names differ from the bindings")


# ============================================================================================
# proposal processes
# ============================================================================================
_PROCESS_OPTIONS = {"select": (), "shared": ("cohort_size",), "sweep": ("cohort_size",),
                    "parallel_trial": ("pt_exclude_self_loops",), "nfold": ("nfold_clock",), "colored": ()}


@dataclass(frozen=True)
class ProcessSpec:
    """A validated proposal process and its options.

    ``cohort_size`` (shared, sweep): replicas with equal ``replica_id // cohort_size`` (and equal
    instance id) share the group sequence / permutations; ``None`` = one cohort per instance for
    ``shared`` and one cohort per replica (independent permutations) for ``sweep``.
    ``pt_exclude_self_loops`` (parallel_trial, default True): only state-changing groups compete.
    ``nfold_clock`` (nfold): ``"attempts"`` (geometric attempt-equivalent clock, default) or ``"steps"``.
    """

    name: str = "select"
    cohort_size: int | None = None
    pt_exclude_self_loops: bool = True
    nfold_clock: str = "attempts"

    @classmethod
    def coerce(cls, process="select", options=None) -> "ProcessSpec":
        if isinstance(process, ProcessSpec):
            if options:
                raise ConfigError("give process options inside the ProcessSpec")
            return process
        name = "select" if process is None else str(process)
        if name not in PROCESSES:
            raise ConfigError(f"unknown proposal process {name!r}; known: {PROCESSES}")
        opts = dict(options or {})
        bad = sorted(set(opts) - set(_PROCESS_OPTIONS[name]))
        if bad:
            raise ConfigError(f"process {name!r} does not take options {bad} (allowed: "
                              f"{list(_PROCESS_OPTIONS[name])})")
        cs = opts.get("cohort_size")
        if cs is not None:
            if isinstance(cs, bool) or not isinstance(cs, (int, np.integer)) or int(cs) < 1:
                raise ConfigError(f"cohort_size must be a positive integer or null, got {cs!r}")
            cs = int(cs)
        ex = opts.get("pt_exclude_self_loops", True)
        if not isinstance(ex, (bool, np.bool_)):
            raise ConfigError(f"pt_exclude_self_loops must be true/false, got {ex!r}")
        clock = opts.get("nfold_clock", "attempts")
        if clock not in ("attempts", "steps"):
            raise ConfigError(f"nfold_clock must be 'attempts' or 'steps', got {clock!r}")
        return cls(name, cs, bool(ex), str(clock))

    @property
    def is_select(self) -> bool:
        return self.name == "select"

    @property
    def clock(self) -> str:
        """Units of the run budget B, the schedule index and the ``attempts`` counter."""
        if self.name == "parallel_trial" or (self.name == "nfold" and self.nfold_clock == "steps"):
            return "steps"
        return "attempts"

    @property
    def stepwise(self) -> bool:
        """parallel_trial / nfold: every input settled every step (full library, maintained inputs)."""
        return self.name in ("parallel_trial", "nfold")

    @property
    def colored(self) -> bool:
        """colored: one color class per step, every group of the class tested against its own uniform."""
        return self.name == "colored"

    @property
    def full_library(self) -> bool:
        """Needs every group resident (no capacity batching): parallel_trial, nfold and colored."""
        return self.stepwise or self.colored

    @property
    def ragged(self) -> bool:
        """Replicas may reach different clock values at a chunk end (nfold: geometric advance, freezing)."""
        return self.name == "nfold"

    def self_loop_rule(self) -> str:
        if self.name == "parallel_trial":
            return "excluded" if self.pt_exclude_self_loops else "included"
        if self.name == "nfold":
            return "excluded"
        return "attempt"  # select/shared/sweep/colored: a self-loop is an accepted attempt that changes nothing

    def options(self) -> dict:
        """Non-default options (as given in a config's ``annealing.process_options``)."""
        out = {}
        if self.cohort_size is not None:
            out["cohort_size"] = self.cohort_size
        if self.name == "parallel_trial" and not self.pt_exclude_self_loops:
            out["pt_exclude_self_loops"] = False
        if self.name == "nfold" and self.nfold_clock != "attempts":
            out["nfold_clock"] = self.nfold_clock
        return out

    def describe(self) -> dict:
        """What every run records: process, clock, cohort and self-loop rule."""
        cohort = None
        if self.name == "shared":
            cohort = "instance" if self.cohort_size is None else f"replica_id // {self.cohort_size}"
        elif self.name == "sweep":
            cohort = "replica" if self.cohort_size is None else f"replica_id // {self.cohort_size}"
        out = {"process": self.name, "clock": self.clock, "cohort": cohort, "cohort_size": self.cohort_size,
               "self_loops": self.self_loop_rule()}
        if self.name == "nfold":
            out["nfold_clock"] = self.nfold_clock
        return out


def cohort_ids(spec: ProcessSpec, replica_id) -> np.ndarray:
    """Cohort index per replica (Philox counter word c2 of the SHARED / SWEEP streams)."""
    rid = np.asarray(replica_id, dtype=np.int64)
    if spec.cohort_size is not None:
        return rid // int(spec.cohort_size)
    if spec.name == "sweep":
        return rid.copy()
    return np.zeros_like(rid)


def singleton_routes(model: CompiledModel) -> tuple:
    """(ptr int64[R+1], pos int64[...]): per group, the route positions whose clause is a spin singleton.
    Group r changes the state at s iff one of these routes has alpha = 1."""
    is_spin = np.zeros(model.n_clauses, dtype=bool)
    is_spin[model.readout_clause_of_spin.astype(np.int64)] = True
    m = is_spin[model.route_clause.astype(np.int64)]
    pos = np.flatnonzero(m).astype(np.int64)
    grp = model.route_group().astype(np.int64)[pos]
    ptr = np.zeros(model.n_groups + 1, dtype=np.int64)
    np.add.at(ptr, grp + 1, 1)
    return np.cumsum(ptr), pos


#: Exact per-run architecture event totals of the CPU backend (opt-in ``prepare(event_counters=True)``), stored in
#: ``counters["ev_<name>"]``. Names and definitions equal ``emulator.EVENT_COUNTERS`` (events profile); outside
#: ``COUNTER_NAMES`` like the process counters.
EVENT_COUNTER_NAMES = ("clause_toggles", "toggles_objective", "toggles_gate", "clause_route_events",
                       "clause_route_corrections", "switch_reevaluations", "switch_changes", "switch_route_events",
                       "input_corrections")


def ensure_event_counters(rs: "ReplicaStates"):
    for k in EVENT_COUNTER_NAMES:
        if "ev_" + k not in rs.counters:
            rs.counters["ev_" + k] = np.zeros(rs.n_replicas, dtype=np.int64)


def ensure_process_counters(rs: "ReplicaStates"):
    """Add the process counters to a state that lacks them (older snapshots, other backends): steps and
    candidate_evaluations = attempts (one round and one acceptance test per select attempt), never frozen."""
    att = rs.counters["attempts"]
    if "steps" not in rs.counters:
        rs.counters["steps"] = np.array(att, dtype=np.int64, copy=True)
    if "candidate_evaluations" not in rs.counters:
        rs.counters["candidate_evaluations"] = np.array(att, dtype=np.int64, copy=True)
    if "frozen_attempt" not in rs.counters:
        rs.counters["frozen_attempt"] = np.full(rs.n_replicas, -1, dtype=np.int64)


def check_process_budget(spec: ProcessSpec, B: int, n_groups: int):
    """Counter-word ranges of the process streams: parallel_trial/nfold steps < 2**32, sweep index < 2**32."""
    if spec.stepwise and int(B) > 0xFFFFFFFF:
        raise ConfigError(f"{spec.name}: the budget B = {B} exceeds 2**32 steps (Philox counter word c0)")
    if spec.name == "colored" and int(B) > 0xFFFFFFFF:
        raise ConfigError(f"colored: the budget B = {B} exceeds 2**32 attempts (step index is Philox counter word c0)")
    if spec.name == "sweep" and (int(B) - 1) // max(1, int(n_groups)) > 0xFFFFFFFF:
        raise ConfigError("sweep: more than 2**32 sweeps (Philox counter word c0)")


def termination_labels(rs: "ReplicaStates", was_stopped, hit_now, a1: int, B: int) -> tuple:
    """Per-replica termination of a run call: "stopped" (stopped before the call), "target" (stopped on
    target in this call), "frozen" (nfold: no allowed move, clock jumped to B), "budget" or "chunk"."""
    frozen = rs.counters.get("frozen_attempt")
    out = []
    for i in range(rs.n_replicas):
        if was_stopped[i]:
            out.append("stopped")
        elif hit_now[i]:
            out.append("target")
        elif frozen is not None and frozen[i] >= 0:
            out.append("frozen")
        else:
            out.append("budget" if a1 == B else "chunk")
    return tuple(out)


# ============================================================================================
# plan, step and run records
# ============================================================================================
@dataclass
class BackendPlan:
    """Backend-specific execution plan for one compiled model (layouts, options; no replica state)."""

    backend: str
    model: CompiledModel
    strategy: str
    layout: str
    precision: str
    device: str
    options: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {"backend": self.backend, "strategy": self.strategy, "layout": self.layout,
                "precision": self.precision, "device": self.device, "topology_hash": self.model.topology_hash,
                "options": {k: v for k, v in self.options.items() if isinstance(v, (int, float, str, bool, type(None)))}}


@dataclass
class StepRecord:
    """Outcome of one attempt index for every replica (lock-step comparisons).

    ``active`` is False for replicas that did not take part (stopped on target). ``v`` is the
    selected input read for the decision, ``xi`` the threshold (0 for tau = 0). ``n_toggles`` /
    ``n_switch_changes`` = -1 where the backend has no such quantity. ``events``: per-transaction
    event counts {name: int64[n_rep]} when requested from an event-driven backend.
    """

    attempt: int
    tau: float
    group: np.ndarray
    u: np.ndarray
    xi: np.ndarray
    v: np.ndarray
    accepted: np.ndarray
    changed: np.ndarray
    active: np.ndarray
    n_toggles: np.ndarray
    n_switch_changes: np.ndarray
    events: dict | None = None

    @classmethod
    def empty(cls, attempt: int, tau: float, n_rep: int) -> "StepRecord":
        return cls(attempt=int(attempt), tau=float(tau), group=np.full(n_rep, -1, np.int64),
                   u=np.full(n_rep, np.nan), xi=np.full(n_rep, np.nan), v=np.full(n_rep, np.nan),
                   accepted=np.zeros(n_rep, bool), changed=np.zeros(n_rep, bool), active=np.zeros(n_rep, bool),
                   n_toggles=np.full(n_rep, -1, np.int64), n_switch_changes=np.full(n_rep, -1, np.int64))


@dataclass
class RunResult:
    """Result of one ``run`` call over attempts [start, stop) (a chunk when stop < n_attempts_total).

    ``state``: exported ReplicaStates after the chunk; ``attempts_run``: attempts executed per replica
    in this call; ``termination``: per replica "budget" (reached n_attempts_total), "chunk" (stopped at
    ``stop``), "target" (stopped on target in this call) or "stopped" (already stopped before).
    """

    backend: str
    start: int
    stop: int
    n_attempts_total: int
    seed: int
    state: ReplicaStates
    attempts_run: np.ndarray
    termination: tuple
    recording: object = None
    timings: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def summary(self) -> dict:
        s = self.state
        out = {"backend": self.backend, "start": self.start, "stop": self.stop,
               "n_attempts_total": self.n_attempts_total, "seed": self.seed,
               "termination": list(self.termination), "attempts_run": self.attempts_run.tolist(),
               "best_energy": s.best_energy.tolist(), "best_attempt": s.best_attempt.tolist(),
               "feasible_best_energy": [float(x) if np.isfinite(x) else repr(float(x))
                                        for x in s.feasible_best_energy],
               "feasible_best_attempt": s.feasible_best_attempt.tolist(),
               "final_energy": s.energies[:, 0].tolist(), "timings": dict(self.timings)}
        out.update({k: v.tolist() for k, v in s.counters.items()})
        return out


ChunkResult = RunResult


# ============================================================================================
# backend API and registry
# ============================================================================================
class Backend(abc.ABC):
    """Abstract architecture backend. No problem-specific branches."""

    name: str = "abstract"

    @abc.abstractmethod
    def prepare(self, model: CompiledModel, *, strategy: str = "auto", layout: str = "auto",
                precision: str = "float64", device=None, **options) -> BackendPlan:
        """Build the execution plan (layouts, dependency structures) for one compiled model."""

    @abc.abstractmethod
    def init_state(self, plan: BackendPlan, bindings, spins0, replica_binding=None, replica_id=None,
                   instance_id=None, *, is_feasible: Callable | None = None, targets=None):
        """Create replica states from feasible initial spins (Supplementary Information, Exact local maintenance of move-neuron inputs)."""

    @abc.abstractmethod
    def run(self, plan: BackendPlan, state, schedule, *, n_attempts_total=None, start=None, stop=None,
            seed: int, recording=None, targets=None, stop_on_target: bool = False) -> RunResult:
        """Execute attempts [start, stop) of a run of ``n_attempts_total`` attempts."""

    @abc.abstractmethod
    def export_state(self, state) -> ReplicaStates:
        """Portable numpy snapshot of the state (copies)."""

    @abc.abstractmethod
    def load_state(self, plan: BackendPlan, states: ReplicaStates, bindings=None):
        """Backend state from a snapshot (bindings default to ``states.bindings``)."""

    @abc.abstractmethod
    def evaluate_inputs(self, state, groups=None) -> np.ndarray:
        """v_r from the architecture for the given groups (default all), float64[n_rep, len(groups)]; no commit."""

    def describe(self) -> dict:
        return {"backend": self.name, **host_versions()}


_REGISTRY: dict[str, Callable[..., Backend]] = {}
_LAZY_MODULES = {"oracle": "nimbus.backends.oracle", "emulation": "nimbus.backends.emulator",
                 "cpu": "nimbus.backends.cpu", "gpu": "nimbus.backends.gpu"}
_ALIASES = {"emulator": "emulation", "event": "emulation", "direct": "oracle"}


def register_backend(name: str, factory: Callable[..., Backend], *, replace: bool = False):
    """Register a backend class or zero-argument factory under ``name``."""
    name = str(name)
    if name in _REGISTRY and not replace and _REGISTRY[name] is not factory:
        raise ConfigError(f"backend {name!r} is already registered")
    _REGISTRY[name] = factory


def get_backend(name: str, **kwargs) -> Backend:
    """Backend instance by name ("oracle" | "emulation" | "cpu" | "gpu" | registered names).

    Known backends are imported on first use. A known backend whose module is missing or cannot be
    imported raises BackendUnavailable (never a silent substitution); an unknown name raises ConfigError.
    """
    key = _ALIASES.get(str(name), str(name))
    if key not in _REGISTRY and key in _LAZY_MODULES:
        try:
            importlib.import_module(_LAZY_MODULES[key])
        except ImportError as exc:
            raise BackendUnavailable(f"backend {name!r} cannot be imported ({exc})") from exc
        if key not in _REGISTRY:
            raise BackendUnavailable(f"module {_LAZY_MODULES[key]} did not register backend {key!r}")
    if key not in _REGISTRY:
        raise ConfigError(f"unknown backend {name!r}; known: {available_backends()}")
    return _REGISTRY[key](**kwargs)


def available_backends() -> list[str]:
    """Registered plus known (lazily imported) backend names; known does not mean importable here."""
    return sorted(set(_REGISTRY) | set(_LAZY_MODULES))


def check_host_device(backend: str, device) -> str:
    """Host-only backends accept device None/'cpu'/'host'; anything else raises BackendUnavailable."""
    if device is None or str(device).lower() in ("cpu", "host"):
        return "host"
    raise BackendUnavailable(f"backend {backend!r} runs on the host only; device {device!r} is not available to it")


def host_versions() -> dict:
    import nimbus

    return {"python": sys.version.split()[0], "numpy": np.__version__, "nimbus": nimbus.__version__,
            "platform": platform.platform()}


# ============================================================================================
# shared input coercion and initialization (Supplementary Information, Exact local maintenance of move-neuron inputs)
# ============================================================================================
@dataclass(frozen=True)
class SlotLayout:
    """Slot roles shared by all bindings of a run."""

    slot_names: tuple
    score_slot: int
    violation_slot: int | None
    scale: np.ndarray  # float64[n_bindings, n_slots]

    @property
    def penalty(self) -> bool:
        return self.violation_slot is not None

    @classmethod
    def from_bindings(cls, bindings: Sequence[WeightBinding]) -> "SlotLayout":
        names = tuple(bindings[0].slot_names)
        score = names.index("objective") if "objective" in names else 0
        viol = names.index("violation") if "violation" in names else None
        scale = np.stack([np.maximum(1.0, np.abs(b.J).sum(axis=1) + np.abs(b.const)) for b in bindings])
        return cls(names, score, viol, scale)


def coerce_bindings(model: CompiledModel, bindings) -> tuple:
    if isinstance(bindings, WeightBinding):
        bindings = (bindings,)
    bindings = tuple(bindings)
    if not bindings:
        raise ConfigError("at least one WeightBinding is required")
    for b in bindings:
        if not isinstance(b, WeightBinding):
            raise ConfigError(f"not a WeightBinding: {b!r}")
        b.validate_for(model)
    names = bindings[0].slot_names
    if any(b.slot_names != names for b in bindings):
        raise ConfigError("all bindings of one state must have the same slot names")
    return bindings


def coerce_replicas(model: CompiledModel, n_bindings: int, spins0, replica_binding=None, replica_id=None,
                    instance_id=None):
    """Validated (spins int8[n_rep,N], replica_binding int32, replica_id int64, instance_id int64)."""
    s = np.asarray(spins0)
    if s.ndim == 1:
        s = s[None, :]
    if s.ndim != 2 or s.shape[1] != model.n_spins:
        raise ConfigError(f"initial spins must have shape (n_rep, {model.n_spins}), got {s.shape}")
    if s.shape[0] < 1:
        raise ConfigError("at least one replica is required")
    if not np.all((s == 1) | (s == -1)):
        raise ConfigError("initial spins must be -1/+1")
    s = np.ascontiguousarray(s, dtype=np.int8).copy()
    n = s.shape[0]
    rb = np.zeros(n, np.int32) if replica_binding is None else np.asarray(replica_binding).astype(np.int32).reshape(-1)
    if rb.size != n or (n and (rb.min() < 0 or rb.max() >= n_bindings)):
        raise ConfigError("replica_binding must give one valid binding index per replica")
    rid = np.arange(n, dtype=np.int64) if replica_id is None else np.asarray(replica_id).astype(np.int64).reshape(-1)
    iid = rb.astype(np.int64) if instance_id is None else np.asarray(instance_id).astype(np.int64).reshape(-1)
    if rid.size != n or iid.size != n:
        raise ConfigError("replica_id and instance_id need one entry per replica")
    for name, a in (("replica_id", rid), ("instance_id", iid)):
        if np.any(a < 0) or np.any(a > 0xFFFFFFFF):
            raise ConfigError(f"{name} must be in [0, 2**32) (Philox counter word)")
    pairs = set(zip(rid.tolist(), iid.tolist()))
    if len(pairs) != n:
        raise ConfigError("(replica_id, instance_id) pairs must be unique: equal pairs share one solver stream")
    return s, rb, rid, iid


def check_initial_feasibility(spins, is_feasible: Callable | None):
    """Raise InfeasibleState if ``is_feasible(spins_row)`` is False for some replica (None = no check)."""
    if is_feasible is None:
        return
    for i, row in enumerate(np.asarray(spins)):
        if not bool(is_feasible(row.copy())):
            raise InfeasibleState(f"initial state of replica {i} is not feasible: {row.tolist()}")


def clause_signs(model: CompiledModel, spins) -> np.ndarray:
    """T_k = prod_{i in support k} s_i for all stored clauses through the clause CSR (int8[n_rep, M~])."""
    s = np.asarray(spins)
    if s.ndim == 1:
        s = s[None, :]
    if model.n_clauses == 0:
        return np.ones((s.shape[0], 0), dtype=np.int8)
    neg = (s < 0).astype(np.int64)
    cnt = np.add.reduceat(neg[:, model.clause_spins.astype(np.int64)], model.clause_ptr[:-1], axis=1)
    return (1 - 2 * (cnt & 1)).astype(np.int8)


def route_switches(model: CompiledModel, T, positions=None) -> np.ndarray:
    """alpha of routes from stored signs: 1 for ctrl == -1, else (1 + pol T[ctrl]) / 2; int8[n_rep, n]."""
    T = np.asarray(T)
    if T.ndim == 1:
        T = T[None, :]
    ctrl = model.route_ctrl.astype(np.int64)
    pol = model.route_pol.astype(np.int64)
    if positions is not None:
        ctrl, pol = ctrl[positions], pol[positions]
    out = np.ones((T.shape[0], ctrl.size), dtype=np.int8)
    m = ctrl >= 0
    if np.any(m):
        # one replica at a time in int8 (values in {0, 1}): the all-replica int64 temporaries needed
        # ~16 x n_rep x n_routes bytes (125 GB at n = 30 complete, 256 replicas); same result
        cm, pm = ctrl[m], pol[m].astype(np.int8)
        for i in range(T.shape[0]):
            out[i, m] = (1 + pm * T[i, cm].astype(np.int8)) // 2
    return out


def _weighted_positions(model: CompiledModel, groups=None) -> np.ndarray:
    rp, we = model.route_ptr, model.route_wend
    gs = range(model.n_groups) if groups is None else groups
    parts = [np.arange(int(rp[g]), int(we[g]), dtype=np.int64) for g in gs]
    return np.concatenate(parts) if parts else np.zeros(0, np.int64)


def encoder_inputs(model: CompiledModel, J0_rows, T, alpha=None, groups=None) -> np.ndarray:
    """Transition-neuron inputs v_r = sum over weighted routes of J_k alpha T_k (Supplementary Information, Exact local maintenance of move-neuron inputs).

    J0_rows: float64[n_rep, M~] (slot-0 weights per replica); T: int8[n_rep, M~]; alpha: optional
    int8[n_rep, nnz] switch states (default: from T). Each v_r is the sequential sum over its weighted
    routes in route order (deterministic). Returns float64[n_rep, R] (groups=None) or
    float64[n_rep, len(groups)].
    """
    T = np.asarray(T)
    J0_rows = np.asarray(J0_rows, dtype=np.float64)
    if T.ndim == 1:
        T, J0_rows = T[None, :], J0_rows.reshape(1, -1)
    glist = None if groups is None else [int(g) for g in groups]
    pos = _weighted_positions(model, glist)
    rc = model.route_clause.astype(np.int64)[pos]
    if glist is None:
        gid = model.route_group()[pos]
        n_out = model.n_groups
    else:
        lens = [int(model.route_wend[g] - model.route_ptr[g]) for g in glist]
        gid = np.repeat(np.arange(len(glist), dtype=np.int64), lens)
        n_out = len(glist)
    al = route_switches(model, T, pos) if alpha is None else np.asarray(alpha)[:, pos]
    out = np.zeros((T.shape[0], n_out), dtype=np.float64)
    for i in range(T.shape[0]):
        contrib = J0_rows[i, rc] * al[i].astype(np.float64) * T[i, rc].astype(np.float64)
        out[i] = np.bincount(gid, weights=contrib, minlength=n_out)
    return out


def slot_energies(bindings: Sequence[WeightBinding], replica_binding, T) -> np.ndarray:
    """E[rep, slot] = const[slot] - sum_k J[slot, k] T_k (Methods, Derivation of the transition-programmable architecture)."""
    T = np.asarray(T)
    out = np.empty((T.shape[0], bindings[0].n_slots), dtype=np.float64)
    for i in range(T.shape[0]):
        b = bindings[int(replica_binding[i])]
        out[i] = b.const - b.J @ T[i].astype(np.float64)
    return out


def _feasible_rows(layout: SlotLayout, replica_binding, energies) -> np.ndarray:
    if layout.violation_slot is None:
        return np.ones(energies.shape[0], dtype=bool)
    tol = FEASIBLE_RTOL * layout.scale[np.asarray(replica_binding, dtype=np.int64), layout.violation_slot]
    return np.abs(energies[:, layout.violation_slot]) <= tol


def initial_replica_states(backend: str, model: CompiledModel, bindings, spins, replica_binding, replica_id,
                           instance_id, T, energies, v=None, extras=None) -> ReplicaStates:
    """ReplicaStates at attempt 0: best = initial state (attempt count 0), counters zero, no targets."""
    layout = SlotLayout.from_bindings(bindings)
    n = T.shape[0]
    # initialization: T was set from feasible initial spins; the singleton readout must reproduce them
    readout = model.spin_readout(T)
    if not np.array_equal(readout, np.asarray(spins, dtype=np.int8)):
        raise ArchitectureValidationError("initial clause state does not read out the initial spins")
    feas = _feasible_rows(layout, replica_binding, energies)
    fb = np.where(feas, energies[:, layout.score_slot], np.inf)
    counters = {k: np.zeros(n, dtype=np.int64) for k in COUNTER_NAMES}
    counters["target_hit_attempt"][:] = -1
    return ReplicaStates(
        backend=backend, topology_hash=model.topology_hash, weight_hashes=tuple(b.weight_hash for b in bindings),
        slot_names=tuple(bindings[0].slot_names), replica_binding=replica_binding.copy(),
        replica_id=replica_id.copy(), instance_id=instance_id.copy(),
        spin_clause=model.readout_clause_of_spin.astype(np.int32).copy(), T=T.copy(),
        v=None if v is None else np.asarray(v, dtype=np.float64).copy(), energies=energies.copy(),
        best_energy=energies[:, 0].copy(), best_spins=readout.copy(), best_attempt=np.zeros(n, np.int64),
        feasible_best_energy=fb, feasible_best_spins=readout.copy(),
        feasible_best_attempt=np.where(feas, 0, -1).astype(np.int64), counters=counters,
        next_attempt=np.zeros(n, np.int64), targets=np.full(n, np.nan), stopped=np.zeros(n, bool),
        extras=dict(extras or {}), bindings=tuple(bindings))


def apply_targets(rs: ReplicaStates, layout: SlotLayout, targets, stop_on_target: bool = False):
    """Set per-replica targets (float, array or None = keep) and check the current state against them."""
    if targets is None:
        return
    t = np.broadcast_to(np.asarray(targets, dtype=np.float64), (rs.n_replicas,)).copy()
    changed = ~((t == rs.targets) | (np.isnan(t) & np.isnan(rs.targets)))
    rs.targets[changed] = t[changed]
    rs.counters["target_hit_attempt"][changed] = -1
    feas = _feasible_rows(layout, rs.replica_binding, rs.energies)
    b = rs.replica_binding.astype(np.int64)
    tol = TARGET_RTOL * layout.scale[b, layout.score_slot]
    met = changed & np.isfinite(rs.targets) & feas & (rs.energies[:, layout.score_slot] <= rs.targets + tol)
    rs.counters["target_hit_attempt"][met] = rs.next_attempt[met]
    if stop_on_target:
        rs.stopped |= met


def resolve_window(rs: ReplicaStates, schedule, n_attempts_total=None, start=None, stop=None, *,
                   ragged: bool = False) -> tuple[int, int, int]:
    """(B, a0, a1): run budget and the attempt window; refuses to repeat or skip attempts.

    ``ragged`` (nfold, whose replicas advance their clocks by different amounts): every active replica
    continues from its own next attempt, which must be >= start (default start = the smallest one);
    replicas already at or beyond ``stop`` do nothing in this call."""
    B = int(schedule.n_total) if n_attempts_total is None else int(n_attempts_total)
    if B != int(schedule.n_total):
        raise ConfigError(f"n_attempts_total={B} differs from the schedule's n_total={schedule.n_total}")
    active = ~rs.stopped
    nxt = rs.next_attempt[active]
    if ragged:
        a1 = B if stop is None else int(stop)
        a0 = (min(int(nxt.min()), a1) if nxt.size else 0) if start is None else int(start)
        if nxt.size and np.any(nxt < a0):
            raise ConfigError(f"start={a0} is beyond the next attempt of some active replica "
                              f"({int(nxt.min())}); resuming must not skip attempts")
        if not 0 <= a0 <= a1 <= B:
            raise ConfigError(f"need 0 <= start <= stop <= n_attempts_total, got {a0}, {a1}, {B}")
        return B, a0, a1
    a0 = int(nxt[0]) if (start is None and nxt.size) else (0 if start is None else int(start))
    if nxt.size and np.any(nxt != a0):
        raise ConfigError(f"start={a0} does not match the next attempt index of every active replica "
                          f"({sorted(set(nxt.tolist()))}); resuming must not repeat or skip attempts")
    a1 = B if stop is None else int(stop)
    if not 0 <= a0 <= a1 <= B:
        raise ConfigError(f"need 0 <= start <= stop <= n_attempts_total, got {a0}, {a1}, {B}")
    return B, a0, a1


# ============================================================================================
# acceptance law and small numeric helpers
# ============================================================================================
def metropolis_threshold(tau: float, u: float) -> float:
    """xi = -(tau / 2) log u (Methods, Stochastic transition dynamics and FN annealing); 0 for tau == 0 (limiting rule)."""
    if tau == 0.0:
        return 0.0
    return -(tau / 2.0) * math.log(u)


def metropolis_accept(v: float, tau: float, u: float) -> bool:
    """Accept iff v <= xi (tau > 0) or v <= 0 (tau == 0) (Methods, Stochastic transition dynamics and FN annealing)."""
    if tau == 0.0:
        return v <= 0.0
    return v <= -(tau / 2.0) * math.log(u)


def csr_gather(ptr: np.ndarray, data: np.ndarray, rows: np.ndarray):
    """Concatenate CSR rows: (data[ptr[r]:ptr[r+1]] for r in rows, row position of every element)."""
    starts = ptr[rows]
    lens = ptr[rows + 1] - starts
    total = int(lens.sum())
    if total == 0:
        e = np.zeros(0, dtype=np.int64)
        return data[e], e
    rowpos = np.repeat(np.arange(rows.size, dtype=np.int64), lens)
    offs = np.arange(total, dtype=np.int64) + np.repeat(starts - (np.cumsum(lens) - lens), lens)
    return data[offs], rowpos


def csr_rows(ptr: np.ndarray, data: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Concatenation of CSR rows ``data[ptr[r]:ptr[r+1]]`` for r in rows (in order), without row ids."""
    starts = ptr[rows]
    lens = ptr[rows + 1] - starts
    cum = np.cumsum(lens)
    total = int(cum[-1]) if cum.size else 0
    if total == 0:
        return data[:0]
    return data[np.arange(total, dtype=np.int64) + np.repeat(starts - cum + lens, lens)]


# ============================================================================================
# per-run bookkeeping shared by host backends (best, feasible best, targets, anytime, trace)
# ============================================================================================
class RunBook:
    """Best/feasible-best/target updates and optional anytime/trace collection for one run call.

    The basic counters are maintained by the backend loop; this class only handles state-dependent
    bookkeeping, called after accepted transactions that changed the state (a singleton clause
    toggled). Best assignments are read out from the singleton clauses of the clause state.
    """

    def __init__(self, rs: ReplicaStates, layout: SlotLayout, spec: RecordingSpec, n_total: int, start: int,
                 n_spins: int, n_clauses: int, n_groups: int):
        self.rs = rs
        self.layout = layout
        self.spec = spec
        b = rs.replica_binding.astype(np.int64)
        self.best_tol = (BEST_RTOL * layout.scale[b, 0]).tolist()
        self.uphill_tol = (UPHILL_RTOL * layout.scale[b, 0]).tolist()
        self.score = layout.score_slot
        self.viol = layout.violation_slot
        self.score_tol = (BEST_RTOL * layout.scale[b, self.score]).tolist()
        self.viol_tol = (FEASIBLE_RTOL * layout.scale[b, self.viol]).tolist() if self.viol is not None else None
        self.target_tol = (TARGET_RTOL * layout.scale[b, self.score]).tolist()
        self.anytime = None
        self.cp: list[int] = []
        self.cp_ptr = [0] * rs.n_replicas
        if spec.want_anytime:
            cps = spec.checkpoints(n_total)
            self.anytime = AnytimeLog(rs.n_replicas, cps, spec.improvement_cap)
            self.cp = cps.tolist()
            first = int(np.searchsorted(cps, start, side="right"))
            self.cp_ptr = [first] * rs.n_replicas
        self.trace = TraceLog(spec, n_spins, n_clauses, n_groups) if spec.want_trace else None

    def next_checkpoint(self, rep: int) -> int:
        """Attempt count of the replica's next anytime checkpoint (-1 if none)."""
        j = self.cp_ptr[rep]
        return self.cp[j] if j < len(self.cp) else -1

    def checkpoint(self, rep: int, current_energy: float) -> int:
        """Record the checkpoint reached now; return the next checkpoint count (-1 if none)."""
        j = self.cp_ptr[rep]
        rs = self.rs
        self.anytime.record_checkpoint(rep, j, float(rs.best_energy[rep]), float(current_energy),
                                       float(rs.feasible_best_energy[rep]))
        self.cp_ptr[rep] = j + 1
        return self.next_checkpoint(rep)

    def on_change(self, rep: int, count: int, E_row: np.ndarray, T_row: np.ndarray) -> bool:
        """Update best/feasible best/target after a state change at attempt count ``count``.

        ``T_row`` is the replica's clause state; best assignments are its singleton readout.
        Returns True when the replica's target is hit by this state for the first time.
        """
        rs = self.rs
        sc = rs.spin_clause
        e0 = float(E_row[0])
        if e0 < rs.best_energy[rep] - self.best_tol[rep]:
            rs.best_energy[rep] = e0
            rs.best_spins[rep] = T_row[sc]
            rs.best_attempt[rep] = count
            if self.anytime is not None:
                self.anytime.record_improvement(rep, count, e0)
        if self.viol is not None and abs(float(E_row[self.viol])) > self.viol_tol[rep]:
            return False
        es = float(E_row[self.score])
        if es < rs.feasible_best_energy[rep] - self.score_tol[rep] or rs.feasible_best_attempt[rep] < 0:
            rs.feasible_best_energy[rep] = es
            rs.feasible_best_spins[rep] = T_row[sc]
            rs.feasible_best_attempt[rep] = count
        t = rs.targets[rep]
        if t == t and rs.counters["target_hit_attempt"][rep] < 0 and es <= t + self.target_tol[rep]:
            rs.counters["target_hit_attempt"][rep] = count
            return True
        return False

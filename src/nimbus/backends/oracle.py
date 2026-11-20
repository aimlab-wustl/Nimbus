"""Direct recomputation oracle (backend name ``"oracle"``): independent validation, not a production backend.

Methods, Derivation of the transition-programmable architecture: "A direct correctness oracle applies the selected transformation to a
copy of the spin assignment and recomputes every stored clause and candidate energy difference."
The oracle therefore holds only the physical spins (plus the slot energies recomputed from them)
and, for every attempt n:

1. draws (r_n, u_n) with :func:`nimbus.core.rng.solver_draw` (scalar reference RNG) and tau_n with
   :meth:`Schedule.tau`;
2. applies the spin action of group r to a *copy* of the spins, using the TransitionLibrary
   descriptors when the library is supplied to :meth:`OracleBackend.prepare`, otherwise a separate
   element-by-element implementation of the compiled ``act_*`` arrays (never the compiled routes);
3. recomputes T_k(s) and T_k(mu_r(s)) for every stored clause by direct products over the clause
   supports, and v_r = (E(mu_r(s)) - E(s)) / 2 = (1/2) sum_k J_k (T_k(s) - T_k(mu_r(s)))
   (Methods, Derivation of the transition-programmable architecture);
4. accepts iff v_r <= xi_n, xi_n = -(tau_n/2) log u_n (tau_n > 0), or v_r <= 0 (tau_n == 0);
5. on acceptance replaces the spins by mu_r(s) and recomputes all slot energies directly.

The oracle is a separately labelled reference path: unlike the architecture backends (where the clause
state is authoritative and spins are read out from singleton clauses), it keeps its own
private spin array and applies mu_r explicitly. Its exported ``ReplicaStates`` hold T recomputed from
those spins, so their singleton readout equals the oracle's spins by construction.

Nothing is maintained incrementally, so a lock-step comparison against an architecture backend
checks clause signs, switches, inputs, decoder toggles and energies independently. Slow by design.
Finite-table (READOUT) groups need the library: their physical action is defined by the table,
and deriving it from the compiled decoder routes would not be independent.
"""

from __future__ import annotations

import math
import time

import numpy as np

from nimbus.annealing.schedule import Schedule
from nimbus.backends.base import (
    COUNTER_NAMES,
    UPHILL_RTOL,
    Backend,
    BackendPlan,
    ReplicaStates,
    RunBook,
    RunResult,
    SlotLayout,
    StepRecord,
    apply_targets,
    check_host_device,
    check_initial_feasibility,
    coerce_bindings,
    coerce_replicas,
    host_versions,
    initial_replica_states,
    register_backend,
    resolve_window,
)
from nimbus.core import rng
from nimbus.core.errors import ConfigError, IncompatibleCheckpoint
from nimbus.core.model import KIND_CONDMASK, KIND_MASK, KIND_PERM, KIND_READOUT, CompiledModel
from nimbus.experiments.recording import RecordingData, RecordingSpec

__all__ = ["OracleBackend", "OraclePlan", "OracleState", "direct_clause_signs", "direct_action"]


# ============================================================================================
# independent direct evaluations
# ============================================================================================
def _padded_supports(model: CompiledModel) -> np.ndarray:
    """int64[M~, dmax] clause supports padded with index N (a constant +1 spin)."""
    M, N = model.n_clauses, model.n_spins
    deg = np.diff(model.clause_ptr).astype(np.int64)
    dmax = int(deg.max()) if M else 1
    idx = np.full((M, max(dmax, 1)), N, dtype=np.int64)
    cs = model.clause_spins.astype(np.int64)
    for k in range(M):
        a, b = int(model.clause_ptr[k]), int(model.clause_ptr[k + 1])
        idx[k, : b - a] = cs[a:b]
    return idx


def direct_clause_signs(sup_idx: np.ndarray, spins) -> np.ndarray:
    """T_k(s) = prod over the support by np.prod (int64[..., M~]); spins (N,) or (S, N)."""
    s = np.asarray(spins, dtype=np.int64)
    pad = np.ones(s.shape[:-1] + (1,), dtype=np.int64)
    ext = np.concatenate([s, pad], axis=-1)
    return np.prod(ext[..., sup_idx], axis=-1)


def direct_action(model: CompiledModel, r: int, spins: np.ndarray) -> np.ndarray:
    """Spin action of group r applied element by element from the compiled ``act_*`` arrays (copy).

    Independent of ``CompiledModel.apply_action`` and of the compiled routes. READOUT actions are
    defined by the decoder and are not supported here (supply the library to the oracle).
    """
    s = np.asarray(spins)
    out = s.copy()
    kind = int(model.act_kind[r])
    a0, a1 = int(model.act_ptr[r]), int(model.act_ptr[r + 1])
    idx = model.act_idx[a0:a1].tolist()
    if kind == KIND_MASK:
        for i in idx:
            out[i] = -s[i]
    elif kind == KIND_CONDMASK:
        prod = 1
        for g in model.cond_idx[int(model.cond_ptr[r]):int(model.cond_ptr[r + 1])].tolist():
            prod *= int(s[g])
        if int(model.cond_pol[r]) * prod == 1:
            for i in idx:
                out[i] = -s[i]
    elif kind == KIND_PERM:
        for i, j, e in zip(idx, model.act_src[a0:a1].tolist(), model.act_sgn[a0:a1].tolist()):
            out[i] = e * int(s[j])
    else:
        raise ConfigError(f"group {r}: READOUT (finite-table) actions need the TransitionLibrary descriptors")
    return out


# ============================================================================================
# plan and state
# ============================================================================================
class OraclePlan(BackendPlan):
    """Oracle plan: padded clause supports and the spin-action source (descriptors or act_* arrays)."""

    def __init__(self, model: CompiledModel, library=None, device="host"):
        super().__init__(backend="oracle", model=model, strategy="direct", layout="supports",
                         precision="float64", device=device,
                         options={"action_source": "library" if library is not None else "act_arrays"})
        if library is not None:
            if library.n_groups != model.n_groups or library.n_spins != model.n_spins:
                raise ConfigError("library does not match the model (groups/spins)")
            if model.library_hash and library.hash != model.library_hash:
                raise ConfigError("library hash differs from the model's library_hash")
        elif np.any(model.act_kind == KIND_READOUT):
            raise ConfigError("the model has READOUT (finite-table) groups: pass library= to the oracle")
        self.library = library
        self.sup_idx = _padded_supports(model)

    def mu(self, r: int, s: np.ndarray) -> np.ndarray:
        if self.library is not None:
            return np.asarray(self.library.groups[r].apply(s), dtype=np.int8)
        return direct_action(self.model, r, s).astype(np.int8)

    def T(self, spins) -> np.ndarray:
        return direct_clause_signs(self.sup_idx, spins)


class OracleState:
    """The oracle's private spin array (independent reference only), directly recomputed clause signs
    and energies (inside ``rs``), plus the weights."""

    def __init__(self, plan: OraclePlan, rs: ReplicaStates, bindings, spins):
        self.plan = plan
        self.rs = rs
        self.spins = np.array(spins, dtype=np.int8, copy=True)
        self.bindings = tuple(bindings)
        self.layout = SlotLayout.from_bindings(self.bindings)
        self.J = [np.asarray(b.J, dtype=np.float64) for b in self.bindings]
        self.const = [np.asarray(b.const, dtype=np.float64) for b in self.bindings]

    @property
    def n_replicas(self) -> int:
        return self.rs.n_replicas

    def energies_of(self, rep: int, T) -> np.ndarray:
        b = int(self.rs.replica_binding[rep])
        return self.const[b] - self.J[b] @ np.asarray(T, dtype=np.float64)

    def direct_input(self, rep: int, r: int) -> tuple[float, np.ndarray, np.ndarray]:
        """(v_r, mu_r(s), T(mu_r(s))) with v_r = (E(mu) - E(s))/2 from direct products."""
        s = self.spins[rep]
        mu = self.plan.mu(r, s)
        T0 = self.plan.T(s)
        T1 = self.plan.T(mu)
        J0 = self.J[int(self.rs.replica_binding[rep])][0]
        dE = float(J0 @ (T0 - T1).astype(np.float64))
        return 0.5 * dE, mu, T1


# ============================================================================================
# backend
# ============================================================================================
class OracleBackend(Backend):
    """Direct recomputation oracle (backend API plus :meth:`step`)."""

    name = "oracle"

    def prepare(self, model: CompiledModel, *, strategy="auto", layout="auto", precision="float64", device=None,
                library=None, **options) -> OraclePlan:
        if strategy not in ("auto", "direct"):
            raise ConfigError(f"oracle strategy must be 'direct', got {strategy!r}")
        if layout not in ("auto", "supports"):
            raise ConfigError(f"oracle layout must be 'supports', got {layout!r}")
        if precision != "float64":
            raise ConfigError("the oracle computes in float64 only")
        if options:
            raise ConfigError(f"unknown oracle options {sorted(options)}")
        return OraclePlan(model, library=library, device=check_host_device(self.name, device))

    # ------------------------------------------------------------------------------------------
    def init_state(self, plan: OraclePlan, bindings, spins0, replica_binding=None, replica_id=None,
                   instance_id=None, *, is_feasible=None, targets=None) -> OracleState:
        model = plan.model
        bindings = coerce_bindings(model, bindings)
        spins, rb, rid, iid = coerce_replicas(model, len(bindings), spins0, replica_binding, replica_id, instance_id)
        check_initial_feasibility(spins, is_feasible)
        T = plan.T(spins).astype(np.int8)
        E = np.stack([bindings[int(rb[i])].const - bindings[int(rb[i])].J @ T[i].astype(np.float64)
                      for i in range(spins.shape[0])])
        rs = initial_replica_states(self.name, model, bindings, spins, rb, rid, iid, T, E, v=None)
        st = OracleState(plan, rs, bindings, spins)
        apply_targets(rs, st.layout, targets)
        return st

    # ------------------------------------------------------------------------------------------
    def _attempt(self, st: OracleState, rep: int, n: int, schedule: Schedule, seed: int):
        """One attempt of one replica: returns (r, u, tau, xi, v, accepted, changed).

        ``uphill_accepted`` counts accepted state changes with v > 1e-12 * scale (base module docstring).
        """
        rs = st.rs
        R = st.plan.model.n_groups
        r, u = rng.solver_draw(seed, n, int(rs.replica_id[rep]), int(rs.instance_id[rep]), R)
        tau = float(schedule.tau(n))
        v, mu, T1 = st.direct_input(rep, r)
        if tau > 0.0:
            xi = -(tau / 2.0) * math.log(u)
            accepted = v <= xi
        else:
            xi = 0.0
            accepted = v <= 0.0
        changed = bool(accepted and not np.array_equal(mu, st.spins[rep]))
        c = rs.counters
        c["attempts"][rep] += 1
        if accepted:
            c["accepted"][rep] += 1
            if changed:
                c["accepted_changed"][rep] += 1
                if v > UPHILL_RTOL * float(st.layout.scale[int(rs.replica_binding[rep]), 0]):
                    c["uphill_accepted"][rep] += 1
                st.spins[rep] = mu
                rs.T[rep] = T1.astype(np.int8)
                rs.energies[rep] = st.energies_of(rep, T1)
            else:
                c["self_loops"][rep] += 1
        else:
            c["rejected"][rep] += 1
        rs.next_attempt[rep] = n + 1
        return r, u, tau, xi, v, bool(accepted), changed

    def step(self, state: OracleState, n: int, schedule: Schedule, seed: int) -> StepRecord:
        """Attempt index n for every active replica (lock-step API); updates best/targets as ``run``."""
        rs = state.rs
        n = int(n)
        rec = StepRecord.empty(n, float(schedule.tau(n)), rs.n_replicas)
        book = RunBook(rs, state.layout, RecordingSpec(profile="none"), int(schedule.n_total), n,
                       state.plan.model.n_spins, state.plan.model.n_clauses, state.plan.model.n_groups)
        for rep in range(rs.n_replicas):
            if rs.stopped[rep]:
                continue
            if int(rs.next_attempt[rep]) != n:
                raise ConfigError(f"replica {rep}: next attempt is {int(rs.next_attempt[rep])}, not {n}")
            r, u, tau, xi, v, acc, chg = self._attempt(state, rep, n, schedule, seed)
            rec.group[rep], rec.u[rep], rec.xi[rep], rec.v[rep] = r, u, xi, v
            rec.accepted[rep], rec.changed[rep], rec.active[rep] = acc, chg, True
            if chg:
                book.on_change(rep, n + 1, rs.energies[rep], rs.T[rep])
        return rec

    # ------------------------------------------------------------------------------------------
    def run(self, plan: OraclePlan, state: OracleState, schedule: Schedule, *, n_attempts_total=None, start=None,
            stop=None, seed: int, recording=None, targets=None, stop_on_target: bool = False) -> RunResult:
        if state.plan is not plan:
            raise ConfigError("state was initialized with a different plan")
        spec = RecordingSpec.coerce(recording)
        rs = state.rs
        apply_targets(rs, state.layout, targets, stop_on_target)
        B, a0, a1 = resolve_window(rs, schedule, n_attempts_total, start, stop)
        model = plan.model
        book = RunBook(rs, state.layout, spec, B, a0, model.n_spins, model.n_clauses, model.n_groups)
        trace = book.trace
        was_stopped = rs.stopped.copy()
        att0 = rs.counters["attempts"].copy()
        hit_now = np.zeros(rs.n_replicas, dtype=bool)
        t0 = time.perf_counter()
        for rep in range(rs.n_replicas):
            if rs.stopped[rep]:
                continue
            nxt_cp = book.next_checkpoint(rep) if book.anytime is not None else -1
            for n in range(a0, a1):
                r, u, tau, xi, v, acc, chg = self._attempt(state, rep, n, schedule, seed)
                hit = book.on_change(rep, n + 1, rs.energies[rep], rs.T[rep]) if chg else False
                if trace is not None and trace.wants(n, rep):
                    trace.append((n, rep, int(rs.replica_id[rep]), r, u, tau, xi, v, acc, chg,
                                  float(rs.energies[rep, 0]), -1, -1, -1),
                                 spins=state.spins[rep], T=rs.T[rep], v=None)
                if nxt_cp == n + 1:
                    nxt_cp = book.checkpoint(rep, float(rs.energies[rep, 0]))
                if hit and stop_on_target:
                    rs.stopped[rep] = True
                    hit_now[rep] = True
                    break
        elapsed = time.perf_counter() - t0
        term = tuple("stopped" if was_stopped[i] else "target" if hit_now[i] else "budget" if a1 == B else "chunk"
                     for i in range(rs.n_replicas))
        notes = {}
        if spec.want_events:
            notes["events"] = "not applicable: the direct oracle has no architecture events"
        rec = RecordingData(spec=spec, anytime=book.anytime, events=None, trace=trace, notes=notes)
        return RunResult(backend=self.name, start=a0, stop=a1, n_attempts_total=B, seed=int(seed),
                         state=self.export_state(state), attempts_run=rs.counters["attempts"] - att0,
                         termination=term, recording=rec, timings={"run_s": elapsed},
                         metadata={"plan": plan.describe(), "schedule": schedule.to_dict()})

    # ------------------------------------------------------------------------------------------
    def export_state(self, state: OracleState) -> ReplicaStates:
        rs = state.rs.copy()
        rs.bindings = state.bindings
        return rs

    def load_state(self, plan: OraclePlan, states: ReplicaStates, bindings=None) -> OracleState:
        bindings = coerce_bindings(plan.model, bindings if bindings is not None else states.bindings)
        states.check_compatible(plan.model, bindings)
        rs = states.copy()
        rs.backend = self.name
        rs.v = None
        rs.extras = {}
        spins = rs.spins  # singleton readout of the snapshot's clause state
        T = plan.T(spins).astype(np.int8)
        if not np.array_equal(T, rs.T):
            raise IncompatibleCheckpoint("snapshot clause state is inconsistent with its singleton readout")
        missing = [k for k in COUNTER_NAMES if k not in rs.counters]
        if missing:
            raise IncompatibleCheckpoint(f"snapshot lacks counters {missing}")
        st = OracleState(plan, rs, bindings, spins)
        rs.bindings = st.bindings
        for i in range(rs.n_replicas):
            rs.energies[i] = st.energies_of(i, T[i])
        return st

    def evaluate_inputs(self, state: OracleState, groups=None) -> np.ndarray:
        """v_r = (E(mu_r(s)) - E(s))/2 by direct recomputation, float64[n_rep, len(groups)]."""
        plan = state.plan
        R = plan.model.n_groups
        gs = list(range(R)) if groups is None else [int(g) for g in groups]
        out = np.empty((state.n_replicas, len(gs)))
        if not gs:
            return out
        for rep in range(state.n_replicas):
            s = state.spins[rep]
            mus = np.stack([plan.mu(g, s) for g in gs])
            dT = (plan.T(s)[None, :] - plan.T(mus)).astype(np.float64)  # T_k(s) - T_k(mu_g(s))
            out[rep] = 0.5 * (dT @ state.J[int(state.rs.replica_binding[rep])][0])
        return out

    def describe(self) -> dict:
        return {"backend": self.name, "strategy": "direct", "layout": "supports", "precision": "float64",
                "device": "host", "role": "independent validation oracle (not a production backend)",
                **host_versions()}


register_backend("oracle", OracleBackend, replace=True)

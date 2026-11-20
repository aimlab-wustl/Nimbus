"""Event-driven neuromorphic emulation (backend name ``"emulation"``).

Persistent architecture state per replica (Methods, Derivation of the transition-programmable architecture,
Supplementary Information, Algorithmic implementation, Supplementary Information, ON-OFF neuromorphic realization):

* stored clause signs ``T`` (int8[M~]) for every objective, gate, readout and score clause -- the
  authoritative state. Every spin s_i is a stored singleton clause, so the spins are not a separate
  state: they are read out from the singleton clauses. And,
  when tracked, the complementary ON-OFF pair ``T+ = (1+T)/2``, ``T- = (1-T)/2`` (Supplementary Information, ON-OFF neuromorphic realization);
* switch states ``alpha`` (int8[n_routes], 0/1) of every retained connection of the loaded groups,
  derived from the controlling stored sign at initialization/reload (Methods, Derivation of the transition-programmable architecture) and then
  maintained by events;
* transition-neuron inputs ``v`` (float64[R]) of the loaded groups (Supplementary Information, Exact local maintenance of move-neuron inputs);

Attempt n: draw (r, u) from the SOLVER stream; if r is not in
the loaded batch, load its batch (switches and inputs initialized from the current clause state,
Supplementary Information, Exact local maintenance of move-neuron inputs); read the settled v_r; accept iff v_r <= xi = -(tau/2) log u (tau > 0)
or v_r <= 0 (tau = 0). An accepted transition is one transaction:

1. latch sigma_k = alpha_{k,r} over ALL routes of r (weighted, gate, readout) from the
   pre-transition switches (Methods, Derivation of the transition-programmable architecture);
2. toggle the latched clauses, T_k <- (1 - 2 sigma_k) T_k (Methods, Derivation of the transition-programmable architecture), including the
   spin singletons: this is how the transformation mu^(r) is executed (no separate spin-update path).
   The transaction changed the state iff a singleton toggled; no toggle is recomputed midway;
3. stage 1, clause-sign events with the OLD switches (Supplementary Information, Exact local maintenance of move-neuron inputs): for every toggled
   objective clause k and every weighted route (k, r') of a loaded group with open switch:
   v_r' += J_k alpha_old (T_new - T_old);
4. stage 2, switch events with the NEW clause signs (Supplementary Information, Exact local maintenance of move-neuron inputs): every switch
   controlled by a toggled clause is re-evaluated, alpha = (1 + pol T_ctrl)/2; for every changed
   switch on a weighted route (k', r'): v_r' += J_k' T_k'^new (alpha_new - alpha_old). Stages 3 and 4
   sum to Supplementary Information, Exact local maintenance of move-neuron inputs; switch changes never cause further toggles in this transaction;
5. settle: energies of every slot E += 2 sum_{sigma_k = 1} J[slot, k] T_k^old (the decoder output),
   best state, counters; optional validation of Methods, Derivation of the transition-programmable architecture.

Event counters (``events`` profile; names in ``EVENT_COUNTERS``, Supplementary Information, Size of the compiled machine and
Methods, Benchmark problems and simulation methodology): accepted transactions and state changes; spin changes; decoder routes
read and toggles delivered (= sum of sigma); toggles split by clause class (objective, gate = GATE
and not OBJECTIVE, readout-only, score-only) and of any READOUT clause; ON/OFF events e+ / e-;
clause-route events (weighted routes reached by toggled objective clauses) and the corrections they
cause (open switches); switch reevaluations, actual changes (split by controller class: objective
clause reused as gate vs gate-only clause), weighted switch-route events; total weighted input
corrections; routes whose sign and switch both changed in one transaction; distinct neuron inputs
updated; full-scan routes a complete recomputation of all loaded inputs would read; batch reloads with
the switches and inputs they initialize. Histograms per accepted state change: clause toggles,
switch changes, input corrections (bounded, last bin = overflow).

Batching. With ``capacity`` C_max (or an explicit partition ``batches``), only one batch of groups is
loaded per replica. The uniformly drawn r selects its batch, which is equivalent in law to choosing
batch b with probability L_b/R and then a group uniformly within b (Supplementary Information, Algorithmic implementation), so the
proposal sequence equals that of the fully loaded architecture. A batch change keeps all stored
signs and initializes the newly loaded switches and inputs from them.

Validation (``validate="full"`` after every settled transaction and reload, or ``"sampled"`` with
rate ``validate_rate`` drawn from the DIAG stream, which never feeds the solver): T_k == prod s^H for
every stored clause; 2 v_r == E(mu_r(s)) - E(s) for every loaded group (direct products, spin action
from the library descriptors when given); switch states equal their controller signs; the ON-OFF
pair satisfies T = T+ - T-, T+ + T- = 1 and the toggle polarity T_new - T_old = 2 (e+ - e-); slot
energies equal direct recomputation. The first mismatch raises ArchitectureValidationError.
"""

from __future__ import annotations

import bisect
import math
import time

import numpy as np

from nimbus.annealing.schedule import Schedule
from nimbus.backends.coloring import color_groups
from nimbus.backends.base import (
    COUNTER_NAMES,
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
    cohort_ids,
    coerce_replicas,
    csr_rows,
    encoder_inputs,
    ensure_process_counters,
    host_versions,
    initial_replica_states,
    register_backend,
    resolve_window,
    route_switches,
    singleton_routes,
    termination_labels,
)
from nimbus.backends.oracle import _padded_supports, direct_action, direct_clause_signs
from nimbus.core import rng
from nimbus.core.errors import ConfigError, IncompatibleCheckpoint
from nimbus.core.model import (
    KIND_READOUT,
    ROLE_GATE,
    ROLE_OBJECTIVE,
    ROLE_READOUT,
    CompiledModel,
)
from nimbus.experiments.recording import EventLog, RecordingData, RecordingSpec

__all__ = ["EmulatorBackend", "EmulatorPlan", "EmulatorState", "EVENT_COUNTERS", "HIST_NAMES",
           "CLASS_OBJECTIVE", "CLASS_GATE", "CLASS_READOUT", "CLASS_SCORE"]

EVENT_COUNTERS = (
    "transactions", "state_changes", "spin_changes",
    "decoder_routes_read", "clause_toggles",
    "toggles_objective", "toggles_gate", "toggles_readout", "toggles_score", "readout_toggles",
    "onoff_on_events", "onoff_off_events",
    "clause_route_events", "clause_route_corrections",
    "switch_reevaluations", "switch_changes", "switch_changes_ctrl_objective", "switch_changes_ctrl_gate",
    "switch_route_events", "input_corrections",
    "sign_and_switch_routes", "sign_and_switch_weighted", "neurons_updated", "full_scan_routes",
    "batch_reloads", "reload_switch_inits", "reload_input_inits",
)
HIST_NAMES = ("clause_toggles", "switch_changes", "input_corrections")
_I = {n: i for i, n in enumerate(EVENT_COUNTERS)}
_H_TOG, _H_SW, _H_CORR = 0, 1, 2

# clause classes (counting convention of Supplementary Algorithm 1 (compilation of a move library))
CLASS_OBJECTIVE, CLASS_GATE, CLASS_READOUT, CLASS_SCORE = 0, 1, 2, 3
_BLOCK = 4096
_VALIDATE_MODES = ("off", "full", "sampled")
# outcome of one parallel_trial / nfold step
STEP_COMMIT, STEP_NOFIRE, STEP_FROZEN, STEP_OVERSHOOT = 0, 1, 2, 3


def _clause_classes(role: np.ndarray) -> np.ndarray:
    role = role.astype(np.int64)
    cls = np.full(role.size, CLASS_SCORE, dtype=np.int64)
    cls[(role & ROLE_READOUT) != 0] = CLASS_READOUT
    cls[(role & ROLE_GATE) != 0] = CLASS_GATE
    cls[(role & ROLE_OBJECTIVE) != 0] = CLASS_OBJECTIVE
    return cls


# ============================================================================================
# plan
# ============================================================================================
class EmulatorPlan(BackendPlan):
    """Dependency arrays, spin-action data, batch plan and validation settings for one model."""

    def __init__(self, model: CompiledModel, *, capacity=None, batches=None, validate="off", validate_rate=1.0,
                 validate_tol=1e-9, onoff=None, library=None, device="host", process="select", process_options=None):
        self.process = ProcessSpec.coerce(process, process_options)
        if validate not in _VALIDATE_MODES:
            raise ConfigError(f"validate must be one of {_VALIDATE_MODES}, got {validate!r}")
        if not 0.0 < float(validate_rate) <= 1.0:
            raise ConfigError("validate_rate must be in (0, 1]")
        if library is not None:
            if library.n_groups != model.n_groups or library.n_spins != model.n_spins:
                raise ConfigError("library does not match the model (groups/spins)")
            if model.library_hash and library.hash != model.library_hash:
                raise ConfigError("library hash differs from the model's library_hash")
        self.library = library
        N, M, R = model.n_spins, model.n_clauses, model.n_groups
        self.N, self.M, self.R, self.nnz = N, M, R, model.n_routes
        # routes and dependencies (int64 copies for fast indexing)
        self.rp = model.route_ptr.astype(np.int64).tolist()
        self.route_clause = model.route_clause.astype(np.int64)
        self.route_ctrl = model.route_ctrl.astype(np.int64)
        self.route_pol = model.route_pol.astype(np.int8)
        self.route_group = model.route_group().astype(np.int64)
        self.route_weighted = model.weighted_route_mask()
        self.cw_ptr = model.clause_wroute_ptr.astype(np.int64)
        self.cw = model.clause_wroute.astype(np.int64)
        self.cr_ptr = model.ctrl_route_ptr.astype(np.int64)
        self.cr = model.ctrl_route.astype(np.int64)
        role = model.clause_role
        self.clause_class = _clause_classes(role)
        self.is_obj = (role & ROLE_OBJECTIVE) != 0
        self.is_readout = (role & ROLE_READOUT) != 0
        self.has_ctrl = np.diff(self.cr_ptr) > 0
        ctrl_safe = np.where(self.route_ctrl >= 0, self.route_ctrl, 0)
        self.route_ctrl_gate = (self.route_ctrl >= 0) & (self.clause_class[ctrl_safe] == CLASS_GATE)
        # spin singletons: the clause state is authoritative. Descriptor kinds are kept
        # only for the validation mode's independent reference mu (never for runtime updates).
        self.kinds = model.act_kind.astype(np.int64).tolist()
        self.is_spin_clause = np.zeros(M, dtype=bool)
        self.is_spin_clause[model.readout_clause_of_spin.astype(np.int64)] = True
        # batches
        if batches is not None and capacity is not None:
            raise ConfigError("give either capacity or batches, not both")
        if batches is not None:
            parts = [np.asarray(sorted(int(g) for g in b), dtype=np.int64) for b in batches]
            flat = np.concatenate(parts) if parts else np.zeros(0, np.int64)
            if any(p.size == 0 for p in parts) or flat.size != R or not np.array_equal(np.sort(flat), np.arange(R)):
                raise ConfigError("batches must partition the group ids 0..R-1 into non-empty sets")
            self.capacity = max(p.size for p in parts)
        else:
            cap = R if capacity is None else int(capacity)
            if cap < 1:
                raise ConfigError("capacity C_max must be >= 1")
            cap = min(cap, R)
            parts = [np.arange(g, min(g + cap, R), dtype=np.int64) for g in range(0, R, cap)]
            self.capacity = cap
        self.batch_groups = parts
        self.n_batches = len(parts)
        self.batched = self.n_batches > 1
        self.batch_of_group = np.empty(R, dtype=np.int64)
        for b, p in enumerate(parts):
            self.batch_of_group[p] = b
        self.batch_of_group_list = self.batch_of_group.tolist()
        rpa = model.route_ptr.astype(np.int64)
        wea = model.route_wend.astype(np.int64)
        self.batch_routes, self.batch_wroutes, self.batch_wlocal = [], [], []
        for p in parts:
            allr = [np.arange(rpa[g], rpa[g + 1]) for g in p]
            wr = [np.arange(rpa[g], wea[g]) for g in p]
            self.batch_routes.append(np.concatenate(allr).astype(np.int64) if allr else np.zeros(0, np.int64))
            self.batch_wroutes.append(np.concatenate(wr).astype(np.int64) if wr else np.zeros(0, np.int64))
            self.batch_wlocal.append(np.repeat(np.arange(p.size, dtype=np.int64), [w.size for w in wr]))
        self.batch_wcount = [int(w.size) for w in self.batch_wroutes]
        if self.process.full_library and self.batched:
            raise ConfigError(f"process {self.process.name!r} needs every input v_r settled at every step: it "
                              "runs on the full library only (no capacity batching)")
        # spin-singleton routes per group: group r changes the state iff one of them has alpha = 1
        self.sr_ptr, self.sr_pos = singleton_routes(model)
        # colored: compile-time coloring of the groups; the cyclic class visit is a function of the clock
        self.coloring = color_groups(model) if self.process.colored else None
        # validation
        self.validate = validate
        self.validate_rate = float(validate_rate)
        self.validate_tol = float(validate_tol)
        self.onoff = bool(validate != "off") if onoff is None else bool(onoff)
        self.sup_idx = _padded_supports(model)
        super().__init__(backend="emulation", model=model, strategy="event", layout="csr", precision="float64",
                         device=device,
                         options={"capacity": self.capacity, "n_batches": self.n_batches, "validate": validate,
                                  "validate_rate": self.validate_rate, "validate_tol": self.validate_tol,
                                  "onoff": self.onoff,
                                  "action_source": "library" if library is not None else "act_arrays",
                                  **({"coloring_" + k: v for k, v in self.coloring.summary().items() if v is not None and not isinstance(v, list)}
                                     if self.coloring is not None else {}),
                                  "process": self.process.name, **{"process_" + k: v for k, v in
                                                                   self.process.describe().items() if k != "process"}})

    def mu(self, r: int, s: np.ndarray, T_direct=None) -> np.ndarray:
        """Independent reference transformation for the validation mode only (never a runtime update):
        library descriptor if given, else the descriptor arrays; READOUT goes through the decoder."""
        if self.library is not None:
            return np.asarray(self.library.groups[r].apply(s), dtype=np.int8)
        if self.kinds[r] == KIND_READOUT:
            return self.model.apply_action(r, s, T_direct).astype(np.int8)
        return direct_action(self.model, r, s).astype(np.int8)


# ============================================================================================
# state
# ============================================================================================
class EmulatorState:
    """Persistent architecture state of all replicas (stored signs, switches, inputs, ON-OFF pair)."""

    def __init__(self, plan: EmulatorPlan, rs: ReplicaStates, bindings, alpha, loaded_batch, Tp=None, Tm=None,
                 batch_reloads=None):
        self.plan = plan
        self.rs = rs
        self.bindings = tuple(bindings)
        self.layout = SlotLayout.from_bindings(self.bindings)
        self.J = [np.asarray(b.J, dtype=np.float64) for b in self.bindings]
        self.J0 = [j[0] for j in self.J]
        self.const = [np.asarray(b.const, dtype=np.float64) for b in self.bindings]
        self.alpha = alpha
        self.loaded_batch = np.asarray(loaded_batch, dtype=np.int64)
        n = rs.n_replicas
        self.loaded_group = None
        if plan.batched:
            self.loaded_group = np.zeros((n, plan.R), dtype=bool)
            for i in range(n):
                self.loaded_group[i, plan.batch_groups[int(self.loaded_batch[i])]] = True
        self.Tp = Tp
        self.Tm = Tm
        self.batch_reloads = np.zeros(n, np.int64) if batch_reloads is None else np.asarray(batch_reloads, np.int64)
        self.mark = np.zeros(plan.M, dtype=bool)

    @property
    def n_replicas(self) -> int:
        return self.rs.n_replicas

    def loaded_groups(self, rep: int) -> np.ndarray:
        return self.plan.batch_groups[int(self.loaded_batch[rep])]


# ============================================================================================
# backend
# ============================================================================================
class EmulatorBackend(Backend):
    """Event-driven neuromorphic emulation of the compiled architecture (backend API + step)."""

    name = "emulation"

    def prepare(self, model: CompiledModel, *, strategy="auto", layout="auto", precision="float64", device=None,
                capacity=None, batches=None, validate="off", validate_rate=1.0, validate_tol=1e-9, onoff=None,
                library=None, process="select", process_options=None, **options) -> EmulatorPlan:
        if strategy not in ("auto", "event"):
            raise ConfigError(f"the emulation is event-driven (strategy 'event'), got {strategy!r}")
        if layout not in ("auto", "csr"):
            raise ConfigError(f"emulation layout must be 'csr', got {layout!r}")
        if precision != "float64":
            raise ConfigError("the emulation computes in float64 only")
        if options:
            raise ConfigError(f"unknown emulation options {sorted(options)}")
        return EmulatorPlan(model, capacity=capacity, batches=batches, validate=validate, validate_rate=validate_rate,
                            validate_tol=validate_tol, onoff=onoff, library=library,
                            device=check_host_device(self.name, device), process=process,
                            process_options=process_options)

    # ------------------------------------------------------------------------------------------
    def init_state(self, plan: EmulatorPlan, bindings, spins0, replica_binding=None, replica_id=None,
                   instance_id=None, *, is_feasible=None, targets=None) -> EmulatorState:
        model = plan.model
        bindings = coerce_bindings(model, bindings)
        spins, rb, rid, iid = coerce_replicas(model, len(bindings), spins0, replica_binding, replica_id, instance_id)
        check_initial_feasibility(spins, is_feasible)
        T = clause_signs(model, spins)
        E = np.stack([bindings[int(rb[i])].const - bindings[int(rb[i])].J @ T[i].astype(np.float64)
                      for i in range(spins.shape[0])])
        n = spins.shape[0]
        v = np.full((n, plan.R), np.nan)
        rs = initial_replica_states(self.name, model, bindings, spins, rb, rid, iid, T, E, v=v)
        ensure_process_counters(rs)
        alpha = route_switches(model, T)  # every switch derived from its controller sign at initialization
        Tp = Tm = None
        if plan.onoff:
            Tp = ((1 + T.astype(np.int16)) // 2).astype(np.int8)
            Tm = ((1 - T.astype(np.int16)) // 2).astype(np.int8)
        st = EmulatorState(plan, rs, bindings, alpha, np.zeros(n, np.int64), Tp, Tm)
        for i in range(n):
            self._init_batch(st, i, 0)
        apply_targets(rs, st.layout, targets)
        if plan.validate != "off":
            for i in range(n):
                self._validate(st, i, -1, -1, "initialization")
        return st

    # ------------------------------------------------------------------------------------------
    # batch loading (Supplementary Information, Exact local maintenance of move-neuron inputs for the newly loaded groups)
    # ------------------------------------------------------------------------------------------
    def _init_batch(self, st: EmulatorState, rep: int, b: int):
        P = st.plan
        T = st.rs.T[rep]
        al = st.alpha[rep]
        routes = P.batch_routes[b]
        if routes.size:
            ctrl = P.route_ctrl[routes]
            pol = P.route_pol[routes]
            a_new = np.ones(routes.size, dtype=np.int8)
            m = ctrl >= 0
            a_new[m] = (1 + pol[m] * T[ctrl[m]]) >> 1
            al[routes] = a_new
        groups = P.batch_groups[b]
        wr = P.batch_wroutes[b]
        J0 = st.J0[int(st.rs.replica_binding[rep])]
        rc = P.route_clause[wr]
        contrib = J0[rc] * al[wr] * T[rc]
        st.rs.v[rep, groups] = np.bincount(P.batch_wlocal[b], weights=contrib, minlength=groups.size)
        st.loaded_batch[rep] = b
        if st.loaded_group is not None:
            st.loaded_group[rep, :] = False
            st.loaded_group[rep, groups] = True

    def _reload(self, st: EmulatorState, rep: int, b: int, ev):
        P = st.plan
        old = int(st.loaded_batch[rep])
        st.rs.v[rep, P.batch_groups[old]] = np.nan  # neurons reassigned: inputs no longer held
        self._init_batch(st, rep, b)
        st.batch_reloads[rep] += 1
        if ev is not None:
            c = ev.counts[rep]
            c[_I["batch_reloads"]] += 1
            c[_I["reload_switch_inits"]] += int(P.batch_routes[b].size)
            c[_I["reload_input_inits"]] += int(P.batch_groups[b].size)

    # ------------------------------------------------------------------------------------------
    # one transaction (step 4)
    # ------------------------------------------------------------------------------------------
    def _commit(self, st: EmulatorState, rep: int, r: int, ev):
        """Accepted transaction of group r; returns (changed, n_toggles, n_switch_changes, n_corrections)."""
        P = st.plan
        rs = st.rs
        T = rs.T[rep]
        al = st.alpha[rep]
        a, e = P.rp[r], P.rp[r + 1]
        # (a) latch sigma over all routes of r from the pre-transition switches
        Kt = P.route_clause[a:e][al[a:e] != 0]
        # (b) state change = a spin singleton receives a toggle (the spins are those clauses)
        nflip = int(np.count_nonzero(P.is_spin_clause[Kt]))
        changed = nflip > 0
        nt = int(Kt.size)
        c = None
        if ev is not None:
            c = ev.counts[rep]
            c[_I["transactions"]] += 1
            c[_I["decoder_routes_read"]] += e - a
            c[_I["spin_changes"]] += nflip
            if changed:
                c[_I["state_changes"]] += 1
                c[_I["full_scan_routes"]] += P.batch_wcount[int(st.loaded_batch[rep])]
        if nt == 0:
            if ev is not None and changed:
                ev.add_hist(rep, _H_TOG, 0)
                ev.add_hist(rep, _H_SW, 0)
                ev.add_hist(rep, _H_CORR, 0)
            return changed, 0, 0, 0
        # (c) latched clause toggles through the decoder, singletons included: T <- (1 - 2 sigma) T
        Told = T[Kt]
        T[Kt] = -Told
        if st.Tp is not None:
            e_on = st.Tm[rep, Kt]   # e+ = T- sigma (an OFF clause receiving a toggle emits an ON event)
            e_off = st.Tp[rep, Kt]  # e- = T+ sigma (an ON clause emits an OFF event)
            st.Tp[rep, Kt] += e_on - e_off
            st.Tm[rep, Kt] += e_off - e_on
            if P.validate != "off" and not np.array_equal(-2 * Told.astype(np.int16),
                                                          2 * (e_on.astype(np.int16) - e_off)):
                raise ArchitectureValidationError(
                    "ON-OFF pulse polarity violated: T_new - T_old != 2 (e+ - e-) (Supplementary Information, ON-OFF neuromorphic realization)",
                    {"identity": "T_new - T_old == 2 (e+ - e-)", "replica": rep, "group": r, "clauses": Kt.tolist()})
        Toldf = Told.astype(np.float64)
        J = st.J[int(rs.replica_binding[rep])]
        # (d) stage 1: clause-sign events through the OLD switches (Supplementary Information, Exact local maintenance of move-neuron inputs)
        n_corr1 = 0
        p_open = None
        Kw = Kt[P.is_obj[Kt]]
        if Kw.size:
            pos = csr_rows(P.cw_ptr, P.cw, Kw)  # weighted routes (k, r') of the toggled objective clauses
            if st.loaded_group is not None and pos.size:
                pos = pos[st.loaded_group[rep, P.route_group[pos]]]
            p_open = pos[al[pos] != 0]  # closed switches deliver no event to the neuron
            n_corr1 = int(p_open.size)
            if n_corr1:
                k = P.route_clause[p_open]
                # J_k alpha_old (T_new - T_old) with alpha_old = 1 and T_new - T_old = 2 T_new
                np.add.at(rs.v[rep], P.route_group[p_open], 2.0 * J[0, k] * T[k])
            if c is not None:
                c[_I["clause_route_events"]] += int(pos.size)
                c[_I["clause_route_corrections"]] += n_corr1
        # (d) stage 2: switch re-evaluation with the NEW clause signs (Supplementary Information, Exact local maintenance of move-neuron inputs)
        nsw, p_w = self._propagate_switches(st, rep, Kt, ev)
        n_corr = n_corr1 + (0 if p_w is None else int(p_w.size))
        # (e) settle: energies of every slot from the latched decoder toggles
        rs.energies[rep] += 2.0 * (J[:, Kt] @ Toldf)
        if c is not None:
            cls = np.bincount(P.clause_class[Kt], minlength=4)
            c[_I["clause_toggles"]] += nt
            c[_I["toggles_objective"]] += int(cls[CLASS_OBJECTIVE])
            c[_I["toggles_gate"]] += int(cls[CLASS_GATE])
            c[_I["toggles_readout"]] += int(cls[CLASS_READOUT])
            c[_I["toggles_score"]] += int(cls[CLASS_SCORE])
            c[_I["readout_toggles"]] += int(np.count_nonzero(P.is_readout[Kt]))
            n_on = int(np.count_nonzero(Told < 0))
            c[_I["onoff_on_events"]] += n_on
            c[_I["onoff_off_events"]] += nt - n_on
            c[_I["input_corrections"]] += n_corr
            if n_corr:
                parts = [P.route_group[x] for x in (p_open, p_w) if x is not None and x.size]
                c[_I["neurons_updated"]] += int(np.unique(np.concatenate(parts)).size)
            if changed:
                ev.add_hist(rep, _H_TOG, nt)
                ev.add_hist(rep, _H_SW, nsw)
                ev.add_hist(rep, _H_CORR, n_corr)
        return changed, nt, nsw, n_corr

    def _propagate_switches(self, st: EmulatorState, rep: int, Kt: np.ndarray, ev):
        """Stage 2: re-evaluate every switch controlled by a toggled clause and apply
        Supplementary Information, Exact local maintenance of move-neuron inputs on weighted routes. Returns (switch changes, weighted changed routes)."""
        P = st.plan
        Kc = Kt[P.has_ctrl[Kt]]
        if not Kc.size:
            return 0, None
        T = st.rs.T[rep]
        al = st.alpha[rep]
        pos = csr_rows(P.cr_ptr, P.cr, Kc)  # every route controlled by a toggled clause
        if st.loaded_group is not None and pos.size:
            pos = pos[st.loaded_group[rep, P.route_group[pos]]]
        if pos.size == 0:
            return 0, None
        a_old = al[pos]
        a_new = (1 + P.route_pol[pos] * T[P.route_ctrl[pos]]) >> 1  # from the NEW controller signs
        al[pos] = a_new
        d = a_new - a_old
        chg = d != 0
        wm = P.route_weighted[pos] & chg
        p_w = pos[wm]
        if p_w.size:
            kc = P.route_clause[p_w]
            v = st.rs.v[rep]
            J0 = st.J0[int(st.rs.replica_binding[rep])]
            corr = J0[kc] * T[kc] * d[wm]  # J_k' T_k'^new (alpha_new - alpha_old)
            np.add.at(v, P.route_group[p_w], corr)
        nsw = int(np.count_nonzero(chg))
        if ev is not None:
            c = ev.counts[rep]
            c[_I["switch_reevaluations"]] += int(pos.size)
            c[_I["switch_changes"]] += nsw
            pc = pos[chg]
            ng = int(np.count_nonzero(P.route_ctrl_gate[pc]))
            c[_I["switch_changes_ctrl_gate"]] += ng
            c[_I["switch_changes_ctrl_objective"]] += nsw - ng
            c[_I["switch_route_events"]] += int(p_w.size)
            mark = st.mark
            mark[Kt] = True
            both = mark[P.route_clause[pc]]
            c[_I["sign_and_switch_routes"]] += int(np.count_nonzero(both))
            c[_I["sign_and_switch_weighted"]] += int(np.count_nonzero(both & P.route_weighted[pc]))
            mark[Kt] = False
        return nsw, p_w

    # ------------------------------------------------------------------------------------------
    # one attempt (shared by run and step)
    # ------------------------------------------------------------------------------------------
    def _attempt(self, st: EmulatorState, rep: int, n: int, r: int, u: float, tau: float, ev, seed: int):
        """Returns (xi, v_r, accepted, changed, n_toggles, n_switch_changes)."""
        P = st.plan
        if P.batched and P.batch_of_group_list[r] != st.loaded_batch[rep]:
            self._reload(st, rep, P.batch_of_group_list[r], ev)
            if P.validate == "full":
                self._validate(st, rep, n, r, "batch reload")
        vr = float(st.rs.v[rep, r])
        if tau > 0.0:
            xi = -(tau / 2.0) * math.log(u)
            accepted = vr <= xi
        else:
            xi = 0.0
            accepted = vr <= 0.0
        if not accepted:
            return xi, vr, False, False, 0, 0
        changed, nt, nsw, _ = self._commit(st, rep, r, ev)
        if P.validate == "full" or (P.validate == "sampled" and self._diag_uniform(st, rep, n, seed) < P.validate_rate):
            self._validate(st, rep, n, r, "settled transaction")
        return xi, vr, True, changed, nt, nsw

    @staticmethod
    def _diag_uniform(st: EmulatorState, rep: int, n: int, seed: int) -> float:
        rs = st.rs
        x = rng.philox4x32_scalar(n & 0xFFFFFFFF, n >> 32, int(rs.replica_id[rep]) & 0xFFFFFFFF,
                                  int(rs.instance_id[rep]) & 0xFFFFFFFF, int(seed), rng.DIAG)
        return rng.uniform52(x[2], x[3])

    def step(self, state: EmulatorState, n: int, schedule: Schedule, seed: int, *, collect_events: bool = False) -> StepRecord:
        """Clock index n for every active replica (lock-step API). ``collect_events`` returns the
        per-transaction event counts in ``StepRecord.events``.

        select/shared/sweep: attempt n. parallel_trial: step n (``u``/``xi`` are NaN; ``group`` is the
        committed group or -1 when nothing fired). nfold advances replicas by different numbers of
        attempts and has no lock-step form: use ``run``."""
        rs = state.rs
        n = int(n)
        proc = state.plan.process
        if proc.name == "nfold":
            raise ConfigError("nfold replicas advance by different numbers of attempts; use run() (chunks)")
        if proc.colored:
            raise ConfigError("colored commits a whole color class per step (one attempt per group, bookkeeping at "
                              "the class end); use run() (chunks)")
        tau = float(schedule.tau(n))
        rec = StepRecord.empty(n, tau, rs.n_replicas)
        ev = EventLog(rs.n_replicas, EVENT_COUNTERS, HIST_NAMES, 65) if collect_events else None
        book = RunBook(rs, state.layout, RecordingSpec(profile="none"), int(schedule.n_total), n,
                       state.plan.N, state.plan.M, state.plan.R)
        c = rs.counters
        cid = cohort_ids(proc, rs.replica_id)
        for rep in range(rs.n_replicas):
            if rs.stopped[rep]:
                continue
            if int(rs.next_attempt[rep]) != n:
                raise ConfigError(f"replica {rep}: next attempt is {int(rs.next_attempt[rep])}, not {n}")
            if proc.name == "parallel_trial":
                self._step_pt(state, rep, n, tau, seed, book, ev, rec)
                continue
            r, u = rng.solver_draw(seed, n, int(rs.replica_id[rep]), int(rs.instance_id[rep]), state.plan.R)
            if proc.name == "shared":
                r = rng.shared_group(seed, n, int(cid[rep]), int(rs.instance_id[rep]), state.plan.R)
            elif proc.name == "sweep":
                r = int(rng.sweep_groups(seed, [n], int(cid[rep]), int(rs.instance_id[rep]), state.plan.R)[0])
            xi, vr, acc, chg, nt, nsw = self._attempt(state, rep, n, r, u, tau, ev, seed)
            c["attempts"][rep] += 1
            c["steps"][rep] += 1
            c["candidate_evaluations"][rep] += 1
            if acc:
                c["accepted"][rep] += 1
                if chg:
                    c["accepted_changed"][rep] += 1
                    c["uphill_accepted"][rep] += vr > book.uphill_tol[rep]
                    book.on_change(rep, n + 1, rs.energies[rep], rs.T[rep])
                else:
                    c["self_loops"][rep] += 1
            else:
                c["rejected"][rep] += 1
            rs.next_attempt[rep] = n + 1
            rec.group[rep], rec.u[rep], rec.xi[rep], rec.v[rep] = r, u, xi, vr
            rec.accepted[rep], rec.changed[rep], rec.active[rep] = acc, chg, True
            rec.n_toggles[rep], rec.n_switch_changes[rep] = nt, nsw
        if ev is not None:
            rec.events = {k: ev.counter(k).copy() for k in EVENT_COUNTERS}
        return rec

    # ------------------------------------------------------------------------------------------
    def run(self, plan: EmulatorPlan, state: EmulatorState, schedule: Schedule, *, n_attempts_total=None,
            start=None, stop=None, seed: int, recording=None, targets=None, stop_on_target: bool = False) -> RunResult:
        if state.plan is not plan:
            raise ConfigError("state was initialized with a different plan")
        spec = RecordingSpec.coerce(recording)
        rs = state.rs
        apply_targets(rs, state.layout, targets, stop_on_target)
        proc = plan.process
        B, a0, a1 = resolve_window(rs, schedule, n_attempts_total, start, stop, ragged=proc.ragged)
        check_process_budget(proc, B, plan.R)
        book = RunBook(rs, state.layout, spec, B, a0, plan.N, plan.M, plan.R)
        ev = None
        if spec.want_events:
            ev = EventLog(rs.n_replicas, EVENT_COUNTERS, HIST_NAMES, spec.event_hist_bins,
                          meta={"weighted_routes_per_batch": plan.batch_wcount, "n_batches": plan.n_batches,
                                "capacity": plan.capacity, "routes_total": plan.nnz,
                                "weighted_routes_total": int(plan.route_weighted.sum())})
        was_stopped = rs.stopped.copy()
        att0 = rs.counters["attempts"].copy()
        hit_now = np.zeros(rs.n_replicas, dtype=bool)
        t0 = time.perf_counter()
        cid = cohort_ids(proc, rs.replica_id)
        for rep in range(rs.n_replicas):
            if not rs.stopped[rep]:
                if proc.colored:
                    hit_now[rep] = self._run_replica_colored(state, rep, a0, a1, schedule, int(seed), book, ev,
                                                             stop_on_target)
                elif proc.stepwise:
                    hit_now[rep] = self._run_replica_steps(state, rep, a1, B, schedule, int(seed), book, ev,
                                                           stop_on_target)
                else:
                    hit_now[rep] = self._run_replica(state, rep, a0, a1, schedule, int(seed), book, ev,
                                                     stop_on_target, int(cid[rep]))
        elapsed = time.perf_counter() - t0
        term = termination_labels(rs, was_stopped, hit_now, a1, B)
        rec = RecordingData(spec=spec, anytime=book.anytime, events=ev, trace=book.trace)
        return RunResult(backend=self.name, start=a0, stop=a1, n_attempts_total=B, seed=int(seed),
                         state=self.export_state(state), attempts_run=rs.counters["attempts"] - att0,
                         termination=term, recording=rec, timings={"run_s": elapsed},
                         metadata={"plan": plan.describe(), "schedule": schedule.to_dict(),
                                   "process": {**proc.describe(), **({"coloring": plan.coloring.summary()} if proc.colored else {})},
                                   "batch_reloads": state.batch_reloads.tolist()})

    def _run_replica(self, st: EmulatorState, rep: int, a0: int, a1: int, schedule: Schedule, seed: int,
                     book: RunBook, ev, stop_on_target: bool, cohort: int = 0) -> bool:
        """Attempt-clock processes: select (solver draw), shared (cohort group sequence + the replica's
        solver uniform) and sweep (per-sweep permutation + the replica's solver uniform)."""
        rs = st.rs
        R = st.plan.R
        pname = st.plan.process.name
        rid, iid = int(rs.replica_id[rep]), int(rs.instance_id[rep])
        trace = book.trace
        nxt_cp = book.next_checkpoint(rep) if book.anytime is not None else -1
        tau_of = schedule.tau
        attempt = self._attempt
        up_tol = book.uphill_tol[rep]
        att = acc = chg = slf = rej = up = 0
        hit = False
        n = a0
        try:
            while n < a1 and not hit:
                m = min(_BLOCK, a1 - n)
                rr, uu = rng.solver_draws(seed, np.arange(n, n + m, dtype=np.uint64), rid, iid, R)
                if pname == "shared":
                    rr = rng.shared_groups(seed, np.arange(n, n + m, dtype=np.uint64), cohort, iid, R)
                elif pname == "sweep":
                    rr = rng.sweep_groups(seed, np.arange(n, n + m, dtype=np.int64), cohort, iid, R)
                rr, uu = rr.tolist(), uu.tolist()
                for j in range(m):
                    nn = n + j
                    r = rr[j]
                    tau = tau_of(nn)
                    xi, vr, a_, c_, nt, nsw = attempt(st, rep, nn, r, uu[j], tau, ev, seed)
                    att += 1
                    if a_:
                        acc += 1
                        if c_:
                            chg += 1
                            if vr > up_tol:
                                up += 1
                            hit = book.on_change(rep, nn + 1, rs.energies[rep], rs.T[rep])
                        else:
                            slf += 1
                    else:
                        rej += 1
                    if trace is not None and trace.wants(nn, rep):
                        trace.append((nn, rep, rid, r, uu[j], tau, xi, vr, a_, c_, float(rs.energies[rep, 0]), nt, nsw,
                                      int(st.loaded_batch[rep])),
                                     spins=rs.spins_of(rep), T=rs.T[rep], v=rs.v[rep])
                    if nxt_cp == nn + 1:
                        nxt_cp = book.checkpoint(rep, float(rs.energies[rep, 0]))
                    if hit and stop_on_target:
                        rs.stopped[rep] = True
                        break
                    hit = False
                n += m
        finally:
            cnt = rs.counters
            cnt["attempts"][rep] += att
            cnt["accepted"][rep] += acc
            cnt["accepted_changed"][rep] += chg
            cnt["self_loops"][rep] += slf
            cnt["rejected"][rep] += rej
            cnt["uphill_accepted"][rep] += up
            cnt["steps"][rep] += att
            cnt["candidate_evaluations"][rep] += att
            rs.next_attempt[rep] = a0 + att
        return bool(rs.stopped[rep])

    def _run_replica_colored(self, st: EmulatorState, rep: int, a0: int, a1: int, schedule: Schedule, seed: int,
                             book: RunBook, ev, stop_on_target: bool) -> bool:
        """colored. The clock is the attempt count a; one cycle of R
        attempts visits the groups in position order (class 0 first). Attempt a belongs to color step
        ``(a // R) * C + class_of_pos[a % R]``; the step's tau is the schedule at the clock value of its first
        position. Each group of the visited class draws its own uniform (stream COLOR, counter (step, r,
        replica, instance)) and is tested against its own v_r; the accepted groups commit (ascending r; the
        classes are non-interacting so the order is immaterial, tests/test_colored.py). Best / feasible /
        target bookkeeping and anytime checkpoints see only the state at the end of the class (the run budget
        or window may cut the last class short: the visited prefix is committed)."""
        rs = st.rs
        P = st.plan
        R, col = P.R, P.coloring
        C = col.n_colors
        c = rs.counters
        rid, iid = int(rs.replica_id[rep]), int(rs.instance_id[rep])
        trace = book.trace
        nxt_cp = book.next_checkpoint(rep) if book.anytime is not None else -1
        up_tol = book.uphill_tol[rep]
        order, cptr, cop = col.order, col.class_ptr, col.class_of_pos
        a = int(rs.next_attempt[rep])
        while a < a1:
            cyc, p = divmod(a, R)
            k = int(cop[p])
            lo, hi = int(cptr[k]), int(cptr[k + 1])
            a_start = a - (p - lo)
            step = cyc * C + k
            tau = float(schedule.tau(a_start))
            n_do = min(hi - p, a1 - a)
            a_new = a + n_do
            E0 = float(rs.energies[rep, 0])
            acc_n = chg_n = slf_n = up_n = 0
            for q in range(p, p + n_do):
                r = int(order[q])
                u = rng.color_uniform(seed, step, r, rid, iid)
                xi, vr, acc, chg, nt, nsw = self._attempt(st, rep, a + (q - p), r, u, tau, ev, seed)
                if acc:
                    acc_n += 1
                    if chg:
                        chg_n += 1
                        up_n += vr > up_tol
                    else:
                        slf_n += 1
                if trace is not None and trace.wants(a + (q - p), rep):
                    trace.append((a + (q - p), rep, rid, r, u, tau, xi, vr, acc, chg, float(rs.energies[rep, 0]), nt,
                                  nsw, int(st.loaded_batch[rep])), spins=rs.spins_of(rep), T=rs.T[rep], v=rs.v[rep])
            c["attempts"][rep] += n_do
            c["candidate_evaluations"][rep] += n_do
            c["steps"][rep] += 1 if p == lo else 0
            c["accepted"][rep] += acc_n
            c["accepted_changed"][rep] += chg_n
            c["self_loops"][rep] += slf_n
            c["rejected"][rep] += n_do - acc_n
            c["uphill_accepted"][rep] += up_n
            while nxt_cp != -1 and nxt_cp < a_new:  # checkpoints inside the class see the pre-commit state
                nxt_cp = book.checkpoint(rep, E0)
            hit = book.on_change(rep, a_new, rs.energies[rep], rs.T[rep]) if chg_n else False
            if nxt_cp == a_new:
                nxt_cp = book.checkpoint(rep, float(rs.energies[rep, 0]))
            a = a_new
            rs.next_attempt[rep] = a
            if hit and stop_on_target:
                rs.stopped[rep] = True
                break
        return bool(rs.stopped[rep])

    # ------------------------------------------------------------------------------------------
    # parallel_trial / nfold: every input settled every step
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _state_changing(st: EmulatorState, rep: int) -> np.ndarray:
        """bool[R]: group r changes the state iff some route of r to a spin singleton has alpha = 1
        (pre-transition switches). Evaluated locally from the maintained switches."""
        P = st.plan
        if P.sr_pos.size == 0:
            return np.zeros(P.R, dtype=bool)
        cs = np.zeros(P.sr_pos.size + 1, dtype=np.int64)
        np.cumsum(st.alpha[rep, P.sr_pos], out=cs[1:])
        return (cs[P.sr_ptr[1:]] - cs[P.sr_ptr[:-1]]) > 0

    def _pt_decide(self, st: EmulatorState, rep: int, n: int, tau: float, seed: int):
        """parallel_trial step n: (committed group or -1, candidates evaluated, |F|). Candidate r fires iff
        v_r <= -(tau/2) log u_{n,r} (tau > 0; v_r <= 0 fires for every u, so no draw is needed) or
        v_r <= 0 (tau = 0); the arbiter picks one member of F (ascending r) uniformly."""
        P = st.plan
        rs = st.rs
        v = rs.v[rep]
        cand = np.flatnonzero(self._state_changing(st, rep)) if P.process.pt_exclude_self_loops else np.arange(P.R)
        rid, iid = int(rs.replica_id[rep]), int(rs.instance_id[rep])
        vc = v[cand]
        pos = vc > 0.0
        fired = vc <= 0.0
        if tau > 0.0 and np.any(pos):
            idx = np.flatnonzero(pos)
            uu = rng.pt_uniforms(seed, n, rid, iid, cand[idx]).tolist()
            vv = vc[idx].tolist()
            for j, x, u in zip(idx.tolist(), vv, uu):
                if x <= -(tau / 2.0) * math.log(u):
                    fired[j] = True
        F = cand[fired]
        if F.size == 0:
            return -1, int(cand.size), 0
        return int(F[rng.pt_arbiter(seed, n, rid, iid, int(F.size))]), int(cand.size), int(F.size)

    def _nfold_decide(self, st: EmulatorState, rep: int, k: int, n: int, tau: float, seed: int, B: int):
        """nfold step k at clock n: (outcome, group, W, candidates evaluated).

        a_r = 1 if v_r <= 0, exp(-2 v_r / tau) if tau > 0, else 0, over the state-changing groups; prefix
        sums S in ascending r (sequential); r = first candidate with S_r > u_sel * A (if rounding leaves
        none, the last candidate whose interval is non-empty). Attempt clock: p = A / R; W = 1 if p >= 1,
        else q = log(u_w) / log1p(-p) and W = 1 + floor(q); if the change would land beyond the budget
        (q >= B - n) nothing is committed and the clock ends at B (outcome OVERSHOOT). A = 0: FROZEN."""
        P = st.plan
        rs = st.rs
        cand = np.flatnonzero(self._state_changing(st, rep)).tolist()
        v = rs.v[rep]
        S = []
        A = 0.0
        for r in cand:
            x = float(v[r])
            if x <= 0.0:
                a = 1.0
            elif tau > 0.0:
                a = math.exp(-2.0 * x / tau)
            else:
                a = 0.0
            A += a
            S.append(A)
        m = len(S)
        if m == 0 or A == 0.0:
            return STEP_FROZEN, -1, 0, m
        u_sel, u_w = rng.nfold_uniforms(seed, k, int(rs.replica_id[rep]), int(rs.instance_id[rep]))
        i = bisect.bisect_right(S, u_sel * A)
        if i == m:
            i = m - 1
            while i > 0 and S[i] <= S[i - 1]:
                i -= 1
        r = cand[i]
        if P.process.nfold_clock != "attempts":
            return STEP_COMMIT, r, 1, m
        p = A / P.R
        if p >= 1.0:
            return STEP_COMMIT, r, 1, m
        q = math.log(u_w) / math.log1p(-p)
        if q >= float(B - n):
            return STEP_OVERSHOOT, r, B - n, m
        return STEP_COMMIT, r, int(math.floor(q)) + 1, m

    def _step_pt(self, st: EmulatorState, rep: int, n: int, tau: float, seed: int, book: RunBook, ev, rec):
        """One parallel_trial step of one replica (lock-step ``step`` API)."""
        rs = st.rs
        c = rs.counters
        r, n_eval, _ = self._pt_decide(st, rep, n, tau, seed)
        c["attempts"][rep] += 1
        c["steps"][rep] += 1
        c["candidate_evaluations"][rep] += n_eval
        chg, nt, nsw, vr = False, 0, 0, float("nan")
        if r >= 0:
            vr = float(rs.v[rep, r])
            chg, nt, nsw, _ = self._commit(st, rep, r, ev)
            c["accepted"][rep] += 1
            if chg:
                c["accepted_changed"][rep] += 1
                c["uphill_accepted"][rep] += vr > book.uphill_tol[rep]
                book.on_change(rep, n + 1, rs.energies[rep], rs.T[rep])
            else:
                c["self_loops"][rep] += 1
            self._maybe_validate(st, rep, n, r, seed)
        else:
            c["rejected"][rep] += 1
        rs.next_attempt[rep] = n + 1
        rec.group[rep], rec.v[rep] = r, vr
        rec.accepted[rep], rec.changed[rep], rec.active[rep] = r >= 0, chg, True
        rec.n_toggles[rep], rec.n_switch_changes[rep] = nt, nsw

    def _maybe_validate(self, st: EmulatorState, rep: int, n: int, r: int, seed: int):
        P = st.plan
        if P.validate == "full" or (P.validate == "sampled" and self._diag_uniform(st, rep, n, seed) < P.validate_rate):
            self._validate(st, rep, n, r, "settled transaction")

    def _run_replica_steps(self, st: EmulatorState, rep: int, a1: int, B: int, schedule: Schedule, seed: int,
                           book: RunBook, ev, stop_on_target: bool) -> bool:
        """parallel_trial / nfold steps of one replica while its clock n < a1. The schedule is evaluated at
        the clock value where the step starts (steps, or attempt-equivalent for nfold_clock: attempts).
        Counters: attempts = clock units; nfold charges W attempts per step (W - 1 skipped attempts count
        as rejected, so attempts = accepted + rejected); steps = rounds; candidate_evaluations = groups
        whose acceptance or weight was evaluated."""
        P = st.plan
        rs = st.rs
        c = rs.counters
        is_pt = P.process.name == "parallel_trial"
        rid = int(rs.replica_id[rep])
        trace = book.trace
        nxt_cp = book.next_checkpoint(rep) if book.anytime is not None else -1
        up_tol = book.uphill_tol[rep]
        n = int(rs.next_attempt[rep])
        k = int(c["steps"][rep])
        while n < a1:
            tau = float(schedule.tau(n))
            if is_pt:
                r, n_eval, _ = self._pt_decide(st, rep, n, tau, seed)
                outcome, W = (STEP_COMMIT if r >= 0 else STEP_NOFIRE), 1
            else:
                outcome, r, W, n_eval = self._nfold_decide(st, rep, k, n, tau, seed, B)
            k += 1
            c["steps"][rep] += 1
            c["candidate_evaluations"][rep] += n_eval
            E0 = float(rs.energies[rep, 0])
            if outcome in (STEP_FROZEN, STEP_OVERSHOOT):
                W = B - n
                while nxt_cp != -1 and nxt_cp <= B:
                    nxt_cp = book.checkpoint(rep, E0)
                c["attempts"][rep] += W
                c["rejected"][rep] += W
                if outcome == STEP_FROZEN:
                    c["frozen_attempt"][rep] = n
                if trace is not None and trace.wants(n, rep):
                    trace.append((n, rep, rid, -1, math.nan, tau, math.nan, math.nan, False, False, E0, 0, 0, 0),
                                 spins=rs.spins_of(rep), T=rs.T[rep], v=rs.v[rep])
                n = B
                break
            # checkpoints passed during the skipped attempts see the pre-commit state
            while nxt_cp != -1 and nxt_cp < n + W:
                nxt_cp = book.checkpoint(rep, E0)
            c["attempts"][rep] += W
            hit = False
            chg, nt, nsw, vr = False, 0, 0, math.nan
            if outcome == STEP_NOFIRE:
                c["rejected"][rep] += 1
            else:
                vr = float(rs.v[rep, r])
                chg, nt, nsw, _ = self._commit(st, rep, r, ev)
                c["accepted"][rep] += 1
                c["rejected"][rep] += W - 1
                if chg:
                    c["accepted_changed"][rep] += 1
                    if vr > up_tol:
                        c["uphill_accepted"][rep] += 1
                    hit = book.on_change(rep, n + W, rs.energies[rep], rs.T[rep])
                else:
                    c["self_loops"][rep] += 1
                self._maybe_validate(st, rep, n, r, seed)
            if trace is not None and trace.wants(n, rep):
                trace.append((n, rep, rid, r, math.nan, tau, math.nan, vr, outcome == STEP_COMMIT, chg,
                              float(rs.energies[rep, 0]), nt, nsw, 0),
                             spins=rs.spins_of(rep), T=rs.T[rep], v=rs.v[rep])
            n += W
            if nxt_cp == n:
                nxt_cp = book.checkpoint(rep, float(rs.energies[rep, 0]))
            if hit and stop_on_target:
                rs.stopped[rep] = True
                break
        rs.next_attempt[rep] = n
        return bool(rs.stopped[rep])

    # ------------------------------------------------------------------------------------------
    # validation mode (Methods, Derivation of the transition-programmable architecture and derived invariants)
    # ------------------------------------------------------------------------------------------
    def _validate(self, st: EmulatorState, rep: int, n: int, r: int, where: str):
        P = st.plan
        model = P.model
        rs = st.rs
        s = rs.spins_of(rep)  # readout of the singleton clauses
        T = rs.T[rep]
        b = int(rs.replica_binding[rep])
        ctx = {"where": where, "attempt": int(n), "replica": int(rep), "replica_id": int(rs.replica_id[rep]),
               "instance_id": int(rs.instance_id[rep]), "group": int(r), "loaded_batch": int(st.loaded_batch[rep])}

        def fail(identity, msg, **extra):
            ctx.update(identity=identity, **extra)
            raise ArchitectureValidationError(f"{identity} violated after {where} (attempt {n}, replica {rep}, "
                                              f"group {r}): {msg}", ctx)

        Td = direct_clause_signs(P.sup_idx, s)
        bad = np.flatnonzero(Td != T)
        if bad.size:
            fail("T_k == prod_i s_i^H_ki", f"{bad.size} stored clause signs differ from the product of their "
                 "singleton readouts",
                 clauses=bad[:20].tolist(), stored=T[bad[:20]].tolist(), direct=Td[bad[:20]].tolist(),
                 supports=[model.clause_support(int(k)) for k in bad[:5]])
        groups = st.loaded_groups(rep)
        J = st.J[b]
        scale = float(st.layout.scale[b, 0])
        tol = P.validate_tol * scale
        if groups.size:
            mus = np.stack([P.mu(int(g), s, Td) for g in groups.tolist()])
            Tm = direct_clause_signs(P.sup_idx, mus)
            dE = (Td[None, :] - Tm).astype(np.float64) @ J[0]  # E(mu_g(s)) - E(s)
            v = rs.v[rep, groups]
            dev = np.abs(2.0 * v - dE)
            badg = np.flatnonzero(~(dev <= tol))
            if badg.size:
                g0 = int(groups[badg[0]])
                fail("2 v_r == E(mu_r(s)) - E(s)", f"{badg.size} loaded transition-neuron inputs are wrong "
                     f"(first: group {g0} {model.group_label[g0]!r}: 2v={2 * v[badg[0]]!r}, dE={dE[badg[0]]!r})",
                     groups=groups[badg[:20]].tolist(), two_v=(2 * v[badg[:20]]).tolist(), dE=dE[badg[:20]].tolist(),
                     tolerance=tol)
        routes = P.batch_routes[int(st.loaded_batch[rep])]
        if routes.size:
            want = route_switches(model, T, routes)[0]
            got = st.alpha[rep, routes]
            badr = np.flatnonzero(want != got)
            if badr.size:
                p = routes[badr[:20]]
                fail("alpha == (1 + pol T_ctrl)/2", f"{badr.size} switch states differ from their controller signs",
                     routes=p.tolist(), clause=P.route_clause[p].tolist(), ctrl=P.route_ctrl[p].tolist(),
                     group_of_route=P.route_group[p].tolist(), stored=got[badr[:20]].tolist(),
                     expected=want[badr[:20]].tolist())
        if st.Tp is not None:
            Tp, Tm_ = st.Tp[rep].astype(np.int16), st.Tm[rep].astype(np.int16)
            badc = np.flatnonzero((Tp - Tm_ != T) | (Tp + Tm_ != 1))
            if badc.size:
                fail("T = T+ - T-, T+ + T- = 1", f"{badc.size} ON-OFF pairs are inconsistent", clauses=badc[:20].tolist())
        Ed = st.const[b] - J @ Td.astype(np.float64)
        E = rs.energies[rep]
        etol = P.validate_tol * st.layout.scale[b]
        bads = np.flatnonzero(~(np.abs(E - Ed) <= etol))
        if bads.size:
            fail("E[slot] == E_const - sum_k J_k T_k(s)", "incremental slot energies drifted from direct recomputation",
                 slots=[st.layout.slot_names[i] for i in bads], energies=E[bads].tolist(), direct=Ed[bads].tolist())

    def validate_state(self, state: EmulatorState):
        """Check every replica against the validation identities now (raises on the first mismatch)."""
        for rep in range(state.n_replicas):
            self._validate(state, rep, int(state.rs.next_attempt[rep]) - 1, -1, "explicit check")

    def drift_report(self, state: EmulatorState) -> dict:
        """Numerical drift audit: maintained inputs vs a fresh encoder sum over the
        current stored signs (Supplementary Information, Exact local maintenance of move-neuron inputs), and incremental slot energies vs direct
        recomputation. Read-only: nothing is corrected, so trajectories are unaffected.

        Returns per-replica arrays ``v_abs_max`` (max |v_maintained - v_fresh| over loaded groups),
        ``v_rel_max`` (divided by the slot-0 scale) and ``energy_abs_max`` (max over slots).
        """
        P = state.plan
        rs = state.rs
        n = rs.n_replicas
        v_abs = np.zeros(n)
        e_abs = np.zeros(n)
        for rep in range(n):
            groups = state.loaded_groups(rep)
            b = int(rs.replica_binding[rep])
            fresh = encoder_inputs(P.model, state.J0[b][None, :], rs.T[rep:rep + 1], groups=groups.tolist())[0]
            v_abs[rep] = float(np.max(np.abs(rs.v[rep, groups] - fresh))) if groups.size else 0.0
            Td = direct_clause_signs(P.sup_idx, rs.spins_of(rep)).astype(np.float64)
            e_abs[rep] = float(np.max(np.abs(rs.energies[rep] - (state.const[b] - state.J[b] @ Td))))
        scale = state.layout.scale[rs.replica_binding.astype(np.int64), 0]
        return {"v_abs_max": v_abs, "v_rel_max": v_abs / scale, "energy_abs_max": e_abs,
                "attempts": rs.next_attempt.copy()}

    # ------------------------------------------------------------------------------------------
    def export_state(self, state: EmulatorState) -> ReplicaStates:
        rs = state.rs.copy()
        rs.bindings = state.bindings
        rs.extras = {"alpha": state.alpha.copy(), "loaded_batch": state.loaded_batch.copy(),
                     "batch_reloads": state.batch_reloads.copy(),
                     "n_batches": np.asarray(state.plan.n_batches, dtype=np.int64),
                     "capacity": np.asarray(state.plan.capacity, dtype=np.int64)}
        if state.Tp is not None:
            rs.extras["Tplus"] = state.Tp.copy()
            rs.extras["Tminus"] = state.Tm.copy()
        return rs

    def load_state(self, plan: EmulatorPlan, states: ReplicaStates, bindings=None) -> EmulatorState:
        """Restore a snapshot. An emulation snapshot with the same batch plan is restored exactly
        (switches, maintained inputs); otherwise switches and loaded inputs are rebuilt from the stored
        signs (Supplementary Information, Exact local maintenance of move-neuron inputs), which can differ from maintained inputs by rounding."""
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
        n = rs.n_replicas
        ex = rs.extras
        exact = (states.backend == self.name and "alpha" in ex and rs.v is not None and rs.v.shape == (n, plan.R)
                 and int(np.asarray(ex.get("n_batches", -1))) == plan.n_batches
                 and int(np.asarray(ex.get("capacity", -1))) == plan.capacity)
        if exact:
            alpha = np.array(ex["alpha"], dtype=np.int8, copy=True)
            loaded = np.array(ex["loaded_batch"], dtype=np.int64, copy=True)
            reloads = np.array(ex.get("batch_reloads", np.zeros(n, np.int64)), dtype=np.int64, copy=True)
        else:
            alpha = route_switches(model, rs.T)
            loaded = np.zeros(n, np.int64)
            reloads = None
            rs.v = np.full((n, plan.R), np.nan)
        Tp = Tm = None
        if plan.onoff:
            if exact and "Tplus" in ex:
                Tp, Tm = np.array(ex["Tplus"], np.int8, copy=True), np.array(ex["Tminus"], np.int8, copy=True)
            else:
                Tp = ((1 + rs.T.astype(np.int16)) // 2).astype(np.int8)
                Tm = ((1 - rs.T.astype(np.int16)) // 2).astype(np.int8)
        rs.backend = self.name
        rs.extras = {}
        st = EmulatorState(plan, rs, bindings, alpha, loaded, Tp, Tm, reloads)
        rs.bindings = st.bindings
        if not exact:
            for i in range(n):
                self._init_batch(st, i, 0)
        if plan.validate != "off":
            for i in range(n):
                self._validate(st, i, int(rs.next_attempt[i]) - 1, -1, "load_state")
        return st

    def evaluate_inputs(self, state: EmulatorState, groups=None) -> np.ndarray:
        """Maintained v_r for loaded groups; for groups outside the loaded batch, the encoder sum over
        the current stored signs (Supplementary Information, Exact local maintenance of move-neuron inputs, no reload and no commit)."""
        P = state.plan
        gs = np.arange(P.R) if groups is None else np.asarray([int(g) for g in groups], dtype=np.int64)
        out = np.empty((state.n_replicas, gs.size))
        for rep in range(state.n_replicas):
            vals = state.rs.v[rep, gs].copy()
            if state.loaded_group is not None:
                miss = ~state.loaded_group[rep, gs]
                if np.any(miss):
                    J0 = state.J0[int(state.rs.replica_binding[rep])][None, :]
                    vals[miss] = encoder_inputs(P.model, J0, state.rs.T[rep:rep + 1], groups=gs[miss].tolist())[0]
            out[rep] = vals
        return out

    def describe(self) -> dict:
        return {"backend": self.name, "strategy": "event", "layout": "csr", "precision": "float64", "device": "host",
                "role": "event-driven neuromorphic emulation (architecture events and resources)", **host_versions()}


register_backend("emulation", EmulatorBackend, replace=True)

"""Event-driven emulation: diagnostic negative test, batching, zero temperature, RNG streams,
recording independence, event-counter identities, ON-OFF pair, checkpoint/resume and performance."""

from __future__ import annotations

import time

import numpy as np
import pytest

from helpers_core import (
    cardinality_identities,
    exact_cardinality_states,
    onehot_identities,
    random_distances,
    tour_length,
    cyclic_assignment_objective,
)
from test_replay_core import (
    case_cardinality_random,
    case_cardinality_toy,
    case_onehot,
    case_section29,
    case_table_kempe,
    mixing_schedule,
)
from nimbus.annealing import schedule as S
from nimbus.backends import ArchitectureValidationError, ReplicaStates, get_backend
from nimbus.backends.emulator import CLASS_GATE, EVENT_COUNTERS, EmulatorBackend
from nimbus.compiler import compile_model
from nimbus.core.errors import BackendUnavailable, ConfigError, IncompatibleCheckpoint, InfeasibleState
from nimbus.core.model import ROLE_GATE, ROLE_OBJECTIVE, ObjectiveStructure, bind_weights
from nimbus.transitions import builders as B


def _same_state(a: ReplicaStates, b: ReplicaStates, v=True):
    assert np.array_equal(a.spins, b.spins)
    assert np.array_equal(a.T, b.T)
    assert np.array_equal(a.energies, b.energies)
    for k in a.counters:
        assert np.array_equal(a.counters[k], b.counters[k]), k
    assert np.array_equal(a.best_energy, b.best_energy) and np.array_equal(a.best_attempt, b.best_attempt)
    assert np.array_equal(a.next_attempt, b.next_attempt)
    if v:
        assert np.array_equal(a.v, b.v, equal_nan=True)


# ============================================================================================
# (b) diagnostic negative test (Fig. 1 a)
# ============================================================================================
class GateSwitchesNotMaintained(EmulatorBackend):
    """Deliberately broken variant: gate-clause signs toggle, but the switches they control are
    never re-evaluated (stage 2 skipped for gate-only controllers)."""

    def _propagate_switches(self, st, rep, Kt, ev):
        keep = st.plan.clause_class[Kt] != CLASS_GATE
        return super()._propagate_switches(st, rep, Kt[keep], ev)


@pytest.mark.parametrize("builder", [case_cardinality_toy, lambda: case_onehot(4, "path", 3, True), case_section29])
def test_broken_switch_maintenance_is_detected(builder):
    case = builder()
    assert np.any((case.model.clause_role & ROLE_GATE) & ~(case.model.clause_role & ROLE_OBJECTIVE))
    sched = mixing_schedule(case, 400)
    spins0 = case.initial(4, 1)
    broken = GateSwitchesNotMaintained()
    plan = broken.prepare(case.model, validate="full", library=case.library)
    st = broken.init_state(plan, case.bindings[0], spins0)
    with pytest.raises(ArchitectureValidationError) as info:
        broken.run(plan, st, sched, seed=3)
    ctx = info.value.context
    assert ctx["identity"] in ("2 v_r == E(mu_r(s)) - E(s)", "alpha == (1 + pol T_ctrl)/2",
                               "T_k == prod_i s_i^H_ki")
    assert ctx["attempt"] >= 0 and "replica" in ctx
    # the same run with correct maintenance validates at every settled transaction
    good = get_backend("emulation")
    gplan = good.prepare(case.model, validate="full", library=case.library)
    res = good.run(gplan, good.init_state(gplan, case.bindings[0], spins0), sched, seed=3)
    assert res.state.counters["accepted_changed"].sum() > 0
    # without validation the broken variant silently corrupts the architecture: stale switches give
    # wrong inputs and wrong latched toggles, so stored signs drift away from the spins
    nplan = broken.prepare(case.model, validate="off")
    nst = broken.init_state(nplan, case.bindings[0], spins0)
    broken.run(nplan, nst, sched, seed=3)
    ex = broken.export_state(nst)
    o = get_backend("oracle")
    oplan = o.prepare(case.model, library=case.library)
    if np.array_equal(case.model.clause_values(ex.spins), ex.T):
        v_true = o.evaluate_inputs(o.load_state(oplan, ex))
        assert np.max(np.abs(v_true - broken.evaluate_inputs(nst))) > 1e-6
    else:
        with pytest.raises(IncompatibleCheckpoint):
            o.load_state(oplan, ex)


def test_broken_variant_fails_at_the_first_gate_toggle():
    """The diagnostic raises exactly at the first accepted transaction that toggles a gate-only clause
    (until then the broken and the correct emulation are identical), naming the violated identity."""
    case = case_cardinality_toy()
    sched = mixing_schedule(case, 600)
    spins0 = case.initial(2, 5)
    gates = np.flatnonzero((case.model.clause_role & ROLE_GATE) & ~(case.model.clause_role & ROLE_OBJECTIVE))
    good = get_backend("emulation")
    gp = good.prepare(case.model)
    gs = good.init_state(gp, case.bindings[0], spins0)
    first = None
    for n in range(600):
        before = gs.rs.T.copy()
        good.step(gs, n, sched, 11)
        if np.any(before[:, gates] != gs.rs.T[:, gates]):
            first = n
            break
    assert first is not None
    broken = GateSwitchesNotMaintained()
    bp = broken.prepare(case.model, validate="full", library=case.library)
    bs = broken.init_state(bp, case.bindings[0], spins0)
    with pytest.raises(ArchitectureValidationError) as info:
        for n in range(600):
            broken.step(bs, n, sched, 11)
    assert info.value.context["attempt"] == first
    assert info.value.context["identity"] in ("2 v_r == E(mu_r(s)) - E(s)", "alpha == (1 + pol T_ctrl)/2")


# ============================================================================================
# (c) batching: capacity-limited emulation reproduces the fully loaded trajectory
# ============================================================================================
@pytest.mark.parametrize("builder,opts", [
    (case_cardinality_toy, {"capacity": 3}),
    (case_cardinality_toy, {"batches": [[0, 5, 10, 15], [1, 2, 3, 4], [6, 7, 8, 9, 11, 12, 13, 14]]}),
    (lambda: case_onehot(5, "complete", 4, False), {"capacity": 4}),
    (case_table_kempe, {"capacity": 1}),
])
def test_batched_emulation_matches_full_load(builder, opts):
    case = builder()
    sched = mixing_schedule(case, 1500)
    e = get_backend("emulation")
    spins0 = case.initial(4, 2)
    rb = [0, 0, 1, 1]
    full = e.prepare(case.model, validate="full", library=case.library)
    bat = e.prepare(case.model, validate="full", library=case.library, **opts)
    assert bat.n_batches > 1 and full.n_batches == 1
    sf = e.init_state(full, case.bindings, spins0, rb)
    sb = e.init_state(bat, case.bindings, spins0, rb)
    reloads = 0
    for n in range(1500):
        rf = e.step(sf, n, sched, 77)
        rbat = e.step(sb, n, sched, 77, collect_events=True)
        assert np.array_equal(rf.group, rbat.group) and np.array_equal(rf.accepted, rbat.accepted)
        assert np.array_equal(rf.changed, rbat.changed)
        assert np.array_equal(sf.rs.spins, sb.rs.spins) and np.array_equal(sf.rs.T, sb.rs.T)
        assert np.allclose(rf.v, rbat.v, rtol=0, atol=1e-9 * float(sf.layout.scale.max()))
        reloads += int(rbat.events["batch_reloads"].sum())
        assert rbat.events["reload_input_inits"].sum() <= rbat.events["batch_reloads"].sum() * bat.capacity
    assert reloads > 0
    assert int(sb.batch_reloads.sum()) == reloads
    # only the loaded batch holds inputs; evaluate_inputs still returns the encoder value for all groups
    ex = e.export_state(sb)
    for i in range(4):
        loaded = bat.batch_groups[int(ex.extras["loaded_batch"][i])]
        assert np.all(np.isfinite(ex.v[i, loaded]))
        assert np.isnan(np.delete(ex.v[i], loaded)).all()
    assert np.allclose(e.evaluate_inputs(sb), e.evaluate_inputs(sf), atol=1e-9 * float(sf.layout.scale.max()))
    # same trajectory through run() as well
    rr_full = e.run(full, e.init_state(full, case.bindings, spins0, rb), sched, seed=77)
    rr_bat = e.run(bat, e.init_state(bat, case.bindings, spins0, rb), sched, seed=77, recording="events")
    assert np.array_equal(rr_full.state.spins, rr_bat.state.spins)
    for k in rr_full.state.counters:
        assert np.array_equal(rr_full.state.counters[k], rr_bat.state.counters[k])
    assert rr_bat.recording.events.counter("batch_reloads").sum() == rr_bat.state.extras["batch_reloads"].sum() > 0


def test_batch_options_validated():
    case = case_cardinality_toy()
    e = get_backend("emulation")
    with pytest.raises(ConfigError):
        e.prepare(case.model, batches=[[0, 1], [1, 2]])
    with pytest.raises(ConfigError):
        e.prepare(case.model, capacity=0)
    with pytest.raises(ConfigError):
        e.prepare(case.model, capacity=2, batches=[list(range(16))])
    assert e.prepare(case.model, capacity=1000).n_batches == 1


# ============================================================================================
# (d) zero temperature, neutral moves, RNG streams, recording independence
# ============================================================================================
def _neutral_case():
    """Exactly-3-of-6 with an objective on spins 0..2 only (dyadic weights): exchanges among spins
    3..5 that change the state are exactly neutral (v = 0)."""
    structure = ObjectiveStructure(n_spins=6, terms=[(0,), (1,), (0, 1), (1, 2), (0, 1, 2)])
    w = np.array([0.5, -0.25, 1.0, 0.75, -0.5])
    lib = B.pair_exchanges(range(6))
    model = compile_model(structure, lib, identities=cardinality_identities(6, 3))
    return structure, w, lib, model, bind_weights(model, w, 0.0), exact_cardinality_states(6, 3)


def test_zero_temperature_rule_and_neutral_moves():
    structure, w, lib, model, bnd, Om = _neutral_case()
    e, o = get_backend("emulation"), get_backend("oracle")
    pe = e.prepare(model, validate="full", library=lib)
    po = o.prepare(model, library=lib)
    spins0 = Om[[0, 5, 11, 19]]
    se = e.init_state(pe, bnd, spins0)
    so = o.init_state(po, bnd, spins0)
    sched = S.zero(3000)
    neutral_changed = uphill = 0
    for n in range(3000):
        re = e.step(se, n, sched, 5)
        ro = o.step(so, n, sched, 5)
        assert np.array_equal(re.accepted, re.v <= 0.0)  # limiting rule: accept iff v <= 0
        assert np.all(re.xi == 0.0)
        assert np.array_equal(re.accepted, ro.accepted) and np.array_equal(re.v, ro.v)  # exact (dyadic weights)
        neutral_changed += int(np.count_nonzero(re.changed & (re.v == 0.0)))
        uphill += int(np.count_nonzero(re.accepted & (re.v > 0)))
    assert neutral_changed > 0  # neutral transitions are accepted and recorded as state changes
    assert uphill == 0
    assert np.all(se.rs.counters["uphill_accepted"] == 0)
    E = np.array([bnd.const[0] - bnd.J[0] @ model.clause_values(s).astype(float) for s in se.rs.spins])
    E_all = np.array([bnd.const[0] - bnd.J[0] @ model.clause_values(s).astype(float) for s in Om])
    assert np.all(E >= E_all.min() - 1e-12)
    assert np.array_equal(se.rs.counters["accepted"],
                          se.rs.counters["accepted_changed"] + se.rs.counters["self_loops"])


def test_rng_streams_reproduce_and_separate():
    case = case_cardinality_random()
    sched = mixing_schedule(case, 1500)
    e = get_backend("emulation")
    plan = e.prepare(case.model)
    s0 = case.initial(4, 9)
    a = e.run(plan, e.init_state(plan, case.bindings, s0, [0, 0, 1, 1]), sched, seed=17)
    b = e.run(plan, e.init_state(plan, case.bindings, s0, [0, 0, 1, 1]), sched, seed=17)
    _same_state(a.state, b.state)
    c = e.run(plan, e.init_state(plan, case.bindings, s0, [0, 0, 1, 1]), sched, seed=18)
    assert not np.array_equal(a.state.counters["accepted"], c.state.counters["accepted"])
    # the stream depends on (replica_id, instance_id), not on the replica's row position
    d = e.run(plan, e.init_state(plan, case.bindings, s0[::-1], [1, 1, 0, 0], [3, 2, 1, 0], [1, 1, 0, 0]),
              sched, seed=17)
    _same_state(a.state, _reverse(d.state))


def _reverse(rs: ReplicaStates) -> ReplicaStates:
    out = rs.copy()
    for name in ("T", "v", "energies", "best_energy", "best_spins", "best_attempt", "next_attempt"):
        setattr(out, name, getattr(rs, name)[::-1].copy())
    out.counters = {k: v[::-1].copy() for k, v in rs.counters.items()}
    return out


@pytest.mark.parametrize("builder", [case_cardinality_toy, case_section29])
def test_recording_profiles_do_not_change_trajectories(builder):
    case = builder()
    sched = mixing_schedule(case, 1200)
    e = get_backend("emulation")
    plan = e.prepare(case.model)
    s0 = case.initial(3, 4)
    results = {}
    for prof in ("none", "summary", "anytime", "events", "trace"):
        rec = {"profile": prof, "trace_snapshots": ("spins", "T", "v")} if prof == "trace" else prof
        results[prof] = e.run(plan, e.init_state(plan, case.bindings[0], s0), sched, seed=21, recording=rec)
    for prof in results:
        _same_state(results["none"].state, results[prof].state)
    assert results["none"].recording.events is None and results["none"].recording.trace is None
    assert results["events"].recording.events is not None and results["events"].recording.trace is None
    tr = results["trace"].recording.trace
    assert tr.n_records == 3 * 1200 and not tr.truncated
    # validation mode does not change trajectories either (it never draws from the solver stream)
    pv = e.prepare(case.model, validate="sampled", validate_rate=0.3)
    rv = e.run(pv, e.init_state(pv, case.bindings[0], s0), sched, seed=21)
    _same_state(results["none"].state, rv.state)


# ============================================================================================
# (e) event-counter identities against independent recomputation
# ============================================================================================
@pytest.mark.parametrize("builder", [case_cardinality_toy, lambda: case_onehot(4, "complete", 3, True),
                                     case_section29, case_table_kempe])
def test_event_counter_identities(builder):
    case = builder()
    model = case.model
    sched = mixing_schedule(case, 800)
    e = get_backend("emulation")
    plan = e.prepare(case.model, validate="full", library=case.library)
    st = e.init_state(plan, case.bindings[0], case.initial(3, 6))
    rp = model.route_ptr
    wmask = model.weighted_route_mask()
    role = model.clause_role
    obj = (role & ROLE_OBJECTIVE) != 0
    gate_only = ((role & ROLE_GATE) != 0) & ~obj
    readout = (role & 4) != 0
    wfan = np.diff(model.clause_wroute_ptr)
    cfan = np.diff(model.ctrl_route_ptr)
    ctrl = model.route_ctrl.astype(np.int64)
    tot = {k: 0 for k in EVENT_COUNTERS}
    exp = {k: 0 for k in ("toggles", "on", "off", "readout", "decoder_reads", "clause_route_events",
                          "clause_route_corr", "reevals", "switch_changes", "switch_w", "gate_changes",
                          "sign_and_switch", "spin_changes", "transactions", "changes", "toggles_gate")}
    for n in range(800):
        T0, s0 = st.rs.T.copy(), st.rs.spins.copy()
        al0 = model.route_alpha(T0)
        rec = e.step(st, n, sched, 31, collect_events=True)
        T1, s1 = st.rs.T, st.rs.spins
        al1 = model.route_alpha(T1)
        for k in EVENT_COUNTERS:
            tot[k] += int(rec.events[k].sum())
        for i in np.flatnonzero(rec.accepted):
            r = int(rec.group[i])
            tog = np.flatnonzero(T0[i] != T1[i])
            exp["transactions"] += 1
            exp["changes"] += int(rec.changed[i])
            exp["toggles"] += tog.size
            exp["toggles_gate"] += int(gate_only[tog].sum())
            exp["on"] += int(np.count_nonzero(T0[i, tog] == -1))
            exp["off"] += int(np.count_nonzero(T0[i, tog] == 1))
            exp["readout"] += int(readout[tog].sum())
            exp["decoder_reads"] += int(rp[r + 1] - rp[r])
            exp["spin_changes"] += int(np.count_nonzero(s0[i] != s1[i]))
            tw = tog[obj[tog]]
            exp["clause_route_events"] += int(wfan[tw].sum())
            hit = np.isin(model.route_clause, tw) & wmask
            exp["clause_route_corr"] += int(np.count_nonzero(hit & (al0[i] == 1)))
            exp["reevals"] += int(cfan[tog].sum())
            dal = al0[i] != al1[i]
            exp["switch_changes"] += int(dal.sum())
            exp["switch_w"] += int((dal & wmask).sum())
            exp["gate_changes"] += int((dal & (ctrl >= 0) & gate_only[np.maximum(ctrl, 0)]).sum())
            exp["sign_and_switch"] += int((dal & np.isin(model.route_clause, tog)).sum())
            assert rec.n_toggles[i] == tog.size and rec.n_switch_changes[i] == int(dal.sum())
    # identities with independently recomputed quantities
    assert tot["transactions"] == exp["transactions"] and tot["state_changes"] == exp["changes"]
    assert tot["clause_toggles"] == exp["toggles"] == tot["onoff_on_events"] + tot["onoff_off_events"]
    assert tot["onoff_on_events"] == exp["on"] and tot["onoff_off_events"] == exp["off"]
    assert tot["clause_toggles"] == (tot["toggles_objective"] + tot["toggles_gate"] + tot["toggles_readout"]
                                     + tot["toggles_score"])
    assert tot["toggles_gate"] == exp["toggles_gate"]
    assert tot["readout_toggles"] == exp["readout"]
    assert tot["decoder_routes_read"] == exp["decoder_reads"]
    assert tot["spin_changes"] == exp["spin_changes"]
    assert tot["clause_route_events"] == exp["clause_route_events"]
    assert tot["clause_route_corrections"] == exp["clause_route_corr"]
    assert tot["switch_reevaluations"] == exp["reevals"] == tot["switch_changes"] == exp["switch_changes"]
    assert tot["switch_changes"] == tot["switch_changes_ctrl_gate"] + tot["switch_changes_ctrl_objective"]
    assert tot["switch_changes_ctrl_gate"] == exp["gate_changes"]
    assert tot["switch_route_events"] == exp["switch_w"]
    assert tot["input_corrections"] == tot["clause_route_corrections"] + tot["switch_route_events"]
    assert tot["sign_and_switch_routes"] == exp["sign_and_switch"]
    assert tot["full_scan_routes"] == tot["state_changes"] * int(wmask.sum())
    assert tot["clause_toggles"] > 0 and tot["input_corrections"] > 0
    rcs = model.readout_clause_of_spin
    if np.all(rcs >= 0) and np.all(readout[rcs]):  # full singleton readout: one readout toggle per changed spin
        assert tot["readout_toggles"] == tot["spin_changes"] > 0


def test_event_histograms_and_run_counters():
    case = case_cardinality_toy()
    sched = mixing_schedule(case, 2000)
    e = get_backend("emulation")
    plan = e.prepare(case.model)
    res = e.run(plan, e.init_state(plan, case.bindings, case.initial(4, 3), [0, 0, 1, 1]), sched, seed=8,
                recording={"profile": "events", "event_hist_bins": 400})
    ev = res.recording.events
    c = res.state.counters
    assert np.array_equal(ev.counter("transactions"), c["accepted"])
    assert np.array_equal(ev.counter("state_changes"), c["accepted_changed"])
    for h in ("clause_toggles", "switch_changes", "input_corrections"):
        H = ev.histogram(h)
        assert np.array_equal(H.sum(axis=1), c["accepted_changed"])
        assert H[:, -1].sum() == 0  # no overflow with 400 bins
    vals = np.arange(400)
    assert np.array_equal(ev.histogram("clause_toggles") @ vals, ev.counter("clause_toggles"))
    assert np.array_equal(ev.histogram("switch_changes") @ vals, ev.counter("switch_changes"))
    assert np.array_equal(ev.histogram("input_corrections") @ vals, ev.counter("input_corrections"))
    # bounded histograms: a small cap folds the tail into the overflow bin
    small = e.run(plan, e.init_state(plan, case.bindings, case.initial(4, 3), [0, 0, 1, 1]), sched, seed=8,
                  recording={"profile": "events", "event_hist_bins": 4}).recording.events
    assert small.hist.shape[-1] == 4 and np.array_equal(small.histogram("clause_toggles").sum(axis=1),
                                                         c["accepted_changed"])
    assert small.histogram("clause_toggles")[:, 3].sum() > 0
    # basic counter identities
    assert np.array_equal(c["attempts"], c["accepted"] + c["rejected"])
    assert np.array_equal(c["accepted"], c["accepted_changed"] + c["self_loops"])
    assert np.all(c["uphill_accepted"] <= c["accepted_changed"])
    assert np.all(c["attempts"] == 2000)


# ============================================================================================
# ON-OFF pair, validation modes, state import/export
# ============================================================================================
def test_onoff_pair_tracks_clause_signs():
    case = case_section29()
    e = get_backend("emulation")
    plan = e.prepare(case.model, onoff=True)
    st = e.init_state(plan, case.bindings[0], case.initial(3, 2))
    e.run(plan, st, mixing_schedule(case, 1500), seed=4)
    ex = e.export_state(st)
    Tp, Tm = ex.extras["Tplus"].astype(int), ex.extras["Tminus"].astype(int)
    assert np.array_equal(Tp - Tm, ex.T) and np.all(Tp + Tm == 1)  # Supplementary Information, ON-OFF neuromorphic realization
    assert "Tplus" not in e.export_state(e.init_state(e.prepare(case.model), case.bindings[0], case.initial(1))).extras


def test_sampled_validation_and_explicit_check():
    case = case_onehot(4, "complete", 3, True)
    e = get_backend("emulation")
    plan = e.prepare(case.model, validate="sampled", validate_rate=0.25, library=case.library)
    st = e.init_state(plan, case.bindings[0], case.initial(2, 1))
    e.run(plan, st, mixing_schedule(case, 1000), seed=2)
    e.validate_state(st)
    # corrupt one maintained input: the explicit check reports the identity, group and values
    st.rs.v[0, 1] += 0.5
    with pytest.raises(ArchitectureValidationError) as info:
        e.validate_state(st)
    assert info.value.context["identity"] == "2 v_r == E(mu_r(s)) - E(s)" and info.value.context["groups"] == [1]
    st.rs.v[0, 1] -= 0.5
    st.alpha[1, 0] ^= 1
    with pytest.raises(ArchitectureValidationError, match="switch states"):
        e.validate_state(st)
    with pytest.raises(ConfigError):
        e.prepare(case.model, validate="sometimes")
    with pytest.raises(ConfigError):
        e.prepare(case.model, validate="sampled", validate_rate=0.0)


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    case = case_cardinality_random()
    sched = mixing_schedule(case, 2000)
    e = get_backend("emulation")
    for opts in ({}, {"capacity": 5}):
        plan = e.prepare(case.model, **opts)
        s0 = case.initial(4, 8)
        full = e.run(plan, e.init_state(plan, case.bindings, s0, [0, 1, 0, 1]), sched, seed=1)
        st = e.init_state(plan, case.bindings, s0, [0, 1, 0, 1])
        r1 = e.run(plan, st, sched, seed=1, stop=700)
        assert r1.termination == ("chunk",) * 4 and np.all(r1.state.next_attempt == 700)
        path = r1.state.save(tmp_path / f"ckpt{len(opts)}")
        back = ReplicaStates.load(path)
        assert back.bindings == ()
        st2 = e.load_state(plan, back, bindings=case.bindings)
        r2 = e.run(plan, st2, sched, seed=1, stop=1300)
        r3 = e.run(plan, st2, sched, seed=1)
        assert r3.termination == ("budget",) * 4
        _same_state(full.state, r3.state)
        # resuming at the wrong attempt index is refused (no repeated or skipped attempts)
        with pytest.raises(ConfigError):
            e.run(plan, e.load_state(plan, r2.state), sched, seed=1, start=0)
    # rebuilding caches from a snapshot without emulator extras reproduces the trajectory (exact weights)
    case = case_cardinality_toy()
    sched = mixing_schedule(case, 1000)
    plan = e.prepare(case.model, validate="full", library=case.library)
    s0 = case.initial(2, 1)
    full = e.run(plan, e.init_state(plan, case.bindings[0], s0), sched, seed=5)
    half = e.run(plan, e.init_state(plan, case.bindings[0], s0), sched, seed=5, stop=500).state
    half.extras = {}
    half.v = None
    rest = e.run(plan, e.load_state(plan, half), sched, seed=5)
    _same_state(full.state, rest.state)


def test_load_state_rejects_mismatches():
    case = case_cardinality_toy()
    e = get_backend("emulation")
    plan = e.prepare(case.model)
    st = e.init_state(plan, case.bindings[0], case.initial(2))
    snap = e.export_state(st)
    with pytest.raises(IncompatibleCheckpoint):
        e.load_state(plan, snap, bindings=[case.bindings[1]])
    bad = snap.copy()
    bad.T[0, 0] *= -1
    with pytest.raises(IncompatibleCheckpoint):
        e.load_state(plan, bad)
    other = case_cardinality_random()
    with pytest.raises((IncompatibleCheckpoint, ValueError)):
        e.load_state(e.prepare(other.model), snap, bindings=other.bindings[:1])


def test_init_state_checks():
    case = case_cardinality_toy()
    e = get_backend("emulation")
    plan = e.prepare(case.model)
    bad = np.array([1, 1, 1, 1, -1, -1], dtype=np.int8)  # 4 selected: infeasible
    with pytest.raises(InfeasibleState):
        e.init_state(plan, case.bindings[0], bad, is_feasible=case.is_feasible)
    with pytest.raises(ConfigError):
        e.init_state(plan, case.bindings[0], np.zeros(6, np.int8))
    with pytest.raises(ConfigError):
        e.init_state(plan, case.bindings[0], case.initial(2), replica_id=[0, 0])  # shared solver stream
    with pytest.raises(BackendUnavailable):
        e.prepare(case.model, device="cuda:0")
    with pytest.raises(ConfigError):
        e.prepare(case.model, precision="float32")
    with pytest.raises(ConfigError):
        e.prepare(case.model, strategy="recompute")


def test_targets_and_stop_on_target():
    case = case_cardinality_toy()
    sched = mixing_schedule(case, 3000)
    e = get_backend("emulation")
    plan = e.prepare(case.model)
    s0 = case.initial(4, 12)
    E_opt = case.energy(0, case.states).min()
    res = e.run(plan, e.init_state(plan, case.bindings[0], s0), sched, seed=6, recording="trace", targets=E_opt)
    hit = res.state.counters["target_hit_attempt"]
    tr = res.recording.trace.columns()
    E0 = case.energy(0, s0)
    for i in range(4):
        rows = tr["replica"] == i
        reached = np.flatnonzero(tr["energy"][rows] <= E_opt + 1e-9)
        want = 0 if E0[i] <= E_opt + 1e-9 else (tr["attempt"][rows][reached[0]] + 1 if reached.size else -1)
        assert hit[i] == want
    assert np.any(hit > 0)
    st = e.init_state(plan, case.bindings[0], s0)
    stopped = e.run(plan, st, sched, seed=6, targets=E_opt, stop_on_target=True)
    for i in range(4):
        if hit[i] > 0:
            assert stopped.termination[i] == "target" and stopped.state.next_attempt[i] == hit[i]
            assert stopped.state.counters["attempts"][i] == hit[i]
            assert abs(stopped.state.energies[i, 0] - E_opt) < 1e-9
    again = e.run(plan, st, sched, seed=6)
    assert all(t in ("stopped", "budget") for t in again.termination)
    # an initial state already at the target hits at attempt count 0
    best = case.states[[np.argmin(case.energy(0, case.states))]]
    r0 = e.run(plan, e.init_state(plan, case.bindings[0], best, targets=E_opt), sched, seed=1, stop=10)
    assert r0.state.counters["target_hit_attempt"][0] == 0


# ============================================================================================
# performance (~1e5 attempts on a P=8 one-hot assignment model)
# ============================================================================================
def test_cyclic_assignment_p8_throughput_and_validation():
    P = 8
    d = random_distances(P, 1)
    structure, w, c = cyclic_assignment_objective(d)
    lib = B.row_exchanges(P, "complete")
    model = compile_model(structure, lib, identities=onehot_identities(P))
    bnd = bind_weights(model, w, c)
    s0 = -np.ones(P * P, np.int8)
    for p, a in enumerate(np.random.default_rng(0).permutation(P)):
        s0[B.onehot_index(a, p, P)] = 1
    e = get_backend("emulation")
    plan = e.prepare(model)
    st = e.init_state(plan, bnd, s0)
    t = time.perf_counter()
    res = e.run(plan, st, S.fn_endpoints(20.0, 1.0, 50.0, 30000), seed=1)
    dt = time.perf_counter() - t
    assert res.state.counters["attempts"][0] == 30000 and res.state.counters["accepted_changed"][0] > 100
    assert dt < 30.0, f"emulation too slow: {dt:.1f} s for 3e4 attempts"
    e.validate_state(e.load_state(e.prepare(model, validate="full"), res.state))  # settled state is exact
    x = res.state.spins[0].reshape(P, P)
    tour = [int(np.flatnonzero(x[:, p] == 1)[0]) for p in range(P)]
    assert abs(tour_length(d, tour) - res.state.energies[0, 0]) < 1e-9
    assert np.all(x.sum(axis=0) == -(P - 2)) and np.all(x.sum(axis=1) == -(P - 2))  # still a permutation

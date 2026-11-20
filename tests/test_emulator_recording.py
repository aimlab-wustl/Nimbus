"""Recording profiles: spec semantics, anytime checkpoints and
improvement caps, trace sampling and byte cap with truncation metadata, event logs across chunks,
serialization to dict/numpy, and that disabled diagnostics are not computed."""

from __future__ import annotations

import json

import numpy as np
import pytest

from test_replay_core import case_cardinality_toy, case_section29, mixing_schedule
from nimbus.backends import base as base_mod
from nimbus.backends import emulator as emu_mod
from nimbus.backends import get_backend
from nimbus.core.errors import ConfigError
from nimbus.experiments.recording import (
    PROFILES,
    AnytimeLog,
    EventLog,
    RecordingData,
    RecordingSpec,
    TraceLog,
    log_checkpoints,
)


def test_spec_profiles_and_switches():
    want = {"none": (0, 0, 0), "summary": (0, 0, 0), "anytime": (1, 0, 0), "events": (1, 1, 0), "trace": (1, 1, 1)}
    for p in PROFILES:
        s = RecordingSpec(profile=p)
        assert (s.want_anytime, s.want_events, s.want_trace) == tuple(map(bool, want[p]))
        assert not RecordingSpec(profile=p, collect=False).want_anytime
    assert not RecordingSpec(profile="none").should_persist and RecordingSpec().should_persist
    assert not RecordingSpec(persist=False).should_persist
    assert RecordingSpec(profile="summary", events=True).want_events
    assert not RecordingSpec(profile="trace", events=False).want_events
    assert not RecordingSpec(profile="trace", collect=False, trace=True).want_trace
    assert RecordingSpec.coerce(None).profile == "summary"
    assert RecordingSpec.coerce("events").want_events
    s = RecordingSpec(profile="trace", trace_snapshots=("spins", "v"), trace_replicas=[2, 0], anytime_checkpoints=[50, 5])
    assert s.trace_replicas == (0, 2) and s.anytime_checkpoints == (5, 50)
    back = RecordingSpec.coerce(json.loads(json.dumps(s.to_dict())))
    assert back == s
    for bad in ({"profile": "verbose"}, {"trace_snapshots": ("energy",)}, {"event_hist_bins": 1},
                {"anytime_checkpoints": (0,)}, {"trace_every": 0}):
        with pytest.raises(ConfigError):
            RecordingSpec(**bad)
    with pytest.raises(ConfigError):
        RecordingSpec.from_dict({"profile": "summary", "colour": "red"})
    with pytest.raises(ConfigError):
        RecordingSpec.coerce(3.5)


def test_log_checkpoints():
    for B, k in ((1, 10), (10, 64), (1000, 16), (123457, 40)):
        c = log_checkpoints(B, k)
        assert c[0] >= 1 and c[-1] == B and np.all(np.diff(c) > 0) and c.size <= max(k, 1) + 1
    assert log_checkpoints(0, 5).size == 0
    assert RecordingSpec(anytime_checkpoints=(3, 7, 500)).checkpoints(100).tolist() == [3, 7]


def _run(case, recording, B=1500, seed=5, n_rep=3, **kw):
    e = get_backend("emulation")
    plan = e.prepare(case.model)
    st = e.init_state(plan, case.bindings[0], case.initial(n_rep, 4))
    return e, plan, st, e.run(plan, st, mixing_schedule(case, B), seed=seed, recording=recording, **kw)


def test_anytime_checkpoints_and_improvements():
    case = case_cardinality_toy()
    e, plan, st, res = _run(case, {"profile": "trace", "anytime_points": 20})
    any_ = res.recording.anytime
    tr = res.recording.trace.columns()
    cp = any_.checkpoints
    assert cp[-1] == 1500 and np.all(np.isfinite(any_.best))
    for i in range(3):
        e_i = tr["energy"][tr["replica"] == i]  # slot-0 energy after attempt n (index n)
        assert np.allclose(any_.current[i], e_i[cp - 1])
        E0 = case.energy(0, case.initial(3, 4)[i:i + 1])[0]
        run_best = np.minimum.accumulate(np.concatenate([[E0], e_i]))
        assert np.allclose(any_.best[i], run_best[cp], atol=1e-9)
        assert np.all(np.diff(any_.best[i]) <= 1e-12)
        att, en = any_.improvements(i)
        assert np.all(np.diff(en) < 0)
        if att.size:
            assert att[-1] == res.state.best_attempt[i] and en[-1] == pytest.approx(res.state.best_energy[i])
        else:
            assert res.state.best_attempt[i] == 0
    # chunked: checkpoints outside the chunk stay NaN, and the two chunks together cover all
    e2 = get_backend("emulation")
    st2 = e2.init_state(plan, case.bindings[0], case.initial(3, 4))
    sched = mixing_schedule(case, 1500)
    a = e2.run(plan, st2, sched, seed=5, stop=600, recording={"profile": "anytime", "anytime_points": 20})
    b = e2.run(plan, st2, sched, seed=5, recording={"profile": "anytime", "anytime_points": 20})
    ina = cp <= 600
    assert np.all(np.isfinite(a.recording.anytime.best[:, ina])) and np.all(np.isnan(a.recording.anytime.best[:, ~ina]))
    assert np.all(np.isnan(b.recording.anytime.best[:, ina])) and np.all(np.isfinite(b.recording.anytime.best[:, ~ina]))
    merged = np.where(ina[None, :], a.recording.anytime.best, b.recording.anytime.best)
    assert np.allclose(merged, any_.best)


def test_improvement_cap_truncation():
    log = AnytimeLog(2, [1, 10], improvement_cap=3)
    for k in range(5):
        log.record_improvement(0, k + 1, -float(k))
    att, en = log.improvements(0)
    assert att.tolist() == [1, 2, 3] and log.improvements_dropped.tolist() == [2, 0] and log.truncated
    back = AnytimeLog.from_arrays(log.to_arrays())
    assert back.improvements(0)[0].tolist() == [1, 2, 3] and back.improvements_dropped.tolist() == [2, 0]
    back2 = AnytimeLog.from_dict(json.loads(json.dumps(log.to_dict())))
    assert np.array_equal(back2.best, log.best, equal_nan=True)


def test_trace_sampling_byte_cap_and_metadata():
    case = case_section29()
    spec = {"profile": "trace", "trace_every": 3, "trace_first": 900, "trace_replicas": (1,), "trace_snapshots": ("T",)}
    e, plan, st, res = _run(case, spec)
    tr = res.recording.trace
    cols = tr.columns()
    assert set(cols["replica"].tolist()) == {1}
    assert np.all(cols["attempt"] % 3 == 0) and cols["attempt"].max() < 900
    assert tr.n_records == 300 and not tr.truncated
    assert cols["snap_T"].shape == (300, case.model.n_clauses)
    # the trace's selected inputs equal the lock-step step() records
    e2 = get_backend("emulation")
    st2 = e2.init_state(plan, case.bindings[0], case.initial(3, 4))
    sched = mixing_schedule(case, 1500)
    recs = {n: e2.step(st2, n, sched, 5) for n in range(900)}
    for j, n in enumerate(cols["attempt"].tolist()):
        r = recs[n]
        assert cols["group"][j] == r.group[1] and cols["v"][j] == r.v[1] and cols["accepted"][j] == r.accepted[1]
        assert cols["u"][j] == r.u[1] and cols["xi"][j] == r.xi[1]
    # byte cap: records beyond the cap are dropped and counted, never silently
    cap_spec = {"profile": "trace", "trace_max_bytes": 5000, "trace_snapshots": ("spins", "T", "v")}
    _, _, _, res2 = _run(case, cap_spec)
    t2 = res2.recording.trace
    meta = t2.metadata()
    assert t2.truncated and meta["bytes_used"] <= 5000 and meta["n_records"] == 5000 // t2.record_bytes
    assert meta["n_dropped"] == 3 * 1500 - meta["n_records"] and meta["first_dropped_attempt"] >= 0
    acc_only = _run(case, {"profile": "trace", "trace_accepted_only": True})[3].recording.trace.columns()
    assert acc_only["accepted"].all() and acc_only["accepted"].size == int(res.state.counters["accepted"].sum())


def test_recording_data_serialization(tmp_path):
    case = case_cardinality_toy()
    _, _, _, res = _run(case, {"profile": "trace", "trace_snapshots": ("spins",), "anytime_points": 8})
    rec = res.recording
    d = json.loads(json.dumps(rec.to_dict()))
    back = RecordingData.from_dict(d)
    assert back.spec == rec.spec
    assert np.array_equal(back.events.counts, rec.events.counts) and np.array_equal(back.events.hist, rec.events.hist)
    assert np.allclose(back.anytime.best, rec.anytime.best)
    assert np.array_equal(back.trace.columns()["v"], rec.trace.columns()["v"])
    np.savez(tmp_path / "rec.npz", **rec.to_arrays())
    with np.load(tmp_path / "rec.npz", allow_pickle=False) as z:
        back2 = RecordingData.from_arrays({k: z[k] for k in z.files})
    assert back2.events.counter_names == rec.events.counter_names
    assert np.array_equal(back2.events.counts, rec.events.counts)
    assert np.array_equal(back2.trace.columns()["snap_spins"], rec.trace.columns()["snap_spins"])
    assert back2.trace.metadata() == rec.trace.metadata()
    assert np.array_equal(back2.anytime.checkpoints, rec.anytime.checkpoints)
    tl = TraceLog.from_dict(json.loads(json.dumps(rec.trace.to_dict())))
    assert tl.n_records == rec.trace.n_records


def test_event_logs_merge_across_chunks():
    case = case_cardinality_toy()
    e = get_backend("emulation")
    plan = e.prepare(case.model)
    sched = mixing_schedule(case, 1500)
    s0 = case.initial(3, 4)
    full = e.run(plan, e.init_state(plan, case.bindings[0], s0), sched, seed=5, recording="events").recording.events
    st = e.init_state(plan, case.bindings[0], s0)
    parts = [e.run(plan, st, sched, seed=5, stop=s, recording="events").recording.events for s in (400, 1100, 1500)]
    merged = parts[0].merge(parts[1]).merge(parts[2])
    assert np.array_equal(merged.counts, full.counts) and np.array_equal(merged.hist, full.hist)
    with pytest.raises(ConfigError):
        merged.merge(EventLog(3, ("a",)))
    assert full.meta["weighted_routes_total"] == int(case.model.weighted_route_mask().sum())


def test_disabled_diagnostics_are_not_computed(monkeypatch):
    """With the default profile no event log, anytime log or trace object is ever created."""
    def boom(*a, **k):
        raise AssertionError("a disabled diagnostic was constructed")

    monkeypatch.setattr(emu_mod, "EventLog", boom)
    monkeypatch.setattr(base_mod, "AnytimeLog", boom)
    monkeypatch.setattr(base_mod, "TraceLog", boom)
    case = case_cardinality_toy()
    for prof in ("none", "summary", {"profile": "trace", "collect": False}):
        _, _, _, res = _run(case, prof, B=300)
        assert res.recording.events is None and res.recording.anytime is None and res.recording.trace is None
    with pytest.raises(AssertionError, match="disabled diagnostic"):
        _run(case, "events", B=10)

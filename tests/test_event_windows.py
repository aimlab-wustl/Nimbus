"""Per-window event counters of the CPU select kernels.

Checks: (1) dynamics are unchanged: states, best states, energies and every ordinary counter are bit-identical with
and without windows (and with event counters off entirely); (2) window sums equal the run totals for every counter
that has a total; (3) per-window attempts / accepted / neutral / fan-in equal an independent per-attempt trace of the
same trajectory; (4) a run split into two calls (state hand-over cutting a window) gives the same window arrays.
"""
from __future__ import annotations

import numpy as np
import pytest

from nimbus.annealing import schedule as S
from nimbus.backends import _numba_kernels as K
from nimbus.backends.base import get_backend
from nimbus.compiler import compile_model
from nimbus.core.model import bind_weights
from nimbus.problems.matching import MatchingProblem

from test_matching import _tensor

B = 3000


def _case():
    t, _ = _tensor(5, seed=4)
    prob = MatchingProblem(t)
    lib = prob.transition_library("path", double_exchanges={"count": 2, "seed": 1})
    model = compile_model(prob.structure(), lib, identities=prob.identities())
    binding = bind_weights(model, prob.term_weights(), prob.const())
    return model, binding, prob.random_feasible(3, 6)


def _run(model, binding, spins0, *, splits=(0, B), recording=None, **prep):
    be = get_backend("cpu")
    plan = be.prepare(model, **prep)
    st = be.init_state(plan, [binding], spins0)
    sched = S.geometric(3.0, 0.05, B)
    res = None
    for a, b in zip(splits[:-1], splits[1:]):
        res = be.run(plan, st, sched, n_attempts_total=B, start=a, stop=b, seed=9, recording=recording)
    return be, st, res


def _same_dynamics(r1, r2, ev=True):
    a, b = r1.state, r2.state
    for k in ("T", "best_spins", "best_attempt", "best_energy", "energies", "feasible_best_energy", "next_attempt"):
        assert np.array_equal(np.asarray(getattr(a, k)), np.asarray(getattr(b, k))), k
    for k in a.counters:
        if k.startswith("ev_") and not ev:
            continue
        assert np.array_equal(np.asarray(a.counters[k]), np.asarray(b.counters[k])), k


def test_log_windows_edges():
    e = K.log_windows(2 ** 16, per_doubling=8, min_width=16)
    assert e[0] == 0 and e[-1] == 2 ** 16 and np.all(np.diff(e) > 0)
    assert np.all(np.diff(e)[:-1] >= 16)
    # log part: ratio of consecutive edges ~ 2^(1/8)
    r = e[-20:-1][1:] / e[-20:-1][:-1]
    assert np.allclose(r, 2 ** 0.125, rtol=0.01)
    assert np.array_equal(K.linear_windows(100, 4), [0, 25, 50, 75, 100])


@pytest.mark.parametrize("strategy,layout", [("recompute", "gate_blocked"), ("incremental", "flat"), ("lazy", "gate_blocked")])
def test_windows_bit_identical_and_sums(strategy, layout):
    model, binding, spins0 = _case()
    edges = K.log_windows(B, per_doubling=8, min_width=32)
    _, _, r_plain = _run(model, binding, spins0, strategy=strategy, layout=layout)
    _, _, r_ev = _run(model, binding, spins0, strategy=strategy, layout=layout, event_counters=True)
    be, st, r_win = _run(model, binding, spins0, strategy=strategy, layout=layout, event_counters=True, event_windows=edges)
    _same_dynamics(r_plain, r_win, ev=False)
    _same_dynamics(r_ev, r_win, ev=True)
    w = be.window_counters(st)
    c = r_win.state.counters
    for k in ("attempts", "accepted", "accepted_changed", "uphill_accepted"):
        assert np.array_equal(w[k].sum(axis=1), np.asarray(c[k])), k
    for k in K.EVENT_NAMES:
        assert np.array_equal(w[k].sum(axis=1), np.asarray(c["ev_" + k])), k
    assert w["clause_toggles"].sum() > 0 and w["input_corrections"].sum() > 0
    # best energy at the last window = run best; monotone non-increasing
    assert np.array_equal(w["best_energy"][:, -1], np.asarray(r_win.state.best_energy))
    assert np.all(np.diff(w["best_energy"], axis=1) <= 0)


def test_windows_equal_independent_trace():
    model, binding, spins0 = _case()
    edges = K.log_windows(B, per_doubling=4, min_width=64)
    be, st, r_win = _run(model, binding, spins0, strategy="recompute", layout="gate_blocked", event_windows=edges)
    w = be.window_counters(st)
    _, _, r_tr = _run(model, binding, spins0, strategy="recompute", layout="gate_blocked", recording="trace")
    cols = r_tr.recording.trace.columns()
    fan_w = model.route_wend - model.route_ptr[:-1]
    fan_all = np.diff(model.route_ptr)
    from nimbus.backends.base import UPHILL_RTOL
    tol = UPHILL_RTOL * float(st.layout.scale[0, 0])  # the kernel's uphill/neutral tolerance (binding 0)
    win_of = np.searchsorted(edges, cols["attempt"], side="right") - 1
    for rep in range(spins0.shape[0]):
        m = cols["replica"] == rep
        wi, g = win_of[m], cols["group"][m]
        acc, chg, v = cols["accepted"][m], cols["changed"][m], cols["v"][m]
        nw = edges.size - 1
        assert np.array_equal(np.bincount(wi, minlength=nw), w["attempts"][rep])
        assert np.array_equal(np.bincount(wi, weights=acc, minlength=nw).astype(np.int64), w["accepted"][rep])
        assert np.array_equal(np.bincount(wi, weights=acc & chg, minlength=nw).astype(np.int64), w["accepted_changed"][rep])
        assert np.array_equal(np.bincount(wi, weights=fan_w[g], minlength=nw).astype(np.int64), w["fanin_weighted"][rep])
        assert np.array_equal(np.bincount(wi, weights=fan_all[g], minlength=nw).astype(np.int64), w["fanin_routes"][rep])
        neutral = acc & chg & (np.abs(v) <= tol)
        assert np.array_equal(np.bincount(wi, weights=neutral, minlength=nw).astype(np.int64), w["neutral_accepted"][rep])


def test_windows_split_run_and_errors():
    model, binding, spins0 = _case()
    edges = K.log_windows(B, per_doubling=8, min_width=32)
    be1, st1, r1 = _run(model, binding, spins0, strategy="recompute", layout="gate_blocked", event_windows=edges)
    cut = int((edges[10] + edges[11]) // 2)  # inside a window
    be2, st2, r2 = _run(model, binding, spins0, splits=(0, cut, B), strategy="recompute", layout="gate_blocked",
                        event_windows=edges)
    _same_dynamics(r1, r2)
    w1, w2 = be1.window_counters(st1), be2.window_counters(st2)
    for k in w1:
        assert np.array_equal(w1[k], w2[k], equal_nan=True) if w1[k].dtype.kind == "f" else np.array_equal(w1[k], w2[k]), k
    with pytest.raises(Exception):
        _run(model, binding, spins0, strategy="recompute", layout="gate_blocked", event_windows=edges, recording="trace")
    with pytest.raises(Exception):
        get_backend("cpu").prepare(model, event_windows=[5, 3])

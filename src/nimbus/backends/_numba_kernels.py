"""Numba (nopython) kernels for the optimized CPU architecture backend (``nimbus.backends.cpu``).

These kernels implement exactly the transaction of the architecture, mirroring ``nimbus.backends.emulator``
(see its module docstring) with three interchangeable *strategies* for computing the transition-
neuron input ``v_r`` and two *layouts* for how a group's routes are visited:

* strategy ``recompute`` (0): ``v_r`` is recomputed on every attempt from the weighted routes of
  the *selected* group only (Methods, Derivation of the transition-programmable architecture); nothing is maintained between attempts.
* strategy ``incremental`` (1): ``v`` is maintained for every group (array ``v_row``) and an
  explicit per-route switch cache ``alpha_row`` is maintained alongside it; on acceptance, stage 1
  (clause-sign events, old switches) and stage 2 (switch re-evaluation, new clause signs) apply the
  exact corrections of the transaction / Supplementary Information, Exact local maintenance of move-neuron inputs, mirroring
  ``EmulatorBackend._commit`` / ``_propagate_switches``.
* strategy ``lazy`` (2): ``v`` is maintained but entries touched by an accepted transaction are
  marked stale (``NaN``) instead of corrected; a stale entry is recomputed (and cached) the next
  time its group is read. This trades the exact algebra of ``incremental`` for a cheaper,
  conservative mark (every weighted route of a toggled objective clause, and every weighted route
  controlled by a toggled clause, regardless of whether its switch actually changed).

Layouts choose how a group's routes are visited when a fresh sum *is* needed (recompute always;
incremental only once at initialization; lazy when a stale group is read):

* layout ``flat`` (0): the model's own per-group CSR order (weighted routes ``[route_ptr[r],
  route_wend[r])``, all routes ``[route_ptr[r], route_ptr[r+1])``); every route's controller sign is
  read individually.
* layout ``gate_blocked`` (1): routes are pre-grouped (at plan build time, ``cpu.py``) by their
  *controlling (clause, polarity)* pair; a closed block's controller sign is read once and the
  whole block is skipped (a backend plan may reorder routes within a group
  provided v_r, sigma and the toggles are unchanged).

Both layouts give bit-identical ``v_r`` and toggle sets; only the traversal cost differs. Coding rule: every
array index in these loops is cast to a signed 64-bit "intp" at the Python/numpy boundary in
``cpu.py`` and used as plain Python ints inside ``@njit`` loops, so Numba never inserts
negative-index wraparound checks (that cast is what the comparison measured, not an unsigned dtype
inside the kernel, which numba's range()/indexing does not need).

Every function is ``@njit(cache=True)``; the per-attempt kernel ``attempt_kernel`` is also called,
attempt by attempt, by the plain-Python bookkeeping path (``cpu.py``'s ``_run_sequential``, used
for recording profiles beyond ``summary`` and for ``CPUBackend.step``), so there is exactly one
implementation of the architecture transaction. The ``prange`` bulk path (``run_block_prange``)
re-implements only the counters/best/feasible-best bookkeeping of ``RunBook`` in array form (no
Python objects under ``nogil``); its numeric results must equal the sequential path exactly, and
both are checked against the oracle and the emulator in ``tests/test_cpu_backend.py``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from nimbus.backends._numba_common import schedule_tau, solver_draw

STRAT_RECOMPUTE, STRAT_INCREMENTAL, STRAT_LAZY = 0, 1, 2
LAYOUT_FLAT, LAYOUT_BLOCKED = 0, 1
NAN = np.nan

__all__ = [
    "STRAT_RECOMPUTE", "STRAT_INCREMENTAL", "STRAT_LAZY", "LAYOUT_FLAT", "LAYOUT_BLOCKED",
    "group_v_flat", "group_kt_flat", "group_v_blocked", "group_kt_blocked",
    "toggles_spin_clause", "commit_update", "apply_energy_update", "attempt_kernel", "attempt_given",
    "run_block_prange", "event_counts_commit", "EVENT_NAMES",
]


# ============================================================================================
# group traversal: flat layout (model's own per-group CSR order)
# ============================================================================================
@njit(cache=True, inline="always")
def group_v_flat(r, T, J0, route_ptr, route_wend, route_clause, route_ctrl, route_pol):
    """v_r = sum over weighted routes [route_ptr[r], route_wend[r]) of J0[k] alpha T[k]."""
    v = 0.0
    a, b = route_ptr[r], route_wend[r]
    for p in range(a, b):
        k = route_clause[p]
        al = (1 + route_pol[p] * T[route_ctrl[p]]) >> 1
        if al:
            v += J0[k] * T[k]
    return v


@njit(cache=True, inline="always")
def group_kt_flat(r, T, route_ptr, route_ctrl, route_pol, route_clause, kt_buf):
    """Latch sigma over ALL routes [route_ptr[r], route_ptr[r+1]) into kt_buf; returns the count."""
    a, b = route_ptr[r], route_ptr[r + 1]
    n = 0
    for p in range(a, b):
        al = (1 + route_pol[p] * T[route_ctrl[p]]) >> 1
        if al:
            kt_buf[n] = route_clause[p]
            n += 1
    return n


# ============================================================================================
# group traversal: gate-blocked layout (routes pre-grouped by (ctrl, pol) within each group)
# ============================================================================================
@njit(cache=True, inline="always")
def group_v_blocked(r, T, J0, blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, plan_route_clause):
    v = 0.0
    for b in range(blk_group_ptr[r], blk_group_ptr[r + 1]):
        al = (1 + blk_pol[b] * T[blk_ctrl[b]]) >> 1
        if al:
            for p in range(blk_start[b], blk_wend[b]):
                k = plan_route_clause[p]
                v += J0[k] * T[k]
    return v


@njit(cache=True, inline="always")
def group_kt_blocked(r, T, blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_end, plan_route_clause, kt_buf):
    n = 0
    for b in range(blk_group_ptr[r], blk_group_ptr[r + 1]):
        al = (1 + blk_pol[b] * T[blk_ctrl[b]]) >> 1
        if al:
            for p in range(blk_start[b], blk_end[b]):
                kt_buf[n] = plan_route_clause[p]
                n += 1
    return n


# ============================================================================================
# state change: spins are stored singleton clauses updated by the decoder like every
# other clause, so a transaction changed the state iff a latched toggle hits a spin singleton.
# There is no separate spin-update path.
# ============================================================================================
@njit(cache=True, inline="always")
def toggles_spin_clause(kt_buf, kt_n, is_spin_clause):
    """True iff the latched toggle list contains a spin singleton clause (the state changes)."""
    for i in range(kt_n):
        if is_spin_clause[kt_buf[i]]:
            return True
    return False


# ============================================================================================
# commit: clause toggles + (incremental) exact propagation / (lazy) conservative stale marks
# ============================================================================================
@njit(cache=True, inline="always")
def commit_update(strategy_code, T_row, alpha_row, v_row, kt_buf, kt_n, told_buf, J0,
                  route_clause, route_ctrl, route_pol, route_group, route_weighted,
                  is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route):
    """Capture Told, toggle T[Kt], then update the maintained input per ``strategy_code``."""
    for i in range(kt_n):
        told_buf[i] = T_row[kt_buf[i]]
    for i in range(kt_n):
        k = kt_buf[i]
        T_row[k] = -T_row[k]
    if strategy_code == STRAT_INCREMENTAL:
        # stage 1 (Supplementary Information, Exact local maintenance of move-neuron inputs): old switches, new - old clause sign
        for i in range(kt_n):
            k = kt_buf[i]
            if is_obj[k]:
                dT = T_row[k] - told_buf[i]
                for q in range(clause_wroute_ptr[k], clause_wroute_ptr[k + 1]):
                    p = clause_wroute[q]
                    if alpha_row[p]:
                        v_row[route_group[p]] += J0[k] * dT
        # stage 2 (Supplementary Information, Exact local maintenance of move-neuron inputs): re-evaluate switches controlled by a toggled clause
        for i in range(kt_n):
            k = kt_buf[i]
            if has_ctrl[k]:
                for q in range(ctrl_route_ptr[k], ctrl_route_ptr[k + 1]):
                    p = ctrl_route[q]
                    a_old = alpha_row[p]
                    a_new = (1 + route_pol[p] * T_row[k]) >> 1
                    if a_new != a_old:
                        if route_weighted[p]:
                            kc = route_clause[p]
                            v_row[route_group[p]] += J0[kc] * T_row[kc] * (a_new - a_old)
                        alpha_row[p] = a_new
    elif strategy_code == STRAT_LAZY:
        for i in range(kt_n):
            k = kt_buf[i]
            if is_obj[k]:
                for q in range(clause_wroute_ptr[k], clause_wroute_ptr[k + 1]):
                    p = clause_wroute[q]
                    v_row[route_group[p]] = NAN
            if has_ctrl[k]:
                for q in range(ctrl_route_ptr[k], ctrl_route_ptr[k + 1]):
                    p = ctrl_route[q]
                    if route_weighted[p]:
                        v_row[route_group[p]] = NAN
    # STRAT_RECOMPUTE: nothing is maintained


@njit(cache=True, inline="always")
def apply_energy_update(J_row, kt_buf, told_buf, kt_n, energies_row):
    """energies_row[s] += 2 * sum_i J_row[s, kt_buf[i]] * told_buf[i], for every slot s
    (``E[slot] += 2 sum_{k in Kt} J[slot,k] T_k^old``). The single
    implementation used by the prange bulk path, the sequential path and ``CPUBackend.step``."""
    n_slots = J_row.shape[0]
    for s in range(n_slots):
        acc = 0.0
        for i in range(kt_n):
            acc += J_row[s, kt_buf[i]] * told_buf[i]
        energies_row[s] += 2.0 * acc


# ============================================================================================
# exact per-run event counters (opt-in, ``CPUBackend.prepare(event_counters=True)``)
# ============================================================================================
#: counter order of the ``ev_row`` vector filled by :func:`event_counts_commit`; names and definitions
#: equal ``nimbus.backends.emulator.EVENT_COUNTERS`` (events profile) so the two can be compared exactly.
EVENT_NAMES = ("clause_toggles", "toggles_objective", "toggles_gate", "clause_route_events",
               "clause_route_corrections", "switch_reevaluations", "switch_changes", "switch_route_events",
               "input_corrections")


@njit(cache=True, inline="always")
def event_counts_commit(kt_buf, kt_n, T_row, is_obj, is_gate_only, has_ctrl, clause_wroute_ptr, clause_wroute,
                        ctrl_route_ptr, ctrl_route, route_ctrl, route_pol, route_weighted, ev_row):
    """Add the event counts of ONE committed transaction to ``ev_row`` (int64[len(EVENT_NAMES)]).

    Called after ``commit_update`` (``T_row`` holds the NEW clause signs, ``kt_buf[:kt_n]`` the toggled
    clauses). Strategy independent: the old switch of a route is rebuilt from the old controller sign
    (the toggled clauses are flipped back, counted, and flipped forward again), the new one from the
    new sign, so no maintained switch cache is needed. Definitions (emulator ``_commit``):

    * stage 1 (clause-route events): every weighted route of a toggled objective clause is an event;
      it is a correction iff its OLD switch is open;
    * stage 2: every route controlled by a toggled clause is re-evaluated; the controller sign flipped,
      so its switch always changes; the change is a weighted switch-route event (an input correction)
      iff the route is weighted.
    """
    ev_row[0] += kt_n
    for i in range(kt_n):
        k = kt_buf[i]
        if is_obj[k]:
            ev_row[1] += 1
        elif is_gate_only[k]:
            ev_row[2] += 1
    for i in range(kt_n):  # back to the OLD signs
        k = kt_buf[i]
        T_row[k] = -T_row[k]
    n_ev = 0
    n_corr = 0
    for i in range(kt_n):
        k = kt_buf[i]
        if is_obj[k]:
            for q in range(clause_wroute_ptr[k], clause_wroute_ptr[k + 1]):
                p = clause_wroute[q]
                n_ev += 1
                if ((1 + route_pol[p] * T_row[route_ctrl[p]]) >> 1) != 0:
                    n_corr += 1
    for i in range(kt_n):  # forward to the NEW signs
        k = kt_buf[i]
        T_row[k] = -T_row[k]
    n_reeval = 0
    n_wsw = 0
    for i in range(kt_n):
        k = kt_buf[i]
        if has_ctrl[k]:
            for q in range(ctrl_route_ptr[k], ctrl_route_ptr[k + 1]):
                n_reeval += 1
                if route_weighted[ctrl_route[q]]:
                    n_wsw += 1
    ev_row[3] += n_ev
    ev_row[4] += n_corr
    ev_row[5] += n_reeval
    ev_row[6] += n_reeval
    ev_row[7] += n_wsw
    ev_row[8] += n_corr + n_wsw


# ============================================================================================
# one attempt of one replica (shared by the sequential/bookkeeping path and the prange bulk path)
# ============================================================================================
@njit(cache=True)
def attempt_given(r, u, strategy_code, layout_code,
                  T_row, alpha_row, v_row, kt_buf, told_buf,
                  J0, tau,
                  route_ptr, route_wend, route_clause, route_ctrl, route_pol, route_group, route_weighted,
                  blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, blk_end, plan_route_clause,
                  is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route,
                  is_spin_clause):
    """One attempt with a given group r and acceptance uniform u (read v, accept/reject, commit).
    Returns (xi, v, accepted, changed, kt_n). The same body as ``attempt_kernel`` after its solver
    draw (kept as a separate copy so that the select kernel's compiled code is untouched); used by the
    shared/sweep processes, which draw r from their own streams."""
    if strategy_code == STRAT_RECOMPUTE:
        cached = NAN
    else:
        cached = v_row[r]
    if cached == cached:  # not NaN: maintained value is valid (incremental always; lazy if fresh)
        v = cached
    else:
        if layout_code == LAYOUT_FLAT:
            v = group_v_flat(r, T_row, J0, route_ptr, route_wend, route_clause, route_ctrl, route_pol)
        else:
            v = group_v_blocked(r, T_row, J0, blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend,
                                plan_route_clause)
        if strategy_code == STRAT_LAZY:
            v_row[r] = v
    if tau > 0.0:
        xi = -(tau / 2.0) * math.log(u)
        accepted = v <= xi
    else:
        xi = 0.0
        accepted = v <= 0.0
    if not accepted:
        return xi, v, False, False, 0
    if layout_code == LAYOUT_FLAT:
        kt_n = group_kt_flat(r, T_row, route_ptr, route_ctrl, route_pol, route_clause, kt_buf)
    else:
        kt_n = group_kt_blocked(r, T_row, blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_end,
                                plan_route_clause, kt_buf)
    changed = toggles_spin_clause(kt_buf, kt_n, is_spin_clause)
    if kt_n:
        commit_update(strategy_code, T_row, alpha_row, v_row, kt_buf, kt_n, told_buf, J0,
                      route_clause, route_ctrl, route_pol, route_group, route_weighted,
                      is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route)
    return xi, v, True, changed, kt_n


@njit(cache=True)
def attempt_kernel(seed, n, replica_id, instance_id, R, strategy_code, layout_code,
                   T_row, alpha_row, v_row, kt_buf, told_buf,
                   J0, tau,
                   route_ptr, route_wend, route_clause, route_ctrl, route_pol, route_group, route_weighted,
                   blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, blk_end, plan_route_clause,
                   is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route,
                   is_spin_clause):
    """One attempt: draw, read v, accept/reject, commit. Returns (r, u, xi, v, accepted, changed, kt_n).

    The commit toggles every latched clause, T <- (1 - 2 sigma) T, spin singletons included; that is
    the whole state update (no spin array)."""
    r, u = solver_draw(seed, n, replica_id, instance_id, R)
    if strategy_code == STRAT_RECOMPUTE:
        cached = NAN
    else:
        cached = v_row[r]
    if cached == cached:  # not NaN: maintained value is valid (incremental always; lazy if fresh)
        v = cached
    else:
        if layout_code == LAYOUT_FLAT:
            v = group_v_flat(r, T_row, J0, route_ptr, route_wend, route_clause, route_ctrl, route_pol)
        else:
            v = group_v_blocked(r, T_row, J0, blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend,
                                plan_route_clause)
        if strategy_code == STRAT_LAZY:
            v_row[r] = v
    if tau > 0.0:
        xi = -(tau / 2.0) * math.log(u)
        accepted = v <= xi
    else:
        xi = 0.0
        accepted = v <= 0.0
    if not accepted:
        return r, u, xi, v, False, False, 0
    if layout_code == LAYOUT_FLAT:
        kt_n = group_kt_flat(r, T_row, route_ptr, route_ctrl, route_pol, route_clause, kt_buf)
    else:
        kt_n = group_kt_blocked(r, T_row, blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_end,
                                plan_route_clause, kt_buf)
    changed = toggles_spin_clause(kt_buf, kt_n, is_spin_clause)
    if kt_n:
        commit_update(strategy_code, T_row, alpha_row, v_row, kt_buf, kt_n, told_buf, J0,
                      route_clause, route_ctrl, route_pol, route_group, route_weighted,
                      is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route)
    return r, u, xi, v, True, changed, kt_n


# ============================================================================================
# parallel bulk run (summary-profile fast path): counters/best/feasible-best in array form
# ============================================================================================
@njit(cache=True, parallel=True, nogil=True)
def run_block_prange(seed, a0, a1, replica_id, instance_id, R, strategy_code, layout_code,
                     T, alpha, v, kt_buf, told_buf, J0_of_rep, J_of_rep,
                     sched_kind, sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total,
                     route_ptr, route_wend, route_clause, route_ctrl, route_pol, route_group, route_weighted,
                     blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, blk_end, plan_route_clause,
                     is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route,
                     is_spin_clause, spin_clause, energies, best_energy, best_spins, best_attempt,
                     feasible_best_energy, feasible_best_spins, feasible_best_attempt, targets,
                     stopped, target_hit_attempt, score_slot, viol_slot, has_viol, best_tol, viol_tol,
                     score_tol, target_tol, uphill_tol, cnt_attempts, cnt_accepted, cnt_accepted_changed,
                     cnt_self_loops, cnt_rejected, cnt_uphill_accepted, stop_on_target, next_attempt,
                     ev_on, ev_counts, is_gate_only):
    """``RunBook``-equivalent bookkeeping done in plain arrays, one replica per thread (no GIL, no
    Python objects): same best/feasible-best/target/counter semantics as ``base.RunBook.on_change``
    and the counters of ``EmulatorBackend.step`` (base module docstring).
    Best assignments are read out from the singleton clauses (``spin_clause``) of the clause state.
    """
    n_rep = T.shape[0]
    for rep in prange(n_rep):
        if stopped[rep]:
            continue
        rid, iid = replica_id[rep], instance_id[rep]
        T_row, alpha_row, v_row = T[rep], alpha[rep], v[rep]
        kt_row, told_row = kt_buf[rep], told_buf[rep]
        J0 = J0_of_rep[rep]
        att = a0
        for n in range(a0, a1):
            tau = schedule_tau(sched_kind, sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total, n)
            r, u, xi, vr, accepted, changed, kt_n = attempt_kernel(
                seed, n, rid, iid, R, strategy_code, layout_code, T_row, alpha_row, v_row,
                kt_row, told_row, J0, tau, route_ptr, route_wend, route_clause, route_ctrl,
                route_pol, route_group, route_weighted, blk_group_ptr, blk_ctrl, blk_pol, blk_start,
                blk_wend, blk_end, plan_route_clause, is_obj, has_ctrl, clause_wroute_ptr, clause_wroute,
                ctrl_route_ptr, ctrl_route, is_spin_clause)
            cnt_attempts[rep] += 1
            att = n + 1
            if not accepted:
                cnt_rejected[rep] += 1
                continue
            cnt_accepted[rep] += 1
            # energy from the decoder toggles alone (no-op when sigma is empty)
            if kt_n > 0:
                apply_energy_update(J_of_rep[rep], kt_row, told_row, kt_n, energies[rep])
                if ev_on:
                    event_counts_commit(kt_row, kt_n, T_row, is_obj, is_gate_only, has_ctrl, clause_wroute_ptr,
                                        clause_wroute, ctrl_route_ptr, ctrl_route, route_ctrl, route_pol,
                                        route_weighted, ev_counts[rep])
            if not changed:
                cnt_self_loops[rep] += 1
                continue
            cnt_accepted_changed[rep] += 1
            if vr > uphill_tol[rep]:
                cnt_uphill_accepted[rep] += 1
            e0 = energies[rep, 0]
            if e0 < best_energy[rep] - best_tol[rep]:
                best_energy[rep] = e0
                for i in range(spin_clause.shape[0]):
                    best_spins[rep, i] = T_row[spin_clause[i]]
                best_attempt[rep] = n + 1
            feasible = True
            if has_viol:
                feasible = abs(energies[rep, viol_slot]) <= viol_tol[rep]
            if feasible:
                es = energies[rep, score_slot]
                if es < feasible_best_energy[rep] - score_tol[rep] or feasible_best_attempt[rep] < 0:
                    feasible_best_energy[rep] = es
                    for i in range(spin_clause.shape[0]):
                        feasible_best_spins[rep, i] = T_row[spin_clause[i]]
                    feasible_best_attempt[rep] = n + 1
                t = targets[rep]
                if t == t and target_hit_attempt[rep] < 0 and es <= t + target_tol[rep]:
                    target_hit_attempt[rep] = n + 1
                    if stop_on_target:
                        stopped[rep] = True
                        break
        next_attempt[rep] = att


# ============================================================================================
# per-window event counters (opt-in, ``CPUBackend.prepare(event_counters=True, event_windows=edges)``;
# per-window event counters). A separate copy of ``run_block_prange`` so the
# compiled code of the ordinary fast path is untouched; dynamics are the same calls in the same order
# (bit-identity with and without windows is tested in tests/test_event_windows.py).
# ============================================================================================
#: per-window int64 fields, in this order; then len(EVENT_NAMES) event counters of
#: :func:`event_counts_commit` (same definitions, attributed to the window of the attempt); then the two
#: feasibility-occupancy counters (appended so earlier field positions are unchanged):
#: ``feasible_attempts`` = attempts after which the CURRENT state is feasible (violation slot within
#: tolerance, or no violation slot), ``accepted_changed_feasible`` = accepted state-changing moves whose new
#: state is feasible.
WIN_FIELDS = ("attempts", "accepted", "accepted_changed", "neutral_accepted", "uphill_accepted",
              "fanin_weighted", "fanin_routes") + EVENT_NAMES + ("feasible_attempts", "accepted_changed_feasible")
WIN_EV0 = 7
WIN_FEAS0 = WIN_EV0 + len(EVENT_NAMES)


@njit(cache=True, parallel=True, nogil=True)
def run_block_prange_windowed(seed, a0, a1, replica_id, instance_id, R, strategy_code, layout_code,
                              T, alpha, v, kt_buf, told_buf, J0_of_rep, J_of_rep,
                              sched_kind, sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total,
                              route_ptr, route_wend, route_clause, route_ctrl, route_pol, route_group, route_weighted,
                              blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, blk_end, plan_route_clause,
                              is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route,
                              is_spin_clause, spin_clause, energies, best_energy, best_spins, best_attempt,
                              feasible_best_energy, feasible_best_spins, feasible_best_attempt, targets,
                              stopped, target_hit_attempt, score_slot, viol_slot, has_viol, best_tol, viol_tol,
                              score_tol, target_tol, uphill_tol, cnt_attempts, cnt_accepted, cnt_accepted_changed,
                              cnt_self_loops, cnt_rejected, cnt_uphill_accepted, stop_on_target, next_attempt,
                              ev_counts, is_gate_only, win_edges, win, win_best, win_fbest):
    """``run_block_prange`` (event counters on) plus per-window accumulation: attempt ``n`` (global
    index) belongs to window ``w`` with ``win_edges[w] <= n < win_edges[w + 1]``; ``win[rep, w, :]``
    accumulates :data:`WIN_FIELDS`, ``win_best[rep, w]`` / ``win_fbest[rep, w]`` hold the best energy /
    feasible-best score after the last attempt run inside the window. Attempts outside the edges are
    run and counted in the totals but not in any window.

    neutral_accepted: accepted, state changed and ``|v| <= uphill_tol`` (dE = 0 to rounding; uphill =
    ``v > uphill_tol`` as the ordinary counter). fanin_weighted / fanin_routes: weighted routes / all
    routes of the PROPOSED group, summed over attempts (the recompute-selected fan-in).
    feasible_attempts / accepted_changed_feasible: see :data:`WIN_FIELDS` (read-only, no effect on dynamics)."""
    n_rep = T.shape[0]
    n_win = win_edges.shape[0] - 1
    n_ev = ev_counts.shape[1]
    for rep in prange(n_rep):
        if stopped[rep]:
            continue
        rid, iid = replica_id[rep], instance_id[rep]
        T_row, alpha_row, v_row = T[rep], alpha[rep], v[rep]
        kt_row, told_row = kt_buf[rep], told_buf[rep]
        J0 = J0_of_rep[rep]
        ev_row = ev_counts[rep]
        prev = np.empty(n_ev, dtype=np.int64)
        w = -1
        w_lo = 1
        w_hi = 0  # empty: forces a lookup at the first attempt
        att = a0
        for n in range(a0, a1):
            if n < w_lo or n >= w_hi:
                w = np.searchsorted(win_edges, n, side="right") - 1
                if w >= 0 and w < n_win:
                    w_lo, w_hi = win_edges[w], win_edges[w + 1]
                else:
                    w = -1
                    w_lo, w_hi = n, n + 1
            in_win = w >= 0
            tau = schedule_tau(sched_kind, sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total, n)
            r, u, xi, vr, accepted, changed, kt_n = attempt_kernel(
                seed, n, rid, iid, R, strategy_code, layout_code, T_row, alpha_row, v_row,
                kt_row, told_row, J0, tau, route_ptr, route_wend, route_clause, route_ctrl,
                route_pol, route_group, route_weighted, blk_group_ptr, blk_ctrl, blk_pol, blk_start,
                blk_wend, blk_end, plan_route_clause, is_obj, has_ctrl, clause_wroute_ptr, clause_wroute,
                ctrl_route_ptr, ctrl_route, is_spin_clause)
            cnt_attempts[rep] += 1
            att = n + 1
            if in_win:
                win[rep, w, 0] += 1
                win[rep, w, 5] += route_wend[r] - route_ptr[r]
                win[rep, w, 6] += route_ptr[r + 1] - route_ptr[r]
            if accepted:
                cnt_accepted[rep] += 1
                if in_win:
                    win[rep, w, 1] += 1
                if kt_n > 0:
                    apply_energy_update(J_of_rep[rep], kt_row, told_row, kt_n, energies[rep])
                    for i in range(n_ev):
                        prev[i] = ev_row[i]
                    event_counts_commit(kt_row, kt_n, T_row, is_obj, is_gate_only, has_ctrl, clause_wroute_ptr,
                                        clause_wroute, ctrl_route_ptr, ctrl_route, route_ctrl, route_pol,
                                        route_weighted, ev_row)
                    if in_win:
                        for i in range(n_ev):
                            win[rep, w, WIN_EV0 + i] += ev_row[i] - prev[i]
                if not changed:
                    cnt_self_loops[rep] += 1
                else:
                    cnt_accepted_changed[rep] += 1
                    if in_win:
                        win[rep, w, 2] += 1
                        if abs(vr) <= uphill_tol[rep]:
                            win[rep, w, 3] += 1
                    if vr > uphill_tol[rep]:
                        cnt_uphill_accepted[rep] += 1
                        if in_win:
                            win[rep, w, 4] += 1
                    e0 = energies[rep, 0]
                    if e0 < best_energy[rep] - best_tol[rep]:
                        best_energy[rep] = e0
                        for i in range(spin_clause.shape[0]):
                            best_spins[rep, i] = T_row[spin_clause[i]]
                        best_attempt[rep] = n + 1
                    feasible = True
                    if has_viol:
                        feasible = abs(energies[rep, viol_slot]) <= viol_tol[rep]
                    if feasible:
                        es = energies[rep, score_slot]
                        if es < feasible_best_energy[rep] - score_tol[rep] or feasible_best_attempt[rep] < 0:
                            feasible_best_energy[rep] = es
                            for i in range(spin_clause.shape[0]):
                                feasible_best_spins[rep, i] = T_row[spin_clause[i]]
                            feasible_best_attempt[rep] = n + 1
                        t = targets[rep]
                        if t == t and target_hit_attempt[rep] < 0 and es <= t + target_tol[rep]:
                            target_hit_attempt[rep] = n + 1
                            if stop_on_target:
                                stopped[rep] = True
            else:
                cnt_rejected[rep] += 1
            if in_win:
                win_best[rep, w] = best_energy[rep]
                win_fbest[rep, w] = feasible_best_energy[rep]
                f_now = True  # feasibility of the CURRENT state after this attempt (read only, no dynamics)
                if has_viol:
                    f_now = abs(energies[rep, viol_slot]) <= viol_tol[rep]
                if f_now:
                    win[rep, w, WIN_FEAS0] += 1
                    if accepted and changed:
                        win[rep, w, WIN_FEAS0 + 1] += 1
            if stopped[rep]:
                break
        next_attempt[rep] = att


def log_windows(n_total: int, per_doubling: int = 8, min_width: int = 1, start: int = 0) -> np.ndarray:
    """Window edges ``start = e_0 < e_1 < ... < e_K = n_total`` for the per-window counters: log-spaced
    (``per_doubling`` windows per factor 2 of the attempt index) with a linear fallback where the log
    spacing would be narrower than ``min_width`` attempts (early in the run): there the edges step by
    ``min_width``. Integer, strictly increasing, last edge = ``n_total``."""
    n_total, min_width, start = int(n_total), max(1, int(min_width)), int(start)
    f = 2.0 ** (1.0 / per_doubling)
    edges = [start]
    x = float(max(start, 0))
    while True:
        nxt = max(x * f, x + min_width)
        e = int(round(nxt))
        if e <= edges[-1]:
            e = edges[-1] + 1
        if e >= n_total:
            break
        edges.append(e)
        x = float(e)
    edges.append(n_total)
    return np.asarray(edges, dtype=np.int64)


def linear_windows(n_total: int, n_windows: int, start: int = 0) -> np.ndarray:
    """Equal-width window edges (the linear alternative to :func:`log_windows`)."""
    return np.unique(np.linspace(int(start), int(n_total), int(n_windows) + 1).round().astype(np.int64))

"""Numba kernels of the proposal processes on the CPU backend.

The architecture step is unchanged in every process: v_r from stored clauses and
switches, single commit, latched sigma over all routes of the committed group, T <- (1 - 2 sigma) T
(spin singletons included), no spin array. Only the choice of the committed group and the clock
differ. Structures:

* ``shared`` / ``sweep`` - per-replica path: the select attempt (``_numba_kernels.attempt_given``,
  any strategy/layout) with the group drawn from the SHARED stream (cohort sequence) or from the
  replica's (or cohort's) sweep permutation, and the acceptance uniform from the replica's SOLVER draw
  (words x2, x3). One replica per ``prange`` lane (``run_attempts_prange``).
* ``shared`` - lock-step path (``run_shared_lockstep``): the replicas of one cohort advance together;
  clause signs are stored replica-minor (``T3[lane, k, j]``, j = replica within the lane), so one read
  of the group's route arrays serves every replica and the inner loop over replicas is contiguous
  (SIMD). v_r is recomputed (flat route order, branch-free: sum of J_k T_k alpha), bit-identical to the
  per-replica ``recompute``/``flat`` value. Lanes (a cohort, or a block of it) run in parallel.
* ``parallel_trial`` / ``nfold`` - every v_r settled at every step: maintained inputs (the incremental
  stage 1/2 algebra, latch in the model's flat route order) plus an exact per-group count of open
  spin-singleton routes (``open_cnt``; group r changes the state iff open_cnt[r] > 0), updated in
  stage 2 whenever a singleton route's switch changes. Each step is one scan over R (fired test or
  a_r with a sequential prefix sum) and at most one commit. One replica per ``prange`` lane.

Counter rows of the stacked counter array ``cnt`` (int64[N_CNT, n_rep]) are the ``C_*`` constants.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from nimbus.backends._numba_common import (
    color_uniform,
    nfold_uniforms,
    pt_arbiter,
    pt_uniform,
    schedule_tau,
    shared_group,
    solver_uniform,
    sweep_permutation_into,
)
from nimbus.backends._numba_kernels import apply_energy_update, attempt_given

MODE_SHARED, MODE_SWEEP = 1, 2
PROC_PT, PROC_NFOLD = 0, 1
OUT_COMMIT, OUT_NOFIRE, OUT_FROZEN, OUT_OVERSHOOT = 0, 1, 2, 3
C_ATT, C_ACC, C_CHG, C_SELF, C_REJ, C_UP, C_STEPS, C_CAND, C_FROZEN, C_HIT = range(10)
N_CNT = 10
CNT_NAMES = ("attempts", "accepted", "accepted_changed", "self_loops", "rejected", "uphill_accepted", "steps",
             "candidate_evaluations", "frozen_attempt", "target_hit_attempt")
T_BEST, T_VIOL, T_SCORE, T_TARGET, T_UPHILL = range(5)


# ============================================================================================
# bookkeeping after a state change (array form of base.RunBook.on_change)
# ============================================================================================
@njit(cache=True)
def book_change(rep, count, T_row, spin_clause, energies, best_energy, best_spins, best_attempt,
                fb_energy, fb_spins, fb_attempt, targets, cnt, tol, score_slot, viol_slot, has_viol):
    """Best / feasible-best / target after a state change reached at attempt count ``count``; True when
    the target is hit for the first time. T_row may be strided (replica-minor lock-step layout)."""
    e0 = energies[rep, 0]
    if e0 < best_energy[rep] - tol[T_BEST, rep]:
        best_energy[rep] = e0
        for i in range(spin_clause.shape[0]):
            best_spins[rep, i] = T_row[spin_clause[i]]
        best_attempt[rep] = count
    if has_viol:
        if not abs(energies[rep, viol_slot]) <= tol[T_VIOL, rep]:
            return False
    es = energies[rep, score_slot]
    if es < fb_energy[rep] - tol[T_SCORE, rep] or fb_attempt[rep] < 0:
        fb_energy[rep] = es
        for i in range(spin_clause.shape[0]):
            fb_spins[rep, i] = T_row[spin_clause[i]]
        fb_attempt[rep] = count
    t = targets[rep]
    if t == t and cnt[C_HIT, rep] < 0 and es <= t + tol[T_TARGET, rep]:
        cnt[C_HIT, rep] = count
        return True
    return False


# ============================================================================================
# shared / sweep: per-replica path
# ============================================================================================
@njit(cache=True)
def process_draw(mode, seed, n, rid, iid, cid, R, perm_row, perm_sweep, rep):
    """(r, u) of attempt n: r from the SHARED stream (cohort cid) or the sweep permutation of sweep
    n // R (cached in perm_row, its sweep index in perm_sweep[rep]); u from the replica's SOLVER draw."""
    u = solver_uniform(seed, n, rid, iid)
    if mode == MODE_SHARED:
        return shared_group(seed, n, cid, iid, R), u
    s = n // R
    if perm_sweep[rep] != s:
        sweep_permutation_into(seed, s, cid, iid, perm_row)
        perm_sweep[rep] = s
    return perm_row[n - s * R], u


@njit(cache=True, parallel=True, nogil=True)
def run_attempts_prange(mode, seed, a0, a1, replica_id, instance_id, cohort_id, R, strategy_code, layout_code,
                        T, alpha, v, kt_buf, told_buf, perm_buf, perm_sweep, J0_of_rep, J_of_rep,
                        sched_kind, sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total,
                        route_ptr, route_wend, route_clause, route_ctrl, route_pol, route_group, route_weighted,
                        blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, blk_end, plan_route_clause,
                        is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route,
                        is_spin_clause, spin_clause, energies, best_energy, best_spins, best_attempt,
                        fb_energy, fb_spins, fb_attempt, targets, stopped, cnt, tol, score_slot, viol_slot,
                        has_viol, stop_on_target, next_attempt):
    """shared / sweep attempts [a0, a1), one replica per prange lane (select's bookkeeping)."""
    n_rep = T.shape[0]
    for rep in prange(n_rep):
        if stopped[rep]:
            continue
        rid, iid, cid = replica_id[rep], instance_id[rep], cohort_id[rep]
        T_row, alpha_row, v_row = T[rep], alpha[rep], v[rep]
        kt_row, told_row = kt_buf[rep], told_buf[rep]
        J0 = J0_of_rep[rep]
        att = a0
        for n in range(a0, a1):
            tau = schedule_tau(sched_kind, sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total, n)
            r, u = process_draw(mode, seed, n, rid, iid, cid, R, perm_buf[rep], perm_sweep, rep)
            xi, vr, accepted, changed, kt_n = attempt_given(
                r, u, strategy_code, layout_code, T_row, alpha_row, v_row, kt_row, told_row, J0, tau,
                route_ptr, route_wend, route_clause, route_ctrl, route_pol, route_group, route_weighted,
                blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, blk_end, plan_route_clause,
                is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route, is_spin_clause)
            cnt[C_ATT, rep] += 1
            cnt[C_STEPS, rep] += 1
            cnt[C_CAND, rep] += 1
            att = n + 1
            if not accepted:
                cnt[C_REJ, rep] += 1
                continue
            cnt[C_ACC, rep] += 1
            if kt_n > 0:
                apply_energy_update(J_of_rep[rep], kt_row, told_row, kt_n, energies[rep])
            if not changed:
                cnt[C_SELF, rep] += 1
                continue
            cnt[C_CHG, rep] += 1
            if vr > tol[T_UPHILL, rep]:
                cnt[C_UP, rep] += 1
            hit = book_change(rep, n + 1, T_row, spin_clause, energies, best_energy, best_spins, best_attempt,
                              fb_energy, fb_spins, fb_attempt, targets, cnt, tol, score_slot, viol_slot, has_viol)
            if hit and stop_on_target:
                stopped[rep] = True
                break
        next_attempt[rep] = att


# ============================================================================================
# colored: one color class per step
# ============================================================================================
@njit(cache=True)
def color_segment(a, a1, R, C, seed, rid, iid, order, class_ptr, class_of_pos, strategy_code, layout_code, T_row,
                  alpha_row, v_row, kt_buf, told_buf, J0, J, energy_row, sched_kind, sched_p0, sched_p1,
                  sched_p2, sched_p3, sched_n_total, route_ptr, route_wend, route_clause, route_ctrl, route_pol,
                  route_group, route_weighted, blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, blk_end,
                  plan_route_clause, is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr,
                  ctrl_route, is_spin_clause, uphill_tol):
    """One color step of one replica starting at attempt clock ``a`` (cut at ``a1``). Every visited group
    draws its own uniform and is tested and committed in turn (ascending r); the class is non-interacting,
    so this equals deciding all groups first and committing the accepted set together. Returns
    (n_do, accepted, changed, self_loops, uphill, step_started)."""
    cyc = a // R
    p = a - cyc * R
    k = class_of_pos[p]
    lo = class_ptr[k]
    hi = class_ptr[k + 1]
    a_start = a - (p - lo)
    step = cyc * C + k
    tau = schedule_tau(sched_kind, sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total, a_start)
    n_do = min(hi - p, a1 - a)
    acc_n = chg_n = slf_n = up_n = 0
    for q in range(p, p + n_do):
        r = order[q]
        u = color_uniform(seed, step, r, rid, iid)
        xi, vr, accepted, changed, kt_n = attempt_given(
            r, u, strategy_code, layout_code, T_row, alpha_row, v_row, kt_buf, told_buf, J0, tau,
            route_ptr, route_wend, route_clause, route_ctrl, route_pol, route_group, route_weighted,
            blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, blk_end, plan_route_clause,
            is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route, is_spin_clause)
        if not accepted:
            continue
        acc_n += 1
        if kt_n > 0:
            apply_energy_update(J, kt_buf, told_buf, kt_n, energy_row)
        if changed:
            chg_n += 1
            if vr > uphill_tol:
                up_n += 1
        else:
            slf_n += 1
    return n_do, acc_n, chg_n, slf_n, up_n, 1 if p == lo else 0


@njit(cache=True, parallel=True, nogil=True)
def run_colored_prange(seed, a0, a1, R, C, replica_id, instance_id, order, class_ptr, class_of_pos, strategy_code,
                       layout_code, T, alpha, v, kt_buf, told_buf, J0_of_rep, J_of_rep,
                       sched_kind, sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total,
                       route_ptr, route_wend, route_clause, route_ctrl, route_pol, route_group, route_weighted,
                       blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, blk_end, plan_route_clause,
                       is_obj, has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route,
                       is_spin_clause, spin_clause, energies, best_energy, best_spins, best_attempt,
                       fb_energy, fb_spins, fb_attempt, targets, stopped, cnt, tol, score_slot, viol_slot,
                       has_viol, stop_on_target, next_attempt):
    """colored attempts [a0, a1) of every replica, one replica per prange lane. Bookkeeping (best, feasible
    best, target) is evaluated once per class, at the clock value where the class ends."""
    n_rep = T.shape[0]
    for rep in prange(n_rep):
        if stopped[rep]:
            continue
        rid, iid = replica_id[rep], instance_id[rep]
        T_row, alpha_row, v_row = T[rep], alpha[rep], v[rep]
        kt_row, told_row = kt_buf[rep], told_buf[rep]
        a = next_attempt[rep]
        while a < a1:
            n_do, acc_n, chg_n, slf_n, up_n, st_n = color_segment(
                a, a1, R, C, seed, rid, iid, order, class_ptr, class_of_pos, strategy_code, layout_code, T_row,
                alpha_row, v_row, kt_row, told_row, J0_of_rep[rep], J_of_rep[rep], energies[rep], sched_kind,
                sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total, route_ptr, route_wend, route_clause,
                route_ctrl, route_pol, route_group, route_weighted, blk_group_ptr, blk_ctrl, blk_pol, blk_start,
                blk_wend, blk_end, plan_route_clause, is_obj, has_ctrl, clause_wroute_ptr, clause_wroute,
                ctrl_route_ptr, ctrl_route, is_spin_clause, tol[T_UPHILL, rep])
            a += n_do
            cnt[C_ATT, rep] += n_do
            cnt[C_STEPS, rep] += st_n
            cnt[C_CAND, rep] += n_do
            cnt[C_ACC, rep] += acc_n
            cnt[C_REJ, rep] += n_do - acc_n
            cnt[C_CHG, rep] += chg_n
            cnt[C_SELF, rep] += slf_n
            cnt[C_UP, rep] += up_n
            if chg_n > 0:
                hit = book_change(rep, a, T_row, spin_clause, energies, best_energy, best_spins, best_attempt,
                                  fb_energy, fb_spins, fb_attempt, targets, cnt, tol, score_slot, viol_slot,
                                  has_viol)
                if hit and stop_on_target:
                    stopped[rep] = True
                    break
        next_attempt[rep] = a


# ============================================================================================
# shared: lock-step path over the replicas of a cohort (replica-minor clause signs)
# ============================================================================================
@njit(cache=True, parallel=True, nogil=True)
def run_shared_lockstep(seed, a0, a1, R, lane_ptr, lane_reps, lane_cid, lane_iid, T3, vbuf, albuf, kt_buf,
                        told_buf, replica_id, J0_of_rep, J_of_rep, sched_kind, sched_p0, sched_p1, sched_p2,
                        sched_p3, sched_n_total, blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, blk_end,
                        plan_route_clause, is_spin_clause, spin_clause, energies, best_energy, best_spins,
                        best_attempt, fb_energy, fb_spins, fb_attempt, targets, stopped, cnt, tol, score_slot,
                        viol_slot, has_viol, stop_on_target, next_attempt):
    """shared attempts [a0, a1) for every lane (replicas of one cohort with one binding) in lock step,
    gate-blocked layout. T3[l, k, j] = clause k of the j-th replica of lane l (k = M: pinned +1 dummy).

    Per attempt: one SHARED draw per lane; per (ctrl, pol) block the switch of every replica of the lane
    (contiguous int8 loads); a block closed for the whole lane is skipped, otherwise its weighted routes
    add J_k T_k alpha to every replica's v (alpha = 0 adds a signed zero, so v is bit-identical to the
    per-replica recompute/gate_blocked sum). Accepted replicas latch their own open blocks (same order
    as ``group_kt_blocked``) and toggle their own column."""
    n_lanes = lane_ptr.shape[0] - 1
    for l in prange(n_lanes):
        j0 = lane_ptr[l]
        w = lane_ptr[l + 1] - j0
        Tl = T3[l]
        vl = vbuf[l]
        al = albuf[l]
        kt_l = kt_buf[l]
        told_l = told_buf[l]
        J0 = J0_of_rep[lane_reps[j0]]
        cid, iid = lane_cid[l], lane_iid[l]
        n_active = 0
        for j in range(w):
            if not stopped[lane_reps[j0 + j]]:
                n_active += 1
        for n in range(a0, a1):
            if n_active == 0:
                break
            tau = schedule_tau(sched_kind, sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total, n)
            r = shared_group(seed, n, cid, iid, R)
            for j in range(w):
                vl[j] = 0.0
            for b in range(blk_group_ptr[r], blk_group_ptr[r + 1]):
                if blk_wend[b] == blk_start[b]:
                    continue
                Tc = Tl[blk_ctrl[b]]
                pol = blk_pol[b]
                n_open = 0
                for j in range(w):
                    a = (1 + pol * Tc[j]) >> 1
                    al[j] = a
                    n_open += a
                if n_open == 0:
                    continue
                for p in range(blk_start[b], blk_wend[b]):
                    k = plan_route_clause[p]
                    jk = J0[k]
                    Tk = Tl[k]
                    for j in range(w):
                        vl[j] += jk * Tk[j] * al[j]
            for j in range(w):
                rep = lane_reps[j0 + j]
                if stopped[rep]:
                    continue
                u = solver_uniform(seed, n, replica_id[rep], iid)
                vr = vl[j]
                cnt[C_ATT, rep] += 1
                cnt[C_STEPS, rep] += 1
                cnt[C_CAND, rep] += 1
                next_attempt[rep] = n + 1
                if tau > 0.0:
                    accepted = vr <= -(tau / 2.0) * math.log(u)
                else:
                    accepted = vr <= 0.0
                if not accepted:
                    cnt[C_REJ, rep] += 1
                    continue
                cnt[C_ACC, rep] += 1
                # latch sigma over the open blocks of r from the pre-transition column, then toggle
                kt_n = 0
                changed = False
                for b in range(blk_group_ptr[r], blk_group_ptr[r + 1]):
                    if (1 + blk_pol[b] * Tl[blk_ctrl[b], j]) >> 1:
                        for p in range(blk_start[b], blk_end[b]):
                            k = plan_route_clause[p]
                            kt_l[kt_n] = k
                            kt_n += 1
                            if is_spin_clause[k]:
                                changed = True
                for i in range(kt_n):
                    k = kt_l[i]
                    told_l[i] = Tl[k, j]
                    Tl[k, j] = -Tl[k, j]
                if kt_n > 0:
                    apply_energy_update(J_of_rep[rep], kt_l, told_l, kt_n, energies[rep])
                if not changed:
                    cnt[C_SELF, rep] += 1
                    continue
                cnt[C_CHG, rep] += 1
                if vr > tol[T_UPHILL, rep]:
                    cnt[C_UP, rep] += 1
                hit = book_change(rep, n + 1, Tl[:, j], spin_clause, energies, best_energy, best_spins,
                                  best_attempt, fb_energy, fb_spins, fb_attempt, targets, cnt, tol, score_slot,
                                  viol_slot, has_viol)
                if hit and stop_on_target:
                    stopped[rep] = True
                    n_active -= 1


# ============================================================================================
# parallel_trial / nfold: maintained inputs, open singleton-route counts, one scan per step
# ============================================================================================
@njit(cache=True)
def pt_decide(seed, n, rid, iid, R, tau, v_row, open_cnt, exclude_self_loops, cand_buf):
    """parallel_trial step n: (committed group or -1, candidates evaluated, |F|). Fired iff v_r <= 0 or
    (tau > 0 and v_r <= -(tau/2) log u_{n,r}); the arbiter draws an index into F (ascending r)."""
    nf = 0
    n_eval = 0
    for r in range(R):
        if exclude_self_loops and open_cnt[r] == 0:
            continue
        n_eval += 1
        x = v_row[r]
        if x <= 0.0:
            cand_buf[nf] = r
            nf += 1
        elif tau > 0.0:
            if x <= -(tau / 2.0) * math.log(pt_uniform(seed, n, r, rid, iid)):
                cand_buf[nf] = r
                nf += 1
    if nf == 0:
        return -1, n_eval, 0
    return cand_buf[pt_arbiter(seed, n, rid, iid, nf)], n_eval, nf


@njit(cache=True)
def nfold_decide(seed, k, n, rid, iid, R, tau, v_row, open_cnt, cand_buf, pref_buf, B, attempts_clock):
    """nfold step k at clock n: (outcome, group, W, candidates evaluated); see
    ``EmulatorBackend._nfold_decide`` for the exact rule (identical expressions and order)."""
    A = 0.0
    m = 0
    for r in range(R):
        if open_cnt[r] == 0:
            continue
        x = v_row[r]
        if x <= 0.0:
            a = 1.0
        elif tau > 0.0:
            a = math.exp(-2.0 * x / tau)
        else:
            a = 0.0
        A += a
        cand_buf[m] = r
        pref_buf[m] = A
        m += 1
    if m == 0 or A == 0.0:
        return OUT_FROZEN, -1, 0, m
    u_sel, u_w = nfold_uniforms(seed, k, rid, iid)
    target = u_sel * A
    lo = 0
    hi = m  # first index with pref > target (pref is non-decreasing)
    while lo < hi:
        mid = (lo + hi) >> 1
        if pref_buf[mid] > target:
            hi = mid
        else:
            lo = mid + 1
    i = lo
    if i == m:
        i = m - 1
        while i > 0 and pref_buf[i] <= pref_buf[i - 1]:
            i -= 1
    r = cand_buf[i]
    if not attempts_clock:
        return OUT_COMMIT, r, 1, m
    p = A / R
    if p >= 1.0:
        return OUT_COMMIT, r, 1, m
    q = math.log(u_w) / math.log1p(-p)
    if q >= np.float64(B - n):
        return OUT_OVERSHOOT, r, B - n, m
    return OUT_COMMIT, r, np.int64(math.floor(q)) + 1, m


@njit(cache=True)
def commit_stepwise(r, T_row, alpha_row, v_row, open_cnt, kt_buf, told_buf, J0, route_ptr, route_clause,
                    route_ctrl, route_pol, route_group, route_weighted, is_obj, has_ctrl, clause_wroute_ptr,
                    clause_wroute, ctrl_route_ptr, ctrl_route, is_spin_clause):
    """Commit group r: latch sigma from the maintained switches in model route order, toggle, stage 1
    (old switches) and stage 2 (new signs) input corrections, and the open singleton-route counts.
    Returns (changed, kt_n). Same operations and order as the emulator's event commit."""
    kt_n = 0
    changed = False
    for p in range(route_ptr[r], route_ptr[r + 1]):
        if alpha_row[p]:
            k = route_clause[p]
            kt_buf[kt_n] = k
            kt_n += 1
            if is_spin_clause[k]:
                changed = True
    for i in range(kt_n):
        told_buf[i] = T_row[kt_buf[i]]
    for i in range(kt_n):
        k = kt_buf[i]
        T_row[k] = -T_row[k]
    # stage 1 (Supplementary Information, Exact local maintenance of move-neuron inputs): old switches, new - old clause sign
    for i in range(kt_n):
        k = kt_buf[i]
        if is_obj[k]:
            dT = T_row[k] - told_buf[i]
            for q in range(clause_wroute_ptr[k], clause_wroute_ptr[k + 1]):
                p = clause_wroute[q]
                if alpha_row[p]:
                    v_row[route_group[p]] += J0[k] * dT
    # stage 2 (Supplementary Information, Exact local maintenance of move-neuron inputs): switches controlled by a toggled clause, with the new signs
    for i in range(kt_n):
        k = kt_buf[i]
        if has_ctrl[k]:
            for q in range(ctrl_route_ptr[k], ctrl_route_ptr[k + 1]):
                p = ctrl_route[q]
                a_old = alpha_row[p]
                a_new = (1 + route_pol[p] * T_row[k]) >> 1
                if a_new != a_old:
                    kc = route_clause[p]
                    if route_weighted[p]:
                        v_row[route_group[p]] += J0[kc] * T_row[kc] * (a_new - a_old)
                    if is_spin_clause[kc]:
                        open_cnt[route_group[p]] += a_new - a_old
                    alpha_row[p] = a_new
    return changed, kt_n


@njit(cache=True)
def refresh_inputs(R, T_row, J0, v_row, oc, need_selfloop_test, sr_ptr, sr_pos, route_ctrl, route_pol,
                   blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, plan_route_clause):
    """Recompute strategy of parallel_trial / nfold: fresh v_r of every (state-changing) group from the
    stored signs (gate-blocked sum, Methods, Derivation of the transition-programmable architecture) and oc[r] = 1 iff r changes the state (first open
    spin-singleton route). Groups that are self-loops get no v when need_selfloop_test
    (they are excluded from the step); otherwise every v_r is computed and oc is still filled."""
    for r in range(R):
        o = 0
        for q in range(sr_ptr[r], sr_ptr[r + 1]):
            p = sr_pos[q]
            if (1 + route_pol[p] * T_row[route_ctrl[p]]) >> 1:
                o = 1
                break
        oc[r] = o
        if o == 0 and need_selfloop_test:
            continue
        v = 0.0
        for b in range(blk_group_ptr[r], blk_group_ptr[r + 1]):
            if (1 + blk_pol[b] * T_row[blk_ctrl[b]]) >> 1:
                for p in range(blk_start[b], blk_wend[b]):
                    k = plan_route_clause[p]
                    v += J0[k] * T_row[k]
        v_row[r] = v


@njit(cache=True)
def commit_toggle_blocked(r, T_row, kt_buf, told_buf, blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_end,
                          plan_route_clause, is_spin_clause):
    """Recompute strategy commit: latch sigma over the open blocks of r (``group_kt_blocked`` order) and
    toggle; nothing is maintained. Returns (changed, kt_n)."""
    kt_n = 0
    changed = False
    for b in range(blk_group_ptr[r], blk_group_ptr[r + 1]):
        if (1 + blk_pol[b] * T_row[blk_ctrl[b]]) >> 1:
            for p in range(blk_start[b], blk_end[b]):
                k = plan_route_clause[p]
                kt_buf[kt_n] = k
                kt_n += 1
                if is_spin_clause[k]:
                    changed = True
    for i in range(kt_n):
        k = kt_buf[i]
        told_buf[i] = T_row[k]
        T_row[k] = -T_row[k]
    return changed, kt_n


@njit(cache=True, parallel=True, nogil=True)
def run_steps_prange(proc, exclude_self_loops, attempts_clock, maintained, seed, a1, B, R, replica_id, instance_id,
                     T, alpha, v, open_cnt, kt_buf, told_buf, cand_buf, pref_buf, J0_of_rep, J_of_rep,
                     sched_kind, sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total,
                     route_ptr, route_clause, route_ctrl, route_pol, route_group, route_weighted, is_obj,
                     has_ctrl, clause_wroute_ptr, clause_wroute, ctrl_route_ptr, ctrl_route, is_spin_clause,
                     sr_ptr, sr_pos, blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, blk_end,
                     plan_route_clause, spin_clause, energies, best_energy, best_spins, best_attempt, fb_energy,
                     fb_spins, fb_attempt, targets, stopped, cnt, tol, score_slot, viol_slot, has_viol,
                     stop_on_target, next_attempt):
    """parallel_trial / nfold steps of every replica while its clock < a1 (one replica per lane).
    ``maintained``: inputs and open singleton counts maintained by the incremental commit; else every
    input is recomputed at the start of each step (``refresh_inputs``) and the commit only toggles."""
    need_sl = exclude_self_loops or proc != PROC_PT
    n_rep = T.shape[0]
    for rep in prange(n_rep):
        if stopped[rep]:
            continue
        rid, iid = replica_id[rep], instance_id[rep]
        T_row, alpha_row, v_row, oc = T[rep], alpha[rep], v[rep], open_cnt[rep]
        kt_row, told_row, cb, pb = kt_buf[rep], told_buf[rep], cand_buf[rep], pref_buf[rep]
        J0 = J0_of_rep[rep]
        n = next_attempt[rep]
        k = cnt[C_STEPS, rep]
        while n < a1:
            tau = schedule_tau(sched_kind, sched_p0, sched_p1, sched_p2, sched_p3, sched_n_total, n)
            if not maintained:
                refresh_inputs(R, T_row, J0, v_row, oc, need_sl, sr_ptr, sr_pos, route_ctrl, route_pol,
                               blk_group_ptr, blk_ctrl, blk_pol, blk_start, blk_wend, plan_route_clause)
            if proc == PROC_PT:
                r, n_eval, nf = pt_decide(seed, n, rid, iid, R, tau, v_row, oc, exclude_self_loops, cb)
                W = 1
                outcome = OUT_COMMIT if r >= 0 else OUT_NOFIRE
            else:
                outcome, r, W, n_eval = nfold_decide(seed, k, n, rid, iid, R, tau, v_row, oc, cb, pb, B,
                                                     attempts_clock)
            k += 1
            cnt[C_STEPS, rep] += 1
            cnt[C_CAND, rep] += n_eval
            if outcome == OUT_FROZEN or outcome == OUT_OVERSHOOT:
                cnt[C_ATT, rep] += B - n
                cnt[C_REJ, rep] += B - n
                if outcome == OUT_FROZEN:
                    cnt[C_FROZEN, rep] = n
                n = B
                break
            cnt[C_ATT, rep] += W
            if outcome == OUT_NOFIRE:
                cnt[C_REJ, rep] += 1
                n += 1
                continue
            vr = v_row[r]
            if maintained:
                changed, kt_n = commit_stepwise(r, T_row, alpha_row, v_row, oc, kt_row, told_row, J0, route_ptr,
                                                route_clause, route_ctrl, route_pol, route_group, route_weighted,
                                                is_obj, has_ctrl, clause_wroute_ptr, clause_wroute,
                                                ctrl_route_ptr, ctrl_route, is_spin_clause)
            else:
                changed, kt_n = commit_toggle_blocked(r, T_row, kt_row, told_row, blk_group_ptr, blk_ctrl, blk_pol,
                                                      blk_start, blk_end, plan_route_clause, is_spin_clause)
            if kt_n > 0:
                apply_energy_update(J_of_rep[rep], kt_row, told_row, kt_n, energies[rep])
            cnt[C_ACC, rep] += 1
            cnt[C_REJ, rep] += W - 1
            n += W
            if not changed:
                cnt[C_SELF, rep] += 1
                continue
            cnt[C_CHG, rep] += 1
            if vr > tol[T_UPHILL, rep]:
                cnt[C_UP, rep] += 1
            hit = book_change(rep, n, T_row, spin_clause, energies, best_energy, best_spins, best_attempt,
                              fb_energy, fb_spins, fb_attempt, targets, cnt, tol, score_slot, viol_slot, has_viol)
            if hit and stop_on_target:
                stopped[rep] = True
                break
        next_attempt[rep] = n

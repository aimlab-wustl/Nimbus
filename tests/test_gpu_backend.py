"""Tests for the GPU architecture backend (``nimbus.backends.gpu``).

Two tiers:

* Host-only tests (no CUDA needed, run without CUDA): backend registration/``BackendUnavailable``
  without CuPy, :class:`~nimbus.backends.gpu.GPUPlan` construction (CSR shapes, max fan-in/perm), and
  a pure-numpy mirror of the kernel's per-attempt algorithm (:func:`_host_kernel_step`, a direct
  transcription of ``kernels/anneal.cu``'s math) replayed against the emulator/oracle on the shared
  case builders of ``tests/test_replay_core.py``. This checks the *algorithm* encoded in the plan and
  kernel without a GPU; it is not a substitute for an executed device run.
* ``@pytest.mark.gpu`` tests: the real :class:`~nimbus.backends.gpu.GPUBackend` through CuPy/NVRTC on
  a CUDA device (skipped with reason 'not executed: no CUDA device' otherwise), replaying the same
  cases against the emulator and oracle, split-run continuation, and batch sizes 1..4096.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers_core import all_states
from test_replay_core import CASE_BUILDERS, Case, _finish, mixing_schedule

from nimbus.backends import get_backend
from nimbus.backends.gpu import GPUBackend, cuda_available
from nimbus.core import rng
from nimbus.core.errors import BackendUnavailable

HAVE_GPU = cuda_available()


def case_onehot_cyclic(P: int = 4) -> Case:
    """One-hot P x P with path row exchanges plus a NON-involutive cyclic column shift (compiled with
    allow_non_involutive): its group has routes whose controller is itself toggled by the same group,
    so sigma must be latched from the pre-transition state before any toggle. (For involutive groups
    alpha_A(mu s) = alpha_A(s), so an intra-group controller is never toggled on feasible states and
    latching order is unobservable; none of the replay families exercises it.)"""
    from helpers_core import onehot_identities, permutation_states, random_distances, cyclic_assignment_objective
    from nimbus.compiler import compile_model
    from nimbus.compiler.compile import CompileOptions
    from nimbus.transitions import builders as B
    from nimbus.transitions.library import TransitionLibrary

    structure, w, c = cyclic_assignment_objective(random_distances(P, seed=P))
    s2, w2, c2 = cyclic_assignment_objective(random_distances(P, seed=P + 100))
    shift = list(range(1, P)) + [0]
    lib = B.row_exchanges(P, "path") + TransitionLibrary(P * P, [B.column_permutation(P, shift)], ["col_shift"],
                                                         name="col_shift")
    model = compile_model(structure, lib, identities=onehot_identities(P),
                          options=CompileOptions(allow_non_involutive=True))
    S_, _ = permutation_states(P)
    case = Case(f"onehot_P{P}_cyclic", structure, lib, model, S_, [w, w2], [c, c2])
    case.has_selfloops = False
    return _finish(case)


GPU_CASES = dict(CASE_BUILDERS, onehot_cyclic=case_onehot_cyclic)


def test_cyclic_case_has_intra_group_control():
    m = case_onehot_cyclic().model
    intra = sum(int(np.isin(m.route_ctrl[m.route_ptr[r]:m.route_ptr[r + 1]],
                            m.route_clause[m.route_ptr[r]:m.route_ptr[r + 1]]).sum()) for r in range(m.n_groups))
    assert intra > 0
_skip_no_gpu = pytest.mark.skipif(not HAVE_GPU, reason="not executed: no CUDA device")


def gpu_or_skip(fn):
    """Mark as a device test and skip (reported as not executed) without a CUDA device."""
    return pytest.mark.gpu(_skip_no_gpu(fn))


# ============================================================================================
# host-only: registration / plan
# ============================================================================================
def test_gpu_backend_registered_and_unavailable_without_device():
    b = get_backend("gpu")
    assert isinstance(b, GPUBackend)
    if not HAVE_GPU:
        with pytest.raises(BackendUnavailable):
            b.prepare(None)


def test_gpu_plan_shapes():
    case = CASE_BUILDERS["cardinality_toy"]()
    from nimbus.backends.gpu import GPUPlan
    plan = GPUPlan(case.model, device="cuda:0")
    assert plan.route_ptr.shape == (case.model.n_groups + 1,)
    assert plan.max_fanin >= 1
    # no transformation descriptors reach the device: spins are singleton clauses
    assert not any(hasattr(plan, a) for a in ("act_kind", "act_idx", "act_src", "cond_idx"))
    np.testing.assert_array_equal(plan.spin_clause, case.model.readout_clause_of_spin)
    assert plan.options["kernel_variant"] == "auto"
    # packed route records: clause | spin bit, (controller << 1) | neg, constant switch -> dummy slot M~
    m = case.model
    x, y = plan.rec[:, 0].astype(np.int64), plan.rec[:, 1].astype(np.int64)
    np.testing.assert_array_equal(x & 0x7FFFFFFF, m.route_clause)
    is_spin = np.isin(m.route_clause, m.readout_clause_of_spin)
    np.testing.assert_array_equal(x < 0, is_spin)
    np.testing.assert_array_equal(y >> 1, np.where(m.route_ctrl < 0, m.n_clauses, m.route_ctrl))
    np.testing.assert_array_equal(y & 1, (m.route_ctrl >= 0) & (m.route_pol < 0))


def test_gpu_auto_rule_and_shared_memory_rule():
    from nimbus.backends.gpu import choose_variant, resolve_kernel_config
    tiny = {"n_clauses": 300, "fanin_mean": 6.0, "fanin_max": 12}
    small = {"n_clauses": 1008, "fanin_mean": 45.0, "fanin_max": 205}
    mid = {"n_clauses": 15600, "fanin_mean": 1596.0, "fanin_max": 3000}
    big = {"n_clauses": 53100, "fanin_mean": 6840.0, "fanin_max": 6840}
    lim = 227 * 1024 - 1024
    # the auto rule against the measured winners
    assert choose_variant(tiny, 8192)[0] == "thread" and choose_variant(tiny, 64)[0] == "warp"
    assert [choose_variant(small, n)[0] for n in (1, 64, 1024, 4096)] == ["block", "block", "warp", "warp"]
    assert [choose_variant(mid, n)[0] for n in (1, 64, 1024, 4096)] == ["block", "block", "warp", "warp"]
    assert [choose_variant(big, n)[0] for n in (1, 64, 1024, 4096)] == ["block"] * 4
    k = resolve_kernel_config(tiny, 8192, smem_limit=lim)
    assert (k.variant, k.lanes, k.latch, k.t_shared, k.block_threads) == ("thread", 1, "mask", False, 32)
    k = resolve_kernel_config(small, 4096, smem_limit=lim)
    assert (k.variant, k.lanes, k.latch, k.t_shared) == ("warp", 32, "mask", False)  # small M~: global
    k = resolve_kernel_config(mid, 4096, smem_limit=lim)
    assert (k.variant, k.lanes, k.latch, k.t_shared) == ("warp", 32, "mark", True)
    assert k.replicas_per_block == min(4, lim // k.t_smem_stride)
    assert k.smem_bytes == k.replicas_per_block * k.t_smem_stride <= lim
    k = resolve_kernel_config(big, 64, smem_limit=lim)
    assert (k.variant, k.t_shared, k.replicas_per_block, k.latch) == ("block", True, 1, "mask")
    assert resolve_kernel_config(big, 64, smem_limit=48 * 1024).t_shared is False
    k = resolve_kernel_config(small, 4096, smem_limit=lim, kernel_variant="thread", t_memory="shared")
    assert k.t_shared and k.replicas_per_block % 32 == 0 and k.smem_bytes <= lim
    with pytest.raises(Exception):
        resolve_kernel_config(mid, 64, smem_limit=lim, kernel_variant="thread", t_memory="shared")
    with pytest.raises(Exception):
        resolve_kernel_config(big, 64, smem_limit=48 * 1024, t_memory="shared")
    with pytest.raises(Exception):
        resolve_kernel_config(mid, 64, smem_limit=lim, kernel_variant="thread", latch="mask")


def test_gpu_plan_strategies_and_rejects_float32():
    from nimbus.backends.gpu import GPUPlan
    case = CASE_BUILDERS["cardinality_toy"]()
    for strat in ("recompute", "incremental", "adaptive"):
        plan = GPUPlan(case.model, strategy=strat)
        assert plan.strategy == strat and plan.needs_v == (strat != "recompute")
    assert GPUPlan(case.model).strategy in ("recompute", "incremental", "adaptive")  # auto resolves
    with pytest.raises(Exception):
        GPUPlan(case.model, strategy="lazy")
    with pytest.raises(Exception):
        GPUPlan(case.model, precision="float32")


# ============================================================================================
# host-only: numpy mirror of the kernel algorithm (kernels/anneal.cu), replayed vs the emulator
# ============================================================================================
def _butterfly(vals):
    """The kernel's xor-shuffle reduction over 32 lanes (Python floats = IEEE doubles, round to nearest)."""
    x = [float(a) for a in vals]
    for o in (16, 8, 4, 2, 1):
        x = [x[i] + x[i ^ o] for i in range(32)]
    assert all(a == x[0] for a in x)  # every lane holds the bit-identical sum
    return x[0]


def _lane_sum(partials, lanes):
    if lanes == 1:
        return float(partials[0])
    if lanes == 32:
        return _butterfly(partials)
    warps = [_butterfly(partials[w * 32:(w + 1) * 32]) for w in range(lanes // 32)]
    return _butterfly(warps + [0.0] * (32 - len(warps)))


def _alpha(T, y):
    tc = int(T[y >> 1])
    return int((-tc if y & 1 else tc) > 0)


def _host_kernel_step(plan, lanes, latch, Jw, J0, Jall, T, r, u, tau):
    """Transcription of one attempt of ``tpi_anneal_recompute`` for one replica with TPI_LANES = lanes:
    packed route records, dummy controller slot T[M~] = +1, lane-strided pass 1, fixed reduction,
    sigma latched before any toggle (register bits or sign-keeping marks), commit pass.

    T has M~ + 1 entries (int8). Returns (xi, v, accepted, changed, T_new, dE)."""
    rec = plan.rec
    M = plan.model.n_clauses
    rb0, rbw, rb1 = int(plan.route_ptr[r]), int(plan.route_wend[r]), int(plan.route_ptr[r + 1])
    jo = int(plan.jw_off[r])
    n_slots = Jall.shape[0]
    vp = [0.0] * lanes
    for lane in range(lanes):
        for t in range(rb0 + lane, rbw, lanes):
            x, y = int(rec[t, 0]), int(rec[t, 1])
            if _alpha(T, y):
                k = x & 0x7FFFFFFF
                Jk = float(Jw[jo + t]) if Jw is not None else float(J0[k])
                vp[lane] = vp[lane] + Jk * float(T[k])
    v = _lane_sum(vp, lanes)
    xi = -(tau * 0.5) * math.log(u)
    accepted = (v <= 0.0) if tau == 0.0 else (v <= xi)
    if not accepted:
        return xi, v, False, False, T, np.zeros(n_slots)
    T = T.copy()
    dEp = np.zeros((lanes, n_slots))
    spin = False
    if latch == "mask":  # every lane latches its routes' alphas (pre-transition) before any toggle
        toggles = []
        for lane in range(lanes):
            routes = list(range(rb0 + lane, rb1, lanes))
            assert len(routes) <= 64  # one 64-bit register of alpha bits per lane
            toggles += [(lane, t) for t in routes if _alpha(T, int(rec[t, 1]))]
    else:  # mark pass (lanes in sequence: a marked controller keeps its sign), then commit by marks
        for lane in range(lanes):
            for t in range(rb0 + lane, rb1, lanes):
                if _alpha(T, int(rec[t, 1])):
                    k = int(rec[t, 0]) & 0x7FFFFFFF
                    T[k] = 2 * T[k]
        toggles = [(lane, t) for lane in range(lanes) for t in range(rb0 + lane, rb1, lanes)
                   if abs(int(T[int(rec[t, 0]) & 0x7FFFFFFF])) == 2]
    for lane, t in toggles:
        x = int(rec[t, 0])
        k = x & 0x7FFFFFFF
        told = int(T[k]) // 2 if latch == "mark" else int(T[k])
        for sl in range(1, n_slots):
            dEp[lane, sl] = dEp[lane, sl] + 2.0 * (float(Jall[sl, k]) * float(told))
        T[k] = -told
        spin = spin or x < 0
    assert T[M] == 1 and np.all(np.abs(T) == 1)
    dE = np.zeros(n_slots)
    dE[0] = 2.0 * v
    for sl in range(1, n_slots):
        dE[sl] = _lane_sum(dEp[:, sl], lanes)
    return xi, v, True, spin, T, dE


# ---- incremental strategy: fixed-point maintained inputs, stage 1/2 ----
def _few_bit_dyadic(J, bits=16):
    """Weights whose float sums are exact (multiples of 2^-bits, small): there the float inputs of the
    emulator equal the exact fixed-point inputs bit for bit; otherwise they differ in the last bits."""
    J = np.asarray(J, dtype=np.float64)
    return bool(np.all(np.ldexp(J, bits) == np.rint(np.ldexp(J, bits))) and np.all(np.abs(J) < 2.0 ** 20))


def _rebuild_V(plan, Jc, T):
    """V[r] = sum over weighted routes of Jc[k] * alpha * T[k], exact (Python ints)."""
    rec = plan.rec
    V = []
    for r in range(plan.model.n_groups):
        acc = 0
        for t in range(int(plan.route_ptr[r]), int(plan.route_wend[r])):
            if _alpha(T, int(rec[t, 1])):
                k = int(rec[t, 0]) & 0x7FFFFFFF
                acc += int(Jc[k]) * int(T[k])
        V.append(acc)
    return V


def _host_incremental_step(plan, Jc, vscale, Jall, V, T, r, u, tau):
    """Transcription of the TPI_INCREMENTAL path of ``tpi_anneal_recompute`` for one replica:
    v from the maintained V, mark pass, stage 1/2 from the marked state, commit pass."""
    rec, deps = plan.rec, plan.deps
    rb0, rb1 = int(plan.route_ptr[r]), int(plan.route_ptr[r + 1])
    v = float(V[r]) * vscale
    xi = -(tau * 0.5) * math.log(u)
    accepted = (v <= 0.0) if tau == 0.0 else (v <= xi)
    if not accepted:
        return v, False, False, T, V, np.zeros(Jall.shape[0])
    T = T.copy()
    V = list(V)
    for t in range(rb0, rb1):  # mark pass
        if _alpha(T, int(rec[t, 1])):
            k = int(rec[t, 0]) & 0x7FFFFFFF
            T[k] = 2 * T[k]
    for t in range(rb0, rb1):  # stage 1 + 2 from the marked state
        k = int(rec[t, 0]) & 0x7FFFFFFF
        tv = int(T[k])
        if abs(tv) != 2:
            continue
        told = tv // 2
        for q in range(int(deps["cw_ptr"][k]), int(deps["cw_ptr"][k + 1])):
            r2, y = int(deps["cw_rec"][q, 0]), int(deps["cw_rec"][q, 1])
            if _alpha(T, y):
                V[r2] += int(Jc[k]) * (-2 * told)
        for q in range(int(deps["cc_ptr"][k]), int(deps["cc_ptr"][k + 1])):
            r2, x = int(deps["cc_rec"][q, 0]), int(deps["cc_rec"][q, 1])
            k2 = x & 0x7FFFFFFF
            a_old = int(((-told) if x < 0 else told) > 0)
            t2 = int(T[k2])
            tnew = -(t2 // 2) if abs(t2) == 2 else t2
            V[r2] += int(Jc[k2]) * tnew * (1 - 2 * a_old)
    dE = np.zeros(Jall.shape[0])
    spin = False
    for t in range(rb0, rb1):  # commit pass
        x = int(rec[t, 0])
        k = x & 0x7FFFFFFF
        if abs(int(T[k])) == 2:
            told = int(T[k]) // 2
            for sl in range(1, Jall.shape[0]):
                dE[sl] += 2.0 * (float(Jall[sl, k]) * told)
            T[k] = -told
            spin = spin or x < 0
    dE[0] = 2.0 * v
    return v, True, spin, T, V, dE


@pytest.mark.parametrize("name", ["cardinality_toy", "cardinality_random", "parity_masks", "onehot_P4",
                                 "section29", "table_kempe", "table_chain", "onehot_cyclic"])
def test_host_incremental_mirror_matches_emulator(name):
    """Maintained fixed-point inputs: after every attempt V equals the exact rebuild from T (stage 1/2
    are complete and use the right pre/post signs), and v drives the same decisions as the emulator."""
    from nimbus.backends.base import clause_signs
    from nimbus.backends.gpu import GPUPlan, fixed_point_weights

    case = GPU_CASES[name]()
    plan = GPUPlan(case.model, device="cuda:0", strategy="incremental")
    n_rep = 3
    spins0 = case.initial(n_rep, seed=7)
    rb = np.arange(n_rep) % len(case.bindings)
    emu = get_backend("emulation")
    eplan = emu.prepare(case.model, validate="full", library=case.library)
    est = emu.init_state(eplan, [case.bindings[i] for i in rb], spins0,
                         replica_binding=np.arange(n_rep), replica_id=np.arange(n_rep), instance_id=rb)
    n_attempts = 300
    sched = mixing_schedule(case, n_attempts)
    seed = 123
    M = case.model.n_clauses
    Jalls = [np.asarray(b.J, dtype=np.float64) for b in case.bindings]
    Jc, vscale, _ = fixed_point_weights(case.model, np.stack([j[0] for j in Jalls]))
    exact = [_few_bit_dyadic(j[0]) for j in Jalls]
    hT = np.ones((n_rep, M + 1), dtype=np.int8)
    hT[:, :M] = clause_signs(case.model, spins0)
    hV = [_rebuild_V(plan, Jc[rb[i]], hT[i]) for i in range(n_rep)]
    n_acc = 0
    for n in range(n_attempts):
        tau = float(sched.tau(n))
        rec = emu.step(est, n, sched, seed)
        for rep in range(n_rep):
            b = int(rb[rep])
            r, u = rng.solver_draw(seed, n, rep, b, case.model.n_groups)
            v, acc, chg, T2, V2, dE = _host_incremental_step(plan, Jc[b], vscale[b], Jalls[b], hV[rep], hT[rep],
                                                             r, u, tau)
            assert acc == bool(rec.accepted[rep]) and chg == bool(rec.changed[rep]), (name, n, rep)
            if exact[b]:
                assert v == float(rec.v[rep]), (name, n, rep)
            else:
                assert abs(v - float(rec.v[rep])) <= 1e-9 * max(1.0, abs(v)), (name, n, rep)
            if acc:
                n_acc += 1
                assert V2 == _rebuild_V(plan, Jc[b], T2), (name, n, rep)  # drift-free, complete
            hT[rep], hV[rep] = T2, V2
    assert n_acc > 0
    np.testing.assert_array_equal(hT[:, :M], est.rs.T)


def test_fixed_point_weights_and_dependency_records():
    from nimbus.backends.gpu import FIXED_POINT_BITS, dependency_records, fixed_point_weights
    m = case_onehot_cyclic().model
    case = case_onehot_cyclic()
    J0 = np.stack([b.J[0] for b in case.bindings])
    Jc, vscale, exact = fixed_point_weights(m, J0)
    gid = np.repeat(np.arange(m.n_groups), np.diff(m.route_ptr))
    for b in range(J0.shape[0]):
        S = np.max(np.bincount(gid, weights=np.abs(Jc[b][m.route_clause]) * m.weighted_route_mask()))
        assert S <= 2.0 ** FIXED_POINT_BITS + m.n_routes
        if exact[b]:
            np.testing.assert_array_equal(Jc[b] * vscale[b], J0[b])
    d = dependency_records(m)
    np.testing.assert_array_equal(d["cw_ptr"], m.clause_wroute_ptr)
    np.testing.assert_array_equal(d["cw_rec"][:, 0], gid[m.clause_wroute])
    np.testing.assert_array_equal(d["cc_rec"][:, 1].astype(np.int64) & 0x7FFFFFFF, m.route_clause[m.ctrl_wroute])
    np.testing.assert_array_equal(d["cc_rec"][:, 1] < 0, m.route_pol[m.ctrl_wroute] < 0)


def test_adaptive_cost_model():
    from nimbus.backends.gpu import choose_adaptive_strategy
    c = {"wfan": 1000.0, "dfan": 1200.0, "prop": 50000.0}
    assert choose_adaptive_strategy(c, 0.001, "recompute")[0] == "incremental"
    assert choose_adaptive_strategy(c, 0.9, "incremental")[0] == "recompute"
    # hysteresis: a marginal gain does not switch
    s0, cr, ci = choose_adaptive_strategy(c, 0.0185, "recompute")
    assert s0 == "recompute" and ci < cr


# latch "auto": register bits when the largest group fits 64 * lanes, else marks (as the plan decides)
MIRROR_CONFIGS = [("thread", 1, "auto"), ("thread", 1, "mark"), ("warp", 32, "auto"), ("warp", 32, "mark"),
                  ("block", 64, "auto"), ("block", 256, "mark")]


@pytest.mark.parametrize("variant,lanes,latch", MIRROR_CONFIGS, ids=[f"{v}{n}-{l}" for v, n, l in MIRROR_CONFIGS])
@pytest.mark.parametrize("name", ["cardinality_toy", "cardinality_random", "parity_masks", "onehot_P4",
                                 "section29", "table_kempe", "table_chain", "onehot_cyclic"])
def test_host_kernel_mirror_matches_emulator(name, variant, lanes, latch):
    from nimbus.backends.base import clause_signs
    from nimbus.backends.gpu import GPUPlan, per_route_weights

    case = GPU_CASES[name]()
    plan = GPUPlan(case.model, device="cuda:0", kernel_variant=variant)
    if latch == "auto":
        latch = "mask" if plan.max_fanin <= 64 * lanes else "mark"
    n_rep = 3
    spins0 = case.initial(n_rep, seed=7)
    rb = np.arange(n_rep) % len(case.bindings)
    emu = get_backend("emulation")
    eplan = emu.prepare(case.model, validate="full", library=case.library)
    est = emu.init_state(eplan, [case.bindings[i] for i in rb], spins0,
                         replica_binding=np.arange(n_rep), replica_id=np.arange(n_rep), instance_id=rb)
    n_attempts = 300
    sched = mixing_schedule(case, n_attempts)
    seed = 123
    M = case.model.n_clauses
    hT = np.ones((n_rep, M + 1), dtype=np.int8)
    hT[:, :M] = clause_signs(case.model, spins0)
    hE = np.stack([case.bindings[rb[i]].const - case.bindings[rb[i]].J @ hT[i, :M].astype(np.float64)
                  for i in range(n_rep)])
    Jalls = [np.asarray(case.bindings[i].J, dtype=np.float64) for i in range(len(case.bindings))]
    use_jw = latch == "mask"  # exercise both weight paths
    Jws = per_route_weights(case.model, np.stack([j[0] for j in Jalls]), plan.wpos) if use_jw else None
    n_acc = 0
    for n in range(n_attempts):
        tau = float(sched.tau(n))
        rec = emu.step(est, n, sched, seed)
        for rep in range(n_rep):
            r, u = rng.solver_draw(seed, n, rep, int(rb[rep]), case.model.n_groups)
            assert r == rec.group[rep] and u == rec.u[rep]
            b = int(rb[rep])
            xi, v, acc, chg, T2, dE = _host_kernel_step(plan, lanes, latch, None if Jws is None else Jws[b],
                                                        Jalls[b][0], Jalls[b], hT[rep], r, u, tau)
            assert acc == bool(rec.accepted[rep]), (name, n, rep)
            assert chg == bool(rec.changed[rep]), (name, n, rep)
            assert abs(v - float(rec.v[rep])) <= 1e-9 * max(1.0, abs(v)), (name, n, rep)
            if acc:  # the singleton readout after the decoder equals the descriptor transformation mu_r
                n_acc += 1
                s_pre = case.model.spin_readout(hT[rep, :M])
                mu = np.asarray(case.library.groups[r].apply(s_pre), dtype=np.int8)
                np.testing.assert_array_equal(case.model.spin_readout(T2[:M]), mu, err_msg=f"{name} n={n} r={r}")
            hT[rep] = T2
            hE[rep] = hE[rep] + dE
    assert n_acc > 0
    np.testing.assert_array_equal(case.model.spin_readout(hT[:, :M]), est.rs.spins)
    np.testing.assert_array_equal(hT[:, :M], est.rs.T)
    assert np.allclose(hE, est.rs.energies, atol=1e-7)


# ============================================================================================
# @pytest.mark.gpu: real device tests (skipped without a CUDA device)
# ============================================================================================
# kernel configurations every device replay runs: each variant, both latch modes, T in shared and in
# global memory, per-route and gathered weights; "auto" resolves per model
GPU_CONFIGS = {
    "auto": {},
    "thread": dict(strategy="recompute", kernel_variant="thread"),
    "thread-global-mark-gather": dict(strategy="recompute", kernel_variant="thread", t_memory="global", latch="mark",
                                      jw="off"),
    "warp": dict(strategy="recompute", kernel_variant="warp"),
    "warp-global-mark-gather": dict(strategy="recompute", kernel_variant="warp", t_memory="global", latch="mark",
                                    jw="off"),
    "block": dict(strategy="recompute", kernel_variant="block"),
    "block64-mark": dict(strategy="recompute", kernel_variant="block", block_threads=64, latch="mark"),
    "block-global-gather": dict(strategy="recompute", kernel_variant="block", t_memory="global", jw="off"),
    # incremental: maintained fixed-point inputs, V in shared or global memory
    "inc-thread": dict(strategy="incremental", kernel_variant="thread"),
    "inc-warp": dict(strategy="incremental", kernel_variant="warp", t_memory="shared"),
    "inc-warp-vglobal": dict(strategy="incremental", kernel_variant="warp", t_memory="global", v_memory="global"),
    "inc-block": dict(strategy="incremental", kernel_variant="block"),
    "inc-block64-vglobal": dict(strategy="incremental", kernel_variant="block", block_threads=64, v_memory="global"),
}
_REF_CACHE: dict = {}


def _reference(name, n_rep, n_att, seed):
    """Emulator and oracle states after n_att lock-step attempts (cached across kernel configs)."""
    key = (name, n_rep, n_att, seed)
    if key not in _REF_CACHE:
        case = GPU_CASES[name]()
        spins0 = case.initial(n_rep, seed=3)
        rb = np.arange(n_rep) % len(case.bindings)
        emu = get_backend("emulation")
        eplan = emu.prepare(case.model, validate="full", library=case.library)
        est = emu.init_state(eplan, [case.bindings[i] for i in rb], spins0,
                             replica_binding=np.arange(n_rep), replica_id=np.arange(n_rep), instance_id=rb)
        oracle = get_backend("oracle")
        oplan = oracle.prepare(case.model, library=case.library)
        ost = oracle.init_state(oplan, [case.bindings[i] for i in rb], spins0,
                                replica_binding=np.arange(n_rep), replica_id=np.arange(n_rep), instance_id=rb)
        sched = mixing_schedule(case, n_att)
        for n in range(n_att):
            emu.step(est, n, sched, seed)
            oracle.step(ost, n, sched, seed)
        _REF_CACHE[key] = (case, spins0, rb, sched, est.rs.copy(), ost.rs.copy())
    return _REF_CACHE[key]


@gpu_or_skip
@pytest.mark.parametrize("cfg", list(GPU_CONFIGS))
@pytest.mark.parametrize("name", list(GPU_CASES))
@pytest.mark.parametrize("n_rep", [1, 3, 37])
def test_gpu_replay_vs_emulator_and_oracle(name, n_rep, cfg):
    seed = 99
    case, spins0, rb, sched, ers, ors = _reference(name, n_rep, 300, seed)
    gpu = get_backend("gpu")
    gplan = gpu.prepare(case.model, **GPU_CONFIGS[cfg])
    gst = gpu.init_state(gplan, [case.bindings[i] for i in rb], spins0,
                         replica_binding=np.arange(n_rep), replica_id=np.arange(n_rep), instance_id=rb)
    from nimbus.experiments.recording import RecordingSpec
    gres = gpu.run(gplan, gst, sched, seed=seed, recording=RecordingSpec(profile="trace"))
    if cfg != "auto":
        assert gres.metadata["kernel"]["variant"] == GPU_CONFIGS[cfg]["kernel_variant"]
        assert gres.metadata["kernel"]["strategy"] == GPU_CONFIGS[cfg]["strategy"]
    if GPU_CONFIGS[cfg].get("strategy") == "incremental":  # maintained inputs == the emulator's
        assert gres.state.v is not None and ers.v is not None
        if all(_few_bit_dyadic(b.J[0]) for b in case.bindings):
            np.testing.assert_array_equal(gres.state.v, ers.v)
        else:
            assert np.allclose(gres.state.v, ers.v, rtol=1e-12, atol=1e-9)
    np.testing.assert_array_equal(gst.rs.spins, ers.spins)
    np.testing.assert_array_equal(gst.rs.spins, ors.spins)
    np.testing.assert_array_equal(gst.rs.T, ers.T)
    assert np.allclose(gst.rs.energies, ers.energies, atol=1e-7)
    assert np.allclose(gst.rs.energies, ors.energies, atol=1e-7)
    for k in ("attempts", "accepted", "accepted_changed", "self_loops", "rejected"):
        np.testing.assert_array_equal(gst.rs.counters[k], ers.counters[k])
    np.testing.assert_array_equal(gst.rs.best_attempt, ors.best_attempt)
    np.testing.assert_array_equal(gst.rs.best_spins, ors.best_spins)
    tr = gres.metadata["gpu_trace"]
    assert tr is not None and tr["attempt"].size == n_rep * 300


@gpu_or_skip
@pytest.mark.parametrize("variant", ["thread", "warp", "block"])
@pytest.mark.parametrize("name", ["onehot_cyclic", "cardinality_toy", "section29"])
def test_gpu_adaptive_switching_replays(name, variant, monkeypatch):
    """Adaptive (measured-rate policy): forced to probe incremental every other window, always switch
    and revert after one window, so V is rebuilt on every switch; the trajectory still replays exactly."""
    import nimbus.backends.gpu as G
    seed, n_rep = 99, 37
    case, spins0, rb, sched, ers, ors = _reference(name, n_rep, 300, seed)
    monkeypatch.setattr(G, "ADAPTIVE_PROBE_GAP", 1)
    monkeypatch.setattr(G, "ADAPTIVE_PROBE_RATIO", 1e300)
    monkeypatch.setattr(G, "ADAPTIVE_HYSTERESIS", -1.0)
    monkeypatch.setattr(G, "ADAPTIVE_REVERT_P_FACTOR", 0.0)
    gpu = get_backend("gpu")
    gplan = gpu.prepare(case.model, strategy="adaptive", kernel_variant=variant, adaptive_subwindow=37)
    gst = gpu.init_state(gplan, [case.bindings[i] for i in rb], spins0,
                         replica_binding=np.arange(n_rep), replica_id=np.arange(n_rep), instance_id=rb)
    gres = gpu.run(gplan, gst, sched, seed=seed)
    strategies = [w["strategy"] for w in gres.metadata["strategy_schedule"]]
    assert "incremental" in strategies and "recompute" in strategies
    np.testing.assert_array_equal(gst.rs.T, ers.T)
    np.testing.assert_array_equal(gst.rs.spins, ors.spins)
    for k in ("attempts", "accepted", "accepted_changed", "self_loops", "rejected"):
        np.testing.assert_array_equal(gst.rs.counters[k], ers.counters[k])


@gpu_or_skip
@pytest.mark.parametrize("strategy", ["recompute", "incremental"])
@pytest.mark.parametrize("variant", ["thread", "warp", "block"])
def test_gpu_split_run_continuation(variant, strategy):
    case = CASE_BUILDERS["cardinality_toy"]()
    n_rep = 5
    spins0 = case.initial(n_rep, seed=1)
    rb = np.zeros(n_rep, dtype=np.int64)
    sched = mixing_schedule(case, 500)
    gpu = get_backend("gpu")

    def make():
        plan = gpu.prepare(case.model, kernel_variant=variant, strategy=strategy)
        st = gpu.init_state(plan, case.bindings[0], spins0,
                            replica_binding=rb, replica_id=np.arange(n_rep), instance_id=rb)
        return plan, st

    plan_a, st_a = make()
    gpu.run(plan_a, st_a, sched, seed=5)

    plan_b, st_b = make()
    gpu.run(plan_b, st_b, sched, seed=5, stop=200)
    gpu.run(plan_b, st_b, sched, seed=5, start=200)

    np.testing.assert_array_equal(st_a.rs.spins, st_b.rs.spins)
    np.testing.assert_array_equal(st_a.rs.T, st_b.rs.T)
    assert np.allclose(st_a.rs.energies, st_b.rs.energies)


@gpu_or_skip
@pytest.mark.parametrize("strategy", ["recompute", "incremental"])
@pytest.mark.parametrize("variant", ["thread", "warp", "block"])
@pytest.mark.parametrize("n_rep", [1, 4, 512, 4096])
def test_gpu_batch_sizes_run_without_error(n_rep, variant, strategy):
    case = CASE_BUILDERS["cardinality_random"]()
    spins0 = case.initial(n_rep, seed=2)
    rb = np.arange(n_rep) % len(case.bindings)
    gpu = get_backend("gpu")
    plan = gpu.prepare(case.model, kernel_variant=variant, strategy=strategy)
    st = gpu.init_state(plan, [case.bindings[i] for i in rb], spins0,
                        replica_binding=np.arange(n_rep), replica_id=np.arange(n_rep), instance_id=rb)
    sched = mixing_schedule(case, 50)
    res = gpu.run(plan, st, sched, seed=1)
    assert res.state.n_replicas == n_rep
    assert np.all(res.attempts_run == 50)


LARGE_CONFIGS = {
    "inc-thread": dict(strategy="incremental", kernel_variant="thread"),
    "inc-warp": dict(strategy="incremental", kernel_variant="warp"),
    "inc-block-vglobal": dict(strategy="incremental", kernel_variant="block", v_memory="global"),
    "adaptive": dict(strategy="adaptive", adaptive_subwindow=500),
    "thread": dict(kernel_variant="thread"),
    "thread-shared-optin": dict(kernel_variant="thread", t_memory="shared", block_threads=128),
    "warp": dict(kernel_variant="warp"),
    "warp-shared-4": dict(kernel_variant="warp", t_memory="shared", block_threads=128),
    "block": dict(kernel_variant="block"),
    "block-global-mark": dict(kernel_variant="block", t_memory="global", latch="mark", block_threads=128),
}


@gpu_or_skip
@pytest.mark.parametrize("cfg", list(LARGE_CONFIGS))
def test_gpu_variants_agree_on_large_groups(cfg):
    """A one-hot assignment model with groups of thousands of routes (mark latch, multi-word lane loops) and a
    non-dyadic objective: every variant against the CPU backend (recompute) on the same draws.
    Summation order differs between variants, so v may differ in the last bits; any accept-decision
    difference at the threshold is classified (counted) rather than hidden; with this seed none occurs."""
    from helpers_core import random_distances, cyclic_assignment_objective
    from nimbus.annealing import schedule as S
    from nimbus.compiler import compile_model
    from nimbus.core.model import bind_weights
    from nimbus.transitions import builders as B

    p = 9
    structure, w, c = cyclic_assignment_objective(random_distances(p, seed=4))
    sign = -1 if (p - 1) % 2 else 1
    ids = [(tuple(r), sign) for r in B.onehot_rows(p)] + [(tuple(cc), sign) for cc in B.onehot_columns(p)]
    model = compile_model(structure, B.row_exchanges(p, "complete"), identities=ids)
    binding = bind_weights(model, w, c)
    n_rep = 40
    perm = np.stack([np.random.default_rng(i).permutation(p) for i in range(n_rep)])
    spins0 = -np.ones((n_rep, p * p), dtype=np.int8)
    for i in range(n_rep):
        spins0[i, np.arange(p) * p + perm[i]] = 1
    sched = S.geometric(float(np.max(np.abs(w))) * 4, float(np.max(np.abs(w))) * 0.05, 3000)
    cpu = get_backend("cpu")
    cplan = cpu.prepare(model, strategy="recompute")
    cst = cpu.init_state(cplan, binding, spins0)
    cres = cpu.run(cplan, cst, sched, seed=11)
    gpu = get_backend("gpu")
    gplan = gpu.prepare(model, **LARGE_CONFIGS[cfg])
    gst = gpu.init_state(gplan, binding, spins0)
    gres = gpu.run(gplan, gst, sched, seed=11)
    if cfg == "thread-shared-optin":  # > 48 KB of dynamic shared memory: the opt-in attribute path
        assert gres.metadata["kernel"]["t_shared"] and gres.metadata["kernel"]["smem_bytes"] > 48 * 1024
    auto_latch = "mask" if gplan.max_fanin <= 64 * gres.metadata["kernel"]["lanes"] else "mark"
    want_latch = "mark" if gres.metadata["kernel"]["incremental"] else LARGE_CONFIGS[cfg].get("latch", auto_latch)
    assert gres.metadata["kernel"]["latch"] == want_latch  # incremental kernels always latch by marks
    np.testing.assert_array_equal(gres.state.T, cres.state.T)
    np.testing.assert_array_equal(gres.state.counters["accepted"], cres.state.counters["accepted"])
    if gres.state.v is not None:  # maintained inputs equal the exact rebuild and the float recomputation
        from nimbus.backends.gpu import fixed_point_weights
        Jc, vscale, _ = fixed_point_weights(model, binding.J[0][None, :])
        T1 = np.ones(model.n_clauses + 1, dtype=np.int8)
        for i in range(0, n_rep, 13):
            T1[:-1] = gres.state.T[i]
            V = np.asarray(_rebuild_V(gplan, Jc[0], T1), dtype=np.float64) * vscale[0]
            np.testing.assert_array_equal(gres.state.v[i], V)
        assert np.allclose(gres.state.v, gpu.evaluate_inputs(gst), rtol=1e-12, atol=1e-9)
    assert np.allclose(gres.state.energies, cres.state.energies, rtol=1e-9, atol=1e-6)

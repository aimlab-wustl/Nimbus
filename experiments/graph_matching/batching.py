"""Batched move libraries (Fig. 4g, Supplementary Table S3): the complete library loaded in batches of K shortcuts.

The complete library of ``L(L - 1)/2`` row exchanges is the path backbone (``L - 1`` exchanges, always loaded) plus all
other row pairs. Those shortcuts are ranked by the informed overlap score (``nimbus.problems.matching.informed_pairs``,
computed from the tensor alone) and cut into consecutive batches of ``K``, so batch 0 is the static library "backbone
+ K informed shortcuts" with backbone share 0.5. A run cycles batches 0, 1, ..., 0, ... and replaces the loaded batch
every ``D`` iterations:

* the FN schedule runs over the whole budget ``B``; window ``w`` covers the absolute iterations ``[w D, (w + 1) D)`` of
  that schedule, so the random streams continue across windows;
* at each window the next batch's machine is initialised from the current spins (the clause outputs are a function of
  the spins, so the hand-over is exact);
* the temperatures are calibrated once, on batch 0, with the automatically calibrated schedule (``m = 1``,
  ``kappa = 2``), so batched and static runs have the same temperature at every iteration.

With ``D >= B`` the run is the static library run. Task kind ``matching_dwell`` (``worker.py``); params: instances,
k, share, dwells (log2 D), budgets (log2 B), replicas, seed, kappa, kappa_mode. Arrays per (instance, K, B, D):
``<name>_k<K>_b<B>_D<D>_bestF``.
"""
from __future__ import annotations

import time

import numpy as np

import common as C


def batch_libraries(prob, tensor, k, share=0.5):
    """``[(TransitionLibrary, pairs, copies)]`` of every batch, in loading order."""
    from nimbus.problems.matching import backbone_edges, informed_pairs
    from nimbus.transitions import builders as B
    from nimbus.transitions.library import TransitionLibrary

    n = int(prob.P)
    excl = set(backbone_edges(n))
    ranked = informed_pairs(tensor, n * (n - 1) // 2, excl)       # all shortcuts, informed order
    bb = B.row_exchanges(n, pattern="path")
    out = []
    for b0 in range(0, len(ranked), k):
        pairs = ranked[b0:b0 + k]
        m = max(1, int(round(share / (1.0 - share) * len(pairs) / (n - 1)))) if share is not None else 1
        add = B.row_exchanges(n, pattern="edges", edges=pairs)
        out.append((TransitionLibrary.concat(*([bb] * m + [add]), name=f"batch{b0 // k}"), pairs, m))
    return out


def set_next_attempt(st, a):
    """A freshly initialised state starts at iteration 0; a window resumes the schedule at absolute iteration ``a``."""
    st.rs.next_attempt[:] = a
    if hasattr(st, "d_next_attempt"):          # GPU backend: device copy
        import cupy as cp
        st.d_next_attempt = cp.asarray(st.rs.next_attempt)


def task_matching_dwell(p, ctx):
    from nimbus.annealing import schedule as S
    from nimbus.annealing.protocol import calibrate_temperatures
    from nimbus.backends import get_backend
    from nimbus.compiler import compile_model
    from nimbus.core.model import bind_weights, structural_resources

    be = get_backend(ctx["backend"])
    records, arrays = [], {}
    reps, seed = p["replicas"], p["seed"]
    for name in p["instances"]:
        m, t, prob = C.load_instance(ctx["inst"], name)
        t0 = time.perf_counter()
        libsb = batch_libraries(prob, t, p["k"], p.get("share", 0.5))
        models = [compile_model(prob.structure(), lib, identities=tuple(prob.identities())) for lib, _, _ in libsb]
        binds = [bind_weights(md, prob.term_weights(0), prob.const(0)) for md in models]
        plans = [C.prepare_plan(be, ctx["backend"], md, ctx["threads"]) for md in models]
        res_b = [structural_resources(md) for md in models]
        t_setup = time.perf_counter() - t0
        t0 = time.perf_counter()
        tau_s, tau_e = calibrate_temperatures(models[0], binds[0], prob, seed, min_ratio=p.get("kappa"),
                                              min_ratio_mode=p.get("kappa_mode", "lower_end"))
        t_cal = time.perf_counter() - t0
        kw = dict(replica_id=np.arange(reps, dtype=np.int64), instance_id=np.zeros(reps, dtype=np.int64),
                  is_feasible=prob.is_feasible)
        spins0 = prob.random_feasible(seed, reps, 0)
        for pl, bd in zip(plans, binds):        # kernel warm-up per plan (not charged to the reload)
            be.run(pl, be.init_state(pl, bd, spins0, **kw), S.fn_endpoints(tau_s, tau_e, 0.1, 64), seed=seed)
        nb = 1 if p.get("check_single_batch") else len(plans)   # check: windows on batch 0 only equal one static run
        for lb in p["budgets"]:
            Btot = 2 ** lb
            sched = S.fn_endpoints(tau_s, tau_e, 0.1, Btot)
            for ld in p["dwells"]:
                D = 2 ** ld
                if D >= Btot:
                    continue
                spins = spins0.copy()
                bestE = np.full(reps, np.inf)
                t_run = t_reload = 0.0
                n_win = Btot // D
                acc = att = 0
                for w in range(n_win):
                    b = w % nb
                    t1 = time.perf_counter()
                    st = be.init_state(plans[b], binds[b], spins, **kw)
                    set_next_attempt(st, w * D)
                    t2 = time.perf_counter()
                    res = be.run(plans[b], st, sched, n_attempts_total=Btot, start=w * D, stop=(w + 1) * D, seed=seed)
                    s = res.state
                    E = np.asarray(s.best_energy, dtype=np.float64)
                    spins = np.asarray(s.spins).copy()
                    t3 = time.perf_counter()
                    t_reload += t2 - t1
                    t_run += t3 - t2
                    bestE = np.minimum(bestE, E)
                    c = s.counters
                    acc += int(np.sum(c["accepted"]))
                    att += int(np.sum(c["attempts"]))
                bestF = -bestE
                key = f"{name}_k{p['k']}_b{lb}_D{ld}"
                arrays[key + "_bestF"] = bestF
                records.append(dict(instance=name, family=m["family"], role=m["role"], n=int(m["n"]), k=p["k"],
                                    share=p.get("share", 0.5), n_batches=nb, budget_log2=lb, dwell_log2=ld,
                                    windows=n_win, replicas=reps, seed=seed, backend=ctx["backend"], tau_start=tau_s,
                                    tau_end=tau_e, kappa=p.get("kappa"), wall_run_s=t_run, wall_reload_s=t_reload,
                                    acc_rate=acc / max(1, att), attempts_total=att,
                                    routes_batches=[int(md.n_routes) for md in models],
                                    groups_batches=[int(md.n_groups) for md in models],
                                    mem_bytes_batches=[r.get("memory_bytes") for r in res_b],
                                    bestF_max=float(bestF.max()), bestF_median=float(np.median(bestF)),
                                    t_setup=t_setup, t_calibrate=t_cal))
    return records, arrays

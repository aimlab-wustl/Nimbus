"""Event-driven measurements (Fig. 6, Fig. S9): per-window operation counts along one slow anneal per workload.

  python run_event.py list
  python run_event.py run --task m30c__o0_21 --out RUNS/event [--log2 24] [--replicas 128] [--threads N]
  python run_event.py run --all [--only PREFIX ...] [--skip PREFIX ...] --out RUNS/event

Protocol (every workload). Start temperature tau_start: matching uses the temperatures of the automatically calibrated
schedule of the graph-matching runs (``configs/matching_temperatures.json``); the surface code and the synthetic
problems are calibrated with the same acceptance rule (p_start 0.8, p_end 0.01, short late calibration run; minimum ratio kappa = 2
for the synthetic problems). Then one Fowler-Nordheim schedule (eta = 0.1) from tau_start to tau_start / 1000 over
2^log2 iterations per replica. The run is never stopped at the optimum. The CPU backend runs the recomputation
strategy with per-window event counters (``CPUBackend.prepare(event_counters=True, event_windows=...)``); the counters
are exact and do not depend on the update strategy, so one trajectory gives the operations of event-driven updating
(input updates after accepted moves) and of recomputation (connection reads of the proposed move) at the same time.
Windows: 1/8 octave (eight per doubling of the iteration count), at least 256 iterations wide.

Tasks (workload__instance):
  m30c__o0_21, o0_08, o0_09, o0_13, o0_24   matching, complete library, L = 30 (five test instances); 2^24, 128 replicas
  m50p__o0_06, o0_07, o0_08                 matching, path library, L = 50 (first three test instances); 2^26, 128
  s11__1..4                                 surface code d = 11, depolarizing p_err = 0.12, all d logical operators per
                                            type; 4 shot seeds x 128 shots; 2^26
  pair2d__0..3                              2D +-J spin glass, 16 x 16 torus, single flips; 4 coupling sets; 2^26, 64
  syn_<setting>__0..3                       synthetic family (``synthetic.SETTINGS``), 4 generator seeds; 2^26, 64
Output: ``OUT/<task>.npz`` (skipped if present): JSON metadata, window edges, per-replica per-window counters
``win[replica, window, field]`` (fields in ``meta['win_fields']``), per-window best energies, reference energies,
first iteration at the reference, and run totals.

The matching tasks need the staged instances (``../graph_matching/stage_instances.py``; ``--inst`` or the
environment variable ``NIMBUS_MATCHING_INST``); the surface-code task needs PyMatching for the reference energies.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "graph_matching"))
warnings.filterwarnings("ignore")

from nimbus.annealing import schedule as S  # noqa: E402
from nimbus.annealing.protocol import calibrate_temperatures  # noqa: E402
from nimbus.backends import _numba_kernels as K  # noqa: E402
from nimbus.backends import get_backend  # noqa: E402
from nimbus.compiler import compile_model  # noqa: E402
from nimbus.core.model import bind_weights, structural_resources  # noqa: E402

SEED = 11
KCOLD = 1000.0
ETA = 0.1
PER_DOUBLING = 8
MIN_WIDTH = 256
M30_INST = ("o0_21", "o0_08", "o0_09", "o0_13", "o0_24")
M50_INST = ("o0_06", "o0_07", "o0_08")
S11_SEEDS = (1, 2, 3, 4)
SYN_SEEDS = (0, 1, 2, 3)
#: workload -> (log2 iterations per replica, replicas)
DEFAULTS = {"m30c": (24, 128), "m50p": (26, 128), "s11": (26, 128), "syn": (26, 64), "pair2d": (26, 64)}
TEMPS = HERE / "configs" / "matching_temperatures.json"


def tasks():
    import synthetic as SY
    t = [f"m30c__{i}" for i in M30_INST] + [f"m50p__{i}" for i in M50_INST] + [f"s11__{s}" for s in S11_SEEDS]
    t += [f"{name}__{s}" for name in SY.SETTINGS for s in SYN_SEEDS]
    t += [f"pair2d__{s}" for s in SYN_SEEDS]
    return t


def build(task, M, inst=None):
    """dict(model, binding, spins0, kw, tau_s, tau_e_cal, tau_source, E_ref (per replica or None), ref_kind, info)."""
    wl, ident = task.split("__")
    if wl in ("m30c", "m50p"):
        import common as GM
        n, lib = (30, "complete") if wl == "m30c" else (50, "path")
        name = f"syn_n{n}_s0.05_{ident}"
        m, t, prob = GM.load_instance(GM.inst_dir(inst), name)
        t0 = time.perf_counter()
        model, binding, _ = GM.compile_instance(prob, t, {"base": lib})
        tc = time.perf_counter() - t0
        tau_s, tau_e = json.loads(TEMPS.read_text())[f"{name}|{lib}"]
        fref, kind = GM.f_target(m, GM.psi_ref_table())
        kw = dict(replica_id=np.arange(M, dtype=np.int64), instance_id=np.zeros(M, dtype=np.int64),
                  is_feasible=prob.is_feasible)
        return dict(model=model, binding=binding, spins0=prob.random_feasible(SEED, M, 0), kw=kw, tau_s=tau_s,
                    tau_e_cal=tau_e, tau_source="automatically calibrated schedule (matching runs)",
                    E_ref=None if fref is None else np.full(M, -fref), ref_kind=f"Psi_ref ({kind})",
                    info=dict(problem="matching", n=n, lib=lib, instance=name, psi_ref=fref, t_compile=tc))
    if wl == "s11":
        from nimbus.decoding.baselines import matching_correction
        from nimbus.decoding.code_capacity import sample_shots
        from nimbus.problems.surface_code import SurfaceCodeProblem
        prob = SurfaceCodeProblem(11, 0.12, noise="depolarizing", library="stabilizers_logical_all")
        t0 = time.perf_counter()
        model = compile_model(prob.structure(), prob.transition_library(), identities=())
        binding = bind_weights(model, prob.term_weights(0), prob.const(0))
        tc = time.perf_counter() - t0
        tau_s, tau_e = calibrate_temperatures(model, binding, prob, SEED, late=True)
        _, syn = sample_shots(prob, M, int(ident))
        spins0 = prob.particular_spins(syn)
        c_mw = matching_correction(prob, syn)
        sp_mw = (1 - 2 * c_mw.astype(np.int64)).astype(np.int8)
        E_mw = np.array([prob.energy(sp_mw[i]) for i in range(M)])
        kw = dict(replica_id=np.arange(M, dtype=np.int64), instance_id=np.zeros(M, dtype=np.int64), is_feasible=None)
        return dict(model=model, binding=binding, spins0=spins0, kw=kw, tau_s=tau_s, tau_e_cal=tau_e,
                    tau_source="calibrated", E_ref=E_mw, ref_kind="energy of the PyMatching correction per shot",
                    info=dict(problem="surface", d=11, p=0.12, noise="depolarizing", lib="stabilizers_logical_all",
                              shot_seed=int(ident), t_compile=tc))
    import synthetic as SY
    if wl == "pair2d":
        prob, lo, info = SY.make_spin_glass(16, int(ident))
        E_opt, ref_kind = None, "none"
    else:
        prob, lo, info = SY.make_problem(**SY.SETTINGS[wl], seed=int(ident))
        E_opt = SY.exact_optimum(prob)
        ref_kind = "exact optimum (enumeration)" if E_opt is not None else "none"
    t0 = time.perf_counter()
    model = compile_model(prob.structure(), prob.transition_library(**lo), identities=())
    binding = bind_weights(model, prob.term_weights(0), prob.const(0))
    tc = time.perf_counter() - t0
    tau_s, tau_e = calibrate_temperatures(model, binding, prob, SEED, late=True, min_ratio=2.0, min_ratio_mode="lower_end")
    kw = dict(replica_id=np.arange(M, dtype=np.int64), instance_id=np.zeros(M, dtype=np.int64),
              is_feasible=prob.is_feasible)
    return dict(model=model, binding=binding, spins0=prob.random_feasible(SEED, M, 0), kw=kw, tau_s=tau_s,
                tau_e_cal=tau_e, tau_source="calibrated", E_ref=None if E_opt is None else np.full(M, E_opt),
                ref_kind=ref_kind, info=dict(problem="synthetic", setting=wl, t_compile=tc, **info))


def run_task(task, log2, M, threads, out, inst=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{task}.npz"
    if f.exists():
        print(f"{task}: exists, skipped", flush=True)
        return
    t_all = time.perf_counter()
    w = build(task, M, inst)
    model = w["model"]
    B = 2 ** int(log2)
    edges = K.log_windows(B, PER_DOUBLING, MIN_WIDTH)
    be = get_backend("cpu")
    plan = be.prepare(model, strategy="recompute", layout="gate_blocked", threads=threads, event_counters=True,
                      event_windows=edges)
    targets = None if w["E_ref"] is None else np.asarray(w["E_ref"], dtype=np.float64)
    st = be.init_state(plan, w["binding"], w["spins0"], **w["kw"], targets=targets)
    E0 = np.asarray(st.rs.energies[:, 0]).copy()
    tau_s = float(w["tau_s"])
    tau_end = tau_s / KCOLD
    sched = S.fn_endpoints(tau_s, tau_end, ETA, B)
    # progress reports every two octaves (windows and counters do not depend on where a run is cut)
    cuts = sorted({0, B} | {2 ** k for k in range(10, int(log2), 2)})
    t0 = time.perf_counter()
    res = None
    for a, b in zip(cuts[:-1], cuts[1:]):
        res = be.run(plan, st, sched, n_attempts_total=B, start=a, stop=b, seed=SEED, stop_on_target=False)
        c = res.state.counters
        print(f"{task}: {b}/{B} iterations, acceptance {int(c['accepted'].sum()) / max(1, int(c['attempts'].sum())):.3f}, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    wall = time.perf_counter() - t0
    wc = be.window_counters(st)
    s = res.state
    r = structural_resources(model)
    fan_w = (model.route_wend - model.route_ptr[:-1]).astype(np.int64)
    meta = dict(task=task, budget_log2=int(log2), budget=B, replicas=M, seed=SEED, kcold=KCOLD, eta=ETA,
                per_doubling=PER_DOUBLING, min_width=MIN_WIDTH, tau_start=tau_s, tau_end=tau_end,
                tau_end_calibrated=float(w["tau_e_cal"]), tau_source=w["tau_source"], ref_kind=w["ref_kind"],
                R=int(model.n_groups), routes=int(model.n_routes), routes_weighted=int(fan_w.sum()),
                fan_in_mean=float(fan_w.mean()), fan_in_max=int(fan_w.max()),
                fan_in_routes_mean=float(np.diff(model.route_ptr).mean()),
                decoder_fan_out_mean=r["decoder_fan_out_mean"], wall_run_s=wall,
                wall_total_s=time.perf_counter() - t_all, threads=threads,
                win_fields=list(K.WIN_FIELDS), event_names=list(K.EVENT_NAMES),
                info={k: v for k, v in w["info"].items() if not isinstance(v, np.ndarray)})
    win = np.stack([wc[k] for k in K.WIN_FIELDS], axis=-1)  # [replicas, windows, fields]
    tmp = out / f".{task}.tmp.npz"
    np.savez_compressed(tmp, meta=np.asarray(json.dumps(meta)), edges=wc["edges"], win=win,
                        win_best=wc["best_energy"], win_fbest=wc["feasible_best"], E0=E0,
                        E_ref=np.full(M, np.nan) if w["E_ref"] is None else np.asarray(w["E_ref"], dtype=np.float64),
                        best_energy=np.asarray(s.best_energy), final_energy=np.asarray(s.energies[:, 0]),
                        target_hit_attempt=np.asarray(s.counters["target_hit_attempt"]),
                        tot=np.stack([np.asarray(s.counters[k]) for k in ("attempts", "accepted", "accepted_changed",
                                                                            "uphill_accepted")]
                                     + [np.asarray(s.counters["ev_" + k]) for k in K.EVENT_NAMES]))
    os.replace(tmp, f)
    hit = np.asarray(s.counters["target_hit_attempt"]) >= 0
    print(f"{task}: done, {wall:.0f} s, reference reached by {int(hit.sum())}/{M} replicas", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list")
    r = sp.add_parser("run")
    r.add_argument("--task")
    r.add_argument("--all", action="store_true")
    r.add_argument("--only", nargs="*", default=None, help="task prefixes (with --all)")
    r.add_argument("--skip", nargs="*", default=(), help="task prefixes to skip (with --all)")
    r.add_argument("--log2", type=int, default=None, help="log2 iterations per replica (default: per workload)")
    r.add_argument("--replicas", type=int, default=None)
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--inst", default=None, help="staged matching instances (default: $NIMBUS_MATCHING_INST)")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        print("\n".join(tasks()))
        return
    todo = tasks() if a.all else [a.task]
    if a.all and a.only:
        todo = [t for t in todo if any(t.startswith(p) for p in a.only)]
    todo = [t for t in todo if not any(t.startswith(p) for p in a.skip)]
    for t in todo:
        d = DEFAULTS.get(t.split("__")[0], DEFAULTS["syn"])
        run_task(t, a.log2 or d[0], a.replicas or d[1], a.threads, a.out, a.inst)


if __name__ == "__main__":
    main()

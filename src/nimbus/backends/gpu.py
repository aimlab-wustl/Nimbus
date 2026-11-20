"""GPU architecture backend (``"gpu"``): CUDA C++ compiled at run time by NVRTC through CuPy.

The kernel source lives in
``src/nimbus/backends/kernels/anneal.cu`` (includes ``common.cuh``/``philox.cuh``) and is compiled
once per kernel configuration with :class:`cupy.RawModule`, cached in this module so repeated
``run`` calls reuse the compiled kernel. Strategy ``"recompute"`` (selected-input recomputation), ``"incremental"`` (every v_r maintained per
replica as a 64-bit fixed-point integer, updated by stage 1/2 of the transaction) and
``"adaptive"`` (recompute or incremental per sub-window from the observed acceptance).
Three kernel variants parallelize the work at different granularities:

* ``"thread"``: one thread per replica, routes of the drawn group visited sequentially;
* ``"warp"``: one warp (32 lanes) per replica, lanes stride over the group's routes;
* ``"block"``: one thread block (64..1024 threads) per replica, for very large groups.

All variants execute the identical transaction: v_r over the weighted routes
from the pre-transition clause signs, sigma latched over ALL routes of the group before any toggle,
T <- (1 - 2 sigma) T for every latched clause (spin singletons included), same RNG,
acceptance, counters and best/feasible-best/target semantics. They differ only in the summation
order of v (recorded in ``plan.describe()`` / ``RunResult.metadata["kernel"]``). ``kernel_variant=
"auto"`` picks a variant per model and replica count (:func:`choose_variant`). float64 only.

Import never fails for lack of CuPy or a CUDA device (``backends.base.get_backend`` must still be
able to *import* this module to register the name); :meth:`GPUBackend.prepare` and
:meth:`GPUBackend.init_state` raise :class:`nimbus.core.errors.BackendUnavailable` instead, and the
backend never silently falls back to the CPU.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from nimbus.annealing.schedule import Schedule
from nimbus.backends.base import (
    COUNTER_NAMES,
    PROCESS_COUNTER_NAMES,
    ProcessSpec,
    check_process_budget,
    cohort_ids,
    ensure_process_counters,
    singleton_routes,
    termination_labels,
    Backend,
    BackendPlan,
    ReplicaStates,
    RunResult,
    SlotLayout,
    apply_targets,
    clause_signs,
    coerce_bindings,
    coerce_replicas,
    check_initial_feasibility,
    host_versions,
    initial_replica_states,
    register_backend,
    resolve_window,
    route_switches,
)
from nimbus.core.errors import BackendUnavailable, ConfigError, IncompatibleCheckpoint
from nimbus.core.model import CompiledModel
from nimbus.experiments.recording import RecordingData, RecordingSpec

__all__ = ["GPUBackend", "GPUPlan", "GPUState", "KernelConfig", "cuda_available", "COUNTER_INDEX",
           "VARIANTS", "choose_variant", "resolve_kernel_config", "route_stats", "pack_routes", "weighted_offsets",
           "per_route_weights", "STRATEGIES", "dependency_records", "fixed_point_weights", "choose_strategy",
           "choose_adaptive_strategy", "strategy_costs", "SCHEDULINGS", "PERSISTENT_BLOCKS_PER_SM",
           "resolve_scheduling"]

_KERNELS_DIR = Path(__file__).resolve().parent / "kernels"
_KERNEL_NAME = "tpi_anneal_recompute"

# must match TPI_C_* in kernels/anneal.cu
COUNTER_INDEX = {"attempts": 0, "accepted": 1, "accepted_changed": 2, "self_loops": 3, "rejected": 4,
                  "uphill_accepted": 5, "target_hit_attempt": 6}
assert tuple(sorted(COUNTER_INDEX, key=COUNTER_INDEX.get)) == COUNTER_NAMES
N_COUNTERS = len(COUNTER_INDEX)
# device counter columns (kernels/anneal.cu TPI_C_*): the base counters, then steps,
# candidate_evaluations, frozen_attempt
DEVICE_COUNTERS = COUNTER_NAMES + PROCESS_COUNTER_NAMES
PROC_KIND = {"select": 0, "shared": 1, "sweep": 2, "parallel_trial": 3, "nfold": 4}

VARIANTS = ("thread", "warp", "block")
STRATEGIES = ("recompute", "incremental", "adaptive")
SCHEDULINGS = ("static", "queue", "auto")
# Persistent (queue) scheduling: grid size is capped at
# n_SM * PERSISTENT_BLOCKS_PER_SM resident replica-group workers instead of one per replica, each pulling
# the next replica index from a global atomic cursor when it finishes its current one. The cap is a
# measured-conservative heuristic (not an occupancy-API query); it only
# affects how many workers are in flight, never the result.
PERSISTENT_BLOCKS_PER_SM = 4
FIXED_POINT_BITS = 61                   # |V| <= 2^61: max_r sum |J| * 2^e_b <= 2^61
# adaptive strategy (choose_adaptive_strategy): sub-window = max(MIN, B / DIVISOR) attempts, capped at B
ADAPTIVE_MIN_SUBWINDOW = 4096
ADAPTIVE_SUBWINDOW_DIVISOR = 64
ADAPTIVE_HYSTERESIS = 0.20
ADAPTIVE_C0 = 8.0                       # incremental's fixed per-attempt cost (draw, cached read), route units
ADAPTIVE_PROPAGATION_FACTOR = 1.0       # weight of the stage-1/2 upper bound relative to a route visit
ADAPTIVE_PROBE_RATIO = 3.0              # probe incremental while the structural model predicts <= 3x its cost
ADAPTIVE_PROBE_FRACTION = 8             # a probe window is W / 8 attempts
ADAPTIVE_PROBE_GAP = 4                  # windows between probes (doubled after a failed probe, max 64)
ADAPTIVE_REVERT_P_FACTOR = 2.0          # back to recompute when acceptance rises 2x above the switch point
DEFAULT_BLOCK_THREADS = {"thread": 128, "warp": 128, "block": 256}
# auto rule (choose_variant), calibrated on an NVIDIA H100; f = mean
# routes per group (weighted + gate + readout), n = replicas, n_SM = streaming multiprocessors
AUTO_THREAD_MAX_FANIN = 8.0       # thread only for tiny groups and many replicas (extrapolated: never best measured)
AUTO_THREAD_MIN_REP_PER_SM = 32
AUTO_BLOCK_MIN_FANIN = 2048.0     # block at every replica count from here on
AUTO_BLOCK_SMALL_MIN_FANIN = 32.0  # block also for few replicas (n <= 2 n_SM) once groups have >= 32 routes
AUTO_BLOCK_MAX_REP_PER_SM = 2
AUTO_WARP_SHARED_MIN_BYTES = 12 * 1024  # warp variant: T in shared memory only for M~ + 1 above this
# proposal processes: parallel_trial / nfold scan all
# R groups per step; shared lets the replicas of a warp read one group (replica-minor T, thread variant)
AUTO_STEP_THREAD_REP_PER_SM = 8
AUTO_STEP_BLOCK_WORK = 1.0e5
AUTO_SHARED_MINOR = True
AUTO_SHARED_THREAD_REP_PER_SM = 8
# per-route weight copies Jw[b][pos] are used while n_bindings * weighted_routes * 8 B fits this budget
JW_BUDGET_BYTES = 512 * 2 ** 20
SMEM_DEFAULT_BYTES = 48 * 1024          # without the opt-in attribute
SMEM_RESERVED_BYTES = 1024              # static shared memory (block reduction scratch) + margin
H100_SMEM_OPTIN_BYTES = 227 * 1024      # cudaDevAttrMaxSharedMemoryPerBlockOptin on an H100 (host-side default)
H100_SM_COUNT = 132
_ATTR_SMEM_OPTIN = 97                   # cudaDevAttrMaxSharedMemoryPerBlockOptin
_ATTR_SM_COUNT = 16                     # cudaDevAttrMultiProcessorCount

SUMMATION_ORDER = {
    "thread": "v = sequential sum in route order over the group's weighted routes with alpha = 1",
    "warp": ("v: lane l sums its routes route_ptr[r] + l + 32 j (j ascending, alpha = 1 only), then a "
             "xor-butterfly over the 32 lanes (offsets 16, 8, 4, 2, 1); identical in every lane"),
    "block": ("v: thread l sums its routes route_ptr[r] + l + B j (B = block threads), xor-butterfly within "
              "each warp, then warp 0 butterflies the per-warp sums (missing warps padded with 0.0)"),
}

try:  # CuPy is optional: a CPU-only install must still be able to `import nimbus.backends.gpu`
    import cupy as cp  # noqa: F401
    _CUPY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised without CuPy / CUDA
    cp = None
    _CUPY_IMPORT_ERROR = exc


def cuda_available() -> bool:
    """True iff CuPy is importable and reports at least one usable CUDA device."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _require_cuda():
    if cp is None:
        raise BackendUnavailable(f"backend 'gpu' needs CuPy, which is not importable here ({_CUPY_IMPORT_ERROR!r})")
    if not cuda_available():
        raise BackendUnavailable("backend 'gpu' needs a CUDA device; none is visible to CuPy here")


def _inline_headers(text: str, seen=None) -> str:
    seen = set() if seen is None else seen
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('#include "') and s.endswith('"'):
            name = s.split('"')[1]
            if name in seen:
                continue
            seen.add(name)
            out.append(_inline_headers((_KERNELS_DIR / name).read_text(), seen))
        else:
            out.append(line)
    return "\n".join(out)


_KERNEL_SOURCE_TEMPLATE = _inline_headers((_KERNELS_DIR / "anneal.cu").read_text())

# process-local compile cache: kernel macro tuple -> cupy.RawModule
_MODULE_CACHE: dict = {}


_REBUILD_NAME = "tpi_rebuild_v"


def kernel_source(lanes: int, max_slots: int, t_shared: bool, latch_mark: bool, incremental: bool = False,
                  v_shared: bool = False, t_minor: bool = False, persistent: bool = False) -> str:
    """Full NVRTC source of one kernel configuration (also used by the NVRTC compile check)."""
    return (f"#define TPI_LANES {int(lanes)}\n#define TPI_MAX_SLOTS {int(max_slots)}\n"
            f"#define TPI_T_SHARED {int(bool(t_shared))}\n#define TPI_LATCH_MARK {int(bool(latch_mark))}\n"
            f"#define TPI_INCREMENTAL {int(bool(incremental))}\n#define TPI_V_SHARED {int(bool(v_shared))}\n"
            f"#define TPI_T_MINOR {int(bool(t_minor))}\n#define TPI_PERSISTENT {int(bool(persistent))}\n"
            + _KERNEL_SOURCE_TEMPLATE)


def _compiled_module(lanes: int, max_slots: int, t_shared: bool, latch_mark: bool, incremental: bool = False,
                     v_shared: bool = False, t_minor: bool = False, persistent: bool = False):
    key = (int(lanes), int(max_slots), bool(t_shared), bool(latch_mark), bool(incremental), bool(v_shared),
           bool(t_minor), bool(persistent))
    if key not in _MODULE_CACHE:
        mod = cp.RawModule(code=kernel_source(*key), options=("--std=c++14",),
                           name_expressions=[_KERNEL_NAME, _REBUILD_NAME])
        mod.get_function(_KERNEL_NAME)  # force compilation now (reports NVRTC errors eagerly)
        _MODULE_CACHE[key] = mod
    return _MODULE_CACHE[key]


def _compiled_function(lanes: int, max_slots: int, t_shared: bool, latch_mark: bool, incremental: bool = False,
                       v_shared: bool = False, t_minor: bool = False, persistent: bool = False):
    return _compiled_module(lanes, max_slots, t_shared, latch_mark, incremental, v_shared, t_minor,
                            persistent).get_function(_KERNEL_NAME)


# ============================================================================================
# host-side layout (pure numpy; shared by the plan and the host-side mirror tests)
# ============================================================================================
def pack_routes(model: CompiledModel) -> np.ndarray:
    """One int32 pair per route position: x = clause | (is_spin_singleton << 31),
    y = (controller << 1) | (pol < 0), controller = M~ (the dummy slot T[M~] = +1) for a constant switch."""
    M = int(model.n_clauses)
    if model.n_routes >= 2 ** 31 or M >= 2 ** 30:
        raise ConfigError("backend 'gpu' route records need < 2^31 routes and < 2^30 clauses")
    is_spin = np.zeros(M, dtype=bool)
    is_spin[model.readout_clause_of_spin.astype(np.int64)] = True
    clause = model.route_clause.astype(np.int64)
    ctrl = model.route_ctrl.astype(np.int64)
    neg = (ctrl >= 0) & (model.route_pol.astype(np.int64) < 0)
    x = (clause | (is_spin[clause].astype(np.int64) << 31)).astype(np.uint32).view(np.int32)
    y = ((np.where(ctrl < 0, M, ctrl) << 1) | neg.astype(np.int64)).astype(np.int32)
    return np.ascontiguousarray(np.stack([x, y], axis=1))


def weighted_offsets(model: CompiledModel) -> tuple[np.ndarray, np.ndarray]:
    """(weighted route positions in position order, jw_off[r]) with Jw index = jw_off[r] + pos."""
    ptr = model.route_ptr.astype(np.int64)
    wend = model.route_wend.astype(np.int64)
    cnt = wend - ptr[:-1]
    wstart = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64) if cnt.size else np.zeros(0, np.int64)
    wpos = np.flatnonzero(model.weighted_route_mask()).astype(np.int64)
    return wpos, (wstart - ptr[:-1]).astype(np.int64)


def per_route_weights(model: CompiledModel, J0s: np.ndarray, wpos: np.ndarray) -> np.ndarray:
    """``Jw[b][i] = J[b][0][route_clause[wpos[i]]]`` for every binding b (contiguous per group)."""
    return np.ascontiguousarray(np.asarray(J0s, dtype=np.float64)[:, model.route_clause[wpos].astype(np.int64)])


def dependency_records(model: CompiledModel) -> dict:
    """Stage-1/2 lists for the incremental strategy (dependency CSRs, packed):
    cw: clause k -> its weighted routes, int2 {group r', (controller << 1) | neg} (as in pack_routes);
    cc: controller g -> the weighted routes it controls, int2 {group r', clause k' | (pol < 0) << 31}."""
    group = np.repeat(np.arange(model.n_groups, dtype=np.int64), np.diff(model.route_ptr).astype(np.int64))
    rec = pack_routes(model)
    cw = model.clause_wroute.astype(np.int64)
    cc = model.ctrl_wroute.astype(np.int64)
    cw_rec = np.stack([group[cw], rec[cw, 1].astype(np.int64)], axis=1).astype(np.int32)
    neg = (model.route_pol[cc].astype(np.int64) < 0).astype(np.int64)
    ccx = (model.route_clause[cc].astype(np.int64) | (neg << 31)).astype(np.uint32).view(np.int32)
    cc_rec = np.stack([group[cc].astype(np.int32), ccx], axis=1).astype(np.int32)
    return {"cw_ptr": model.clause_wroute_ptr.astype(np.int64), "cw_rec": np.ascontiguousarray(cw_rec.reshape(-1, 2)),
            "cc_ptr": model.ctrl_wroute_ptr.astype(np.int64), "cc_rec": np.ascontiguousarray(cc_rec.reshape(-1, 2))}


def fixed_point_weights(model: CompiledModel, J0s: np.ndarray):
    """(Jc int64[n_b, M~], vscale float64[n_b], exact bool[n_b]) for the maintained inputs.

    e_b = FIXED_POINT_BITS - ceil(log2 S_b), S_b = max over groups of sum over weighted routes |J0[k]|,
    so |V_r| <= 2^61; Jc = round(J0 * 2^e_b); vscale = 2^-e_b; exact[b]: every Jc * 2^-e_b == J0 (true
    for the dyadic test weights, and for any weights within ~2^-9 S_b of the largest)."""
    J0s = np.atleast_2d(np.asarray(J0s, dtype=np.float64))
    wmask = model.weighted_route_mask()
    rc = model.route_clause.astype(np.int64)
    gid = np.repeat(np.arange(model.n_groups), np.diff(model.route_ptr).astype(np.int64))
    Jc = np.zeros(J0s.shape, dtype=np.int64)
    vscale = np.ones(J0s.shape[0])
    exact = np.ones(J0s.shape[0], dtype=bool)
    for b in range(J0s.shape[0]):
        a = np.abs(J0s[b][rc]) * wmask
        S = float(np.max(np.bincount(gid, weights=a, minlength=model.n_groups))) if model.n_groups else 0.0
        e = 0 if S == 0.0 else int(FIXED_POINT_BITS - np.ceil(np.log2(S)))
        e = int(np.clip(e, -1000, 1000))
        Jc[b] = np.rint(np.ldexp(J0s[b], e)).astype(np.int64)
        vscale[b] = float(np.ldexp(1.0, -e))
        exact[b] = bool(np.array_equal(np.ldexp(Jc[b].astype(np.float64), -e), J0s[b]))
    return Jc, vscale, exact


def strategy_costs(model: CompiledModel) -> dict:
    """Structural per-attempt costs (in route visits) for the strategy choice: wfan (pass 1 of
    recompute), dfan (decoder routes, paid on acceptance by both), prop (mean over groups of the
    stage-1/2 work if every route of the group toggled: sum of |clause_wroute[k]| + |ctrl_wroute[k]|)."""
    fan = np.diff(model.route_ptr).astype(np.int64)
    wfan = (model.route_wend - model.route_ptr[:-1]).astype(np.int64)
    per_clause = np.diff(model.clause_wroute_ptr).astype(np.int64) + np.diff(model.ctrl_wroute_ptr).astype(np.int64)
    gid = np.repeat(np.arange(model.n_groups), fan)
    prop = np.bincount(gid, weights=per_clause[model.route_clause.astype(np.int64)], minlength=model.n_groups)
    m = (lambda a: float(np.mean(a)) if np.size(a) else 0.0)
    return {"wfan": m(wfan), "dfan": m(fan), "prop": m(prop)}


def choose_adaptive_strategy(costs: dict, p_acc: float, prev: str, hysteresis: float = ADAPTIVE_HYSTERESIS):
    """(strategy, cost_recompute, cost_incremental) for the next sub-window, at acceptance p_acc:
    recompute = wfan + p (2 dfan); incremental = C0 + p (3 dfan + F prop). Switch away from ``prev``
    only if the predicted cost drops by more than ``hysteresis``."""
    cr = costs["wfan"] + p_acc * 2.0 * costs["dfan"]
    ci = ADAPTIVE_C0 + p_acc * (3.0 * costs["dfan"] + ADAPTIVE_PROPAGATION_FACTOR * costs["prop"])
    best = "incremental" if ci < cr else "recompute"
    if best != prev:
        old, new = (cr, ci) if prev == "recompute" else (ci, cr)
        if not new < (1.0 - hysteresis) * old:
            best = prev
    return best, cr, ci


def choose_strategy(model: CompiledModel, process: str = "select") -> tuple[str, str]:
    """``strategy="auto"``: (strategy, reason).
    parallel_trial / nfold read every input every step: incremental when one commit's stage-1/2 work
    (prop) is below a full recompute of all inputs (R * wfan), else recompute."""
    if process in ("parallel_trial", "nfold"):
        c = strategy_costs(model)
        full = model.n_groups * c["wfan"]
        pick = "incremental" if c["prop"] < full else "recompute"
        return pick, f"auto ({process}): stage-1/2 work {c['prop']:.0f} vs recompute of all inputs {full:.0f}"
    return "adaptive", "auto: adaptive (recompute at high acceptance, incremental at low; costs " \
        + ", ".join(f"{k} {v:.1f}" for k, v in strategy_costs(model).items()) + ")"


def route_stats(model: CompiledModel) -> dict:
    """Structural numbers the variant choice uses (routes per group = weighted + gate + readout)."""
    fan = np.diff(model.route_ptr).astype(np.int64)
    wfan = (model.route_wend - model.route_ptr[:-1]).astype(np.int64)
    return {"n_groups": int(model.n_groups), "n_clauses": int(model.n_clauses), "n_spins": int(model.n_spins),
            "n_routes": int(model.n_routes), "n_weighted_routes": int(wfan.sum()),
            "fanin_mean": float(fan.mean()) if fan.size else 0.0, "fanin_max": int(fan.max()) if fan.size else 0,
            "wfanin_mean": float(wfan.mean()) if wfan.size else 0.0}


def resolve_scheduling(scheduling: str, stop_on_target: bool, process: str) -> str:
    """``scheduling="auto"`` -> ``"static"`` or ``"queue"`` (pure, no device): persistent (queue) scheduling
    is worth its atomic-counter traffic only when replicas in this run can end before the full budget --
    ``stop_on_target`` (every process), or ``process == "nfold"`` (can freeze early even without a target). Otherwise every worker runs to the same attempt_stop regardless of
    scheduling, so static (no atomics) is strictly cheaper and is used."""
    if scheduling not in SCHEDULINGS:
        raise ConfigError(f"backend 'gpu' scheduling must be one of {SCHEDULINGS}, got {scheduling!r}")
    if scheduling != "auto":
        return scheduling
    return "queue" if (stop_on_target or process == "nfold") else "static"


def choose_variant(stats: dict, n_rep: int | None = None, n_sm: int = H100_SM_COUNT,
                   process: str = "select") -> tuple[str, str]:
    """``kernel_variant="auto"``: (variant, reason) from the mean routes per group f and the replica count n.

    A wider replica group pays a reduction and barriers per attempt but splits the route loop; it
    pays off while there are enough routes per lane and the GPU is not already filled by replicas.
    Measured on an H100: block wins for f >= 2048 at every n and for
    f >= 32 with n <= 2 n_SM; warp wins otherwise (f = 18..1600 at n >= 1024, and f = 18 at every n);
    thread never won a measured point (it ties warp only at f = 18, n = 4096, low acceptance), so it
    is chosen only for f <= 8 with n >= 32 n_SM (an extrapolation)."""
    f = float(stats["fanin_mean"])
    n = 1 if n_rep is None else max(int(n_rep), 1)
    if process in ("parallel_trial", "nfold"):  # one scan over all R groups per step
        R = int(stats.get("n_groups", 1))
        if n >= AUTO_STEP_THREAD_REP_PER_SM * n_sm:
            return "thread", f"{process}: {n} replicas >= {AUTO_STEP_THREAD_REP_PER_SM} x {n_sm} SMs"
        if R * f >= AUTO_STEP_BLOCK_WORK and n <= AUTO_BLOCK_MAX_REP_PER_SM * n_sm:
            return "block", f"{process}: R * f = {R * f:.0f} >= {AUTO_STEP_BLOCK_WORK:g}, {n} replicas"
        return "warp", f"{process}: R = {R}, {n} replicas"
    if process == "shared" and AUTO_SHARED_MINOR and n >= AUTO_SHARED_THREAD_REP_PER_SM * n_sm:
        return "thread", f"shared: {n} replicas >= {AUTO_SHARED_THREAD_REP_PER_SM} x {n_sm} SMs (lock-step cohorts)"
    if f <= AUTO_THREAD_MAX_FANIN and n >= AUTO_THREAD_MIN_REP_PER_SM * n_sm:
        return "thread", f"mean routes per group {f:.1f} <= {AUTO_THREAD_MAX_FANIN:g} and {n} replicas >= " + \
            f"{AUTO_THREAD_MIN_REP_PER_SM} x {n_sm} SMs"
    if f >= AUTO_BLOCK_MIN_FANIN:
        return "block", f"mean routes per group {f:.1f} >= {AUTO_BLOCK_MIN_FANIN:g}"
    if f >= AUTO_BLOCK_SMALL_MIN_FANIN and n <= AUTO_BLOCK_MAX_REP_PER_SM * n_sm:
        return "block", f"mean routes per group {f:.1f} >= {AUTO_BLOCK_SMALL_MIN_FANIN:g} and {n} replicas <= " + \
            f"{AUTO_BLOCK_MAX_REP_PER_SM} x {n_sm} SMs"
    return "warp", f"mean routes per group {f:.1f}, {n} replicas (neither thread nor block conditions hold)"


@dataclass(frozen=True)
class KernelConfig:
    """One resolved kernel configuration (what a launch uses; recorded in RunResult.metadata)."""

    variant: str
    lanes: int                 # threads per replica
    block_threads: int
    replicas_per_block: int
    t_shared: bool             # clause signs T in shared memory for the chunk
    t_smem_stride: int         # bytes per replica in shared memory (M~ + 1 rounded up to 16)
    smem_bytes: int            # dynamic shared memory per block
    latch: str                 # "mask" (register alpha bits) | "mark" (in-place sign-keeping marks)
    reason: str
    summation_order: str
    incremental: bool = False  # maintained fixed-point inputs V[R] (strategy incremental)
    v_shared: bool = False     # V in shared memory for the chunk
    v_smem_base: int = 0       # byte offset of the V region in dynamic shared memory
    t_minor: bool = False      # replica-minor clause signs (thread variant; lock-step cohorts)

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_kernel_config(stats: dict, n_rep: int, *, kernel_variant="auto", t_memory="auto", latch="auto",
                          block_threads=None, smem_limit=H100_SMEM_OPTIN_BYTES, n_sm=H100_SM_COUNT,
                          incremental: bool = False, v_memory="auto", t_layout="auto",
                          process: str = "select") -> KernelConfig:
    """Pure function (no device): variant, launch shape, T placement and latch mode for one state.

    Shared-memory rule: T (M~ + 1 bytes, padded to 16) goes to shared memory when ``t_memory`` is
    'shared' or 'auto' and ``replicas_per_block`` copies fit ``smem_limit`` (the device opt-in
    maximum minus static use; replicas per block are reduced to make it fit: thread variant down to
    32, warp variant down to 1); otherwise T stays in global memory ('shared' then raises).
    Incremental: the maintained inputs V (8 R bytes, padded to 16) also go to shared memory when they
    fit -- tried together with T first, then V alone; the latch is always 'mark'."""
    variant = kernel_variant
    if variant == "auto":
        variant, reason = choose_variant(stats, n_rep, n_sm, process)
        reason = "auto: " + reason
    elif variant in VARIANTS:
        reason = "forced"
    else:
        raise ConfigError(f"backend 'gpu' kernel_variant must be auto/{'/'.join(VARIANTS)}, got {variant!r}")
    if t_memory not in ("auto", "shared", "global"):
        raise ConfigError(f"backend 'gpu' t_memory must be auto/shared/global, got {t_memory!r}")
    if latch not in ("auto", "mask", "mark"):
        raise ConfigError(f"backend 'gpu' latch must be auto/mask/mark, got {latch!r}")
    n_rep = max(int(n_rep), 1)
    if variant == "thread":
        bt = int(block_threads or (DEFAULT_BLOCK_THREADS["thread"] if n_rep >= 128 * n_sm else 32))
        if bt % 32 or not 32 <= bt <= 1024:
            raise ConfigError("thread variant block_threads must be a multiple of 32 in [32, 1024]")
        lanes, rpb = 1, bt
    elif variant == "warp":
        bt = int(block_threads or (DEFAULT_BLOCK_THREADS["warp"] if n_rep >= 4 * n_sm else 32))
        if bt % 32 or not 32 <= bt <= 1024:
            raise ConfigError("warp variant block_threads must be a multiple of 32 in [32, 1024]")
        lanes, rpb = 32, bt // 32
    else:
        bt = int(block_threads or DEFAULT_BLOCK_THREADS["block"])
        if bt % 32 or not 64 <= bt <= 1024:
            raise ConfigError("block variant block_threads must be a multiple of 32 in [64, 1024]")
        lanes, rpb = bt, 1
    stride = (int(stats["n_clauses"]) + 1 + 15) // 16 * 16
    vbytes = (8 * int(stats.get("n_groups", 0)) + 15) // 16 * 16
    if t_memory == "auto":  # measured: block gains from shared T; warp only for large M~; thread never
        t_memory = {"block": "auto", "warp": "auto" if stride > AUTO_WARP_SHARED_MIN_BYTES else "global",
                    "thread": "global"}[variant]
    options = ([(True, True)] if incremental and t_memory != "global" else []) + \
        ([(True, False)] if not incremental and t_memory != "global" else []) + \
        ([(False, True)] if incremental else []) + [(False, False)]
    t_shared = v_shared = False
    if v_memory not in ("auto", "global"):
        raise ConfigError(f"backend 'gpu' v_memory must be auto/global, got {v_memory!r}")
    if v_memory == "global":
        options = [o for o in options if not o[1]]
    for ts, vs in options:
        per = (stride if ts else 0) + (vbytes if vs else 0)
        if per == 0:
            break
        fit = int(smem_limit) // per
        rpb_fit = min(rpb, fit // 32 * 32) if variant == "thread" else min(rpb, fit)
        if rpb_fit >= (32 if variant == "thread" else 1):
            t_shared, v_shared, rpb = ts, vs, rpb_fit
            bt = rpb * lanes if variant != "block" else bt
            break
    if t_memory == "shared" and not t_shared:
        raise ConfigError(f"t_memory='shared': {stride} B of clause signs per replica do not fit the "
                          f"shared-memory limit {smem_limit} B for variant {variant!r}")
    max_fan = int(stats["fanin_max"])
    if incremental:  # stage 1/2 read the marked (pre-transition) state: always the mark latch
        latch = "mark"
    elif latch == "auto":
        latch = "mask" if max_fan <= 64 * lanes else "mark"
    elif latch == "mask" and max_fan > 64 * lanes:
        raise ConfigError(f"latch='mask' needs max routes per group ({max_fan}) <= 64 * lanes ({64 * lanes})")
    if t_layout not in ("auto", "major", "minor"):
        raise ConfigError(f"backend 'gpu' t_layout must be auto/major/minor, got {t_layout!r}")
    if t_layout == "minor" and variant != "thread":
        raise ConfigError("t_layout='minor' (replica-minor clause signs) needs kernel_variant='thread'")
    t_minor = t_layout == "minor" or (t_layout == "auto" and variant == "thread" and process == "shared"
                                      and AUTO_SHARED_MINOR)
    t_bytes = stride * rpb if t_shared else 0
    return KernelConfig(variant=variant, lanes=lanes, block_threads=bt, replicas_per_block=rpb,
                        t_shared=t_shared, t_smem_stride=stride, smem_bytes=t_bytes + (vbytes * rpb if v_shared else 0),
                        latch=latch, reason=reason,
                        summation_order=("v maintained as an exact 64-bit fixed-point sum (order-independent)"
                                         if incremental else SUMMATION_ORDER[variant]),
                        incremental=bool(incremental), v_shared=v_shared, v_smem_base=t_bytes, t_minor=t_minor)


# ============================================================================================
# plan
# ============================================================================================
class GPUPlan(BackendPlan):
    """Device-resident topology (packed route records, weighted-route offsets, spin-singleton map)
    shared by every replica of this model, plus the kernel-variant request.

    No transformation descriptors reach the device: the kernel executes mu^(r) through the decoder
    toggles of the spin singleton clauses."""

    def __init__(self, model: CompiledModel, *, strategy="auto", layout="csr", precision="float64",
                 device="cuda:0", kernel_variant="auto", t_memory="auto", latch="auto", block_threads=None,
                 jw="auto", jw_budget_bytes=JW_BUDGET_BYTES, v_memory="auto", adaptive_subwindow=None,
                 t_layout="auto", process="select", process_options=None, scheduling="auto", color_threads=32):
        if strategy not in ("auto",) + STRATEGIES:
            raise ConfigError(f"backend 'gpu' strategy must be auto/{'/'.join(STRATEGIES)}, got {strategy!r}")
        if scheduling not in SCHEDULINGS:
            raise ConfigError(f"backend 'gpu' scheduling must be one of {SCHEDULINGS}, got {scheduling!r}")
        self.scheduling = scheduling
        if layout not in ("auto", "csr"):
            raise ConfigError(f"backend 'gpu' layout must be 'csr', got {layout!r}")
        if precision != "float64":
            raise ConfigError("backend 'gpu' computes in float64 only in this build (float32 not implemented)")
        if kernel_variant not in ("auto",) + VARIANTS:
            raise ConfigError(f"backend 'gpu' kernel_variant must be auto/{'/'.join(VARIANTS)}, got {kernel_variant!r}")
        if jw not in ("auto", "on", "off"):
            raise ConfigError(f"backend 'gpu' jw must be auto/on/off, got {jw!r}")
        self.process = ProcessSpec.coerce(process, process_options)
        self.coloring = None
        self.color_threads = 32
        if self.process.colored:
            # separate kernel (kernels/anneal_colored.cu, gpu_colored.py); recompute only, one block per replica
            from nimbus.backends import gpu_colored as GC
            from nimbus.backends.coloring import color_groups
            if strategy not in ("auto", "recompute"):
                raise ConfigError("the gpu colored process uses strategy 'recompute' only")
            strategy = "recompute"
            self.color_threads = GC.check_threads(color_threads)
            self.coloring = color_groups(model)
        self.stats = route_stats(model)
        self.max_fanin = max(self.stats["fanin_max"], 1)
        self.kernel_request = {"kernel_variant": kernel_variant, "t_memory": t_memory, "latch": latch,
                               "block_threads": block_threads, "v_memory": v_memory, "t_layout": t_layout,
                               "process": self.process.name}
        self.adaptive_subwindow = None if adaptive_subwindow is None else max(1, int(adaptive_subwindow))
        self.strategy_request = strategy
        if strategy == "auto":
            strategy, self.strategy_reason = choose_strategy(model, self.process.name)
        else:
            self.strategy_reason = "forced"
        if strategy == "adaptive" and self.process.stepwise:
            raise ConfigError(f"the adaptive strategy switches per attempt window; {self.process.name!r} uses "
                              "'recompute' or 'incremental'")
        self.costs = strategy_costs(model)
        # validate the request on a nominal replica count now (raises ConfigError early)
        for inc in self.uses(strategy):
            resolve_kernel_config(self.stats, 1, incremental=inc, **self.kernel_request)
        self.jw_mode = jw
        self.jw_budget_bytes = int(jw_budget_bytes)
        super().__init__(backend="gpu", model=model, strategy=strategy, layout="csr", precision="float64",
                         device=str(device),
                         options={"max_fanin": self.max_fanin, "fanin_mean": self.stats["fanin_mean"],
                                  "kernel_variant": kernel_variant, "t_memory": t_memory, "latch": latch,
                                  "block_threads": block_threads, "jw": jw, "jw_budget_bytes": self.jw_budget_bytes,
                                  "strategy_request": self.strategy_request, "strategy_reason": self.strategy_reason,
                                  "scheduling": self.scheduling,
                                  "process": self.process.name,
                                  **{"process_" + k: v for k, v in self.process.describe().items() if k != "process"},
                                  **{f"cost_{k}": v for k, v in self.costs.items()},
                                  "auto_rule": (f"thread if mean routes/group f <= {AUTO_THREAD_MAX_FANIN:g} and replicas >= "
                                                f"{AUTO_THREAD_MIN_REP_PER_SM}*n_SM; block if f >= {AUTO_BLOCK_MIN_FANIN:g}, "
                                                f"or f >= {AUTO_BLOCK_SMALL_MIN_FANIN:g} and replicas <= "
                                                f"{AUTO_BLOCK_MAX_REP_PER_SM}*n_SM; else warp. T in shared memory: "
                                                f"block when it fits, warp when M~+1 > {AUTO_WARP_SHARED_MIN_BYTES} B "
                                                "and it fits, thread never; latch mask if max routes/group <= "
                                                "64*lanes else mark")})
        self.route_ptr = model.route_ptr.astype(np.int64)
        self.route_wend = model.route_wend.astype(np.int64)
        self.rec = pack_routes(model)
        self.wpos, self.jw_off = weighted_offsets(model)
        self.spin_clause = model.readout_clause_of_spin.astype(np.int32)
        self.deps = dependency_records(model) if self.needs_v else None
        sr_ptr, sr_pos = singleton_routes(model)
        self.sr_ptr = sr_ptr.astype(np.int64)
        self.sr_y = np.ascontiguousarray(self.rec[sr_pos, 1]) if sr_pos.size else np.zeros(1, np.int32)
        self._dev_topology = None  # populated lazily (one upload per plan)

    @property
    def R(self) -> int:
        return int(self.model.n_groups)

    @staticmethod
    def uses(strategy: str) -> tuple:
        """Incremental flags of the kernels a strategy launches."""
        return {"recompute": (False,), "incremental": (True,), "adaptive": (False, True)}[strategy]

    @property
    def needs_v(self) -> bool:
        return self.strategy in ("incremental", "adaptive")

    def use_jw(self, n_bindings: int) -> bool:
        if self.jw_mode == "off":
            return False
        if self.jw_mode == "on":
            return True
        return n_bindings * max(self.wpos.size, 1) * 8 <= self.jw_budget_bytes

    def kernel_config(self, n_rep: int, incremental: bool = False, **request) -> KernelConfig:
        """Kernel configuration for ``n_rep`` replicas on the current device (opt-in shared-memory
        limit and SM count queried from CUDA; H100 defaults without a device)."""
        smem_limit, n_sm = SMEM_DEFAULT_BYTES - SMEM_RESERVED_BYTES, H100_SM_COUNT
        if cuda_available():
            dev = cp.cuda.Device().id
            smem_limit = int(cp.cuda.runtime.deviceGetAttribute(_ATTR_SMEM_OPTIN, dev)) - SMEM_RESERVED_BYTES
            n_sm = int(cp.cuda.runtime.deviceGetAttribute(_ATTR_SM_COUNT, dev))
        return resolve_kernel_config(self.stats, n_rep, smem_limit=smem_limit, n_sm=n_sm, incremental=incremental,
                                     **dict(self.kernel_request, **request))

    def _device_topology(self):
        if self._dev_topology is None:
            self._dev_topology = {
                "route_ptr": cp.asarray(self.route_ptr), "route_wend": cp.asarray(self.route_wend),
                "jw_off": cp.asarray(self.jw_off), "rec": cp.asarray(self.rec.reshape(-1)),
                "spin_clause": cp.asarray(self.spin_clause)}
            deps = self.deps or {"cw_ptr": np.zeros(1, np.int64), "cw_rec": np.zeros((1, 2), np.int32),
                                 "cc_ptr": np.zeros(1, np.int64), "cc_rec": np.zeros((1, 2), np.int32)}
            for k in ("cw_ptr", "cc_ptr"):
                self._dev_topology[k] = cp.asarray(deps[k])
            for k in ("cw_rec", "cc_rec"):
                self._dev_topology[k] = cp.asarray(deps[k].reshape(-1) if deps[k].size else np.zeros(2, np.int32))
            self._dev_topology["sr_ptr"] = cp.asarray(self.sr_ptr)
            self._dev_topology["sr_y"] = cp.asarray(self.sr_y.astype(np.int32))
        return self._dev_topology

    def describe(self) -> dict:
        d = super().describe()
        d["summation_order"] = dict(SUMMATION_ORDER)
        d["route_stats"] = dict(self.stats)
        return d


# ============================================================================================
# state
# ============================================================================================
class GPUState:
    """Per-replica device arrays (T -- the authoritative state, rows of M~ + 1 with the dummy
    controller slot pinned to +1 --, energies, best/feasible-best readouts, counters). There is no
    device spin array."""

    def __init__(self, plan: GPUPlan, rs: ReplicaStates, bindings):
        self.plan = plan
        self.rs = rs
        self.bindings = tuple(bindings)
        self.layout = SlotLayout.from_bindings(self.bindings)
        n_slots = len(self.bindings[0].slot_names)
        n_bindings = len(self.bindings)
        M = plan.model.n_clauses
        J = np.zeros((n_bindings, n_slots, M), dtype=np.float64)
        const = np.zeros((n_bindings, n_slots), dtype=np.float64)
        for i, b in enumerate(self.bindings):
            J[i] = b.J
            const[i] = b.const
        self.d_J = cp.asarray(J)
        self.d_const = cp.asarray(const)
        self.use_jw = plan.use_jw(n_bindings)
        if self.use_jw:
            Jw = per_route_weights(plan.model, J[:, 0, :], plan.wpos)
            self.d_Jw, self.jw_stride = cp.asarray(Jw.reshape(-1) if Jw.size else np.zeros(1)), int(Jw.shape[1])
        else:
            self.d_Jw, self.jw_stride = cp.zeros(1, dtype=cp.float64), 0
        n = rs.n_replicas
        ensure_process_counters(rs)
        self.t_minor = False
        self.kernels = {}
        if plan.needs_v:
            Jc, vscale, exact = fixed_point_weights(plan.model, J[:, 0, :])
            self.vscale, self.v_exact = vscale, exact
            self.d_Jc, self.d_vscale = cp.asarray(Jc), cp.asarray(vscale)
            self.d_v = cp.zeros((n, max(plan.model.n_groups, 1)), dtype=cp.int64)
        else:
            self.vscale, self.v_exact = None, None
            self.d_Jc, self.d_vscale = cp.zeros(1, dtype=cp.int64), cp.ones(1, dtype=cp.float64)
            self.d_v = cp.zeros(1, dtype=cp.int64)
        self.v_valid = False
        T1 = np.ones((n, M + 1), dtype=np.int8)
        T1[:, :M] = rs.T
        self.d_T = cp.asarray(T1)  # replica-major; transposed by reconfigure() for a replica-minor kernel
        self.d_energies = cp.asarray(rs.energies)
        self.d_best_energy = cp.asarray(rs.best_energy)
        self.d_best_spins = cp.asarray(rs.best_spins)
        self.d_best_attempt = cp.asarray(rs.best_attempt)
        self.d_feasible_best_energy = cp.asarray(rs.feasible_best_energy)
        self.d_feasible_best_spins = cp.asarray(rs.feasible_best_spins)
        self.d_feasible_best_attempt = cp.asarray(rs.feasible_best_attempt)
        counters = np.stack([rs.counters[k] for k in DEVICE_COUNTERS], axis=1).astype(np.int64)
        self.d_counters = cp.asarray(counters)
        self.d_next_attempt = cp.asarray(rs.next_attempt)
        self.d_targets = cp.asarray(rs.targets)
        self.d_stopped = cp.asarray(rs.stopped.astype(np.uint8))
        self.d_replica_binding = cp.asarray(rs.replica_binding.astype(np.int32))
        self.d_replica_id = cp.asarray(rs.replica_id.astype(np.uint32))
        self.d_instance_id = cp.asarray(rs.instance_id.astype(np.uint32))
        scale = self.layout.scale[rs.replica_binding.astype(np.int64)]
        self.d_scale = cp.asarray(scale)
        proc = plan.process
        R = max(plan.R, 1)
        self.d_cohort = cp.asarray(cohort_ids(proc, rs.replica_id).astype(np.uint32))
        self.d_perm = cp.zeros((n, R) if proc.name == "sweep" else 1, dtype=cp.int32)
        self.fwords = (R + 31) // 32
        if proc.stepwise:
            self.d_vbuf = cp.zeros((n, R), dtype=cp.float64)
            self.d_abuf = cp.zeros((n, R), dtype=cp.float64)
            self.d_fbuf = cp.zeros((n, self.fwords), dtype=cp.uint32)
        else:
            self.d_vbuf = self.d_abuf = cp.zeros(1, dtype=cp.float64)
            self.d_fbuf = cp.zeros(1, dtype=cp.uint32)
        # persistent (queue) scheduling: a global atomic cursor over [0, n_rep), re-zeroed before every
        # launch that uses it (ignored by the compiled kernel when TPI_PERSISTENT == 0, see kernels/anneal.cu)
        self.d_work_counter = cp.zeros(1, dtype=cp.int64)
        self.color_dev = None
        self.reconfigure()
        if plan.needs_v:
            self.rebuild_v()

    def rebuild_v(self):
        """``V[rep][r]`` from the current clause signs (exact integer sums, device kernel)."""
        model = self.plan.model
        topo = self.plan._device_topology()
        kc = self.kernels.get("incremental") or next(iter(self.kernels.values()))
        fn = _compiled_module(kc.lanes, 1, False, True, True, False).get_function(_REBUILD_NAME)
        total = self.n_replicas * model.n_groups
        if total:
            fn((max(1, (total + 255) // 256),), (256,), (
                topo["route_ptr"], topo["route_wend"], topo["rec"], self.d_Jc, self.d_T, self.d_v,
                self.d_replica_binding, np.int32(model.n_clauses), np.int32(model.n_groups), np.int32(self.n_replicas),
                np.int32(1 if self.t_minor else 0)))
        self.v_valid = True

    def v_float(self) -> np.ndarray | None:
        """The maintained inputs as float64 [n_rep, R] (V * 2^-e_b), or None if not maintained/current."""
        if not self.v_valid:
            return None
        V = cp.asnumpy(self.d_v)[:, :self.plan.model.n_groups]
        return V.astype(np.float64) * self.vscale[self.rs.replica_binding.astype(np.int64)][:, None]

    @property
    def n_replicas(self) -> int:
        return self.rs.n_replicas

    def reconfigure(self, **request) -> KernelConfig:
        """Re-resolve the kernel configuration(s) of this state (benchmarks: one device state, several
        variants). ``request``: kernel_variant/t_memory/latch/block_threads, as for ``prepare``. Every
        variant executes the same transaction; only the summation order of v differs. Returns the
        configuration of the plan's main kernel (recompute for 'adaptive')."""
        for inc in self.plan.uses(self.plan.strategy):
            req = dict(request)
            self.kernels["incremental" if inc else "recompute"] = self.plan.kernel_config(
                self.n_replicas, incremental=inc, **req)
        self.kernel = self.kernels["incremental" if self.plan.strategy == "incremental" else "recompute"]
        self.plan.options["kernel_variant_resolved"] = self.kernel.variant
        if len({kc.t_minor for kc in self.kernels.values()}) > 1:
            raise ConfigError("adaptive kernels must share one clause layout")
        self.set_t_layout(self.kernel.t_minor)
        return self.kernel

    def set_t_layout(self, minor: bool):
        """Device clause signs replica-major [n_rep, M~+1] or replica-minor [M~+1, n_rep] (transposed in place)."""
        if bool(minor) != self.t_minor:
            self.d_T = cp.ascontiguousarray(self.d_T.T)
            self.t_minor = bool(minor)

    def sync_to_host(self):
        """Copy every device array back into ``self.rs`` (host numpy, in place)."""
        rs = self.rs
        T = cp.asnumpy(self.d_T)
        rs.T[...] = (T.T if self.t_minor else T)[:, :-1]
        rs.energies[...] = cp.asnumpy(self.d_energies)
        rs.best_energy[...] = cp.asnumpy(self.d_best_energy)
        rs.best_spins[...] = cp.asnumpy(self.d_best_spins)
        rs.best_attempt[...] = cp.asnumpy(self.d_best_attempt)
        rs.feasible_best_energy[...] = cp.asnumpy(self.d_feasible_best_energy)
        rs.feasible_best_spins[...] = cp.asnumpy(self.d_feasible_best_spins)
        rs.feasible_best_attempt[...] = cp.asnumpy(self.d_feasible_best_attempt)
        counters = cp.asnumpy(self.d_counters)
        for j, k in enumerate(DEVICE_COUNTERS):
            rs.counters[k][...] = counters[:, j]
        rs.next_attempt[...] = cp.asnumpy(self.d_next_attempt)
        rs.targets[...] = cp.asnumpy(self.d_targets)
        rs.stopped[...] = cp.asnumpy(self.d_stopped).astype(bool)


# ============================================================================================
# backend
# ============================================================================================
_PLAN_OPTIONS = ("kernel_variant", "t_memory", "latch", "block_threads", "jw", "jw_budget_bytes", "v_memory",
                 "adaptive_subwindow", "t_layout", "scheduling", "color_threads")


class GPUBackend(Backend):
    """CUDA architecture backend. strategy='recompute'; see module docstring."""

    name = "gpu"

    def prepare(self, model: CompiledModel, *, strategy="auto", layout="auto", precision="float64", device=None,
                process="select", process_options=None, **options) -> GPUPlan:
        """``process`` / ``process_options``: proposal process."""
        _require_cuda()
        unknown = sorted(set(options) - set(_PLAN_OPTIONS))
        if unknown:
            raise ConfigError(f"unknown gpu backend options {unknown} (known: {list(_PLAN_OPTIONS)})")
        dev_index = 0
        if device not in (None, "cuda", "gpu"):
            s = str(device)
            dev_index = int(s.split(":")[1]) if ":" in s else int(s) if s.isdigit() else 0
        try:
            cp.cuda.Device(dev_index).use()
        except Exception as exc:
            raise BackendUnavailable(f"backend 'gpu' could not select CUDA device {dev_index}: {exc}") from exc
        return GPUPlan(model, strategy=strategy, layout=layout, precision=precision, device=f"cuda:{dev_index}",
                       process=process, process_options=process_options, **options)

    # ------------------------------------------------------------------------------------------
    def init_state(self, plan: GPUPlan, bindings, spins0, replica_binding=None, replica_id=None,
                   instance_id=None, *, is_feasible=None, targets=None) -> GPUState:
        _require_cuda()
        model = plan.model
        bindings = coerce_bindings(model, bindings)
        spins, rb, rid, iid = coerce_replicas(model, len(bindings), spins0, replica_binding, replica_id, instance_id)
        check_initial_feasibility(spins, is_feasible)
        T = clause_signs(model, spins)
        E = np.stack([bindings[int(rb[i])].const - bindings[int(rb[i])].J @ T[i].astype(np.float64)
                      for i in range(spins.shape[0])])
        rs = initial_replica_states(self.name, model, bindings, spins, rb, rid, iid, T, E, v=None)
        st = GPUState(plan, rs, bindings)
        apply_targets(rs, st.layout, targets)
        st.d_targets = cp.asarray(rs.targets)  # apply_targets may have rewritten them
        return st

    # ------------------------------------------------------------------------------------------
    def run(self, plan: GPUPlan, state: GPUState, schedule: Schedule, *, n_attempts_total=None, start=None,
            stop=None, seed: int, recording=None, targets=None, stop_on_target: bool = False) -> RunResult:
        _require_cuda()
        if state.plan is not plan:
            raise ConfigError("state was initialized with a different plan")
        spec = RecordingSpec.coerce(recording)
        if spec.profile not in ("none", "summary", "trace"):
            raise ConfigError(f"backend 'gpu' supports recording profiles none/summary/trace in this build, "
                              f"not {spec.profile!r}")
        rs = state.rs
        apply_targets(rs, state.layout, targets, stop_on_target)
        state.d_targets = cp.asarray(rs.targets)
        state.d_stopped = cp.asarray(rs.stopped.astype(np.uint8))
        proc = plan.process
        B, a0, a1 = resolve_window(rs, schedule, n_attempts_total, start, stop, ragged=proc.ragged)
        check_process_budget(proc, B, plan.R)
        n_rep = rs.n_replicas
        # Scheduling: 'auto' uses the work-queue (persistent
        # kernel) launch whenever replicas in this run can end early (stop_on_target, or 'nfold' which can
        # freeze a replica before the budget) -- that is exactly when a static one-replica-per-worker launch
        # would otherwise leave workers idle; 'select'/'shared'/'sweep'/'parallel_trial' without
        # stop_on_target run to the full budget on every replica, so static scheduling already has no idle
        # time to recover and is used instead (cheaper: no atomic traffic).
        effective_scheduling = resolve_scheduling(plan.scheduling, stop_on_target, proc.name)
        persistent = effective_scheduling == "queue"
        n_sm = H100_SM_COUNT
        if persistent:
            try:
                n_sm = int(cp.cuda.runtime.deviceGetAttribute(_ATTR_SM_COUNT, cp.cuda.Device().id))
            except Exception:
                pass
        n_slots = len(state.bindings[0].slot_names)
        score_slot = state.layout.score_slot
        violation_slot = -1 if state.layout.violation_slot is None else state.layout.violation_slot
        topo = plan._device_topology()

        # Trace buffers are always real device arrays (CuPy cannot pass a null pointer); trace_cap = 0
        # disables every trace write (and the shared atomic counter) inside the kernel.
        trace_cap = 0
        if spec.want_trace:
            trace_cap = min(n_rep * max(a1 - a0, 1), max(1, spec.trace_max_bytes // 48))
        alloc = max(trace_cap, 1)
        trace_bufs = {
            "attempt": cp.zeros(alloc, dtype=cp.int64), "replica": cp.zeros(alloc, dtype=cp.int32),
            "group": cp.zeros(alloc, dtype=cp.int32), "u": cp.zeros(alloc, dtype=cp.float64),
            "v": cp.zeros(alloc, dtype=cp.float64), "xi": cp.zeros(alloc, dtype=cp.float64),
            "accepted": cp.zeros(alloc, dtype=cp.uint8), "changed": cp.zeros(alloc, dtype=cp.uint8)}
        trace_count = cp.zeros(1, dtype=cp.uint64)

        kind, p0, p1, p2, p3, n_total = schedule.kernel_params()
        was_stopped = cp.asnumpy(state.d_stopped).astype(bool).copy()
        att0 = rs.counters["attempts"].copy()

        def launch(strategy, wa0, wa1):
            kc = state.kernels[strategy]
            inc = strategy == "incremental"
            if inc and not state.v_valid:
                state.rebuild_v()
            kernel = _compiled_function(kc.lanes, max(n_slots, 1), kc.t_shared, kc.latch == "mark", inc, kc.v_shared,
                                        kc.t_minor, persistent)
            if kc.smem_bytes > SMEM_DEFAULT_BYTES - SMEM_RESERVED_BYTES:  # opt in above the 48 KB default
                kernel.max_dynamic_shared_size_bytes = int(kc.smem_bytes)
            blocks_static = max(1, (n_rep + kc.replicas_per_block - 1) // kc.replicas_per_block)
            if persistent:
                blocks = min(blocks_static, max(1, n_sm * PERSISTENT_BLOCKS_PER_SM))
                state.d_work_counter.fill(0)
            else:
                blocks = blocks_static
            _launch_kernel(kernel, kc, blocks, wa0, wa1)
            if not inc:
                state.v_valid = False  # recompute changes T without maintaining V
            return kc, blocks

        def _launch_kernel(kernel, kc, blocks, wa0, wa1):
            kernel((blocks,), (kc.block_threads,), (
            topo["route_ptr"], topo["route_wend"], topo["jw_off"], topo["rec"], topo["spin_clause"],
            np.int32(plan.model.n_spins), np.int32(plan.model.n_clauses),
            np.int32(plan.model.n_groups), np.int32(n_slots), np.int32(score_slot), np.int32(violation_slot),
            state.d_J, state.d_Jw, np.int32(1 if state.use_jw else 0), np.int64(state.jw_stride),
            state.d_T, state.d_energies,
            state.d_best_energy, state.d_best_spins, state.d_best_attempt,
            state.d_feasible_best_energy, state.d_feasible_best_spins, state.d_feasible_best_attempt,
            state.d_counters, state.d_next_attempt, state.d_targets, state.d_stopped,
            state.d_replica_binding, state.d_replica_id, state.d_instance_id, state.d_scale,
            trace_bufs["attempt"], trace_bufs["replica"], trace_bufs["group"], trace_bufs["u"],
            trace_bufs["v"], trace_bufs["xi"], trace_bufs["accepted"], trace_bufs["changed"],
            np.int64(trace_cap), trace_count,
            np.uint32(seed), np.int32(kind), np.float64(p0), np.float64(p1), np.float64(p2), np.float64(p3),
            np.int64(n_total), np.int64(wa0), np.int64(wa1), np.int32(1 if stop_on_target else 0), np.int32(n_rep),
            np.int64(kc.t_smem_stride),
            state.d_v, state.d_Jc, state.d_vscale, topo["cw_ptr"], topo["cw_rec"], topo["cc_ptr"], topo["cc_rec"],
            np.int64(kc.v_smem_base),
            np.int32(PROC_KIND[proc.name]), state.d_cohort, np.int32(1 if proc.pt_exclude_self_loops else 0),
            np.int32(1 if proc.nfold_clock == "attempts" else 0), state.d_perm, topo["sr_ptr"], topo["sr_y"],
            state.d_vbuf, state.d_abuf, state.d_fbuf, np.int32(state.fwords),
            state.d_work_counter), shared_mem=int(kc.smem_bytes))

        if proc.colored:  # separate kernel, separate launch path (gpu_colored.py)
            from nimbus.backends import gpu_colored as GC
            if state.color_dev is None:
                state.color_dev = GC.ColoredDevice(cp, plan.coloring)
            GC.compiled_function(cp, plan.color_threads, max(n_slots, 1))  # compile outside the timed region
        for name, kcx in ([] if proc.colored else state.kernels.items()):  # compile outside the timed region
            _compiled_function(kcx.lanes, max(n_slots, 1), kcx.t_shared, kcx.latch == "mark", name == "incremental",
                               kcx.v_shared, kcx.t_minor, persistent)
        schedule_log = []
        kc, blocks = state.kernel, 0
        t0 = time.perf_counter()
        if proc.colored:
            args = GC.build_args(cp, plan, state, topo, n_slots=n_slots, score_slot=score_slot,
                                 violation_slot=violation_slot, seed=seed, schedule_params=(kind, p0, p1, p2, p3, n_total),
                                 stop=a1, stop_on_target=stop_on_target, dev=state.color_dev)
            _thr, blocks, _sm = GC.launch(cp, plan, state, args, state.color_dev, n_slots)
            state.v_valid = False
        elif plan.strategy != "adaptive":
            kc, blocks = launch(plan.strategy, a0, a1)
        else:  # measured-rate adaptive: recompute, probing incremental
            W = plan.adaptive_subwindow or max(1, min(max(ADAPTIVE_MIN_SUBWINDOW, B // ADAPTIVE_SUBWINDOW_DIVISOR), B))
            cur, strat, p_acc, p_switch = a0, "recompute", None, None
            rate = {}
            gap, since = ADAPTIVE_PROBE_GAP, 0
            while cur < a1:
                probe = False
                cr = ci = None
                if strat == "recompute" and p_acc is not None and since >= gap and "recompute" in rate:
                    _, cr, ci = choose_adaptive_strategy(plan.costs, p_acc, "recompute", hysteresis=0.0)
                    probe = ci <= ADAPTIVE_PROBE_RATIO * cr
                run = "incremental" if probe else strat
                nxt = min(a1, cur + (max(1, W // ADAPTIVE_PROBE_FRACTION) if probe else W))
                c_before = cp.asnumpy(state.d_counters[:, [0, 1]].sum(axis=0))
                tw = time.perf_counter()
                kc, blocks = launch(run, cur, nxt)
                c_after = cp.asnumpy(state.d_counters[:, [0, 1]].sum(axis=0))  # (synchronizes)
                dt = time.perf_counter() - tw
                d_att, d_acc = (c_after - c_before).tolist()
                if d_att and dt > 0:
                    rate[run] = d_att / dt
                decision = None
                if probe:
                    since = 0
                    if rate.get("incremental", 0.0) > (1.0 + ADAPTIVE_HYSTERESIS) * rate["recompute"]:
                        strat, p_switch, decision = "incremental", p_acc, "switch"
                    else:
                        gap, decision = min(2 * gap, 64), "stay"
                else:
                    since += 1
                schedule_log.append({"start": cur, "stop": nxt, "strategy": run, "probe": probe, "p_acc_in": p_acc,
                                     "rate": rate.get(run), "cost_recompute": cr, "cost_incremental": ci,
                                     "decision": decision})
                p_acc = d_acc / d_att if d_att else p_acc
                if (strat == "incremental" and p_switch is not None and p_acc is not None
                        and p_acc > ADAPTIVE_REVERT_P_FACTOR * max(p_switch, 1e-12)):
                    strat, gap, since = "recompute", ADAPTIVE_PROBE_GAP, 0
                    schedule_log[-1]["decision"] = "revert"
                cur = nxt
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        state.sync_to_host()
        term = termination_labels(rs, was_stopped, rs.stopped & ~was_stopped, a1, B)
        notes = {"anytime": "not implemented by the gpu backend in this build",
                 "events": "not applicable: architecture events are not instrumented by the gpu backend"}
        gpu_trace = None
        if spec.want_trace:
            n_tr = int(min(int(cp.asnumpy(trace_count)[0]), trace_cap))
            gpu_trace = {k: cp.asnumpy(v)[:n_tr] for k, v in trace_bufs.items()}
            if int(cp.asnumpy(trace_count)[0]) > trace_cap:
                notes["trace"] = "truncated: more attempts were traced than the buffer could hold"
        else:
            notes["trace"] = "requires recording profile 'trace'"
        # NOTE: the "trace" entry here is a plain dict of numpy arrays (attempt, replica, group, u,
        # v, xi, accepted, changed), not a nimbus.experiments.recording.TraceLog: this backend's
        # bounded validation trace is simplified in this build, so it is
        # carried in RunResult.metadata rather than RecordingData.trace.
        rec = RecordingData(spec=spec, anytime=None, events=None, trace=None, notes=notes)
        if proc.colored:
            kmeta_extra = {"colored_kernel": "anneal_colored.cu", "color_threads": plan.color_threads,
                           "n_colors": plan.coloring.n_colors}
        else:
            kmeta_extra = {}
        kmeta = dict(kc.to_dict(), **kmeta_extra, use_jw=bool(state.use_jw), blocks=int(blocks), strategy=plan.strategy,
                     strategy_reason=plan.strategy_reason,
                     scheduling=effective_scheduling, persistent=bool(persistent),
                     persistent_blocks_per_sm=PERSISTENT_BLOCKS_PER_SM if persistent else None,
                     v_fixed_point_exact=None if state.v_exact is None else [bool(x) for x in state.v_exact])
        return RunResult(backend=self.name, start=a0, stop=a1, n_attempts_total=B, seed=int(seed),
                         state=self.export_state(state), attempts_run=rs.counters["attempts"] - att0,
                         termination=term, recording=rec, timings={"run_s": elapsed},
                         metadata={"plan": plan.describe(), "kernel": kmeta, "schedule": schedule.to_dict(),
                                   "gpu_trace": gpu_trace, "strategy_schedule": schedule_log,
                                   "process": {**proc.describe(), **({"coloring": plan.coloring.summary()} if proc.colored else {})}})

    # ------------------------------------------------------------------------------------------
    def export_state(self, state: GPUState) -> ReplicaStates:
        state.sync_to_host()
        state.rs.v = state.v_float()  # maintained inputs when current (incremental), else None
        rs = state.rs.copy()
        rs.bindings = state.bindings
        return rs

    def load_state(self, plan: GPUPlan, states: ReplicaStates, bindings=None) -> GPUState:
        _require_cuda()
        model = plan.model
        bindings = coerce_bindings(model, bindings if bindings is not None else states.bindings)
        states.check_compatible(model, bindings)
        missing = [k for k in COUNTER_NAMES if k not in states.counters]
        if missing:
            raise IncompatibleCheckpoint(f"snapshot lacks counters {missing}")
        rs = states.copy()
        ensure_process_counters(rs)
        if not np.array_equal(clause_signs(model, rs.spins), rs.T):
            raise IncompatibleCheckpoint("snapshot clause state is inconsistent with its singleton readout")
        rs.backend = self.name
        rs.extras = {}
        st = GPUState(plan, rs, bindings)
        rs.bindings = st.bindings
        return st

    def evaluate_inputs(self, state: GPUState, groups=None) -> np.ndarray:
        """v_r by direct recomputation from the current host-synced T (no commit; host-side, like the
        oracle's formula, so a pilot/validation call never needs its own kernel launch)."""
        state.sync_to_host()
        model = state.plan.model
        gs = np.arange(model.n_groups) if groups is None else np.asarray([int(g) for g in groups])
        out = np.empty((state.n_replicas, gs.size))
        for rep in range(state.n_replicas):
            b = int(state.rs.replica_binding[rep])
            J0 = state.bindings[b].J[0]
            T = state.rs.T[rep]
            al = route_switches(model, T[None, :])[0]
            for j, g in enumerate(gs.tolist()):
                a, we = int(model.route_ptr[g]), int(model.route_wend[g])
                rc = model.route_clause[a:we]
                out[rep, j] = float(np.sum(J0[rc] * al[a:we].astype(np.float64) * T[rc].astype(np.float64)))
        return out

    def describe(self) -> dict:
        d = {"backend": self.name, "strategies": list(STRATEGIES), "layout": "csr", "precision": "float64",
             "device": "cuda", "kernel_variants": list(VARIANTS),
             "role": "GPU architecture backend (NVRTC/CuPy)", **host_versions()}
        if cuda_available():
            try:
                props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
                d["gpu_name"] = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
                d["cupy"] = cp.__version__
                d["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
            except Exception:
                pass
        else:
            d["cuda_available"] = False
        return d


register_backend("gpu", GPUBackend, replace=True)

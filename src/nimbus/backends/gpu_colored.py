"""Host side of the GPU ``colored`` process (kernels/anneal_colored.cu).

A separate code path: the unchanged select/shared/sweep/parallel_trial/nfold kernel in anneal.cu and its launch code in
``gpu.py`` are untouched; ``GPUBackend.run`` only branches here when ``process == "colored"``. One thread block of
``color_threads`` threads (default 32, one warp) per replica; the groups of one colour class are tested and committed in
parallel by the lanes of that block. Clause signs live in shared memory for the launch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nimbus.core.errors import ConfigError

_KERNEL_NAME = "tpi_color_anneal"
_KERNELS_DIR = Path(__file__).resolve().parent / "kernels"
_MODULE_CACHE: dict = {}
SMEM_DEFAULT_BYTES = 48 * 1024
COLOR_THREADS = (32, 64, 128, 256)


def _inline(text: str, seen: set) -> str:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('#include "') and s.endswith('"'):
            name = s.split('"')[1]
            if name in seen:
                continue
            seen.add(name)
            out.append(_inline((_KERNELS_DIR / name).read_text(), seen))
        else:
            out.append(line)
    return "\n".join(out)


def kernel_source(threads: int, max_slots: int) -> str:
    body = _inline((_KERNELS_DIR / "anneal_colored.cu").read_text(), set())
    return f"#define TPI_CNT {int(threads)}\n#define TPI_MAX_SLOTS {int(max_slots)}\n" + body


def compiled_function(cp, threads: int, max_slots: int):
    key = (int(threads), int(max_slots))
    if key not in _MODULE_CACHE:
        mod = cp.RawModule(code=kernel_source(*key), options=("--std=c++14",), name_expressions=[_KERNEL_NAME])
        mod.get_function(_KERNEL_NAME)
        _MODULE_CACHE[key] = mod
    return _MODULE_CACHE[key].get_function(_KERNEL_NAME)


def check_threads(threads) -> int:
    t = int(threads)
    if t not in COLOR_THREADS:
        raise ConfigError(f"backend 'gpu' color_threads must be one of {COLOR_THREADS}, got {threads!r}")
    return t


def smem_bytes(n_clauses: int, max_class: int, max_slots: int) -> int:
    return 2 * max_class * (8 * max_slots + 8 + 1) + 16 + (n_clauses + 1)


class ColoredDevice:
    """Device copies of the colouring (position order, class pointers, class of position), int32."""

    def __init__(self, cp, coloring):
        self.order = cp.asarray(coloring.order.astype(np.int32))
        self.class_ptr = cp.asarray(coloring.class_ptr.astype(np.int32))
        self.class_of_pos = cp.asarray(coloring.class_of_pos.astype(np.int32))
        self.n_colors = int(coloring.n_colors)
        self.max_class = int(coloring.class_sizes.max()) if coloring.n_colors else 1


def build_args(cp, plan, state, topo, *, n_slots, score_slot, violation_slot, seed, schedule_params, stop, stop_on_target,
               dev: ColoredDevice):
    """Argument tuple of ``tpi_color_anneal`` (window [next_attempt, stop) per replica)."""
    kind, p0, p1, p2, p3, n_total = schedule_params
    m = plan.model
    return (topo["route_ptr"], topo["route_wend"], topo["rec"], topo["spin_clause"],
            np.int32(m.n_spins), np.int32(m.n_clauses), np.int32(m.n_groups), np.int32(n_slots),
            np.int32(score_slot), np.int32(violation_slot), state.d_J,
            state.d_T, state.d_energies, state.d_best_energy, state.d_best_spins, state.d_best_attempt,
            state.d_feasible_best_energy, state.d_feasible_best_spins, state.d_feasible_best_attempt,
            state.d_counters, state.d_next_attempt, state.d_targets, state.d_stopped,
            state.d_replica_binding, state.d_replica_id, state.d_instance_id, state.d_scale,
            np.uint32(seed), np.int32(kind), np.float64(p0), np.float64(p1), np.float64(p2), np.float64(p3),
            np.int64(n_total), np.int64(stop), np.int32(1 if stop_on_target else 0), np.int32(state.n_replicas),
            dev.order, dev.class_ptr, dev.class_of_pos, np.int32(dev.n_colors), np.int32(dev.max_class))


def launch(cp, plan, state, args, dev: ColoredDevice, n_slots: int):
    """Launch one colored window; returns (threads, blocks, smem bytes)."""
    threads = plan.color_threads
    max_slots = max(n_slots, 1)
    fn = compiled_function(cp, threads, max(max_slots, 1))
    # TPI_MAX_SLOTS must be >= n_slots; the kernel is compiled with exactly max(n_slots, 1) slots
    sm = smem_bytes(plan.model.n_clauses, dev.max_class, max_slots)
    if sm > SMEM_DEFAULT_BYTES - 1024:
        fn.max_dynamic_shared_size_bytes = int(sm)
    blocks = max(1, state.n_replicas)
    fn((blocks,), (threads,), args, shared_mem=int(sm))
    return threads, blocks, sm

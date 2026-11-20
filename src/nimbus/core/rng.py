"""Counter-based random numbers: Philox4x32-10 (Random123-compatible) and the
project's documented key/counter mapping.

This pure-Python/numpy module is the reference implementation. The Numba and CUDA
versions must reproduce it bit for bit.

Key = (seed, stream). Counter = (attempt & 0xffffffff, attempt >> 32, replica, instance).
One Philox call per solver attempt yields (x0, x1, x2, x3):
    group   r = (x0*R + ((x1*R) >> 32)) >> 32      (= floor(((x0<<32)|x1) * R / 2**64))
    uniform u = (((x2 << 20) | (x3 >> 12)) + 0.5) * 2**-52   (strictly inside (0, 1))
"""

from __future__ import annotations

import math

import numpy as np

# Streams (key word 1). Never reuse a stream for a different purpose.
SOLVER = 1
INIT = 2
CHANNEL = 3
PILOT = 4
DIAG = 5
DATA = 6
BASELINE = 7
# Proposal-process streams
SHARED = 8   # shared group sequence of a replica cohort
PT = 9       # parallel_trial candidate uniforms and arbiter
NFOLD = 10   # nfold selection and waiting uniforms
SWEEP = 11   # sweep permutations
COLOR = 12   # colored: one acceptance uniform per (step, group, replica, instance)
PT_ARBITER = 0xFFFFFFFF  # counter word c1 of the parallel_trial arbiter draw (never a group index, R < 2**32)

_M0 = 0xD2511F53
_M1 = 0xCD9E8D57
_W0 = 0x9E3779B9
_W1 = 0xBB67AE85
_MASK32 = 0xFFFFFFFF
_TWO_M52 = 2.0 ** -52


def philox4x32_scalar(c0: int, c1: int, c2: int, c3: int, k0: int, k1: int, rounds: int = 10):
    """Philox4x32 on Python ints; returns a 4-tuple of uint32 values."""
    for i in range(rounds):
        if i:
            k0 = (k0 + _W0) & _MASK32
            k1 = (k1 + _W1) & _MASK32
        p0 = _M0 * c0
        p1 = _M1 * c2
        hi0, lo0 = p0 >> 32, p0 & _MASK32
        hi1, lo1 = p1 >> 32, p1 & _MASK32
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & _MASK32, lo1, (hi0 ^ c3 ^ k1) & _MASK32, lo0
    return c0, c1, c2, c3


def philox4x32(c0, c1, c2, c3, k0, k1, rounds: int = 10):
    """Vectorized Philox4x32 over numpy arrays (broadcast). Returns four uint32 arrays."""
    c0 = np.asarray(c0, dtype=np.uint64) & _MASK32
    c1 = np.asarray(c1, dtype=np.uint64) & _MASK32
    c2 = np.asarray(c2, dtype=np.uint64) & _MASK32
    c3 = np.asarray(c3, dtype=np.uint64) & _MASK32
    c0, c1, c2, c3 = np.broadcast_arrays(c0, c1, c2, c3)
    c0, c1, c2, c3 = (a.copy() for a in (c0, c1, c2, c3))
    k0 = np.uint64(int(k0) & _MASK32)
    k1 = np.uint64(int(k1) & _MASK32)
    m0 = np.uint64(_M0)
    m1 = np.uint64(_M1)
    mask = np.uint64(_MASK32)
    s32 = np.uint64(32)
    for i in range(rounds):
        if i:
            k0 = np.uint64((int(k0) + _W0) & _MASK32)
            k1 = np.uint64((int(k1) + _W1) & _MASK32)
        p0 = m0 * c0
        p1 = m1 * c2
        hi0, lo0 = p0 >> s32, p0 & mask
        hi1, lo1 = p1 >> s32, p1 & mask
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & mask, lo1, (hi0 ^ c3 ^ k1) & mask, lo0
    return (c0.astype(np.uint32), c1.astype(np.uint32), c2.astype(np.uint32), c3.astype(np.uint32))


def group_index(x0: int, x1: int, n_groups: int) -> int:
    """Exact floor(((x0 << 32) | x1) * R / 2**64) for R < 2**32."""
    if not 0 < n_groups < 2 ** 32:
        raise ValueError("number of groups must be in [1, 2**32)")
    return (x0 * n_groups + ((x1 * n_groups) >> 32)) >> 32


def group_index_array(x0, x1, n_groups: int):
    if not 0 < n_groups < 2 ** 32:
        raise ValueError("number of groups must be in [1, 2**32)")
    x0 = np.asarray(x0, dtype=np.uint64)
    x1 = np.asarray(x1, dtype=np.uint64)
    r = np.uint64(n_groups)
    return ((x0 * r + ((x1 * r) >> np.uint64(32))) >> np.uint64(32)).astype(np.int64)


def uniform52(x2: int, x3: int) -> float:
    """52-bit uniform strictly inside (0, 1); exactly representable."""
    k = (x2 << 20) | (x3 >> 12)
    return (k + 0.5) * _TWO_M52


def uniform52_array(x2, x3):
    x2 = np.asarray(x2, dtype=np.uint64)
    x3 = np.asarray(x3, dtype=np.uint64)
    k = (x2 << np.uint64(20)) | (x3 >> np.uint64(12))
    return (k.astype(np.float64) + 0.5) * _TWO_M52


def _check_seed(seed: int) -> int:
    seed = int(seed)
    if not 0 <= seed <= _MASK32:
        raise ValueError(f"seed must be a uint32, got {seed}")
    return seed


def solver_draw(seed: int, attempt: int, replica: int, instance: int, n_groups: int):
    """(group index, uniform) for one attempt of one replica (solver stream)."""
    seed = _check_seed(seed)
    x0, x1, x2, x3 = philox4x32_scalar(attempt & _MASK32, attempt >> 32, replica & _MASK32,
                                       instance & _MASK32, seed, SOLVER)
    return group_index(x0, x1, n_groups), uniform52(x2, x3)


def solver_draws(seed: int, attempts, replica: int, instance: int, n_groups: int):
    """Vectorized solver draws for an array of attempt indices of one replica."""
    seed = _check_seed(seed)
    a = np.asarray(attempts, dtype=np.uint64)
    x0, x1, x2, x3 = philox4x32(a & np.uint64(_MASK32), a >> np.uint64(32), replica, instance, seed, SOLVER)
    return group_index_array(x0, x1, n_groups), uniform52_array(x2, x3)


def stream_words(seed: int, stream: int, n_words: int, c2: int = 0, c3: int = 0, c1: int = 0):
    """n_words uint32 values from counters (block, c1, c2, c3), block = 0, 1, ...; 4 words per block."""
    seed = _check_seed(seed)
    n_blocks = (int(n_words) + 3) // 4
    blocks = np.arange(n_blocks, dtype=np.uint64)
    x = philox4x32(blocks, c1, c2, c3, seed, stream)
    words = np.stack(x, axis=1).reshape(-1)
    return words[: int(n_words)]


def stream_uniforms(seed: int, stream: int, n: int, c2: int = 0, c3: int = 0, c1: int = 0):
    """n open-interval 52-bit uniforms; uniform j uses words (2j, 2j+1) of `stream_words`."""
    w = stream_words(seed, stream, 2 * int(n), c2=c2, c3=c3, c1=c1).reshape(-1, 2)
    return uniform52_array(w[:, 0], w[:, 1])


def stream_bits(seed: int, stream: int, n: int, c2: int = 0, c3: int = 0, c1: int = 0):
    """n fair bits (uint8) taken from the low bit of successive words."""
    return (stream_words(seed, stream, int(n), c2=c2, c3=c3, c1=c1) & np.uint32(1)).astype(np.uint8)


def stream_normals(seed: int, stream: int, n: int, c2: int = 0, c3: int = 0, c1: int = 0):
    """n standard normals by Box-Muller on pairs of stream_uniforms (pinned in-house generator)."""
    m = (int(n) + 1) // 2
    u = stream_uniforms(seed, stream, 2 * m, c2=c2, c3=c3, c1=c1).reshape(-1, 2)
    rad = np.sqrt(-2.0 * np.log(u[:, 0]))
    ang = 2.0 * math.pi * u[:, 1]
    z = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1).reshape(-1)
    return z[: int(n)]


def stream_integers(seed: int, stream: int, n: int, high: int, c2: int = 0, c3: int = 0, c1: int = 0):
    """n integers uniform in [0, high) using the exact multiply-high rule on word pairs."""
    w = stream_words(seed, stream, 2 * int(n), c2=c2, c3=c3, c1=c1).reshape(-1, 2)
    return group_index_array(w[:, 0], w[:, 1], int(high))


# ============================================================================================
# proposal-process streams
# ============================================================================================
def _check_u32(x: int, what: str) -> int:
    x = int(x)
    if not 0 <= x <= _MASK32:
        raise ValueError(f"{what} must be in [0, 2**32), got {x}")
    return x


def cohort_id(replica_id: int, cohort_size, *, default_per_replica: bool) -> int:
    """Cohort index of a replica: replica_id // cohort_size; without a cohort size, 0 (one cohort per
    instance, the ``shared`` default) or the replica id itself (``sweep`` default, per-replica draws)."""
    if cohort_size is None:
        return int(replica_id) if default_per_replica else 0
    return int(replica_id) // int(cohort_size)


def shared_group(seed: int, attempt: int, cohort: int, instance: int, n_groups: int) -> int:
    """``shared``: group of attempt n for a cohort. Key (seed, SHARED), counter (n & 0xffffffff, n >> 32,
    cohort, instance); words x0, x1 -> the group-index rule."""
    seed = _check_seed(seed)
    attempt = int(attempt)
    x0, x1, _, _ = philox4x32_scalar(attempt & _MASK32, attempt >> 32, int(cohort) & _MASK32,
                                     int(instance) & _MASK32, seed, SHARED)
    return group_index(x0, x1, n_groups)


def shared_groups(seed: int, attempts, cohort: int, instance: int, n_groups: int):
    """Vectorized :func:`shared_group` over attempt indices."""
    seed = _check_seed(seed)
    a = np.asarray(attempts, dtype=np.uint64)
    x0, x1, _, _ = philox4x32(a & np.uint64(_MASK32), a >> np.uint64(32), int(cohort) & _MASK32,
                              int(instance) & _MASK32, seed, SHARED)
    return group_index_array(x0, x1, n_groups)


def pt_uniform(seed: int, step: int, group: int, replica: int, instance: int) -> float:
    """``parallel_trial``: candidate uniform u_{n,r}. Key (seed, PT), counter (step, r, replica, instance);
    words x0, x1 -> uniform52 (steps < 2**32)."""
    seed = _check_seed(seed)
    x0, x1, _, _ = philox4x32_scalar(_check_u32(step, "parallel_trial step"), int(group) & _MASK32,
                                     int(replica) & _MASK32, int(instance) & _MASK32, seed, PT)
    return uniform52(x0, x1)


def pt_uniforms(seed: int, step: int, replica: int, instance: int, groups):
    """Vectorized :func:`pt_uniform` over group indices (one step of one replica)."""
    seed = _check_seed(seed)
    g = np.asarray(groups, dtype=np.uint64)
    x0, x1, _, _ = philox4x32(_check_u32(step, "parallel_trial step"), g, int(replica) & _MASK32,
                              int(instance) & _MASK32, seed, PT)
    return uniform52_array(x0, x1)


def pt_arbiter(seed: int, step: int, replica: int, instance: int, n_fired: int) -> int:
    """``parallel_trial``: index into the fired list (ascending r). Key (seed, PT), counter
    (step, 0xFFFFFFFF, replica, instance); words x0, x1 -> group-index rule over |F|."""
    seed = _check_seed(seed)
    x0, x1, _, _ = philox4x32_scalar(_check_u32(step, "parallel_trial step"), PT_ARBITER,
                                     int(replica) & _MASK32, int(instance) & _MASK32, seed, PT)
    return group_index(x0, x1, int(n_fired))


def color_uniform(seed: int, step: int, group: int, replica: int, instance: int) -> float:
    """``colored``: acceptance uniform of group r in color step ``step``. Key (seed, COLOR), counter
    (step, r, replica, instance); words x0, x1 -> uniform52 (steps < 2**32)."""
    seed = _check_seed(seed)
    x0, x1, _, _ = philox4x32_scalar(_check_u32(step, "colored step"), int(group) & _MASK32,
                                     int(replica) & _MASK32, int(instance) & _MASK32, seed, COLOR)
    return uniform52(x0, x1)


def nfold_uniforms(seed: int, step: int, replica: int, instance: int) -> tuple[float, float]:
    """``nfold``: (selection uniform, waiting uniform) of step n_step. Key (seed, NFOLD), counter
    (n_step, 0, replica, instance); x0, x1 -> selection, x2, x3 -> waiting (n_step < 2**32)."""
    seed = _check_seed(seed)
    x0, x1, x2, x3 = philox4x32_scalar(_check_u32(step, "nfold step"), 0, int(replica) & _MASK32,
                                       int(instance) & _MASK32, seed, NFOLD)
    return uniform52(x0, x1), uniform52(x2, x3)


def nfold_uniforms_array(seed: int, steps, replica: int, instance: int):
    """Vectorized :func:`nfold_uniforms` over step indices: (u_select[], u_wait[])."""
    seed = _check_seed(seed)
    k = np.asarray(steps, dtype=np.uint64)
    if k.size and int(k.max()) > _MASK32:
        raise ValueError("nfold step must be < 2**32")
    x0, x1, x2, x3 = philox4x32(k, 0, int(replica) & _MASK32, int(instance) & _MASK32, seed, NFOLD)
    return uniform52_array(x0, x1), uniform52_array(x2, x3)


def sweep_words(seed: int, sweep: int, cohort: int, instance: int, n_words: int):
    """Consecutive words of one sweep: word w = lane (w % 4) of Philox with key (seed, SWEEP) and counter
    (sweep, w // 4, cohort, instance); the first block is the counter (sweep, 0, cohort, instance)."""
    seed = _check_seed(seed)
    n_blocks = (int(n_words) + 3) // 4
    blocks = np.arange(n_blocks, dtype=np.uint64)
    x = philox4x32(_check_u32(sweep, "sweep index"), blocks, int(cohort) & _MASK32, int(instance) & _MASK32,
                   seed, SWEEP)
    return np.stack(x, axis=1).reshape(-1)[: int(n_words)]


def sweep_permutation(seed: int, sweep: int, cohort: int, instance: int, n_groups: int) -> np.ndarray:
    """``sweep``: random permutation of 0..R-1 for one sweep (Durstenfeld Fisher-Yates from the identity).
    Swap t = 0..R-2 handles position i = R-1-t and uses the word pair (2t, 2t+1) of :func:`sweep_words`:
    j = group-index rule over i + 1 (so j is uniform in [0, i]); then perm[i] <-> perm[j]."""
    R = int(n_groups)
    if not 0 < R < 2 ** 32:
        raise ValueError("number of groups must be in [1, 2**32)")
    perm = list(range(R))
    if R > 1:
        w = sweep_words(seed, sweep, cohort, instance, 2 * (R - 1)).tolist()
        for t in range(R - 1):
            i = R - 1 - t
            j = group_index(w[2 * t], w[2 * t + 1], i + 1)
            perm[i], perm[j] = perm[j], perm[i]
    return np.asarray(perm, dtype=np.int64)


def sweep_groups(seed: int, attempts, cohort: int, instance: int, n_groups: int) -> np.ndarray:
    """``sweep``: group of every attempt index n: perm_{n // R}[n % R]."""
    R = int(n_groups)
    a = np.asarray(attempts, dtype=np.int64)
    out = np.empty(a.shape, dtype=np.int64)
    s = a // R
    for sw in np.unique(s).tolist():
        m = s == sw
        out[m] = sweep_permutation(seed, sw, cohort, instance, R)[a[m] - sw * R]
    return out

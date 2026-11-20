"""Numba (nopython) helpers shared by the CPU kernels: Philox4x32-10, the solver-draw mapping,
attempt-indexed schedule temperatures and the Metropolis threshold acceptance.

Every function here reproduces a pure-Python reference bit for bit (tests/test_rng_numba.py):

* ``philox4x32_10``, ``group_index``, ``uniform52``, ``solver_draw``, ``stream_*``
  -> ``nimbus.core.rng`` (Random123-compatible Philox4x32-10 and the key/counter mapping:
  key = (seed, stream), counter = (attempt & 0xffffffff,
  attempt >> 32, replica, instance); one call per attempt; group by exact 64-bit multiply-high,
  uniform from 52 bits, strictly inside (0, 1)).
* ``schedule_tau`` -> ``nimbus.annealing.schedule.Schedule.tau`` (closed form from
  (kind, p0..p3, B, n); identical floating-point expression order).
* ``metropolis_threshold`` / ``metropolis_accept`` -> ``nimbus.annealing.policy`` functions of
  the same name (accept iff v <= xi, xi = -(tau/2) log u; tau == 0
  accepts iff v <= 0). ``v`` is the transition-neuron input, Delta E = 2 v.

Integer work is done in uint64 with explicit 32-bit masking. Every integer constant is a typed
numpy scalar so that Numba never mixes signed and unsigned operands (which it would promote to
float64). No fastmath: the floating-point expressions keep the reference evaluation order and
LLVM does not contract them into fused multiply-adds.

Argument conventions: seeds are uint32 values (the caller validates the range; values are masked
here), attempts are non-negative int64, ``n_groups`` is in [1, 2**32) (caller validated).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from nimbus.annealing import schedule as _schedule
from nimbus.core import rng as _rng

# Philox4x32 multipliers and Weyl key increments (Random123).
_M0 = np.uint64(0xD2511F53)
_M1 = np.uint64(0xCD9E8D57)
_W0 = np.uint64(0x9E3779B9)
_W1 = np.uint64(0xBB67AE85)
_MASK32 = np.uint64(0xFFFFFFFF)
_S1 = np.uint64(1)
_S2 = np.uint64(2)
_S12 = np.uint64(12)
_S20 = np.uint64(20)
_S32 = np.uint64(32)
_U1 = np.uint64(1)
_U3 = np.uint64(3)
_TWO_M52 = 2.0 ** -52

# Stream identifiers (key word 1), mirrored from nimbus.core.rng.
SOLVER = _rng.SOLVER
INIT = _rng.INIT
CHANNEL = _rng.CHANNEL
PILOT = _rng.PILOT
DIAG = _rng.DIAG
DATA = _rng.DATA
BASELINE = _rng.BASELINE
SHARED = _rng.SHARED
PT = _rng.PT
NFOLD = _rng.NFOLD
SWEEP = _rng.SWEEP
COLOR = _rng.COLOR
_SOLVER_U = np.uint64(_rng.SOLVER)
_SHARED_U = np.uint64(_rng.SHARED)
_PT_U = np.uint64(_rng.PT)
_NFOLD_U = np.uint64(_rng.NFOLD)
_SWEEP_U = np.uint64(_rng.SWEEP)
_COLOR_U = np.uint64(_rng.COLOR)
_PT_ARBITER_U = np.uint64(_rng.PT_ARBITER)
_ZERO_U = np.uint64(0)

# Schedule kind codes, mirrored from nimbus.annealing.schedule.
ZERO = _schedule.ZERO
CONSTANT = _schedule.CONSTANT
GEOMETRIC = _schedule.GEOMETRIC
FN_ENDPOINT = _schedule.FN_ENDPOINT
FN_LEGACY = _schedule.FN_LEGACY


# --------------------------------------------------------------------------------------------
# Philox4x32-10 and the draw mapping
# --------------------------------------------------------------------------------------------

@njit(cache=True)
def philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10 on one counter (c0..c3) and key (k0, k1); returns four np.uint32 words."""
    x0 = np.uint64(c0) & _MASK32
    x1 = np.uint64(c1) & _MASK32
    x2 = np.uint64(c2) & _MASK32
    x3 = np.uint64(c3) & _MASK32
    key0 = np.uint64(k0) & _MASK32
    key1 = np.uint64(k1) & _MASK32
    for i in range(10):
        if i > 0:
            key0 = (key0 + _W0) & _MASK32
            key1 = (key1 + _W1) & _MASK32
        p0 = _M0 * x0  # < 2**64, exact
        p1 = _M1 * x2
        hi0 = p0 >> _S32
        lo0 = p0 & _MASK32
        hi1 = p1 >> _S32
        lo1 = p1 & _MASK32
        y0 = (hi1 ^ x1 ^ key0) & _MASK32
        y2 = (hi0 ^ x3 ^ key1) & _MASK32
        x0 = y0
        x1 = lo1
        x2 = y2
        x3 = lo0
    return np.uint32(x0), np.uint32(x1), np.uint32(x2), np.uint32(x3)


@njit(cache=True)
def group_index(x0, x1, n_groups):
    """Exact floor(((x0 << 32) | x1) * R / 2**64) as int64, R = n_groups < 2**32."""
    a = np.uint64(x0) & _MASK32
    b = np.uint64(x1) & _MASK32
    r = np.uint64(n_groups)
    return np.int64((a * r + ((b * r) >> _S32)) >> _S32)


@njit(cache=True)
def uniform52(x2, x3):
    """(((x2 << 20) | (x3 >> 12)) + 0.5) * 2**-52: exact, strictly inside (0, 1)."""
    k = ((np.uint64(x2) & _MASK32) << _S20) | ((np.uint64(x3) & _MASK32) >> _S12)
    return (np.float64(k) + 0.5) * _TWO_M52


@njit(cache=True)
def solver_draw(seed, attempt, replica, instance, n_groups):
    """(group index r as int64, uniform u) of one solver-stream attempt."""
    a = np.uint64(attempt)
    x0, x1, x2, x3 = philox4x32_10(a & _MASK32, a >> _S32, np.uint64(replica) & _MASK32,
                                   np.uint64(instance) & _MASK32, np.uint64(seed) & _MASK32, _SOLVER_U)
    return group_index(x0, x1, n_groups), uniform52(x2, x3)


@njit(cache=True)
def solver_uniform(seed, attempt, replica, instance):
    """The uniform of a solver-stream attempt (words x2, x3); used by processes that draw the group
    from another stream (``shared``, ``sweep``)."""
    a = np.uint64(attempt)
    x0, x1, x2, x3 = philox4x32_10(a & _MASK32, a >> _S32, np.uint64(replica) & _MASK32,
                                   np.uint64(instance) & _MASK32, np.uint64(seed) & _MASK32, _SOLVER_U)
    return uniform52(x2, x3)


# --------------------------------------------------------------------------------------------
# proposal-process streams (references in nimbus.core.rng)
# --------------------------------------------------------------------------------------------
@njit(cache=True)
def shared_group(seed, attempt, cohort, instance, n_groups):
    """``rng.shared_group``: key (seed, SHARED), counter (n lo, n hi, cohort, instance), x0, x1 -> group."""
    a = np.uint64(attempt)
    x0, x1, x2, x3 = philox4x32_10(a & _MASK32, a >> _S32, np.uint64(cohort) & _MASK32,
                                   np.uint64(instance) & _MASK32, np.uint64(seed) & _MASK32, _SHARED_U)
    return group_index(x0, x1, n_groups)


@njit(cache=True)
def pt_uniform(seed, step, group, replica, instance):
    """``rng.pt_uniform``: key (seed, PT), counter (step, r, replica, instance), x0, x1 -> uniform52."""
    x0, x1, x2, x3 = philox4x32_10(np.uint64(step) & _MASK32, np.uint64(group) & _MASK32,
                                   np.uint64(replica) & _MASK32, np.uint64(instance) & _MASK32,
                                   np.uint64(seed) & _MASK32, _PT_U)
    return uniform52(x0, x1)


@njit(cache=True)
def color_uniform(seed, step, group, replica, instance):
    """``rng.color_uniform``: key (seed, COLOR), counter (step, r, replica, instance), x0, x1 -> uniform52."""
    x0, x1, x2, x3 = philox4x32_10(np.uint64(step) & _MASK32, np.uint64(group) & _MASK32,
                                   np.uint64(replica) & _MASK32, np.uint64(instance) & _MASK32,
                                   np.uint64(seed) & _MASK32, _COLOR_U)
    return uniform52(x0, x1)


@njit(cache=True)
def pt_arbiter(seed, step, replica, instance, n_fired):
    """``rng.pt_arbiter``: counter (step, 0xFFFFFFFF, replica, instance), x0, x1 -> index in [0, |F|)."""
    x0, x1, x2, x3 = philox4x32_10(np.uint64(step) & _MASK32, _PT_ARBITER_U, np.uint64(replica) & _MASK32,
                                   np.uint64(instance) & _MASK32, np.uint64(seed) & _MASK32, _PT_U)
    return group_index(x0, x1, n_fired)


@njit(cache=True)
def nfold_uniforms(seed, step, replica, instance):
    """``rng.nfold_uniforms``: counter (n_step, 0, replica, instance); (x0, x1) -> selection, (x2, x3) -> wait."""
    x0, x1, x2, x3 = philox4x32_10(np.uint64(step) & _MASK32, _ZERO_U, np.uint64(replica) & _MASK32,
                                   np.uint64(instance) & _MASK32, np.uint64(seed) & _MASK32, _NFOLD_U)
    return uniform52(x0, x1), uniform52(x2, x3)


@njit(cache=True)
def sweep_permutation_into(seed, sweep, cohort, instance, perm):
    """``rng.sweep_permutation`` into perm (int64[R]): Fisher-Yates from the identity; swap t uses words
    (2t, 2t+1) of the sweep's consecutive Philox words (block b = counter (sweep, b, cohort, instance))."""
    R = perm.shape[0]
    for i in range(R):
        perm[i] = i
    su = np.uint64(sweep) & _MASK32
    cu = np.uint64(cohort) & _MASK32
    iu = np.uint64(instance) & _MASK32
    ku = np.uint64(seed) & _MASK32
    y0 = y1 = y2 = y3 = np.uint32(0)
    for t in range(R - 1):
        i = R - 1 - t
        # words 2t, 2t+1 = block t // 2, lanes (0, 1) for even t, (2, 3) for odd t (one Philox per 2 swaps)
        if t & 1:
            j = group_index(y2, y3, i + 1)
        else:
            y0, y1, y2, y3 = philox4x32_10(su, np.uint64(t >> 1) & _MASK32, cu, iu, ku, _SWEEP_U)
            j = group_index(y0, y1, i + 1)
        tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp


@njit(cache=True)
def solver_draws_into(seed, attempt0, replica, instance, n_groups, out_r, out_u):
    """Fill out_r[j], out_u[j] with the solver draws of attempts attempt0 + j."""
    for j in range(out_r.shape[0]):
        r, u = solver_draw(seed, attempt0 + j, replica, instance, n_groups)
        out_r[j] = r
        out_u[j] = u


@njit(cache=True)
def philox4x32_10_rows(ctr, key, out):
    """Row-wise Philox: ctr uint32[n, 4], key uint32[n, 2] -> out uint32[n, 4] (test/cross-check helper)."""
    for j in range(ctr.shape[0]):
        y0, y1, y2, y3 = philox4x32_10(ctr[j, 0], ctr[j, 1], ctr[j, 2], ctr[j, 3], key[j, 0], key[j, 1])
        out[j, 0] = y0
        out[j, 1] = y1
        out[j, 2] = y2
        out[j, 3] = y3


# Generic streams with the layout of nimbus.core.rng.stream_words: word w is lane (w % 4) of the
# Philox block with counter (w // 4, c1, c2, c3) and key (seed, stream); the block index is masked
# to 32 bits exactly as in the reference.

@njit(cache=True)
def _lane(y0, y1, y2, y3, lane):
    if lane == 0:
        return y0
    if lane == 1:
        return y1
    if lane == 2:
        return y2
    return y3


@njit(cache=True)
def stream_word(seed, stream, w, c1, c2, c3):
    """Word w (uint32) of ``rng.stream_words(seed, stream, ..., c2=c2, c3=c3, c1=c1)``."""
    wu = np.uint64(w)
    y0, y1, y2, y3 = philox4x32_10((wu >> _S2) & _MASK32, c1, c2, c3, seed, stream)
    return _lane(y0, y1, y2, y3, np.int64(wu & _U3))


@njit(cache=True)
def _word_pair(seed, stream, j, c1, c2, c3):
    # Pair j = words (2j, 2j+1) = block j // 2, lanes (0, 1) for even j, (2, 3) for odd j.
    ju = np.uint64(j)
    y0, y1, y2, y3 = philox4x32_10((ju >> _S1) & _MASK32, c1, c2, c3, seed, stream)
    if (ju & _U1) == _U1:
        return y2, y3
    return y0, y1


@njit(cache=True)
def stream_uniform(seed, stream, j, c1, c2, c3):
    """Uniform j of ``rng.stream_uniforms`` (words 2j, 2j+1)."""
    a, b = _word_pair(seed, stream, j, c1, c2, c3)
    return uniform52(a, b)


@njit(cache=True)
def stream_integer(seed, stream, j, high, c1, c2, c3):
    """Integer j of ``rng.stream_integers`` (uniform in [0, high), multiply-high on words 2j, 2j+1)."""
    a, b = _word_pair(seed, stream, j, c1, c2, c3)
    return group_index(a, b, high)


@njit(cache=True)
def stream_bit(seed, stream, w, c1, c2, c3):
    """Bit w of ``rng.stream_bits`` (low bit of word w)."""
    return np.uint8(np.uint64(stream_word(seed, stream, w, c1, c2, c3)) & _U1)


# --------------------------------------------------------------------------------------------
# Schedules and acceptance
# --------------------------------------------------------------------------------------------

@njit(cache=True)
def schedule_tau(kind, p0, p1, p2, p3, n_total, n):
    """tau_n of ``Schedule(kind, (p0, p1, p2, p3), n_total).tau(n)`` with the same expressions.

    frac = 0 if B == 1 else n / (B - 1); ZERO: 0; CONSTANT: p0; GEOMETRIC: exp(p0 + (p1 - p0) frac);
    FN_ENDPOINT: p0 / log1p(p1 + (p2 - p1) frac); FN_LEGACY: p0 / log1p((p2 + n p3) / p1).
    An unknown kind raises ValueError (the host already validates kinds in ``Schedule``).
    """
    if n_total == 1:
        frac = 0.0
    else:
        frac = n / (n_total - 1)
    if kind == ZERO:
        return 0.0
    if kind == CONSTANT:
        return p0
    if kind == GEOMETRIC:
        return math.exp(p0 + (p1 - p0) * frac)
    if kind == FN_ENDPOINT:
        return p0 / math.log1p(p1 + (p2 - p1) * frac)
    if kind == FN_LEGACY:
        return p0 / math.log1p((p2 + n * p3) / p1)
    raise ValueError("unknown schedule kind")


@njit(cache=True)
def schedule_taus_into(kind, p0, p1, p2, p3, n_total, n, out):
    """out[j] = schedule_tau(..., n[j]) for an int64 array of attempt indices."""
    for j in range(n.shape[0]):
        out[j] = schedule_tau(kind, p0, p1, p2, p3, n_total, n[j])


@njit(cache=True)
def metropolis_threshold(tau, u):
    """xi = -(tau / 2) * log(u) (Methods, Stochastic transition dynamics and FN annealing)."""
    return -(tau / 2.0) * math.log(u)


@njit(cache=True)
def metropolis_accept(v, tau, u):
    """Accept iff v <= xi (tau > 0) or v <= 0 (tau == 0); v is the neuron input, Delta E = 2 v."""
    if tau == 0.0:
        return v <= 0.0
    return v <= -(tau / 2.0) * math.log(u)


# Short aliases used by kernels and the task description.
accept = metropolis_accept
threshold = metropolis_threshold


@njit(cache=True)
def log_into(u, out):
    """out[j] = math.log(u[j]) through Numba's libm binding (ulp cross-check helper)."""
    for j in range(u.shape[0]):
        out[j] = math.log(u[j])


@njit(cache=True)
def log1p_into(x, out):
    for j in range(x.shape[0]):
        out[j] = math.log1p(x[j])


@njit(cache=True)
def exp_into(x, out):
    for j in range(x.shape[0]):
        out[j] = math.exp(x[j])

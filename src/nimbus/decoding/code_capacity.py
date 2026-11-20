"""Shots of the rotated surface code under code-capacity noise, and logical classes of a correction.

Noise models (``SurfaceCodeProblem.noise``): "independent" (X errors with probability p, one check type) or
"depolarizing" (X, Y, Z each with probability p/3; error bits are (x | z), 2n long, both check types).

The shots of :func:`sample_shots` depend only on (seed, d, p, noise), so every decoder can be run on the same shots by
regenerating them from the seed.
"""
from __future__ import annotations

import numpy as np

__all__ = ["sample_shots", "class_correction", "class_failures"]


def sample_shots(prob, shots, seed):
    """Error bits [shots, n_spins] (uint8) and syndromes of ``prob`` (a SurfaceCodeProblem).

    The generator is ``numpy.random.default_rng(seed * 1000003 + d * 101 + round(p * 1e4) * 7 + noise_id)``,
    noise_id = 0 (independent) or 1 (depolarizing)."""
    nid = 0 if prob.noise == "independent" else 1
    rng = np.random.default_rng(seed * 1000003 + prob.d * 101 + int(round(prob.p * 1e4)) * 7 + nid)
    n = prob.nq
    if prob.noise == "independent":
        e = (rng.random((shots, n)) < prob.p).astype(np.uint8)
    else:
        u = rng.random((shots, n))
        x = ((u < prob.p / 3) | ((u >= 2 * prob.p / 3) & (u < prob.p))).astype(np.uint8)  # X or Y
        z = ((u >= prob.p / 3) & (u < prob.p)).astype(np.uint8)                           # Z or Y
        e = np.concatenate([x, z], axis=1)
    return e, prob.syndrome_of(e)


def class_correction(prob, c0, k):
    """Correction c0 [S, 2n] (depolarizing problem) moved to logical class k [S] relative to c0: bit 0 of k applies one
    X logical line, bit 1 one Z logical line."""
    n = prob.nq
    lx = np.zeros(2 * n, np.uint8); lx[list(prob._checks[0][3][0])] = 1
    lz = np.zeros(2 * n, np.uint8); lz[n + np.array(prob._checks[1][3][0])] = 1
    return c0 ^ (np.asarray(k)[:, None] & 1) * lx ^ ((np.asarray(k)[:, None] >> 1) & 1) * lz


def class_failures(prob, c0, e):
    """Logical failure [S, 4] (bool) of the correction c0 moved to each of the 4 logical classes, against the errors e.
    A decoder that outputs a class k per shot fails on shot s iff ``class_failures(...)[s, k]``."""
    S = len(c0)
    return np.stack([prob.logical_failure(class_correction(prob, c0, np.full(S, k)), e) for k in range(4)], 1)

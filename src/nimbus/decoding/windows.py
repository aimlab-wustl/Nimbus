"""Sliding-window (real-time) decoding of the surface-code memory, for any window decoder.

Memory: N noisy rounds plus one perfect final round (:class:`~nimbus.decoding.phenomenological.DepolarizingMemory`).
Round t owns the data errors that occur just before measurement round t and the measurement errors of round t. With this
choice every detector D(c, t) = s(c, t) xor s(c, t-1) is touched only by variables of round t and by m(c, t-1), so a
committed block of rounds [t0, t0 + C) leaves all its own detectors satisfied and changes exactly one later detector row,
D(., t0 + C), through m(., t0 + C - 1). The data errors after the last noisy round (slice N, before the perfect round)
belong to the final window.

Sliding window (commit C, buffer B, W = C + B): window rounds [t0, t0 + W). A non-final window has an open top (no perfect
round): m(c, t0 + W - 1) keeps only its in-window detector (``memory_model(d, W, open_top=True)``). The bottom is closed:
the residual syndrome already contains the effect of every earlier commit. The window decoder returns a configuration of
the window; its part in the commit rounds [t0, t0 + C) is added to the correction, its syndrome is XORed into the residual
(this clears the rounds < t0 + C and forwards m(., t0 + C - 1) into D(., t0 + C)), then t0 += C. When t0 + W >= N, the final
window [t0, N) plus the perfect round (closed) is decoded and committed whole. Failure = any logical flip of
(true error XOR correction).

A window decoder is a callable ``dec(w, Dxw, Dzw, final, k, t0) -> {variant: (ex_w, ez_w)}``: w is the window model,
Dxw / Dzw the residual detectors of the window (one row per shot), k the window index; it returns local configurations
with H_w e = D_w per sector (one variant for non-final windows). See :mod:`nimbus.decoding.baselines` for the CPU
decoders; the bit-exact hardware decoder is :func:`nimbus.decoding.real_time.decode_stream`.
"""
from __future__ import annotations

import numpy as np

from nimbus.decoding.phenomenological import memory_model

__all__ = ["schedule", "window_maps", "decode_windows", "correction_failures", "particular_decoder"]


def schedule(N: int, C: int, B: int):
    """[(t0, T, final)]: non-final windows of W = C + B rounds while t0 + W < N, then the final window [t0, N) plus the
    perfect round. One window covering all N rounds is the block decoder: ``[(0, N, True)]``."""
    assert C >= 1 and B >= 0
    out, t0 = [], 0
    while t0 + C + B < N:
        out.append((t0, C + B, False))
        t0 += C
    out.append((t0, N - t0, True))
    return out


def window_maps(gc, wc, t0):
    """Global index of each local variable and each local detector of window sector wc (a PhenoCode) starting at round
    t0, in the memory sector gc; also the window round that owns each local variable. Returns (variables, detectors,
    rounds)."""
    n, m = gc.n, gc.m
    tl = np.arange(wc.nx) // n
    vx = (t0 + tl) * n + np.arange(wc.nx) % n
    tm = np.arange(wc.N - wc.nx) // m
    vm = gc.nx + (t0 + tm) * m + np.arange(wc.N - wc.nx) % m
    det = t0 * m + np.arange(wc.ndet)
    rnd = np.concatenate([tl, tm])                       # round that owns each local variable
    return np.concatenate([vx, vm]), det, rnd


def decode_windows(d, N, sched, Dx, Dz, decoder, check=True):
    """Run a window schedule (from :func:`schedule`) on the memory of N noisy rounds with detectors Dx, Dz.
    Returns {variant: (corr_x, corr_z)}: the global corrections of the final window's variants. check=True asserts that
    each commit clears its rounds and that the final corrections reproduce every detector."""
    G = memory_model(d, N)
    gx, gz = G.cx, G.cz
    Rx, Rz = Dx.copy(), Dz.copy()
    cx = np.zeros((len(Dx), gx.N), np.uint8)
    cz = np.zeros((len(Dz), gz.N), np.uint8)
    syn = lambda code, X: ((code.Hsp @ X.T.astype(np.int32)).T % 2).astype(np.uint8)
    for k, (t0, Tw, final) in enumerate(sched):
        w = memory_model(d, Tw, not final)
        vx, dx, rx = window_maps(gx, w.cx, t0)
        vz, dz, rz = window_maps(gz, w.cz, t0)
        out = decoder(w, Rx[:, dx], Rz[:, dz], final, k, t0)
        if not final:
            (ex, ez), = out.values()
            Cc = sched[k + 1][0] - t0
            kx, kz = rx < Cc, rz < Cc
            px = np.zeros_like(cx); px[:, vx[kx]] = ex[:, kx]
            pz = np.zeros_like(cz); pz[:, vz[kz]] = ez[:, kz]
            cx ^= px; cz ^= pz
            Rx ^= syn(gx, px)
            Rz ^= syn(gz, pz)
            if check:
                done = (t0 + Cc) * gx.m
                assert not Rx[:, :done].any() and not Rz[:, :done].any(), "commit left detectors in committed rounds"
        else:
            res = {}
            for v, (ex, ez) in out.items():
                ax, az = cx.copy(), cz.copy()
                ax[:, vx] ^= ex; az[:, vz] ^= ez
                if check:
                    assert (syn(gx, ax) == Dx).all(), f"{v}: X correction misses detectors"
                    assert (syn(gz, az) == Dz).all(), f"{v}: Z correction misses detectors"
                res[v] = (ax, az)
            return res


def correction_failures(d, N, corr, ox, oz):
    """Per-shot logical failure (bool) of each correction {variant: (corr_x, corr_z)} against the true observables."""
    G = memory_model(d, N)
    return {v: (((a.astype(np.int64) @ G.cx.L) % 2 != ox) | ((b.astype(np.int64) @ G.cz.L) % 2 != oz))
            for v, (a, b) in corr.items()}


def particular_decoder():
    """Window decoder that returns any valid configuration (a GF(2) particular solution); tests the commit and
    forwarding logic only."""
    def dec(w, Dxw, Dzw, final, k, t0):
        return {"c0": (w.cx.particular_fast(Dxw), w.cz.particular_fast(Dzw))}
    return dec

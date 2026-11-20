"""PYNQ driver for the real-time decoder board design (ptw_board: streaming wrapper + ptw_top; register map, record layout
and procedures: hardware/README.md).

Runs on the board with numpy only (no nimbus import). Off the board the same code drives the register-level mock
(hardware/host/ptw_mock.py, golden model inside) or records its bus traffic for the xsim board testbench
(hardware/sim/gen_ptw_board_case.py).

A batch = S shots, each N + 1 = 4 d + 1 rounds of raw syndromes (the last one the perfect round), preloaded into the shot
memory; the PL round timer then releases one round every T_ROUND decoder cycles (0 = as fast as the decoder takes them) into
the syndrome ring, and the window FSM decodes the 3 windows of every shot as their rounds arrive. One 16-word record per
window: decision (sigma, frame bits, ML class, best energy, class counts) and timing (trigger, take, commit cycles).

On the RFSoC 4x2:
    from ptw_board import PTWBoard
    brd = PTWBoard.from_overlay("ptw_d7.bit")
    brd.set_noise(0.04)                           # thresholds, swap constants, Wq / wq, rank mode for p (any p < 1/4)
    brd.configure(nsw=4096, tround=0)             # sweeps per window; round period in decoder cycles (0: throughput)
    res = brd.run(syn, seed=1, batch=25)          # syn: uint8 [S, N + 1, M2]; every batch reseeded with seed + batch index

Command line (on the board):
    python3 ptw_board.py --bit ptw_d7.bit --shots shots.npz --p 0.04 --nsw 4096 --tround-us 1000 --batch 25 --out res.npz
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ------------------------------------------------------------------------------------------------ register map (bytes)
CTRL, STATUS, NSW, TROUND, NSHOT, WQ_D, WQ_M, SEED_CTRL, THR_COMMIT = 0x000, 0x004, 0x008, 0x00C, 0x010, 0x014, 0x018, 0x01C, 0x020
ID, CFG0, CFG1, CFG2, CFG3, CFG4, CFG5 = 0x024, 0x028, 0x02C, 0x030, 0x034, 0x038, 0x03C
FRC_LO, FRC_HI, T_START_LO, T_START_HI, T_END_LO, T_END_HI = 0x040, 0x044, 0x048, 0x04C, 0x050, 0x054
MISSES, LATE, RELEASED, WIN_DONE, CFG6, CFG7, CFG8, TMODE = 0x058, 0x05C, 0x060, 0x064, 0x068, 0x06C, 0x070, 0x074
SEED_LANE, SEED_SWAP, THR_STAGE, THETA = 0x080, 0x084, 0x200, 0x400
RES_BASE, SYN_BASE = 0x40000, 0x80000
CTRL_START, CTRL_FILL, CTRL_RECALC, CTRL_ABORT = 1, 2, 4, 8
ST_READY, ST_BUSY, ST_DONE, ST_CORE_BUSY, ST_CORE_READY = 1, 2, 4, 8, 16
ID_VALUE = 0x50545701
REC_WORDS = 16
NWIN = 3                                   # windows per shot (N = 4 d rounds, (C, B) = (d, d): two open, one final)

# fixed design constants (nimbus.decoding.real_time.PTWConfig, nimbus.decoding.one_shot)
GFSR_A, GFSR_B = (2281, 1029), (3217, 576)
EF, QEW, TF, TW, XIW, LF, USW = 9, 12, 3, 20, 20, 16, 12
M64 = 0xFFFFFFFFFFFFFFFF
T1_LEVELS = [(1, -1), (0, 1), (2, -1), (1, 1), (2, 1)]
T2_REST = [(1, 0), (0, 2), (2, 0), (1, 2), (2, 2)]


# ------------------------------------------------------------------------------------------------ tables (= real_time.window_spec)
def ladder(R: int, bmin: float = 0.1) -> np.ndarray:
    return np.geomspace(1.0, bmin, R) if R > 1 else np.array([1.0])


def make_tables(p: float, R: int, KSL: int, B: int = 16, bmin: float = 0.1, betas=None):
    """Runtime tables of the window decoder for noise rate p (same arithmetic as real_time.window_spec; tests/test_ptw_board.py
    checks it): thr [R, KSL + 12] (threshold index r = 1..KSL + 12), theta [R - 1], Wq, wq (fixed-point weights, EF = 9),
    mode (1: p < 1/28, T2 order)."""
    if not 0 < p < 0.25:
        raise ValueError("the rank acceptance is exact for 0 < p < 1/4 only")
    W, wm = float(np.log(3 * (1 - p) / p)), float(np.log((1 - p) / p))
    mode = int(3 * W < 4 * wm)
    t2 = ([(2, -2), (-1, 2)] if mode else [(-1, 2), (2, -2)]) + T2_REST
    dE = [W * k for k in range(1, KSL + 1)] + [W * a + wm * b for a, b in t2] + [W * a + wm * b for a, b in T1_LEVELS]
    dE = np.array([0.0] + dE)
    betas = ladder(R, bmin) if betas is None else np.asarray(betas, np.float64)
    thr = np.floor((1 << B) * np.exp(-betas[:, None] * dE[None, :])).astype(np.int64)
    thr = np.minimum(thr, (1 << B) - 1)
    Wq, wq = int(round(W * (1 << EF))), int(round(wm * (1 << EF)))
    if R > 1:
        th = np.rint(float(1 << (TF + EF)) / (betas[:-1] - betas[1:]))
        theta = np.clip(th, 0, (1 << TW) - 1).astype(np.int64)
    else:
        theta = np.zeros(0, np.int64)
    return thr[:, 1:], theta, Wq, wq, mode


# ------------------------------------------------------------------------------------------------ seeds (= real_time)
def splitmix64_seq(seed: int, count: int):
    out, x = [], seed & M64
    for _ in range(count):
        x = (x + 0x9E3779B97F4A7C15) & M64
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        out.append(z ^ (z >> 31))
    return out


def _bits_from(seed: int, nbits: int) -> np.ndarray:
    z = np.array(splitmix64_seq(seed, (nbits + 63) // 64), dtype=np.uint64)
    b = np.unpackbits(z.view(np.uint8), bitorder="little")[:nbits].astype(np.uint8)
    if not b.any():
        b[0] = 1
    return b


def batch_base(seed: int, TR: int) -> int:
    """Generator base of a batch: real_time.window_bases(seed, [0], 0, TR, False) (shot 0, first window), from which the
    golden model seeds its first window; the generators then run on through every window and shot of the batch."""
    tag = ((TR << 8) | 0) * 0xD6E8FEB86659FD93
    return splitmix64_seq((seed * 0x100000001B3 + tag) & M64, 1)[0]


def seed_streams(base: int, R: int, NB: int, bank_N):
    """Lane generator histories (padded with leading zeros to a multiple of the bank's N = lanes x B bits, oldest first):
    {(which, bank): [R, L] uint8}; swap generators: [R - 1, 128] bits (word k // 32, bit k % 32 of the xoshiro128** state)."""
    lanes = {}
    for which, (P, _) in (("a", GFSR_A), ("b", GFSR_B)):
        for b in range(NB):
            L = -(-P // bank_N[b]) * bank_N[b]
            arr = np.zeros((R, L), np.uint8)
            for i in range(R):
                sd = base ^ (((2 * (i * NB + b) + (1 if which == "a" else 2)) * 0xD1B54A32D192ED03) & M64)
                arr[i, L - P:] = _bits_from(sd, P)
            lanes[(which, b)] = arr
    swb = np.zeros((max(R - 1, 1), 128), np.uint8)
    for t in range(R - 1):
        z = splitmix64_seq(base ^ (((t + 1) * 0xABCDEF0123456789) & M64), 2)
        words = [z[0] & 0xFFFFFFFF, z[0] >> 32, z[1] & 0xFFFFFFFF, z[1] >> 32]
        for k in range(128):
            swb[t, k] = (words[k // 32] >> (k % 32)) & 1
    return lanes, swb


# ------------------------------------------------------------------------------------------------ records
REC_FIELDS = ("shot", "win", "fin", "fr_ml", "frame", "dec_ml", "best_c", "best_e", "counts", "sigma", "t_trig", "d_take",
              "d_done", "cyc_win", "lag")


def unpack_records(w: np.ndarray) -> dict:
    """Window records [nW, 16] uint32 -> fields (layout: hardware/README.md). sigma stays as 4 words (bit i of the
    128-bit value = sigma bit i, sector-major)."""
    w = np.asarray(w, np.uint64).reshape(-1, REC_WORDS)
    w0 = w[:, 0]
    return dict(fr_ml=(w0 & 3).astype(np.int8), frame=((w0 >> 2) & 3).astype(np.int8), dec_ml=((w0 >> 4) & 3).astype(np.int8),
                best_c=((w0 >> 6) & 3).astype(np.int8), fin=((w0 >> 8) & 1).astype(np.int8), win=((w0 >> 9) & 3).astype(np.int8),
                shot=(w0 >> 16).astype(np.int64), best_e=w[:, 1].astype(np.int64), counts=w[:, 2:6].astype(np.int64),
                sigma=w[:, 6:10].astype(np.uint32), t_trig=(w[:, 10] | (w[:, 11] << np.uint64(32))).astype(np.int64),
                d_take=w[:, 12].astype(np.int64), d_done=w[:, 13].astype(np.int64), cyc_win=w[:, 14].astype(np.int64),
                lag=w[:, 15].astype(np.int64))


def pack_records(r: dict) -> np.ndarray:
    """Inverse of unpack_records (used by the mock)."""
    n = len(r["shot"])
    w = np.zeros((n, REC_WORDS), np.uint64)
    g = lambda k: np.asarray(r[k], np.uint64)  # noqa: E731
    w[:, 0] = (g("fr_ml") & 3) | ((g("frame") & 3) << 2) | ((g("dec_ml") & 3) << 4) | ((g("best_c") & 3) << 6) \
        | ((g("fin") & 1) << 8) | ((g("win") & 3) << 9) | ((g("shot") & 0xFFFF) << 16)
    w[:, 1] = g("best_e") & 0xFFFFFFFF
    w[:, 2:6] = np.asarray(r["counts"], np.uint64).reshape(n, 4)
    w[:, 6:10] = np.asarray(r["sigma"], np.uint64).reshape(n, 4)
    t = g("t_trig")
    w[:, 10], w[:, 11] = t & 0xFFFFFFFF, t >> np.uint64(32)
    for j, k in ((12, "d_take"), (13, "d_done"), (14, "cyc_win"), (15, "lag")):
        w[:, j] = g(k) & 0xFFFFFFFF
    return w.astype(np.uint32)


def sigma_words(bits) -> np.ndarray:
    """sigma bits [..., M2] -> [..., 4] uint32 words (record layout)."""
    bits = np.asarray(bits, np.uint8)
    pad = np.zeros(bits.shape[:-1] + (128,), np.uint8)
    pad[..., :bits.shape[-1]] = bits
    return np.packbits(pad.reshape(bits.shape[:-1] + (4, 32)), axis=-1, bitorder="little").view("<u4")[..., 0].astype(np.uint32)


def cycle_model(nsw: int, RS: int, KS: int, NC: int, K: int, GAP_SW: int, LDC: int, NS: int, CSL: int) -> int:
    """Core cycles from the take of a window to its commit (record word 14, cyc_win), as measured in xsim: nsw RS KS NC
    move cycles, GAP_SW idle cycles after every swap sweep but the last, max(LDC, 4) load cycles, the argmin tree
    (TL = ceil(log2 NS)) and 10 (+ 2 with the C-slow loop) cycles of the sweep-end pipeline and commit."""
    TL = max(1, int(NS - 1).bit_length())
    return nsw * RS * KS * NC + GAP_SW * ((nsw - 1) // K) + max(LDC, 4) + TL + 2 * CSL + 10


# ------------------------------------------------------------------------------------------------ driver
class PTWBoard:
    def __init__(self, mmio, poll_timeout_s: float = 600.0):
        self.mmio = mmio
        self.timeout = poll_timeout_s
        idv = self._rd(ID)
        if idv != ID_VALUE:
            raise RuntimeError(f"ptw_board ID {idv:#010x}, expected {ID_VALUE:#010x} (wrong bitstream or address)")
        c = [self._rd(x) for x in (CFG0, CFG1, CFG2, CFG3, CFG4, CFG5, CFG6, CFG7, CFG8)]
        b4 = lambda v: (v >> 24, (v >> 16) & 255, (v >> 8) & 255, v & 255)  # noqa: E731
        self.R, self.B, self.K, self.ROWN = b4(c[0])
        self.d, self.TR, self.M2, self.NC = b4(c[1])
        self.TW, self.EW, self.LW, self.QW = b4(c[2])
        self.SYN_AW, self.RES_AW, self.WPL, self.USW = b4(c[3])
        self.max_shots = c[4]
        self.NE, self.RS, self.NB, self.KS = b4(c[5])
        self.CSL, self.GAP_SW, self.LDC, self.KSL = b4(c[6])
        self.bank_lanes = [c[7] & 0xFFFF, c[7] >> 16][: self.NB]
        self.RD, self.NP, self.RB, self.WPR = b4(c[8])
        self.CM = self.d
        self.NR = 2 * self.TR + 1
        if self.TR != 2 * self.CM or self.M2 > 128 or self.NE * self.NB > 32 or self.NP > 32:
            raise RuntimeError("design outside the wrapper's assumptions (TR = 2 d, M2 <= 128, NE NB <= 32, NP <= 32)")
        self.thr = self.theta = None
        self.Wq = self.wq = self.mode = None
        self.seed_value = None
        self.filled = False
        self.nsw, self.tround = 4096, 0
        self.info = {}

    # ---------------------------------------------------------------- bus helpers
    def _rd(self, off: int) -> int:
        return int(self.mmio.read(off)) & 0xFFFFFFFF

    def _wr(self, off: int, val: int):
        self.mmio.write(off, int(val) & 0xFFFFFFFF)

    def _write_block(self, off: int, words: np.ndarray):
        words = np.ascontiguousarray(words, np.uint32).ravel()
        arr = getattr(self.mmio, "array", None)
        if arr is not None and off % 16 == 0:
            # whole 16-byte groups only (zero padding): the aarch64 memcpy copies a tail that is not a multiple of 16 bytes
            # with overlapping unaligned 16-byte stores, which fault on the device mapping (bus error on the board:
            # a 325-word copy failed, a 260-word copy passed); chunks of 64 words
            n4 = -(-len(words) // 4) * 4
            w = np.zeros(n4, np.uint32); w[:len(words)] = words
            for i in range(0, n4, 64):
                arr[off // 4 + i: off // 4 + min(n4, i + 64)] = w[i:i + 64]
        else:
            for i, v in enumerate(words):
                self._wr(off + 4 * i, int(v))

    def _read_block(self, off: int, n: int) -> np.ndarray:
        arr = getattr(self.mmio, "array", None)
        if arr is not None and off % 16 == 0:
            n4 = -(-n // 4) * 4
            return np.concatenate([np.array(arr[off // 4 + i: off // 4 + min(n4, i + 64)], np.uint32) for i in range(0, n4, 64)]
                                  or [np.zeros(0, np.uint32)])[:n]
        return np.array([self._rd(off + 4 * i) for i in range(n)], np.uint32)

    def _poll(self, off: int, mask: int, val: int, sleep: float = 0.0):
        if hasattr(self.mmio, "poll"):                       # mock / recorder
            return self.mmio.poll(off, mask, val)
        t0 = time.time()
        while (self._rd(off) & mask) != val:
            if time.time() - t0 > self.timeout:
                raise TimeoutError(f"register {off:#x}: waiting for {val:#x} under mask {mask:#x}")
            if sleep:
                time.sleep(sleep)

    def status(self) -> int:
        return self._rd(STATUS)

    def wait_ready(self):
        self._poll(STATUS, ST_READY, ST_READY)

    # ---------------------------------------------------------------- overlay
    @classmethod
    def from_overlay(cls, bitfile: str, ip: str | None = None, info: str | None = None):
        """Load the overlay (PYNQ) and attach to the decoder IP (default: the first IP whose name starts with 'ptw')."""
        from pynq import MMIO, Overlay
        ol = Overlay(bitfile)
        name = ip or next(k for k in ol.ip_dict if k.split("/")[-1].startswith("ptw"))
        d = ol.ip_dict[name]
        brd = cls(MMIO(d["phys_addr"], d["addr_range"]))
        brd.overlay = ol
        ij = info or bitfile.rsplit(".", 1)[0] + ".json"
        try:
            with open(ij) as f:
                brd.info = dict(json.load(f))
        except OSError:
            pass
        if brd.info.get("pl0_board_mhz"):
            brd.fclk0_mhz = cls.set_fclk0(float(brd.info["pl0_board_mhz"]))
        else:
            brd.fclk0_mhz = None
        return brd

    @staticmethod
    def set_fclk0(target_mhz: float) -> float:
        """PL clock 0 at the highest value PYNQ can make that is not above target_mhz (as pt_board.PTBoard.set_fclk0)."""
        import warnings
        from pynq import Clocks
        f = target_mhz
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(40):
                Clocks.fclk0_mhz = f
                got = Clocks.fclk0_mhz
                if got <= target_mhz * 1.0005:
                    break
                f -= 0.5
        time.sleep(1.0)
        return got

    # ---------------------------------------------------------------- tables
    def set_tables(self, thr, theta, Wq: int, wq: int, mode: int):
        """Runtime tables: thr [R, ROWN] (threshold index 1..ROWN; non-increasing within each table, < 2^B), theta [R - 1],
        fixed-point weights Wq, wq (QW bits), rank mode. Only while idle; after FILL, RECALC recomputes the pre-drawn swap
        thresholds from the same uniforms."""
        thr = np.asarray(thr, np.int64)
        if thr.shape == (self.R, self.ROWN + 1):
            thr = thr[:, 1:]
        if thr.shape != (self.R, self.ROWN):
            raise ValueError(f"thr shape {thr.shape}, expected ({self.R}, {self.ROWN})")
        if (thr < 0).any() or (thr >= (1 << self.B)).any():
            raise ValueError("thresholds must be in [0, 2^B)")
        theta = np.asarray(theta, np.int64)
        if theta.shape != (self.R - 1,) or (theta < 0).any() or (theta >= (1 << self.TW)).any():
            raise ValueError(f"theta must have {self.R - 1} entries in [0, 2^{self.TW})")
        if not (0 <= Wq < (1 << self.QW) and 0 <= wq < (1 << self.QW)):
            raise ValueError("Wq, wq out of range")
        if self.status() & ST_BUSY:
            raise RuntimeError("tables can only be written while idle")
        for t in range(self.R):
            for k in range(self.ROWN):
                self._wr(THR_STAGE + 4 * k, int(thr[t, k]))
            self._wr(THR_COMMIT, t)
        for t in range(self.R - 1):
            self._wr(THETA + 4 * t, int(theta[t]))
        self._wr(WQ_D, Wq); self._wr(WQ_M, wq); self._wr(TMODE, mode)
        if self.filled:
            self._wr(CTRL, CTRL_RECALC)
            self.wait_ready()
        self.thr, self.theta, self.Wq, self.wq, self.mode = thr.copy(), theta.copy(), int(Wq), int(wq), int(mode)

    def set_noise(self, p: float, bmin: float = 0.1, betas=None):
        thr, theta, Wq, wq, mode = make_tables(p, self.R, self.KSL, self.B, bmin, betas)
        self.set_tables(thr, theta, Wq, wq, mode)
        self.p, self.bmin = p, bmin
        return thr, theta, Wq, wq, mode

    # ---------------------------------------------------------------- seeds
    def seed(self, seed: int):
        """Seed every lane generator (two GFSR histories per replica and bank, through the engine's bank shift register and
        the per-replica push) and swap generator (xoshiro per pair) from batch_base(seed), then FILL."""
        if self.status() & ST_BUSY:
            raise RuntimeError("seed only while idle")
        bank_N = [bl * self.B for bl in self.bank_lanes]
        lanes, swb = seed_streams(batch_base(seed, self.TR), self.R, self.NB, bank_N)
        for r in range(self.RS):
            for b in range(self.NB):
                for wi, which in enumerate("ab"):
                    arr = lanes[(which, b)]
                    N = bank_N[b]
                    sub = arr[np.arange(self.NE) * self.RS + r]            # engine e's replica r
                    sh = (np.arange(self.NE, dtype=np.uint64) * self.NB + b)
                    words = (sub.astype(np.uint64) << sh[:, None]).sum(0).astype(np.uint64)
                    for k in range(arr.shape[1]):
                        self._wr(SEED_LANE, int(words[k]))
                        if k % N == N - 1:
                            self._wr(SEED_CTRL, ((1 << b) << (8 * wi)) | (r << 16))
        sw_words = (swb.astype(np.uint64) << np.arange(swb.shape[0], dtype=np.uint64)[:, None]).sum(0)
        for k in range(128):
            self._wr(SEED_SWAP, int(sw_words[k]))
        self._wr(CTRL, CTRL_FILL)
        self.wait_ready()
        self.seed_value, self.filled = seed, True

    # ---------------------------------------------------------------- runs
    def configure(self, nsw: int, tround: int = 0):
        """nsw sweeps per window (a power of two, as the golden model); tround = round period in decoder cycles (0: every
        round due at once, throughput limit; otherwise >= WPR + 7)."""
        if nsw < 2 or nsw & (nsw - 1):
            raise ValueError("nsw must be a power of two >= 2")
        if tround and tround < self.WPR + 7:
            raise ValueError(f"tround must be 0 or >= {self.WPR + 7} cycles")
        self._wr(NSW, nsw); self._wr(TROUND, tround)
        self.nsw, self.tround = nsw, tround

    def pack_rounds(self, syn: np.ndarray) -> np.ndarray:
        """syn uint8 [S, N + 1, M2] -> shot memory words (round k of the batch at k 2^WPL)."""
        syn = np.asarray(syn, np.uint8)
        if syn.ndim != 3 or syn.shape[1:] != (self.NR, self.M2):
            raise ValueError(f"syndromes must be [S, {self.NR}, {self.M2}]")
        S = syn.shape[0]
        stride = 1 << self.WPL
        pad = np.zeros((S * self.NR, stride * 32), np.uint8)
        pad[:, :self.M2] = syn.reshape(S * self.NR, self.M2)
        w = np.packbits(pad.reshape(-1, stride, 32), axis=2, bitorder="little").view("<u4")[..., 0]
        return w.reshape(-1).astype(np.uint32)

    def cycles_per_window(self, nsw: int | None = None) -> int:
        return cycle_model(nsw or self.nsw, self.RS, self.KS, self.NC, self.K, self.GAP_SW, self.LDC, self.R, self.CSL)

    def run_batch(self, syn: np.ndarray, seed: int | None = None) -> dict:
        """One batch (<= max_shots): reseed (if seed is given), write the rounds, START, wait for DONE, read the records,
        the batch timers and counters."""
        S = len(syn)
        if not 0 < S <= self.max_shots:
            raise ValueError(f"batch of {S} shots; 1..{self.max_shots}")
        if seed is not None:
            self.seed(seed)
        if not self.filled:
            raise RuntimeError("seed() first")
        self._write_block(SYN_BASE, self.pack_rounds(syn))
        self._wr(NSHOT, S)
        t0 = time.time()
        self._wr(CTRL, CTRL_START)
        self._poll(STATUS, ST_DONE, ST_DONE, sleep=0.002 if self.tround else 0.0)
        wall = time.time() - t0
        r = unpack_records(self._read_block(RES_BASE, S * NWIN * REC_WORDS))
        ts = self._rd(T_START_LO) | (self._rd(T_START_HI) << 32)
        te = self._rd(T_END_LO) | (self._rd(T_END_HI) << 32)
        r["t_trig"] = r["t_trig"] - ts
        r["batch_cycles"] = te - ts
        r["misses"], r["late"] = self._rd(MISSES), self._rd(LATE)
        r["released"], r["win_done"] = self._rd(RELEASED), self._rd(WIN_DONE)
        r["wall_s"] = wall
        r["seed"] = self.seed_value
        return r

    def run(self, syn: np.ndarray, seed: int = 1, batch: int | None = None, reseed: bool = True) -> dict:
        """All shots in batches of up to max_shots; batch b is seeded with seed + b (reseed) or the stream runs on."""
        batch = min(batch or self.max_shots, self.max_shots)
        parts = []
        for b, i in enumerate(range(0, len(syn), batch)):
            sd = (seed + b) if (reseed or b == 0) else None
            parts.append(self.run_batch(syn[i:i + batch], sd))
            parts[-1]["shot"] = parts[-1]["shot"] + i
        out = {k: np.concatenate([p[k] for p in parts]) for k in REC_FIELDS}
        for k in ("batch_cycles", "misses", "late", "released", "win_done", "wall_s", "seed"):
            out["b_" + k] = np.array([p[k] if p[k] is not None else -1 for p in parts])
        out["b_shots"] = np.array([min(batch, len(syn) - i) for i in range(0, len(syn), batch)])
        return out

    def frc(self) -> int:
        lo = self._rd(FRC_LO)
        return lo | (self._rd(FRC_HI) << 32)

    def measure_clock(self, seconds: float = 1.0) -> float:
        c0, t0 = self.frc(), time.perf_counter()
        time.sleep(seconds)
        c1, t1 = self.frc(), time.perf_counter()
        return (c1 - c0) / (t1 - t0) / 1e6

    def settle_clock(self) -> float:
        """Decoder clock once two consecutive 0.5 s readings agree within 0.1 % (the MMCM settles after a PL clock change)."""
        prev = self.measure_clock(0.5)
        for _ in range(20):
            mhz = self.measure_clock(0.5)
            if abs(mhz - prev) <= 1e-3 * mhz:
                return mhz
            prev = mhz
        raise RuntimeError(f"decoder clock did not settle (last readings {prev:.1f}, {mhz:.1f} MHz)")

    def meta(self) -> dict:
        return dict(R=self.R, B=self.B, K=self.K, ROWN=self.ROWN, d=self.d, TR=self.TR, M2=self.M2, NC=self.NC, KS=self.KS,
                    RS=self.RS, NB=self.NB, NE=self.NE, CSL=self.CSL, GAP_SW=self.GAP_SW, LDC=self.LDC, KSL=self.KSL,
                    nsw=self.nsw, tround=self.tround, p=getattr(self, "p", None), bmin=getattr(self, "bmin", None),
                    thr=self.thr, theta=self.theta, Wq=self.Wq, wq=self.wq, mode=self.mode,
                    cyc_model=self.cycles_per_window(), info=json.dumps(self.info))


# ------------------------------------------------------------------------------------------------ output
def save_results(path: str, res: dict, meta: dict, extra: dict | None = None):
    d = {f"r_{k}": np.asarray(v) for k, v in res.items()}
    d.update({f"m_{k}": np.asarray(v if v is not None else -1) for k, v in meta.items()})
    d.update({k: np.asarray(v) for k, v in (extra or {}).items()})
    np.savez_compressed(path, **d)


def timing_summary(res: dict, meta: dict, mhz: float) -> dict:
    """Per-window decode (take to commit), per-round latency (window / d), reaction (trigger to commit) and deadline slack
    (trigger of the next window of the batch - commit), in us."""
    d = int(meta["d"])
    dec = np.asarray(res["d_done"], np.float64) / mhz
    react = (np.asarray(res["d_take"]) + np.asarray(res["d_done"])) / mhz
    tt = np.asarray(res["t_trig"], np.int64)
    done = tt + np.asarray(res["d_take"]) + np.asarray(res["d_done"])
    # slack within each batch: windows are numbered consecutively; batch boundaries: the shot index restarts the timer
    bs = np.cumsum(np.concatenate([[0], np.asarray(res["b_shots"]) * NWIN]))
    sl = []
    for a, b in zip(bs[:-1], bs[1:]):
        sl.append((tt[a + 1:b] - done[a:b - 1]) / mhz)
    slack = np.concatenate(sl) if sl else np.zeros(0)
    q = lambda x: dict(mean=float(np.mean(x)), p50=float(np.median(x)), p99=float(np.quantile(x, 0.99)), max=float(np.max(x)),  # noqa: E731
                       min=float(np.min(x))) if len(x) else {}
    return dict(window_us=q(dec), round_us=q(dec / d), reaction_us=q(react), slack_us=q(slack),
                misses=int(np.sum(res["b_misses"])), late=int(np.sum(res["b_late"])),
                cyc_win=sorted(set(int(x) for x in res["cyc_win"])), cyc_model=int(meta["cyc_model"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Decode real-time batches on the board")
    ap.add_argument("--bit", required=True); ap.add_argument("--shots", required=True, help="npz with syn [S, N + 1, M2]")
    ap.add_argument("--p", type=float, required=True); ap.add_argument("--bmin", type=float, default=0.1)
    ap.add_argument("--nsw", type=int, default=4096); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tround", type=int, default=0, help="round period in decoder cycles (0: throughput)")
    ap.add_argument("--tround-us", type=float, default=0, help="round period in us (converted with the measured clock)")
    ap.add_argument("--batch", type=int, default=0, help="shots per batch (each reseeded with seed + batch index)")
    ap.add_argument("--no-reseed", action="store_true", help="one generator stream across the batches")
    ap.add_argument("--pl0", type=float, default=0); ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    z = np.load(a.shots)
    syn = z["syn"][: a.max or None]
    brd = PTWBoard.from_overlay(a.bit)
    if a.pl0:
        brd.fclk0_mhz = PTWBoard.set_fclk0(a.pl0)
    mhz = brd.settle_clock()
    fmax = float(brd.info.get("dec_fmax_mhz", 0) or 0)
    if fmax and mhz > fmax:
        raise SystemExit(f"decoder clock {mhz:.1f} MHz is above the routed Fmax {fmax:.1f} MHz: results would not be valid")
    tround = a.tround or (int(round(a.tround_us * mhz)) if a.tround_us else 0)
    brd.set_noise(a.p, bmin=a.bmin)
    brd.configure(a.nsw, tround)
    res = brd.run(syn, seed=a.seed, batch=a.batch or None, reseed=not a.no_reseed)
    meta = brd.meta()
    meta.update(seed=a.seed, batch=a.batch or brd.max_shots, reseed=int(not a.no_reseed), tround_us=a.tround_us)
    extra = dict(clock_mhz=mhz, syn=syn)
    for k in ("obs", "ox", "oz"):
        if k in z:
            extra[k] = z[k][: len(syn)]
    ts = timing_summary(res, meta, mhz)
    extra["timing"] = json.dumps(ts)
    save_results(a.out, res, meta, extra)
    if hasattr(os, "sync"):
        os.sync()                                    # on disk before anything else touches the board
    fin = res["fin"] == 1
    msg = (f"{len(syn)} shots, clock {mhz:.2f} MHz, T_round {tround} cycles: window {ts['window_us']['mean']:.1f} us "
           f"({ts['cyc_win'][:3]} cycles, model {ts['cyc_model']}), per round {ts['round_us']['mean']:.2f} us, "
           f"misses {ts['misses']}, late rounds {ts['late']}, min slack {ts['slack_us'].get('min', float('nan')):.1f} us")
    if "obs" in extra:
        ob = extra["obs"]
        frm = np.stack([res["fr_ml"][fin] & 1, res["fr_ml"][fin] >> 1], 1)
        msg += f"; LER(ML) {np.mean((frm != ob).any(1)):.4f}"
    print(msg)


if __name__ == "__main__":
    main()

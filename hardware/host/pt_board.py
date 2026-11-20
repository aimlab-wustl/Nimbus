"""PYNQ driver for the one-syndrome-per-shot decoder board design (pt_board; register map and procedures:
hardware/README.md).

Runs on the board with numpy only (no nimbus import). Off the board the same code drives the register-level mock
(hardware/host/pt_mock.py, golden model inside) or records its bus traffic for the xsim board testbench (hardware/sim/gen_board_case.py).

On the RFSoC 4x2:
    from pt_board import PTBoard
    brd = PTBoard.from_overlay("pt_d7_ns16.bit")       # loads the overlay (bit + hwh with the same name)
    brd.set_noise(0.15)                                 # thresholds and swap constants from (p, ladder): any p, same bitstream
    brd.seed(1)                                         # generator seeds (once per stream), then FILL
    brd.configure(nsw=1024, gmin=0)                     # budget; gmin > 0 enables the on-chip early stop
    res = brd.run(syn)                                  # syn: uint8 [S, M] syndromes; batches of up to brd.max_shots
    save_results("out.npz", res, brd.meta())

Command line (on the board):  python3 pt_board.py --bit pt.bit --shots shots.npz --p 0.15 --nsw 1024 --out res.npz
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
CTRL, STATUS, NSW, ES_GMIN, ES_TH, NSHOT, SHOT_DONE, SEED_CTRL = 0x000, 0x004, 0x008, 0x00C, 0x010, 0x014, 0x018, 0x01C
THR_COMMIT, ID, CFG0, CFG1, CFG2, CFG3, CFG4, CFG5 = 0x020, 0x024, 0x028, 0x02C, 0x030, 0x034, 0x038, 0x03C
FRC_LO, FRC_HI, T_START_LO, T_START_HI, T_END_LO, T_END_HI = 0x040, 0x044, 0x048, 0x04C, 0x050, 0x054
SEED_LANE, SEED_SWAP, THR_STAGE, THETA = 0x080, 0x0A0, 0x200, 0x400
SYN_BASE, RES_BASE = 0x10000, 0x20000
CTRL_START, CTRL_FILL, CTRL_RECALC, CTRL_ABORT = 1, 2, 4, 8
ST_READY, ST_BUSY, ST_DONE, ST_CORE_BUSY, ST_CORE_READY = 1, 2, 4, 8, 16
ID_VALUE = 0x50540200
REC_WORDS = 8

# fixed design constants (nimbus.decoding.one_shot)
GFSR_A, GFSR_B = (2281, 1029), (3217, 576)
LF, TF, USW_DEFAULT = 16, 12, 12
M64 = 0xFFFFFFFFFFFFFFFF


# ------------------------------------------------------------------------------------------------ tables (= one_shot)
def w3_depolarizing(p: float) -> float:
    """Clause weight of the depolarizing code-capacity model, w3 = ln(3 (1 - p) / p)."""
    return math.log(3.0 * (1.0 - p) / p)


def ladder(NS: int, bmin: float = 0.1) -> np.ndarray:
    return np.geomspace(1.0, bmin, NS) if NS > 1 else np.array([1.0])


def make_tables(w3: float, betas, B: int, kmax: int, TF: int = TF, TW: int = 20):
    """Same arithmetic as nimbus.decoding.one_shot.make_tables (tests/test_pt_board.py checks it): thr [NS, kmax + 1]
    (column 0 unused), theta [NS - 1]."""
    betas = np.asarray(betas, np.float64)
    ks = np.arange(kmax + 1)
    thr = np.floor((1 << B) * np.exp(-betas[:, None] * w3 * ks[None, :])).astype(np.int64)
    thr[:, 1:] = np.minimum(thr[:, 1:], (1 << B) - 1)
    thr[:, 0] = 1 << B
    if len(betas) > 1:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            th = np.rint((1 << TF) / ((betas[:-1] - betas[1:]) * w3))
        th = np.where(np.isfinite(th), th, (1 << TW) - 1)
        theta = np.clip(th, 0, (1 << TW) - 1).astype(np.int64)
    else:
        theta = np.zeros(0, np.int64)
    return thr, theta


# ------------------------------------------------------------------------------------------------ seeds (= one_shot.seed_states)
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


def seed_states(NS: int, seed: int):
    """Generator states of one stream (shot index 0 of one_shot.seed_states): ga [NS, 2281], gb [NS, 3217] (oldest bit first),
    sw [NS - 1, 4] uint32 (xoshiro128** state words)."""
    base = (seed * 0x100000001B3) & M64
    ga = np.stack([_bits_from(base ^ (2 * i + 1) * 0xD1B54A32D192ED03, GFSR_A[0]) for i in range(NS)])
    gb = np.stack([_bits_from(base ^ (2 * i + 2) * 0xD1B54A32D192ED03, GFSR_B[0]) for i in range(NS)])
    sw = np.zeros((max(NS - 1, 0), 4), np.uint32)
    for t in range(NS - 1):
        z = splitmix64_seq(base ^ (t + 1) * 0xABCDEF0123456789, 2)
        sw[t] = [z[0] & 0xFFFFFFFF, z[0] >> 32, z[1] & 0xFFFFFFFF, z[1] >> 32]
    return ga, gb, sw


def padded_stream(g: np.ndarray, N: int) -> np.ndarray:
    """Seed stream of a lane generator: leading zeros to a multiple of N bits, then the history oldest first."""
    L = -(-g.shape[1] // N) * N
    return np.concatenate([np.zeros((g.shape[0], L - g.shape[1]), np.uint8), g], 1)


# ------------------------------------------------------------------------------------------------ records
def unpack_records(w: np.ndarray) -> dict:
    """Result records [S, 8] uint32 -> per-shot fields (layout: hardware/README.md)."""
    w = np.asarray(w, np.uint64).reshape(-1, REC_WORDS)
    w0, w4, w5, w6 = w[:, 0], w[:, 4], w[:, 5], w[:, 6]
    cnt = np.stack([w4 & 0xFFFFF, (w4 >> 20) | ((w5 & 0xFF) << 12), (w5 >> 8) & 0xFFFFF, (w5 >> 28) | ((w6 & 0xFFFF) << 4)], 1)
    return dict(ml=(w0 & 3).astype(np.int8), me=((w0 >> 2) & 3).astype(np.int8), stopped=((w0 >> 4) & 1).astype(bool),
                emin=(w0 >> 16).astype(np.int32), sweeps=w[:, 1].astype(np.int64), cyc_start=w[:, 2].astype(np.int64),
                cyc_run=w[:, 3].astype(np.int64), counts=cnt.astype(np.int64), t_done=w[:, 7].astype(np.uint32))


def pack_records(r: dict) -> np.ndarray:
    """Inverse of unpack_records (used by the mock)."""
    S = len(r["ml"])
    c = np.asarray(r["counts"], np.uint64).reshape(S, 4)
    w = np.zeros((S, REC_WORDS), np.uint64)
    w[:, 0] = (np.asarray(r["ml"], np.uint64) & 3) | ((np.asarray(r["me"], np.uint64) & 3) << 2) \
        | (np.asarray(r["stopped"], np.uint64) << 4) | ((np.asarray(r["emin"], np.uint64) & 0xFFFF) << 16)
    w[:, 1] = r["sweeps"]; w[:, 2] = r["cyc_start"]; w[:, 3] = r["cyc_run"]
    w[:, 4] = (c[:, 0] & 0xFFFFF) | ((c[:, 1] & 0xFFF) << 20)
    w[:, 5] = ((c[:, 1] >> 12) & 0xFF) | ((c[:, 2] & 0xFFFFF) << 8) | ((c[:, 3] & 0xF) << 28)
    w[:, 6] = (c[:, 3] >> 4) & 0xFFFF
    w[:, 7] = np.asarray(r.get("t_done", np.zeros(S)), np.uint64) & 0xFFFFFFFF
    return w.astype(np.uint32)


def cycle_model(sweeps, stopped, nsw: int, K: int, NS: int, gmin: int, NCL: int = 6, GAP: int = 4, GAP_CK: int = 8):
    """Cycles from taking the syndrome to the decision (cyc_start + cyc_run + 2), as validated in xsim:
    fixed budget n:          5 + NCL n + GAP (floor((n - 1) / K) - c) + GAP_CK c + TR + 3,  c = checkpoints passed
                             (powers of two in [gmin, n));
    early stop at checkpoint g: 5 + NCL g + GAP (g / K - 1 - c) + GAP_CK (c + 1) + TR - 2,  c = checkpoints before g;
    TR = ceil(log2(NS) / 2) cycles of the minimum-energy argmin tree."""
    sweeps = np.asarray(sweeps, np.int64)
    stopped = np.asarray(stopped, bool)
    TL = max(1, int(NS - 1).bit_length())
    TR = (TL + 1) // 2
    out = np.zeros(len(sweeps), np.int64)
    for i, (n, st) in enumerate(zip(sweeps, stopped)):
        c = 0
        if gmin > 0:
            g = 1
            while g < n:
                if g >= gmin:
                    c += 1
                g <<= 1
        if st:
            out[i] = 5 + NCL * n + GAP * (n // K - 1 - c) + GAP_CK * (c + 1) + TR - 2
        else:
            out[i] = 5 + NCL * n + GAP * ((n - 1) // K - c) + GAP_CK * c + TR + 3
    return out


# ------------------------------------------------------------------------------------------------ driver
class PTBoard:
    def __init__(self, mmio, poll_timeout_s: float = 30.0):
        self.mmio = mmio
        self.timeout = poll_timeout_s
        idv = self._rd(ID)
        if idv != ID_VALUE:
            raise RuntimeError(f"pt_board ID {idv:#010x}, expected {ID_VALUE:#010x} (wrong bitstream or address)")
        c0, c1, c2, c3 = self._rd(CFG0), self._rd(CFG1), self._rd(CFG2), self._rd(CFG3)
        self.NS, self.B, self.K, self.KM = c0 >> 24, (c0 >> 16) & 255, (c0 >> 8) & 255, c0 & 255
        self.N, self.M = c1 >> 16, c1 & 0xFFFF
        self.TW, self.EW, self.NCL, self.LW = c2 >> 24, (c2 >> 16) & 255, (c2 >> 8) & 255, c2 & 255
        self.SYN_AW, self.RES_AW, self.WPL, self.USW = c3 >> 24, (c3 >> 16) & 255, (c3 >> 8) & 255, c3 & 255
        self.max_shots = self._rd(CFG4)
        c5 = self._rd(CFG5)
        self.Lmax, self.NLB = c5 & 0xFFFF, c5 >> 16          # lanes per replica; lane bits drawn per move cycle (= Lmax B)
        self.WPS = (self.M + 31) // 32
        self.thr = self.theta = None
        self.seed_value = None
        self.filled = False
        self.nsw, self.gmin, self.th32 = 1024, 0, 29
        self.info = {}

    # ---------------------------------------------------------------- bus helpers
    def _rd(self, off: int) -> int:
        return int(self.mmio.read(off)) & 0xFFFFFFFF

    def _wr(self, off: int, val: int):
        self.mmio.write(off, int(val) & 0xFFFFFFFF)

    def _write_block(self, off: int, words: np.ndarray):
        words = np.ascontiguousarray(words, np.uint32).ravel()
        arr = getattr(self.mmio, "array", None)
        if arr is not None and off % 4 == 0:
            arr[off // 4: off // 4 + len(words)] = words
        else:
            for i, v in enumerate(words):
                self._wr(off + 4 * i, int(v))

    def _read_block(self, off: int, n: int) -> np.ndarray:
        arr = getattr(self.mmio, "array", None)
        if arr is not None and off % 4 == 0:
            return np.array(arr[off // 4: off // 4 + n], np.uint32)
        return np.array([self._rd(off + 4 * i) for i in range(n)], np.uint32)

    def _poll(self, off: int, mask: int, val: int):
        if hasattr(self.mmio, "poll"):                       # mock / recorder
            return self.mmio.poll(off, mask, val)
        t0 = time.time()
        while (self._rd(off) & mask) != val:
            if time.time() - t0 > self.timeout:
                raise TimeoutError(f"register {off:#x}: waiting for {val:#x} under mask {mask:#x}")

    def status(self) -> int:
        return self._rd(STATUS)

    def wait_ready(self):
        self._poll(STATUS, ST_READY, ST_READY)

    # ---------------------------------------------------------------- overlay
    @classmethod
    def from_overlay(cls, bitfile: str, ip: str | None = None, info: str | None = None):
        """Load the overlay (PYNQ) and attach to the decoder IP (default: the first IP whose name starts with 'pt')."""
        from pynq import MMIO, Overlay
        ol = Overlay(bitfile)
        name = ip or next(k for k in ol.ip_dict if k.split("/")[-1].startswith("pt"))
        # MMIO from the address map: a module-reference RTL cell has no driver attribute on the overlay
        d = ol.ip_dict[name]
        brd = cls(MMIO(d["phys_addr"], d["addr_range"]))
        brd.overlay = ol
        ij = info or bitfile.rsplit(".", 1)[0] + ".json"
        try:
            with open(ij) as f:
                brd.load_info(json.load(f))
        except OSError:
            pass
        info = getattr(brd, "info", {})
        if info.get("pl0_board_mhz"):
            # the MMCM and the timing analysis assume PL clock 0 = pl0_board_mhz on the board
            brd.fclk0_mhz = cls.set_fclk0(float(info["pl0_board_mhz"]))
        elif "axi_mhz_actual" in info:
            brd.fclk0_mhz = cls.set_fclk0(float(info["axi_mhz_actual"]))
        else:
            brd.fclk0_mhz = None
        return brd

    @staticmethod
    def set_fclk0(target_mhz: float) -> float:
        """PL clock 0 at the highest value PYNQ can make that is not above the build's value. PYNQ applies the .hwh dividers
        to the board's own PLL (RFSoC 4x2: 1500 MHz IOPLL, so the build's 96.97 MHz came out as 136.4 MHz and the decoder
        MMCM ran 41 % above its timing). Returns the value set; the decoder clock then scales with it (measure_clock)."""
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
        # the decoder MMCM re-locks after a PL clock change and runs fast meanwhile (decoder held in reset): settle first
        time.sleep(1.0)
        return got

    def load_info(self, info: dict):
        """Build information (hardware/build/build_pt_board.tcl writes <name>.json next to the bitstream: pt_info + clocks)."""
        self.info = dict(info)

    # ---------------------------------------------------------------- tables
    def set_tables(self, thr, theta):
        """Write the runtime tables: thr [NS, KM] or [NS, KM + 1] (k = 1..KM; non-increasing in k, < 2^B), theta [NS - 1].
        Only while idle. If the generators were already filled, RECALC recomputes the pre-drawn swap thresholds from the same
        uniforms, so the random streams continue exactly as in the golden model."""
        thr = np.asarray(thr, np.int64)
        if thr.shape == (self.NS, self.KM + 1):
            thr = thr[:, 1:]
        if thr.shape != (self.NS, self.KM):
            raise ValueError(f"thr shape {thr.shape}, expected ({self.NS}, {self.KM})")
        if (thr < 0).any() or (thr >= (1 << self.B)).any() or (np.diff(thr, axis=1) > 0).any():
            raise ValueError("thresholds must be in [0, 2^B) and non-increasing in k")
        theta = np.asarray(theta, np.int64)
        if theta.shape != (self.NS - 1,) or (theta < 0).any() or (theta >= (1 << self.TW)).any():
            raise ValueError(f"theta must have {self.NS - 1} entries in [0, 2^{self.TW})")
        if self.status() & ST_BUSY:
            raise RuntimeError("tables can only be written while idle")
        for t in range(self.NS):
            for k in range(self.KM):
                self._wr(THR_STAGE + 4 * k, int(thr[t, k]))
            self._wr(THR_COMMIT, t)
        for t in range(self.NS - 1):
            self._wr(THETA + 4 * t, int(theta[t]))
        if self.filled:
            self._wr(CTRL, CTRL_RECALC)
            self.wait_ready()
        self.thr = np.concatenate([np.full((self.NS, 1), 1 << self.B, np.int64), thr], 1)
        self.theta = theta.copy()

    def set_noise(self, p: float, betas=None, bmin: float = 0.1, w3: float | None = None):
        """Tables for depolarizing noise rate p and a temperature ladder (default geomspace(1, bmin, NS))."""
        betas = ladder(self.NS, bmin) if betas is None else np.asarray(betas, np.float64)
        if len(betas) != self.NS:
            raise ValueError(f"ladder needs {self.NS} temperatures")
        thr, theta = make_tables(w3_depolarizing(p) if w3 is None else w3, betas, self.B, self.KM, TF, self.TW)
        self.set_tables(thr, theta)
        self.p, self.betas = p, np.asarray(betas)
        return thr, theta

    # ---------------------------------------------------------------- seeds
    def seed(self, seed: int):
        """Seed every lane generator (two GFSR histories per replica) and swap generator (xoshiro per pair), then FILL."""
        if self.status() & ST_BUSY:
            raise RuntimeError("seed only while idle")
        N = self.NLB
        ga, gb, sw = seed_states(self.NS, seed)
        nwl, nws = (self.NS + 31) // 32, (self.NS - 1 + 31) // 32
        for g, push in ((ga, 2), (gb, 4)):
            st = padded_stream(g, N)
            cols = self._columns(st)
            for k in range(st.shape[1]):
                for j in range(nwl):
                    self._wr(SEED_LANE + 4 * j, int(cols[k, j]))
                if k % N == N - 1:
                    self._wr(SEED_CTRL, push)
        swb = np.zeros((max(self.NS - 1, 1), 128), np.uint8)
        for t in range(self.NS - 1):
            for k in range(128):
                swb[t, k] = (int(sw[t, k // 32]) >> (k % 32)) & 1
        cols = self._columns(swb)
        for k in range(128):
            for j in range(nws):
                self._wr(SEED_SWAP + 4 * j, int(cols[k, j]))
        self._wr(CTRL, CTRL_FILL)
        self.wait_ready()
        self.seed_value, self.filled = seed, True

    @staticmethod
    def _columns(bits: np.ndarray) -> np.ndarray:
        """bits [G, L] -> [L, ceil(G / 32)] uint32, word j bit i = generator 32 j + i."""
        G, L = bits.shape
        nw = (G + 31) // 32
        out = np.zeros((L, nw), np.uint64)
        for i in range(G):
            out[:, i // 32] |= bits[i].astype(np.uint64) << np.uint64(i % 32)
        return out.astype(np.uint32)

    # ---------------------------------------------------------------- runs
    def configure(self, nsw: int, gmin: int = 0, th32: int = 29):
        """Budget nsw sweeps (>= 2; the cap when stopping early; any value: the RTL ends at nsw - 1 and its class-count
        blocks double at powers of two, as the golden model); early stop from gmin sweeps (power of two,
        >= K, K a power of two; 0 = off) when 32 x (largest class count) >= th32 x (block total)."""
        if nsw < 2:
            raise ValueError("nsw must be >= 2")
        if gmin:
            if gmin & (gmin - 1) or gmin < self.K or self.K & (self.K - 1):
                raise ValueError("early stop needs gmin a power of two >= K and K a power of two")
            if not 0 < th32 < 64:
                raise ValueError("th32 in 1..63")
        self._wr(NSW, nsw); self._wr(ES_GMIN, gmin); self._wr(ES_TH, th32)
        self.nsw, self.gmin, self.th32 = nsw, gmin, th32

    def pack_syndromes(self, syn: np.ndarray) -> np.ndarray:
        syn = np.asarray(syn, np.uint8)
        if syn.ndim != 2 or syn.shape[1] != self.M:
            raise ValueError(f"syndromes must be [S, {self.M}]")
        S = syn.shape[0]
        stride = 1 << self.WPL
        pad = np.zeros((S, stride * 32), np.uint8)
        pad[:, :self.M] = syn
        w = np.packbits(pad.reshape(S, stride, 32), axis=2, bitorder="little").view("<u4")[..., 0]
        return w.reshape(S * stride).astype(np.uint32)

    def run_batch(self, syn: np.ndarray) -> dict:
        """One batch (<= max_shots, a multiple of 4 shots): write the syndromes, START, wait for DONE, read the records and the batch timers."""
        S = len(syn)
        if not 0 < S <= self.max_shots:
            raise ValueError(f"batch of {S} shots; 1..{self.max_shots}")
        if S % 4:
            raise ValueError(f"the shot count of a batch must be a multiple of 4 (got {S}): the block copy into the syndrome "
                             "memory moves whole 16-byte groups")
        if not self.filled:
            raise RuntimeError("seed() first")
        self._write_block(SYN_BASE, self.pack_syndromes(syn))
        self._wr(NSHOT, S)
        t0 = time.time()
        self._wr(CTRL, CTRL_START)
        self._poll(STATUS, ST_DONE, ST_DONE)
        wall = time.time() - t0
        r = unpack_records(self._read_block(RES_BASE, S * REC_WORDS))
        ts = self._rd(T_START_LO) | (self._rd(T_START_HI) << 32)
        te = self._rd(T_END_LO) | (self._rd(T_END_HI) << 32)
        r["batch_cycles"] = te - ts
        r["wall_s"] = wall
        return r

    def run(self, syn: np.ndarray, batch: int | None = None) -> dict:
        """All shots, in batches of up to max_shots; records concatenated, per-batch cycles and wall time listed."""
        batch = min(batch or self.max_shots, self.max_shots)
        parts = [self.run_batch(syn[i:i + batch]) for i in range(0, len(syn), batch)]
        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if np.ndim(parts[0][k])}
        out["batch_cycles"] = np.array([p["batch_cycles"] for p in parts], np.int64)
        out["wall_s"] = np.array([p["wall_s"] for p in parts])
        return out

    def frc(self) -> int:
        lo = self._rd(FRC_LO)
        return lo | (self._rd(FRC_HI) << 32)

    def measure_clock(self, seconds: float = 1.0) -> float:
        """Decoder clock in MHz from the free-running counter against the host clock."""
        c0, t0 = self.frc(), time.perf_counter()
        time.sleep(seconds)
        c1, t1 = self.frc(), time.perf_counter()
        return (c1 - c0) / (t1 - t0) / 1e6

    def meta(self) -> dict:
        return dict(NS=self.NS, B=self.B, K=self.K, KM=self.KM, N=self.N, M=self.M, TW=self.TW, NCL=self.NCL, Lmax=self.Lmax,
                    seed=self.seed_value, nsw=self.nsw, gmin=self.gmin, th32=self.th32, p=getattr(self, "p", None),
                    thr=self.thr, theta=self.theta, info=json.dumps(self.info))


# ------------------------------------------------------------------------------------------------ output
def save_results(path: str, res: dict, meta: dict, extra: dict | None = None, csv: bool = False):
    """Compressed npz with every per-shot field, the tables and the configuration (about 20 bytes per shot); optionally a .csv
    with one row per shot next to it (several times larger; the board's file system is nearly full, so off by default)."""
    d = {f"r_{k}": np.asarray(v) for k, v in res.items()}
    d.update({f"m_{k}": np.asarray(v if v is not None else -1) for k, v in meta.items()})
    d.update({k: np.asarray(v) for k, v in (extra or {}).items()})
    np.savez_compressed(path, **d)
    if not csv:
        return
    cols = ["ml", "me", "stopped", "emin", "sweeps", "cyc_start", "cyc_run", "t_done"]
    with open(path.rsplit(".", 1)[0] + ".csv", "w") as f:
        f.write("shot," + ",".join(cols) + ",cnt0,cnt1,cnt2,cnt3\n")
        for i in range(len(res["ml"])):
            f.write(f"{i}," + ",".join(str(int(res[c][i])) for c in cols) + ","
                    + ",".join(str(int(x)) for x in res["counts"][i]) + "\n")


def power_sample(seconds: float = 2.0, interval: float = 0.05, samples: bool = False) -> dict:
    """Mean power per PMBus rail (PYNQ get_rails) over a window; empty if the board exposes no rails.
    samples: also return the number of readings per rail under "n"."""
    try:
        import pynq
        rails = pynq.get_rails()
    except Exception:
        return {}
    acc = {k: [] for k, r in rails.items() if getattr(r, "power", None) is not None}
    t0 = time.time()
    while time.time() - t0 < seconds:
        for k in acc:
            acc[k].append(rails[k].power.value)
        time.sleep(interval)
    out = {k: float(np.mean(v)) for k, v in acc.items() if v}
    if samples:
        out["n"] = min((len(v) for v in acc.values()), default=0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Decode one-syndrome-per-shot batches on the board")
    ap.add_argument("--bit", required=True); ap.add_argument("--shots", required=True, help="npz with syn [S, M] (+ fail_cls)")
    ap.add_argument("--p", type=float, required=True); ap.add_argument("--bmin", type=float, default=0.1)
    ap.add_argument("--nsw", type=int, default=1024); ap.add_argument("--gmin", type=int, default=0)
    ap.add_argument("--th32", type=int, default=29); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pl0", type=float, default=0, help="PL clock 0 in MHz (default: the build's value); lower it to run a "
                    "bitstream that missed timing below its routed Fmax")
    ap.add_argument("--max", type=int, default=0, help="use only the first MAX shots")
    ap.add_argument("--power", action="store_true", help="mean PMBus power idle and during repeated batches (mixes busy and host-gap readings; pt_power.py gives the decoding plateau)")
    ap.add_argument("--csv", action="store_true", help="also write a per-shot .csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    z = np.load(a.shots)
    syn = z["syn"][: a.max or None]
    brd = PTBoard.from_overlay(a.bit)
    if a.pl0:
        brd.fclk0_mhz = PTBoard.set_fclk0(a.pl0)
    # wait until two consecutive 0.5 s readings agree within 0.1 % (the MMCM settles after the PL clock change)
    prev = brd.measure_clock(0.5)
    for _ in range(20):
        mhz = brd.measure_clock(0.5)
        if abs(mhz - prev) <= 1e-3 * mhz:
            break
        prev = mhz
    else:
        raise SystemExit(f"decoder clock did not settle (last readings {prev:.1f}, {mhz:.1f} MHz)")
    fmax = float(brd.info.get("dec_fmax_mhz", 0) or 0) if getattr(brd, "info", None) else 0
    if fmax and mhz > fmax:
        raise SystemExit(f"decoder clock {mhz:.1f} MHz is above the routed Fmax {fmax:.1f} MHz: results would not be valid")
    brd.set_noise(a.p, bmin=a.bmin)
    brd.seed(a.seed)
    brd.configure(a.nsw, a.gmin, a.th32)
    res = brd.run(syn)
    extra = dict(clock_mhz=mhz, syn=syn)
    if "fail_cls" in z:
        fc = z["fail_cls"][: len(syn)]
        extra["fail_ml"] = fc[np.arange(len(syn)), res["ml"]]
        extra["fail_me"] = fc[np.arange(len(syn)), res["me"]]
    if a.power:
        extra["power_idle"] = json.dumps(power_sample(4.0, samples=True))
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))     # sampler in its own process: the MMIO loop holds the GIL
        sampler = subprocess.Popen([sys.executable, "-c", "import json, pt_board; "
                                    "print(json.dumps(pt_board.power_sample(4.0, samples=True)))"],
                                   cwd=here, stdout=subprocess.PIPE, text=True)
        busy, t0 = [0, 0.0], time.time()
        while sampler.poll() is None:
            busy[0] += int(brd.run_batch(syn[: brd.max_shots])["batch_cycles"])
        busy[1] = time.time() - t0
        extra["power_run"] = sampler.stdout.read().strip()
        extra["power_duty"] = busy[0] / (mhz * 1e6) / busy[1]          # decoder-busy fraction during power_run
    save_results(a.out, res, brd.meta(), extra, csv=a.csv)
    lat = (res["cyc_start"] + res["cyc_run"] + 2) / mhz
    print(f"{len(syn)} shots, clock {mhz:.1f} MHz, latency mean {lat.mean():.2f} us, p50 {np.median(lat):.2f}, "
          f"p99 {np.quantile(lat, 0.99):.2f}; throughput {len(syn) / (res['batch_cycles'].sum() / mhz * 1e-6):.0f} shots/s"
          + (f"; LER(ML) {extra['fail_ml'].mean():.4f}" if "fail_ml" in extra else ""))


if __name__ == "__main__":
    main()

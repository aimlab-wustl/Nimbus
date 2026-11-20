"""Register-level model of the one-syndrome-per-shot board design (hardware/rtl/pt_board.sv) with the bit-exact golden model as the decoder.

Host side only (imports nimbus). Used to test the PYNQ driver (hardware/host/pt_board.py) without hardware, and to produce the
expected bus traffic for the xsim board testbench (hardware/sim/gen_board_case.py, RecordingMMIO). It models what the RTL does with
every register write: the seed shift registers and pushes (generator histories are rebuilt from the written bits, not taken from
the driver), the threshold staging row and commit, theta, FILL / RECALC, the syndrome and result memories, and the batch.
Protocol errors that the RTL would turn into silent mismatches raise ProtocolError here (for example a theta change after FILL
without RECALC, or a table write while busy).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nimbus.decoding.one_shot import (GFSR_A, GFSR_B, TW_THETA, PTConfig, build_spec, c0_from_syndrome, run_shot,  # noqa: E402
                                  with_tables)
import pt_board as D  # noqa: E402


class ProtocolError(RuntimeError):
    pass


class PTBoardMock:
    def __init__(self, d: int, NS: int, K: int, B: int, p_gen: float = 0.12, bmin: float = 0.1, SYN_AW: int = 14,
                 RES_AW: int = 15):
        self.spec = build_spec(d, p_gen, PTConfig(NS=NS, bmin=bmin, K=K, B=B))
        s = self.spec
        self.NS, self.NP, self.K, self.B, self.KM = NS, max(NS - 1, 1), K, B, s.kmax
        self.M = s.chains[0].shape[0] + s.chains[1].shape[0]
        self.Lmax = s.cls_mat.shape[1]
        self.NLB = self.Lmax * B
        self.NCL = s.cls_mat.shape[0]
        self.EW = max(1, int(s.n).bit_length())
        self.LW = max(1, int(NS - 1).bit_length())
        self.WPS = (self.M + 31) // 32
        self.WPL = 0 if self.WPS <= 1 else int(self.WPS - 1).bit_length()
        self.SYN_AW, self.RES_AW = SYN_AW, RES_AW
        self.max_shots = min(1 << (SYN_AW - self.WPL), 1 << (RES_AW - 3))
        self.thr = s.thr.copy()                                   # power-on tables (generation p)
        self.theta = s.theta_sw.copy()
        self.stage = np.zeros(self.KM, np.int64)
        self.sd = np.zeros((NS, self.NLB), np.uint8)
        self.ha = np.zeros((NS, GFSR_A[0]), np.uint8)
        self.hb = np.zeros((NS, GFSR_B[0]), np.uint8)
        self.ss = np.zeros((self.NP, 128), np.uint8)
        self.lane_w = np.zeros((NS + 31) // 32, np.uint64)
        self.swap_w = np.zeros((self.NP + 31) // 32, np.uint64)
        self.filled = False
        self.theta_dirty = False
        self.reg = dict(nsw=1024, gmin=0, th=29, nshot=0, irq_en=0)
        self.bdone = False
        self.completed = 0
        self.frc = 0
        self.frc_hi = 0
        self.t_start = self.t_end = 0
        self.smem = np.zeros(1 << SYN_AW, np.uint32)
        self.rmem = np.zeros(1 << RES_AW, np.uint32)
        self.ga = self.gb = self.swst = self.heads = None

    # ---------------------------------------------------------------- bus
    def write(self, off: int, val: int):
        val = int(val) & 0xFFFFFFFF
        self.frc += 7
        if off >= D.RES_BASE:
            return
        if off >= D.SYN_BASE:
            self.smem[(off - D.SYN_BASE) // 4 & ((1 << self.SYN_AW) - 1)] = val
            return
        r = (off & 0xFFF) // 4
        if 0x100 <= r < 0x200:                                    # THETA[t]
            t = r - 0x100
            if t < self.NS - 1:
                self.theta[t] = val & ((1 << TW_THETA) - 1)
                if self.filled:
                    self.theta_dirty = True
        elif 0x80 <= r < 0xC0:                                    # THR_STAGE[k]
            if r - 0x80 < self.KM:
                self.stage[r - 0x80] = val & ((1 << self.B) - 1)
        elif 0x20 <= r < 0x28:                                    # SEED_LANE[j]
            j = r - 0x20
            if j < len(self.lane_w):
                self.lane_w[j] = val
                if j == len(self.lane_w) - 1:
                    bits = np.array([(int(self.lane_w[i // 32]) >> (i % 32)) & 1 for i in range(self.NS)], np.uint8)
                    self.sd = np.concatenate([self.sd[:, 1:], bits[:, None]], 1)
        elif 0x28 <= r < 0x30:                                    # SEED_SWAP[j]
            j = r - 0x28
            if j < len(self.swap_w):
                self.swap_w[j] = val
                if j == len(self.swap_w) - 1:
                    bits = np.array([(int(self.swap_w[i // 32]) >> (i % 32)) & 1 for i in range(self.NP)], np.uint8)
                    self.ss = np.concatenate([self.ss[:, 1:], bits[:, None]], 1)
        elif r == 0:                                              # CTRL
            self.reg["irq_en"] = (val >> 8) & 1
            if val & D.CTRL_FILL:
                self._fill()
            if val & D.CTRL_RECALC:
                if not self.filled:
                    raise ProtocolError("RECALC before FILL")
                self.theta_dirty = False
            if val & D.CTRL_START:
                self._batch()
        elif r == 2:
            self.reg["nsw"] = val
        elif r == 3:
            self.reg["gmin"] = val
        elif r == 4:
            self.reg["th"] = val
        elif r == 5:
            self.reg["nshot"] = val
        elif r == 7:                                              # SEED_CTRL: push A / B
            if val & 2:
                self.ha = np.concatenate([self.ha[:, self.NLB:], self.sd], 1)
            if val & 4:
                self.hb = np.concatenate([self.hb[:, self.NLB:], self.sd], 1)
        elif r == 8:                                              # THR_COMMIT
            t = val & ((1 << self.LW) - 1)
            if t < self.NS:
                self.thr[t, 1:] = self.stage

    def read(self, off: int) -> int:
        self.frc += 7
        if off >= D.RES_BASE:
            return int(self.rmem[(off - D.RES_BASE) // 4 & ((1 << self.RES_AW) - 1)])
        if off >= D.SYN_BASE:
            return int(self.smem[(off - D.SYN_BASE) // 4 & ((1 << self.SYN_AW) - 1)])
        r = (off & 0xFFF) // 4
        if 0x80 <= r < 0xC0:
            return int(self.stage[r - 0x80]) if r - 0x80 < self.KM else 0
        val = {0: self.reg["irq_en"] << 8,
               1: (D.ST_READY | D.ST_CORE_READY | (D.ST_DONE if self.bdone else 0)),
               2: self.reg["nsw"], 3: self.reg["gmin"], 4: self.reg["th"], 5: self.reg["nshot"], 6: self.completed,
               9: D.ID_VALUE,
               0xA: (self.NS << 24) | (self.B << 16) | (self.K << 8) | self.KM,
               0xB: (self.spec.n << 16) | self.M,
               0xC: (TW_THETA << 24) | (self.EW << 16) | (self.NCL << 8) | self.LW,
               0xD: (self.SYN_AW << 24) | (self.RES_AW << 16) | (self.WPL << 8) | self.spec.cfg.USW,
               0xE: self.max_shots, 0xF: (self.NLB << 16) | self.Lmax,
               0x11: self.frc_hi, 0x12: self.t_start & 0xFFFFFFFF, 0x13: self.t_start >> 32,
               0x14: self.t_end & 0xFFFFFFFF, 0x15: self.t_end >> 32}
        if r == 0x10:
            self.frc_hi = self.frc >> 32
            return self.frc & 0xFFFFFFFF
        return int(val.get(r, 0)) & 0xFFFFFFFF

    def poll(self, off: int, mask: int, val: int):
        if (self.read(off) & mask) != val:
            raise ProtocolError(f"poll of {off:#x} would not finish (mask {mask:#x}, want {val:#x})")

    # ---------------------------------------------------------------- behavior
    def _fill(self):
        self.ga, self.gb = self.ha.copy(), self.hb.copy()
        self.heads = np.zeros((self.NS, 2), np.int64)
        self.swst = np.zeros((max(self.NS - 1, 0), 4), np.uint32)
        for t in range(self.NS - 1):
            for w in range(4):
                self.swst[t, w] = int(sum(int(self.ss[t, 32 * w + b]) << b for b in range(32)))
        self.filled, self.theta_dirty = True, False

    def tables_spec(self):
        return with_tables(self.spec, self.thr, self.theta)

    def _batch(self):
        if not self.filled:
            raise ProtocolError("START before seeding / FILL")
        if self.theta_dirty:
            raise ProtocolError("theta changed after FILL without RECALC")
        sp = self.tables_spec()
        nsw, gmin, th = self.reg["nsw"], self.reg["gmin"], self.reg["th"]
        S = self.reg["nshot"]
        if S > self.max_shots:
            raise ProtocolError("NSHOT larger than the memories")
        self.t_start = self.frc
        stride = 1 << self.WPL
        recs = dict(ml=[], me=[], stopped=[], emin=[], sweeps=[], cyc_start=[], cyc_run=[], counts=[], t_done=[])
        t = self.frc
        for s in range(S):
            w = self.smem[s * stride: s * stride + self.WPS].astype(np.uint32)
            bits = np.unpackbits(w.view(np.uint8), bitorder="little")[: self.M]
            c0 = c0_from_syndrome(self.spec, bits[None, :])[0]
            r = run_shot(sp, c0, nsw, self.ga, self.gb, self.heads, self.swst, gmin, th if gmin else 0)
            stop = int(r["stop"])
            kb = stop.bit_length() - 1
            cnt = r["counts"][max(1, stop - 1).bit_length()]
            stopped = stop < nsw
            tot = int(D.cycle_model([stop], [stopped], nsw, self.K, self.NS, gmin, self.NCL)[0])
            t += tot + 2
            for k, v in (("ml", int(np.argmax(cnt))), ("me", int(r["bestC"][np.argmin(r["bestE"])])), ("stopped", stopped),
                         ("emin", int(r["bestE"].min())), ("sweeps", stop), ("cyc_start", 3), ("cyc_run", tot - 5),
                         ("counts", [int(x) for x in cnt]), ("t_done", t & 0xFFFFFFFF)):
                recs[k].append(v)
        if S:
            words = D.pack_records({k: np.array(v) for k, v in recs.items()})
            self.rmem[: S * D.REC_WORDS] = words.ravel()
        self.completed = S
        self.frc = t + 10
        self.t_end = self.frc
        self.bdone = True


class RecordingMMIO:
    """Forwards to an inner MMIO (the mock) and records every access as (op, offset, data, mask): op 0 write, 1 read
    (compare under mask), 2 poll (repeat until (read & mask) == data). The xsim board testbench replays the list."""

    def __init__(self, inner):
        self.inner, self.ops = inner, []

    @staticmethod
    def mask_of(off: int) -> int:
        if off >= D.RES_BASE:
            return 0 if ((off - D.RES_BASE) // 4) % D.REC_WORDS == 7 else 0xFFFFFFFF      # t_done: free-running counter
        if off >= D.SYN_BASE:
            return 0xFFFFFFFF
        if off in (D.FRC_LO, D.FRC_HI, D.T_START_LO, D.T_START_HI, D.T_END_LO, D.T_END_HI):
            return 0
        if off == D.STATUS:
            return D.ST_BUSY | D.ST_DONE
        return 0xFFFFFFFF

    def write(self, off, val):
        self.inner.write(off, val)
        self.ops.append((0, off, int(val) & 0xFFFFFFFF, 0))

    def read(self, off):
        v = self.inner.read(off)
        self.ops.append((1, off, v, self.mask_of(off)))
        return v

    def poll(self, off, mask, val):
        self.inner.poll(off, mask, val)
        self.ops.append((2, off, val, mask))

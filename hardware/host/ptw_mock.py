"""Register-level model of the real-time board design (hardware/rtl/ptw_board.sv) with the bit-exact golden model as the decoder.

Host side only (imports nimbus). Tests the PYNQ driver (hardware/host/ptw_board.py) without hardware and produces the expected bus
traffic for the xsim board testbench (hardware/sim/gen_ptw_board_case.py, RecordingMMIO). It models what the RTL does with every
register write: the engine / bank seed shift registers and the per-replica pushes (generator histories are rebuilt from the
written bits, not taken from the driver), the threshold staging row and commit, theta, Wq / wq / mode, FILL / RECALC, the shot
and result memories, and the batch: decisions from nimbus.decoding.real_time.decode_batch on the generator stream carried
across batches, core cycles from the cycle model, the round timer's trigger / take times when the decoder keeps up.
Protocol errors that the RTL would turn into silent mismatches raise ProtocolError (e.g. theta changed after FILL without
RECALC, a batch before FILL, a timed batch whose deadlines are too tight for the mock to predict the timing words).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nimbus.decoding import real_time as g  # noqa: E402
import ptw_board as D  # noqa: E402


class ProtocolError(RuntimeError):
    pass


class PTWBoardMock:
    def __init__(self, info: dict, p_gen: float | None = None, SYN_AW: int = 17, RES_AW: int = 16):
        """info: the generator's ptw_info.json (hardware/gen/gen_ptw.py) of the core inside the bitstream."""
        self.info = info
        self.d, self.R, self.K, self.B = int(info["d"]), int(info["R"]), int(info["K"]), int(info["B"])
        self.p_gen = float(info["p"]) if p_gen is None else p_gen
        self.cfg = g.PTWConfig(R=self.R, K=self.K, B=self.B, bmin=float(info["bmin"]), k=int(info["k"]),
                               lram=bool(info["lram"]), carry=True)
        self.TR, self.CM = int(info["T"]), int(info["C"])
        ws = g.window_spec(self.d, self.TR, False, self.p_gen, self.cfg)
        self.m = ws.m
        self.M2 = 2 * self.m
        self.KSL, self.ROWN = ws.KSL, int(info["ROWN"])
        self.RS, self.NB, self.KS, self.NC = int(info["RS"]), int(info["NB"]), int(info["k"]), int(info["NC"])
        self.NE = self.R // self.RS
        self.RB = int(info["RB"])
        self.bank_lanes = [int(x) for x in info["bank_lanes"]]
        self.bank_N = [bl * self.B for bl in self.bank_lanes]
        self.LW, self.NP, self.EW = int(info["LW"]), int(info["NP"]), int(info["EW"])
        self.LDC, self.GAP_SW, self.CSL = int(info["LDC"]), int(info["GAP_SW"]), int(info["cs"])
        self.WPR = (self.M2 + 31) // 32
        self.WPL = 0 if self.WPR <= 1 else int(self.WPR - 1).bit_length()
        self.NR = 2 * self.TR + 1
        self.RD = 3 * self.CM + 1
        self.SYN_AW, self.RES_AW = SYN_AW, RES_AW
        self.max_shots = min(((1 << SYN_AW) >> self.WPL) // self.NR, (1 << RES_AW) // (D.REC_WORDS * D.NWIN))
        self.thr = ws.thr[:, 1:].copy()                            # power-on tables (generation p)
        self.theta = ws.theta.copy()
        self.Wq, self.wq, self.mode = ws.Wq, ws.wq, 0
        self.stage = np.zeros(self.ROWN, np.int64)
        self.y = [[[] for _ in range(self.NB)] for _ in range(self.NE)]       # engine / bank seed shift registers
        self.ha = [[[] for _ in range(self.NB)] for _ in range(self.R)]       # replica / bank histories
        self.hb = [[[] for _ in range(self.NB)] for _ in range(self.R)]
        self.ss = [[] for _ in range(self.NP)]
        self.filled = False
        self.theta_dirty = False
        self.state = None
        self.reg = dict(nsw=4096, tround=0, nshot=0, irq_en=0, psel=0)
        self.bdone = False
        self.cnt = dict(misses=0, late=0, released=0, ndone=0)
        self.frc = 0
        self.frc_hi = 0
        self.t_start = self.t_end = 0
        self.smem = np.zeros(1 << SYN_AW, np.uint32)
        self.rmem = np.zeros(1 << RES_AW, np.uint32)
        self.take_known = np.zeros(1 << RES_AW, bool)            # per record: take / lag words predictable (timed mode)
        self.allow_tight = False                                  # tight T_ROUND: decisions only (timing words, counters masked)
        self.counters_known = True

    # ---------------------------------------------------------------- bus
    def write(self, off: int, val: int):
        val = int(val) & 0xFFFFFFFF
        self.frc += 7
        if off >= D.SYN_BASE:
            self.smem[(off - D.SYN_BASE) // 4 & ((1 << self.SYN_AW) - 1)] = val
            return
        if off >= D.RES_BASE:
            return
        r = (off & 0xFFF) // 4
        if 0x100 <= r < 0x200:                                    # THETA[t]
            t = (r - 0x100) & ((1 << self.LW) - 1)
            if t < self.R - 1:
                self.theta[t] = val & ((1 << D.TW) - 1)
                if self.filled:
                    self.theta_dirty = True
        elif 0x80 <= r < 0x100:                                   # THR_STAGE[k]
            if r - 0x80 < self.ROWN:
                self.stage[r - 0x80] = val & ((1 << self.B) - 1)
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
            self.reg["tround"] = val
        elif r == 4:
            self.reg["nshot"] = val
        elif r == 5:
            self.Wq = val & ((1 << D.QEW) - 1)
        elif r == 6:
            self.wq = val & ((1 << D.QEW) - 1)
        elif r == 7:                                              # SEED_CTRL: push A / B per bank, psel
            pa, pb, ps = val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & ((1 << self.RB) - 1)
            for b in range(self.NB):
                for mask, h, P in ((pa, self.ha, D.GFSR_A[0]), (pb, self.hb, D.GFSR_B[0])):
                    if (mask >> b) & 1:
                        for e in range(self.NE):
                            i = e * self.RS + ps
                            if ps < self.RS:
                                h[i][b] = (h[i][b] + self.y[e][b])[-P:]
        elif r == 8:                                              # THR_COMMIT
            t = val & ((1 << self.LW) - 1)
            if t < self.R:
                self.thr[t] = self.stage
        elif r == 0x1D:
            self.mode = val & 1
        elif r == 0x20:                                           # SEED_LANE: shift
            for e in range(self.NE):
                for b in range(self.NB):
                    self.y[e][b] = (self.y[e][b] + [(val >> (e * self.NB + b)) & 1])[-self.bank_N[b]:]
        elif r == 0x21:                                           # SEED_SWAP: shift
            for t in range(self.NP):
                self.ss[t] = (self.ss[t] + [(val >> t) & 1])[-128:]

    def _cfg(self) -> dict:
        b4 = lambda a, b, c, d_: (a << 24) | (b << 16) | (c << 8) | d_  # noqa: E731
        return {0x09: D.ID_VALUE, 0x0A: b4(self.R, self.B, self.K, self.ROWN), 0x0B: b4(self.CM, self.TR, self.M2, self.NC),
                0x0C: b4(D.TW, self.EW, self.LW, D.QEW), 0x0D: b4(self.SYN_AW, self.RES_AW, self.WPL, D.USW),
                0x0E: self.max_shots, 0x0F: b4(self.NE, self.RS, self.NB, self.KS),
                0x1A: b4(self.CSL, self.GAP_SW, self.LDC, self.KSL),
                0x1B: (self.bank_lanes[1] if self.NB > 1 else 0) << 16 | self.bank_lanes[0],
                0x1C: b4(self.RD, self.NP, self.RB, self.WPR)}

    def read(self, off: int) -> int:
        self.frc += 7
        if off >= D.SYN_BASE:
            return int(self.smem[(off - D.SYN_BASE) // 4 & ((1 << self.SYN_AW) - 1)])
        if off >= D.RES_BASE:
            return int(self.rmem[(off - D.RES_BASE) // 4 & ((1 << self.RES_AW) - 1)])
        r = (off & 0xFFF) // 4
        if 0x80 <= r < 0x100:
            return int(self.stage[r - 0x80]) if r - 0x80 < self.ROWN else 0
        if r == 0x10:
            self.frc_hi = self.frc >> 32
            return self.frc & 0xFFFFFFFF
        val = {0: self.reg["irq_en"] << 8, 1: (D.ST_READY | D.ST_CORE_READY | (D.ST_DONE if self.bdone else 0)),
               2: self.reg["nsw"], 3: self.reg["tround"], 4: self.reg["nshot"], 5: self.Wq, 6: self.wq,
               0x11: self.frc_hi, 0x12: self.t_start & 0xFFFFFFFF, 0x13: self.t_start >> 32,
               0x14: self.t_end & 0xFFFFFFFF, 0x15: self.t_end >> 32, 0x16: self.cnt["misses"], 0x17: self.cnt["late"],
               0x18: self.cnt["released"], 0x19: self.cnt["ndone"], 0x1D: self.mode}
        val.update(self._cfg())
        return int(val.get(r, 0)) & 0xFFFFFFFF

    def poll(self, off: int, mask: int, val: int):
        if (self.read(off) & mask) != val:
            raise ProtocolError(f"poll of {off:#x} would not finish (mask {mask:#x}, want {val:#x})")

    # ---------------------------------------------------------------- behavior
    def _fill(self):
        R, NB = self.R, self.NB
        st = g.gen_state(1, R, NB)
        for i in range(R):
            for b in range(NB):
                for h, P, key, j in ((self.ha, D.GFSR_A[0], "ra", 0), (self.hb, D.GFSR_B[0], "rb", 2)):
                    bits = np.zeros(P, np.uint8)
                    hh = np.array(h[i][b], np.uint8)
                    if len(hh):
                        bits[P - len(hh):] = hh
                    ring, P0 = g.ring_from_bits(bits)
                    st[key][0, i, b] = ring
                    st["gp"][0, i, b, j] = st["gp"][0, i, b, j + 1] = P0
        for t in range(R - 1):
            bits = np.zeros(128, np.uint8)
            if self.ss[t]:
                bits[128 - len(self.ss[t]):] = self.ss[t]
            for w in range(4):
                st["sw"][0, t, w] = np.uint32(sum(int(bits[32 * w + b]) << b for b in range(32)))
        st["fresh"] = False
        self.state = st
        self.filled, self.theta_dirty = True, False

    def tables(self):
        return (self.thr.copy(), self.theta.copy(), self.Wq, self.wq, self.mode)

    def cycles(self, nsw: int) -> int:
        return D.cycle_model(nsw, self.RS, self.KS, self.NC, self.K, self.GAP_SW, self.LDC, self.R, self.CSL)

    def unpack_rounds(self, S: int) -> np.ndarray:
        stride = 1 << self.WPL
        w = self.smem[: S * self.NR * stride].reshape(S * self.NR, stride)[:, : self.WPR].astype(np.uint32)
        bits = np.unpackbits(np.ascontiguousarray(w).view(np.uint8), axis=1, bitorder="little")[:, : self.M2]
        return bits.reshape(S, self.NR, self.M2)

    def _batch(self):
        if not self.filled:
            raise ProtocolError("START before seeding / FILL")
        if self.theta_dirty:
            raise ProtocolError("theta changed after FILL without RECALC")
        nsw, T, S = self.reg["nsw"], self.reg["tround"], self.reg["nshot"]
        if S > self.max_shots:
            raise ProtocolError("NSHOT larger than the memories")
        if nsw < 2 or nsw & (nsw - 1):
            raise ProtocolError("the golden model needs a power-of-two budget")
        syn = self.unpack_rounds(S)
        recs, self.state = g.decode_batch(self.d, self.p_gen, syn[..., :self.m], syn[..., self.m:], nsw, self.cfg,
                                          state=self.state, tables=self.tables())
        cyc = self.cycles(nsw)
        self.t_start = self.frc
        # timing (timed mode, decoder keeping up): round k due at t_start + (k + 1) T + 1, written WPR + 5 cycles later; a
        # window is taken need + 3 cycles after its trigger and commits cyc + 1 cycles after the take
        trig_r = [self.TR - 1, self.TR + self.CM - 1, self.NR - 1]
        out = {k: [] for k in D.REC_FIELDS}
        t_last_done = None
        known = T > 0
        for j, rc in enumerate(recs):
            s, w = rc["shot"], rc["win"]
            need = self.TR + 1 if w == 2 else self.TR
            k = s * self.NR + trig_r[w]
            if T > 0:
                trig = self.t_start + (k + 1) * T + 1 + self.WPR + 5
                if t_last_done is not None and t_last_done + 24 > trig:
                    known = False
                take = trig + need + 3
            else:
                trig = take = self.t_start
            t_last_done = take + cyc + 1
            for f, v in (("shot", s), ("win", w), ("fin", rc["fin"]), ("fr_ml", rc["fr_ml"]), ("frame", rc["frame"]),
                         ("dec_ml", rc["dec_ml"]), ("best_c", rc["best_c"]), ("best_e", rc["best_e"]),
                         ("counts", rc["counts"]), ("sigma", D.sigma_words(rc["sigma"])), ("t_trig", trig),
                         ("d_take", take - trig), ("d_done", cyc + 1), ("cyc_win", cyc), ("lag", 0)):
                out[f].append(v)
        if T > 0 and not known and not self.allow_tight:
            raise ProtocolError(f"T_ROUND = {T} too tight for the mock's timing prediction (decode {cyc} cycles per window)")
        self.counters_known = known
        n = len(recs)
        if n:
            self.rmem[: n * D.REC_WORDS] = D.pack_records({k: np.array(v) for k, v in out.items()}).ravel()
            self.take_known[:n] = T > 0 and known
        self.cnt = dict(misses=0, late=0, released=S * self.NR, ndone=n)
        end = (t_last_done if t_last_done is not None else self.t_start) + 20
        self.frc = max(self.frc, end) + 10
        self.t_end = end
        self.bdone = True


class RecordingMMIO:
    """Forwards to an inner MMIO (the mock) and records every access as (op, offset, data, mask): op 0 write, 1 read
    (compare under mask), 2 poll (repeat until (read & mask) == data). The xsim board testbench replays the list."""

    def __init__(self, inner):
        self.inner, self.ops = inner, []

    def mask_of(self, off: int) -> int:
        if off >= D.SYN_BASE:
            return 0xFFFFFFFF
        if off >= D.RES_BASE:
            j = (off - D.RES_BASE) // 4
            wi = j % D.REC_WORDS
            if wi in (10, 11):                                    # trigger time: free-running counter
                return 0
            if wi in (12, 15) and not self.inner.take_known[j // D.REC_WORDS]:
                return 0
            return 0xFFFFFFFF
        if off in (D.FRC_LO, D.FRC_HI, D.T_START_LO, D.T_START_HI, D.T_END_LO, D.T_END_HI):
            return 0
        if off == D.STATUS:
            return D.ST_BUSY | D.ST_DONE
        if off in (D.MISSES, D.LATE) and not self.inner.counters_known:
            return 0
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

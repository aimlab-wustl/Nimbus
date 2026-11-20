"""RTL generator of the real-time core: parallel tempering in sliding windows on the rotated surface code (phenomenological
depolarizing noise). Bit-exact with nimbus.decoding.real_time run with the hardware schedule
(PTWConfig(k, lram) -> hw_schedule: slab-major sweep, one coloring of the physical neurons for both window kinds).

Architecture (binarized clauses): the clause units hold the literals and are the only state; each
neuron's (a, b) is summed every cycle from its clauses' outputs; an accepted neuron flips its clauses' literals directly;
a color of a slab applies in one cycle. Layouts (only where a clause unit keeps its literals changes):
* k = 1 (parallel) and k = 2..4 (FF copies): one FF clause unit per logical clause (the one-shot core's pt_clause for data clauses); the
  physical neuron of a move position reads its slab's copy through a k:1 select (slab one-hot) and flips that copy.
* --lram (k >= 5): one LUTRAM clause unit per (site, row); word a = slice row + a S, address = slab (+ 1 for the colors whose
  neuron reaches it from the slab below). Commit taps: FF mirrors of the observable-line and slice C - 1 literals.

Writes <out>/ptw_core.sv (ptw_rep: one replica; ptw_prep: start state, start counts and summary; ptw_summ: commit summary),
<out>/ptw_params.svh and <out>/ptw_info.json.

  python hardware/gen/gen_ptw.py --d 3 --R 4 --k 1 --out out/ptw_d3
"""
from __future__ import annotations

import argparse
import re
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from nimbus.decoding.real_time import PTWConfig, hw_schedule, window_spec  # noqa: E402
from nimbus.decoding.one_shot import GFSR_A, GFSR_B  # noqa: E402

def gap_sw(cs: bool, swp: bool = False) -> int:
    """Idle item cycles after a swap sweep: energy 3 (lane, groups, nd) + product 1 + decision 1 + label row 1 + registered
    comparator row 1 = 7; the C-slow loop adds its accept register stage (8); --swp (registered swap decision) one more."""
    return (8 if cs else 7) + (1 if swp else 0)


def clog2(n: int) -> int:
    return max(1, int(max(n - 1, 0)).bit_length())


def build(d, R, k, lram, p, C=None, Bw=None, K=1, B=16, bmin=0.1, rs=1, srl=False, cs=False, swp=False, fo=True):
    C = d if C is None else C
    Bw = d if Bw is None else Bw
    T = C + Bw
    if rs > 1 and not lram:
        raise ValueError("replica-shared engines (rs > 1) need LUTRAM clause units")
    if R % rs:
        raise ValueError("R must be a multiple of rs")
    cfg = PTWConfig(R=R, K=K, B=B, bmin=bmin, k=k, lram=lram, carry=True)
    hs = hw_schedule(d, T, k, lram)
    wsn, wsf = window_spec(d, T, False, p, cfg), window_spec(d, T, True, p, cfg)
    if cs and not (lram and rs >= 3):
        raise ValueError("the C-slow loop needs LUTRAM clause units and rs >= 3 replicas per engine")
    return dict(d=d, R=R, k=k, lram=lram, p=p, C=C, T=T, K=K, B=B, cfg=cfg, hs=hs, ws=(wsn, wsf), rs=rs, srl=srl, cs=cs, swp=swp, fo=fo)


# ===================================================================================================== replica
def gen_rep(G):
    hs, (wsn, wsf) = G["hs"], G["ws"]
    d, T, k, S, lram, C, B = G["d"], G["T"], hs["k"], hs["S"], hs["lram"], G["C"], G["B"]
    n, m = wsf.n, wsf.m
    KSL = wsf.KSL
    NC, Lmax = hs["NC"], hs["Lmax"]
    phys = hs["phys"]
    mvs = hs["moves"]
    LB = G["cfg"].LB
    NB = -(-Lmax // LB)
    bank_lanes = [min(LB, Lmax - g * LB) for g in range(NB)]
    pa, qa = GFSR_A
    pb, qb = GFSR_B
    obs = [sorted(int(q) for q in wsf.obs[0]), sorted(int(q) for q in wsf.obs[1])]
    DAW = clog2(KSL + 1) + 1                       # signed accepted data change of one neuron
    NDC = n * (T + 1)                              # data clauses (logical; slice T only in the final window)
    NMC = 2 * m * T
    AW = clog2(k + 1)
    o = []
    w = o.append

    # ---------------------------------------------------------------- term sources per physical neuron
    def tau_of(P, j):
        return j * S + P["row"]

    def lit_set(f, r):
        M_ = mvs[f]
        return {M_["lname"](li) for li, _ in M_["terms"][r]}


    # presence of a term per slab: j -> (cond, literal name); cond "1" (every window kind in which the instance exists),
    # "fin" / "~fin" (one kind only). Slabs without an instance are don't-cares (the neuron is disabled there).
    def term_gate(p, P, tm):
        out = {}
        for j in range(k):
            ex = [f for f in (0, 1) if (p, j) in hs["inst"][f]]
            if not ex:
                continue
            tj = tau_of(P, j)
            if tm[0] == "d":
                nm = ("d", tm[1], tj + tm[2], P["sec"])
            else:
                nm = ("m", tm[1], tm[2], tj)
            pres = [f for f in ex if nm in lit_set(f, hs["inst"][f][(p, j)])]
            if not pres:
                out[j] = ("0", None)
                continue
            if tm[0] == "m":
                assert len(pres) == len(ex), "measurement term missing in an existing instance"
            out[j] = ("1" if len(pres) == len(ex) else ("fin" if pres == [1] else "~fin"), nm)
        return out

    def one_gate(tg):
        """Gate of a term whose source is the same unit in every slab (parallel, LUTRAM): '' = always."""
        if all(cnd == "1" for cnd, _ in tg.values()):
            return ""
        if k == 1:
            return tg[0][0]
        ts = []
        for j in range(k):
            cnd = tg[j][0] if j in tg else "1"
            if cnd == "0":
                continue
            ts.append(f"slab[{j}]" if cnd == "1" else f"(slab[{j}] & {cnd})")
        return "(" + " | ".join(ts) + ")" if ts else "1'b0"
    # existence pattern of each neuron: (slabs in non-final, slabs in final)
    pats = {}
    for p, P in enumerate(phys):
        key = (tuple(P["slabs"][0]), tuple(P["slabs"][1]))
        P["pat"] = pats.setdefault(key, len(pats))

    def mask(js):
        v = 0
        for j in js:
            v |= 1 << j
        return v

    # lanes: tables served and comparator slots
    lane_sl = [0] * Lmax
    lane_t2 = [False] * Lmax
    lane_t1 = [False] * Lmax
    lane_members = [[] for _ in range(Lmax)]
    for p, P in enumerate(phys):
        l = P["lane"]
        lane_members[l].append(p)
        if P["table"] == 0:
            lane_sl[l] = max(lane_sl[l], len(P["dterms"]))
        elif P["table"] == 1:
            lane_t2[l] = True
        else:
            lane_t1[l] = True
    NSLOT = 7 + max(5, KSL - 7)

    w("// Generated by hardware/gen/gen_ptw.py; do not edit.")
    w(f"// d={d} T={T} C={C} k={k} S={S} lram={int(lram)} colors={NC} Lmax={Lmax} banks={NB} neurons={len(phys)} KSL={KSL}")
    RS = G["rs"]
    RB = clog2(RS)
    AWS = AW                                       # slab part of a LUTRAM word address
    AWT = AWS + (RB if RS > 1 else 0)              # word address {local replica, slab + reach}
    xr = bool(hs.get("xr")) and lram               # LUTRAM layout with the perfect slice in an extra FF row
    w("// One engine: RS replicas time-shared (r-major sweeps); their literals are words {r, slab} of the same clause units.")
    w("module ptw_rep (")
    w("    input  wire clk,")
    w("    input  wire ldn, input wire [AW_-1:0] lan,          // next cycle loads the start state (LUTRAM: word lan)")
    w("    input  wire [NLD-1:0] lit0,                       // start literals of the load cycle")
    w("    input  wire mvn, input wire [NC-1:0] coln, input wire [KS-1:0] slabn, input wire [AWS-1:0] slabbn,   // next cycle")
    w("    input  wire [RB-1:0] rn,                          // local replica of the next cycle")
    w("    input  wire fin, input wire tmode,")
    w("    input  wire advn, input wire [RB-1:0] ran,        // generator of local replica ran advances next cycle")
    w("    input  wire sd_sh, input wire [NB-1:0] sd_bit, input wire [NB-1:0] push_a, input wire [NB-1:0] push_b,")
    w("    input  wire [RB-1:0] psel,                        // seeding: local replica whose generators take the push")
    w("    input  wire [NSLOT*B-1:0] tr,                     // comparator row of the table of the move two cycles ahead")
    w("    input  wire [NDW-1:0] nd0, input wire [NMW-1:0] nm0,")
    w("    output wire [RS*NDW-1:0] nd_o, output wire [RS*NMW-1:0] nm_o, output wire [2*RS-1:0] cls_o,")
    w("    output wire [RS*SVW-1:0] sv")
    w(");")
    CS = G.get("cs", False)
    # per-engine control registers (keep: Vivado would otherwise merge the identical copies of all engines into one
    # chip-wide net, the routed critical path of round 2)
    w("    (* keep = \"true\" *) reg ld; (* keep = \"true\" *) reg [NC-1:0] col; (* keep = \"true\" *) reg [KS-1:0] slab;")
    SWP = G.get("swp", False)
    FO = G.get("fo", True)
    NRP = 4 if (CS and SWP and FO) else 1                 # --swp: S2 enables in NRP copies (fan-out of col2 / pe2 at d = 11)
    w("    (* keep = \"true\" *) reg [AW_-1:0] la; (* keep = \"true\" *) reg [RB-1:0] rr; "
      + ("(* max_fanout = 64 *) " if (SWP and FO) else "") + "reg [RS-1:0] adv;")
    w("    integer ir;")
    w(f"    (* keep = \"true\", max_fanout = 32 *) reg [{len(pats)-1}:0] pe;              // neuron existence per (slab, window kind) pattern")
    if CS:
        w("    // C-slow loop (3 stages, replicas interleaved, rs >= 3): S1 read the clause words (registered clause outputs),")
        w("    // S2 neurons (registered accepts), S3 flips written back into the words read in S1. Stage copies of the schedule:")
        w(f"    (* keep = \"true\", max_fanout = 32 *) reg [NC-1:0] col2; (* keep = \"true\" *) reg [KS-1:0] slab2, slab3; (* keep = \"true\", max_fanout = 32 *) reg [{len(pats)-1}:0] pe2;")
        w("    (* keep = \"true\" *) reg [RB-1:0] rr2, rr3; reg advd; reg [RB-1:0] rad;")
        w("    always @(posedge clk) begin col2 <= col; slab2 <= slab; slab3 <= slab2; pe2 <= pe; rr2 <= rr; rr3 <= rr2;")
        w("        advd <= advn; rad <= ran; end")
        w("    wire [NC-1:0] colN = col2; wire [KS-1:0] slabN = slab2, slabF = slab3; wire [RB-1:0] rF = rr3;")
        w(f"    wire [{len(pats)-1}:0] peN = pe2;")
        if NRP > 1:
            for i in range(NRP):
                w(f"    (* keep = \"true\" *) reg [NC-1:0] col2r{i}; (* keep = \"true\" *) reg [{len(pats)-1}:0] pe2r{i};")
                w(f"    always @(posedge clk) begin col2r{i} <= col; pe2r{i} <= pe; end")
                w(f"    wire [NC-1:0] colN{i} = col2r{i}; wire [{len(pats)-1}:0] peN{i} = pe2r{i};")
        w("    always @(posedge clk) begin ld <= ldn; col <= mvn ? coln : {NC{1'b0}}; slab <= slabn; la <= lan; rr <= rn;")
        w("        for (ir = 0; ir < RS; ir = ir + 1) adv[ir] <= advd & (rad == ir); end   // lanes one cycle later (S2)")
    else:
        w("    wire [NC-1:0] colN = col; wire [KS-1:0] slabN = slab, slabF = slab; wire [RB-1:0] rF = rr;")
        w(f"    wire [{len(pats)-1}:0] peN = pe;")
        w("    always @(posedge clk) begin ld <= ldn; col <= mvn ? coln : {NC{1'b0}}; slab <= slabn; la <= lan; rr <= rn;")
        w("        for (ir = 0; ir < RS; ir = ir + 1) adv[ir] <= advn & (ran == ir); end")
    w("    always @(posedge clk) begin")
    for (v0, v1), i in pats.items():
        w(f"        pe[{i}] <= fin ? |(slabn & {k}'d{mask(v1)}) : |(slabn & {k}'d{mask(v0)});")
    w("    end")

    # ---------------------------------------------------------------- generators (the one-shot core's lane generator per bank)
    srl_attr = '(* srl_style = "srl" *) ' if G["srl"] else ""
    for g in range(NB):
        N = bank_lanes[g] * B
        w(f"    // generator bank {g}: lanes {g*LB}..{g*LB + bank_lanes[g] - 1}, N = {N} bits per move cycle, one generator pair")
        w(f"    // per local replica; y = the lane uniforms of the replica that advanced, and the seed staging register")
        w(f"    reg [{N-1}:0] y{g};")
        for r in range(RS):
            s_ = f"{g}_{r}"
            w(f"    {srl_attr}reg [{pa-1}:0] ha{s_}; {srl_attr}reg [{pb-1}:0] hb{s_}; wire [{N-1}:0] na{s_}, nb{s_};")
            w(f"    wire pa{s_} = push_a[{g}] & (psel == {r}), pb{s_} = push_b[{g}] & (psel == {r});")
            for nm_, P_, Q_, h in (("na", pa, qa, "ha"), ("nb", pb, qb, "hb")):
                for j in range(N):
                    if j < Q_:
                        w(f"    assign {nm_}{s_}[{j}] = {h}{s_}[{j}] ^ {h}{s_}[{P_ - Q_ + j}];")
                    else:
                        w(f"    assign {nm_}{s_}[{j}] = {h}{s_}[{j}] ^ {h}{s_}[{j - Q_}] ^ {h}{s_}[{P_ - 2 * Q_ + j}];")
            w("    always @(posedge clk) begin")
            w(f"        if (pa{s_} | adv[{r}]) ha{s_} <= {{pa{s_} ? y{g} : na{s_}, ha{s_}[{pa-1}:{N}]}};")
            w(f"        if (pb{s_} | adv[{r}]) hb{s_} <= {{pb{s_} ? y{g} : nb{s_}, hb{s_}[{pb-1}:{N}]}};")
            w("    end")
        sel = " : ".join(f"adv[{r}] ? (na{g}_{r} ^ nb{g}_{r})" for r in range(RS))
        w(f"    always @(posedge clk) if (|adv) y{g} <= {sel} : y{g}; else if (sd_sh) y{g} <= {{sd_bit[{g}], y{g}[{N-1}:1]}};")

    # ---------------------------------------------------------------- lanes: uniforms, comparators, thermometer counts
    lane_cs = {}
    w("    // lanes: u (first generated bit = MSB), comparators against the current table row, registered counts")
    for l in range(Lmax):
        g, oo = l // LB, l % LB
        bits = ", ".join(f"y{g}[{oo * B + b}]" for b in range(B))
        w(f"    wire [{B-1}:0] u{l} = {{{bits}}};")
        nA = max(min(lane_sl[l], 7), 7 if lane_t2[l] else 0)
        nB = max(5 if lane_t1[l] else 0, lane_sl[l] - 7 if lane_sl[l] > 7 else 0)
        sl_wide = lane_sl[l] > 7
        lane_cs[l] = (nA, nB, sl_wide)
        if nA:
            w(f"    wire [{nA-1}:0] ka{l}; wire [2:0] kaw{l}; reg [2:0] ca{l};")
            for s in range(1, nA + 1):
                w(f"    assign ka{l}[{s-1}] = (u{l} < tr[{s * B - 1}:{(s - 1) * B}]);")
            w(f"    tp_popcnt #(.N({nA}), .W(3)) uka{l} (.in(ka{l}), .cnt(kaw{l}));")
        if nB:
            w(f"    wire [{nB-1}:0] kb{l}; wire [3:0] kbw{l}; reg [3:0] cb{l};")
            for s in range(8, 8 + nB):
                w(f"    assign kb{l}[{s-8}] = (u{l} < tr[{s * B - 1}:{(s - 1) * B}]);")
            w(f"    tp_popcnt #(.N({nB}), .W(4)) ukb{l} (.in(kb{l}), .cnt(kbw{l}));")
        if sl_wide:
            w(f"    reg [4:0] cs{l};")
    w("    always @(posedge clk) begin")
    for l in range(Lmax):
        nA, nB, sw_ = lane_cs[l]
        if nA:
            w(f"        ca{l} <= kaw{l};")
        if nB:
            w(f"        cb{l} <= kbw{l};")
        if sw_:
            w(f"        cs{l} <= kaw{l} + kbw{l};")
    w("    end")

    # ---------------------------------------------------------------- clause units
    flips = {}                     # storage literal name -> list of flip terms

    def addflip(key, expr):
        e_ = re.sub(r"\bslab\[", "slabF[", expr)            # flips happen in the write stage (S3 when C-slow)
        if CS:
            e_ = re.sub(r"\ba(\d+)\b", r"ar[\1]", e_)
        flips.setdefault(key, []).append(e_)

    if not lram:
        w(f"    // clause units: data clause (q, t) = index t n + q (pt_clause, literals x, z), measurement (s, c, t) = s m T + t m + c")
        w(f"    wire [{NDC-1}:0] ep, exm, ezm, cx, cz, fx, fz;")
        w(f"    wire [{NMC-1}:0] ml, fm;")
        for i in range(NDC):
            w(f"    pt_clause cd{i} (.clk(clk), .ld(ld), .c0x(lit0[{2*i}]), .c0z(lit0[{2*i+1}]), .fx(fx[{i}]), .fz(fz[{i}]), "
              f".x(cx[{i}]), .z(cz[{i}]), .v(), .ep(ep[{i}]), .exm(exm[{i}]), .ezm(ezm[{i}]));")
        for i in range(NMC):
            w(f"    ptw_mcl cm{i} (.clk(clk), .ld(ld), .m0(lit0[{2*NDC + i}]), .f(fm[{i}]), .m(ml[{i}]));")

        def dsrc(nm):           # ("d", q, t, sec) -> (ep, dn) signal names
            i = nm[2] * n + nm[1]
            return f"ep[{i}]", (f"exm[{i}]" if nm[3] == 0 else f"ezm[{i}]"), ("dx", i) if nm[3] == 0 else ("dz", i)

        def msrc(nm):
            i = nm[1] * m * T + nm[3] * m + nm[2]
            return f"ml[{i}]", ("m", i)
    else:
        w(f"    // LUTRAM clause units: data (q, row) = index row n + q, word a = slice row + a S; measurement (s, c, row) = (s S + row) m + c")
        NDU, NMU = n * S, 2 * m * S
        if CS:
            w(f"    wire [{NDU-1}:0] epw, exmw, ezmw, cx, cz, fx, fz; wire [{NMU-1}:0] mlw, fm;")
            w(f"    reg [{NDU-1}:0] ep, exm, ezm; reg [{NMU-1}:0] ml;          // S1: registered clause outputs")
            w("    always @(posedge clk) begin ep <= epw; exm <= exmw; ezm <= ezmw; ml <= mlw; end")
        else:
            w(f"    wire [{NDU-1}:0] ep, exm, ezm, cx, cz, fx, fz;")
            w(f"    wire [{NMU-1}:0] ml, fm;")
        # address groups: colors in which a data unit is reached from the slab below (wrap)
        wrapc = [set() for _ in range(NDU)]
        plainc = [set() for _ in range(NDU)]
        for P in phys:
            if P["row"] >= S:                      # extra-row neurons touch only the extra FF units
                continue
            for tm in P["dterms"]:
                rr = P["row"] + tm[2]
                u = (rr % S) * n + tm[1]
                (wrapc if rr >= S else plainc)[u].add(P["col"])
        for u in range(NDU):
            assert not (wrapc[u] & plainc[u]), "LUTRAM unit touched at two words in one color"
        groups = {}
        ugrp = [groups.setdefault(frozenset(wrapc[u]), len(groups)) for u in range(NDU)]
        if frozenset() not in groups:
            groups[frozenset()] = len(groups)
        g0 = groups[frozenset()]
        w(f"    reg [AW_-1:0] adr [0:{len(groups)-1}];      // word address per group of units (slab, + 1 where reached from below)")
        w("    always @(posedge clk) begin")
        for msk, gi in groups.items():
            mv_ = 0
            for c in msk:
                mv_ |= 1 << c
            add = f" + |(coln & {NC}'d{mv_})" if msk else ""
            sa_ = f"AWS'(slabbn{add})"
            w(f"        adr[{gi}] <= ldn ? lan : " + (f"{{rn, {sa_}}};" if RS > 1 else f"{sa_};"))
        w("    end")
        AF = "adr"
        if CS:
            AF = "adrw"
            w(f"    (* keep = \"true\" *) reg [AW_-1:0] adr2 [0:{len(groups)-1}], adrw [0:{len(groups)-1}];   // S3 write addresses")
            w("    integer ig;")
            w(f"    always @(posedge clk) for (ig = 0; ig < {len(groups)}; ig = ig + 1) begin adr2[ig] <= adr[ig]; adrw[ig] <= adr2[ig]; end")
        G["AF"] = AF
        WBD = 2 * NDU
        for u in range(NDU):
            if CS:
                w(f"    ptw_dcl_ram2 #(.AW(AW_)) cd{u} (.clk(clk), .ld(ld), .ra(adr[{ugrp[u]}]), .wa(adrw[{ugrp[u]}]), "
                  f".x0(lit0[{2*u}]), .z0(lit0[{2*u+1}]), .fx(fx[{u}]), .fz(fz[{u}]), .x(cx[{u}]), .z(cz[{u}]), .ep(epw[{u}]), "
                  f".exm(exmw[{u}]), .ezm(ezmw[{u}]));")
            else:
                w(f"    ptw_dcl_ram #(.AW(AW_)) cd{u} (.clk(clk), .ld(ld), .a(adr[{ugrp[u]}]), .x0(lit0[{2*u}]), .z0(lit0[{2*u+1}]), "
                  f".fx(fx[{u}]), .fz(fz[{u}]), .x(cx[{u}]), .z(cz[{u}]), .ep(ep[{u}]), .exm(exm[{u}]), .ezm(ezm[{u}]));")
        for u in range(NMU):
            if CS:
                w(f"    ptw_mcl_ram2 #(.AW(AW_)) cm{u} (.clk(clk), .ld(ld), .ra(adr[{g0}]), .wa(adrw[{g0}]), .m0(lit0[{WBD + u}]), "
                  f".f(fm[{u}]), .m(mlw[{u}]));")
            else:
                w(f"    ptw_mcl_ram #(.AW(AW_)) cm{u} (.clk(clk), .ld(ld), .a(adr[{g0}]), .m0(lit0[{WBD + u}]), .f(fm[{u}]), .m(ml[{u}]));")
        if xr:
            XB = WBD + NMU
            w(f"    // extra row: the perfect slice T (final windows, last slab), FF clause units with one copy per local replica")
            w(f"    wire ldx = ld & (la[AWS-1:0] == {k - 1});")
            w(f"    wire [RB-1:0] xsel = " + ("ld ? la[AW_-1 -: RB] : rr;" if RS > 1 else "1'b0;"))
            w(f"    wire [{RS*n-1}:0] xxa, xza; wire [{n-1}:0] xfx, xfz;")
            if CS:
                w(f"    wire [RB-1:0] xwsel = ld ? la[AW_-1 -: RB] : rr3;")
                w(f"    wire [{n-1}:0] xepw, xexmw, xezmw; reg [{n-1}:0] xep, xexm, xezm;")
                w("    always @(posedge clk) begin xep <= xepw; xexm <= xexmw; xezm <= xezmw; end")
            else:
                w(f"    wire [{n-1}:0] xep, xexm, xezm;")
            for q in range(n):
                if CS:
                    w(f"    ptw_dcl_x2 #(.NR({RS}), .NRB({RB})) cx{q} (.clk(clk), .ld(ldx), .rr(xsel), .rw(xwsel), "
                      f".x0(lit0[{XB + 2*q}]), .z0(lit0[{XB + 2*q + 1}]), .fx(xfx[{q}]), .fz(xfz[{q}]), .ep(xepw[{q}]), "
                      f".exm(xexmw[{q}]), .ezm(xezmw[{q}]), .xa(xxa[{q*RS} +: RS]), .za(xza[{q*RS} +: RS]));")
                else:
                    w(f"    ptw_dcl_x #(.NR({RS}), .NRB({RB})) cx{q} (.clk(clk), .ld(ldx), .r(xsel), .x0(lit0[{XB + 2*q}]), "
                      f".z0(lit0[{XB + 2*q + 1}]), .fx(xfx[{q}]), .fz(xfz[{q}]), .x(), .z(), .ep(xep[{q}]), .exm(xexm[{q}]), "
                      f".ezm(xezm[{q}]), .xa(xxa[{q*RS} +: RS]), .za(xza[{q*RS} +: RS]));")

        def dsrc(nm):
            t = nm[2]
            if xr and t == T:
                q = nm[1]
                return f"xep[{q}]", (f"xexm[{q}]" if nm[3] == 0 else f"xezm[{q}]"), ("xx", q) if nm[3] == 0 else ("xz", q)
            u = (t % S) * n + nm[1]
            return f"ep[{u}]", (f"exm[{u}]" if nm[3] == 0 else f"ezm[{u}]"), ("dx", u) if nm[3] == 0 else ("dz", u)

        def msrc(nm):
            t = nm[3]
            u = (nm[1] * S + (t % S)) * m + nm[2]
            return f"ml[{u}]", ("m", u)
        G["ugrp"], G["groups"] = ugrp, groups

    # ---------------------------------------------------------------- neurons and switch fabric
    w("    // transition neurons: inputs = the clause outputs of the move's support (k:1 slab select in FF-slab layouts)")
    nlog = [[], []]
    for p, P in enumerate(phys):
        en = (f"colN{p % NRP}[{P['col']}] & peN{p % NRP}[{P['pat']}]" if NRP > 1 else f"colN[{P['col']}] & peN[{P['pat']}]")
        l = P["lane"]
        nA, nB, slw = lane_cs[l]
        ups, dns = [], []
        for tm in P["dterms"]:
            tg = term_gate(p, P, tm)
            live = {j: v for j, v in tg.items() if v[0] != "0"}
            if not live:
                ups.append("1'b0"); dns.append("1'b0")
                continue
            srcs = {dsrc(nm) for _, nm in live.values()}
            if (lram or k == 1) and len(srcs) > 1:
                # LUTRAM with the extra row: the term reaches row 0 of the next slab, or the extra row in the last slab
                up_t, dn_t = [], []
                for src in sorted(srcs):
                    js = [j for j, (_, nm) in live.items() if dsrc(nm) == src]
                    gx = "(" + " | ".join(f"slab[{j}]" + ("" if live[j][0] == "1" else f" & {live[j][0]}") for j in js) + ")"
                    up_t.append(f"({gx} & {src[0]})"); dn_t.append(f"({gx} & {src[1]})")
                    addflip(src[2], f"a{p} & {gx}")
                ups.append("(" + " | ".join(up_t) + ")")
                dns.append("(" + " | ".join(dn_t) + ")")
            elif lram or k == 1:
                gx = one_gate(tg)
                su, sdn, fk = srcs.pop()
                ups.append(f"({gx} & {su})" if gx else su)
                dns.append(f"({gx} & {sdn})" if gx else sdn)
                addflip(fk, f"a{p}" + (f" & {gx}" if gx else ""))
            else:
                up_t, dn_t = [], []
                for j, (cnd, nm) in sorted(live.items()):
                    su, sdn, fk = dsrc(nm)
                    gx = f"slab[{j}]" + ("" if cnd == "1" else f" & {cnd}")
                    up_t.append(f"({gx} & {su})"); dn_t.append(f"({gx} & {sdn})")
                    addflip(fk, f"a{p} & {gx}")
                ups.append("(" + " | ".join(up_t) + ")")
                dns.append("(" + " | ".join(dn_t) + ")")
        ups = [re.sub(r"\bslab\[", "slabN[", u) for u in ups]
        dns = [re.sub(r"\bslab\[", "slabN[", u) for u in dns]
        L = len(ups)
        if P["table"] == 0:
            cb = 5 if slw else 3
            cs = f"cs{l}" if slw else f"ca{l}"
            w(f"    wire a{p}; wire [{DAW-1}:0] da{p};")
            w(f"    ptw_sl #(.L({L}), .CB({cb}), .DAW({DAW})) n{p} (.en({en}), .up({{{', '.join(ups)}}}), "
              f".dn({{{', '.join(dns)}}}), .c({cs}), .acc(a{p}), .da(da{p}));")
            if P["kind"] == 2:
                nlog[P["sec"]].append(f"a{p}")
        else:
            ms = []
            for tm in P["mterms"]:
                tg = term_gate(p, P, tm)
                srcs = []
                for j, (g_, nm) in sorted(tg.items()):
                    s_, fk = msrc(nm)
                    gx = "" if (lram or k == 1) else f"slab[{j}]"
                    srcs.append((gx, s_))
                    addflip(fk, f"a{p}" + (f" & {gx}" if gx else ""))
                if lram or k == 1:
                    assert len({s for _, s in srcs}) == 1
                    ms.append(srcs[0][1])
                else:
                    ms.append("(" + " | ".join(f"({g} & {s})" for g, s in srcs) + ")")
            ms = [re.sub(r"\bslab\[", "slabN[", u) for u in ms]
            cc = f"ca{l}" if P["table"] == 1 else f"cb{l}[2:0]"
            w(f"    wire a{p}; wire [2:0] da{p}, db{p};")
            w(f"    ptw_tn #(.ND({L}), .NM({len(ms)})) n{p} (.en({en}), .up({{{', '.join(ups)}}}), .dn({{{', '.join(dns)}}}), "
              f".m({{{', '.join(ms)}}}), .mode(tmode), .c({cc}), .acc(a{p}), .da(da{p}), .db(db{p}));")
    NPH = len(phys)
    if CS:
        w(f"    // S2 -> S3: registered accepts and accepted (a, b) of every neuron")
        w(f"    reg [{NPH-1}:0] ar;")
        for p, P in enumerate(phys):
            w(f"    reg [{(DAW if P['table'] == 0 else 3) - 1}:0] dq{p};" + (f" reg [2:0] bq{p};" if P["table"] else ""))
        w("    always @(posedge clk) begin")
        for p, P in enumerate(phys):
            w(f"        ar[{p}] <= a{p}; dq{p} <= da{p};" + (f" bq{p} <= db{p};" if P["table"] else ""))
        w("    end")
    w("    // switch fabric, neuron -> clause (at most one neuron per clause and color)")
    if not lram:
        nd_, nm__ = NDC, NMC
    else:
        nd_, nm__ = n * S, 2 * m * S
    for i in range(nd_):
        for tag, nmv in (("dx", "fx"), ("dz", "fz")):
            fl = flips.get((tag, i), [])
            w(f"    assign {nmv}[{i}] = " + (" | ".join(f"({e})" for e in fl) if fl else "1'b0") + ";")
    for i in range(nm__):
        fl = flips.get(("m", i), [])
        w(f"    assign fm[{i}] = " + (" | ".join(f"({e})" for e in fl) if fl else "1'b0") + ";")
    if xr:
        for q in range(n):
            for tag, nmv in (("xx", "xfx"), ("xz", "xfz")):
                fl = flips.get((tag, q), [])
                w(f"    assign {nmv}[{q}] = " + (" | ".join(f"({e})" for e in fl) if fl else "1'b0") + ";")

    # ---------------------------------------------------------------- incremental energy and class (off the loop)
    GS = 8
    NG = -(-Lmax // GS)
    GW = DAW + 3
    w("    // incremental energy: per lane the accepted (a, b) of its active neuron, group sums, then nd += a, nm += b")
    w(f"    reg [{DAW-1}:0] la_d [0:{Lmax-1}]; reg [2:0] la_b [0:{Lmax-1}];")
    w(f"    reg signed [{GW-1}:0] ga [0:{NG-1}], gb [0:{NG-1}];")
    w("    always @(posedge clk) begin")
    for l in range(Lmax):
        ds = []
        bs = []
        for p in lane_members[l]:
            P = phys[p]
            DV, BV = ("dq", "bq") if CS else ("da", "db")
            if P["table"] == 0:
                ds.append(f"{DV}{p}")
            else:
                ds.append(f"{{{{{DAW-3}{{{DV}{p}[2]}}}}, {DV}{p}}}" if DAW > 3 else f"{DV}{p}")
                bs.append(f"{BV}{p}")
        w(f"        la_d[{l}] <= " + (" | ".join(ds) if ds else f"{DAW}'d0") + ";")
        w(f"        la_b[{l}] <= " + (" | ".join(bs) if bs else "3'd0") + ";")
    for g in range(NG):
        ls = range(g * GS, min(Lmax, (g + 1) * GS))
        w(f"        ga[{g}] <= " + " + ".join(f"{GW}'($signed(la_d[{l}]))" for l in ls) + ";")
        w(f"        gb[{g}] <= " + " + ".join(f"{GW}'($signed(la_b[{l}]))" for l in ls) + ";")
    w("    end")
    w(f"    wire signed [{GW+4}:0] sa = " + " + ".join(f"{GW+5}'(ga[{g}])" for g in range(NG)) + ";")
    w(f"    wire signed [{GW+4}:0] sb = " + " + ".join(f"{GW+5}'(gb[{g}])" for g in range(NG)) + ";")
    if CS:
        nlog = [[re.sub(r"^a(\d+)$", r"ar[\1]", x) for x in nl] for nl in nlog]
    w("    reg [1:0] lx; reg [RB-1:0] t1, t2;                // replica tags of the lane and group stages")
    w("    reg [NDW-1:0] nd [0:RS-1]; reg [NMW-1:0] nm [0:RS-1]; reg [1:0] cls [0:RS-1];")
    w("    always @(posedge clk) begin")
    w("        lx <= {" + (" ^ ".join(nlog[1]) if nlog[1] else "1'b0") + ", " + (" ^ ".join(nlog[0]) if nlog[0] else "1'b0") + "};")
    w("        t1 <= rF; t2 <= t1;")
    w("        for (ir = 0; ir < RS; ir = ir + 1)")
    w("            if (ld) begin nd[ir] <= nd0; nm[ir] <= nm0; cls[ir] <= 2'd0; end")
    w("            else begin")
    w("                if (t2 == ir) begin nd[ir] <= nd[ir] + NDW'(sa); nm[ir] <= nm[ir] + NMW'(sb); end")
    w("                if (t1 == ir) cls[ir] <= cls[ir] ^ lx;")
    w("            end")
    w("    end")
    w("    genvar gr;")
    w("    generate for (gr = 0; gr < RS; gr = gr + 1) begin : g_o")
    w("        assign nd_o[gr*NDW +: NDW] = nd[gr]; assign nm_o[gr*NMW +: NMW] = nm[gr]; assign cls_o[2*gr +: 2] = cls[gr];")
    w("    end endgenerate")

    # ---------------------------------------------------------------- commit summary vector
    sv_names = sv_list(G)
    if not lram:
        def svsig(nm):
            if nm[0] == "d":
                i = nm[2] * n + nm[1]
                return f"cx[{i}]" if nm[3] == 0 else f"cz[{i}]"
            return f"ml[{nm[1] * m * T + nm[3] * m + nm[2]}]"
        w("    assign sv = {" + ", ".join(svsig(nm) for nm in reversed(sv_names)) + "};")
    else:
        # FF mirrors (one set per local replica) of the literals the commit reads, updated by the same flips
        # (word {r, slice // S}); the extra row's literals are read from its FF units directly
        SVW_ = len(sv_names)
        w(f"    reg [{SVW_-1}:0] mir [0:RS-1];")
        WBD = 2 * n * S
        for r in range(RS):
            bits = []
            w("    always @(posedge clk) begin")
            for i, nm in enumerate(sv_names):
                if nm[0] == "d" and xr and nm[2] == T:
                    bits.append(f"{'xxa' if nm[3] == 0 else 'xza'}[{nm[1]*RS + r}]")
                    continue
                bits.append(f"mir[{r}][{i}]")
                if nm[0] == "d":
                    t, q, sec = nm[2], nm[1], nm[3]
                    u = (t % S) * n + q
                    lb = 2 * u + sec
                    f_ = f"fx[{u}]" if sec == 0 else f"fz[{u}]"
                    a_ = f"{G['AF']}[{G['ugrp'][u]}]"
                else:
                    sec, c, t = nm[1], nm[2], nm[3]
                    u = (sec * S + (t % S)) * m + c
                    lb = WBD + u
                    f_ = f"fm[{u}]"
                    a_ = f"{G['AF']}[{G['groups'][frozenset()]}]"
                wd = (r << AWS) | (t // S) if RS > 1 else t // S
                w(f"        mir[{r}][{i}] <= (ld & (la == {wd})) ? lit0[{lb}] : mir[{r}][{i}] ^ ({f_} & ({a_} == {wd}));")
            w("    end")
            w(f"    assign sv[{r*SVW_} +: {SVW_}] = {{" + ", ".join(reversed(bits)) + "};")
    w("endmodule")
    info = dict(NB=NB, bank_lanes=bank_lanes, Lmax=Lmax, NC=NC, S=S, NSLOT=NSLOT, KSL=KSL, DAW=DAW, n=n, m=m,
                neurons=len(phys), NDC=NDC, NMC=NMC, AW=AWT, AWS=AWS, RB=RB, RS=RS, xr=xr, SVW=len(sv_names),
                comparators=sum(a + b for a, b, _ in lane_cs.values()))
    if lram:
        info["NLD"] = 2 * n * S + 2 * m * S + (2 * n if xr else 0)
        info["groups"] = len(G["groups"])
    else:
        info["NLD"] = 2 * NDC + NMC
    return "\n".join(o) + "\n", info


def sv_list(G):
    """Literals the commit summary reads: X-sector observable line (x) and Z-sector line (z) at every slice 0..T, then the
    measurement literals of slice C - 1 (sector, check)."""
    ws = G["ws"][1]
    T, C, m = G["T"], G["C"], ws.m
    out = []
    for sec in (0, 1):
        for t in range(T + 1):
            for q in sorted(int(x) for x in ws.obs[sec]):
                out.append(("d", q, t, sec))
    for sec in (0, 1):
        for c in range(m):
            out.append(("m", sec, c, C - 1))
    return out


# ===================================================================================================== shared modules
def gen_shared(G, ri):
    """ptw_summ (summary from the summary vector), ptw_prep (start state, start counts)."""
    hs, (wsn, wsf) = G["hs"], G["ws"]
    T, C, k, S, lram = G["T"], G["C"], hs["k"], hs["S"], hs["lram"]
    n, m = wsf.n, wsf.m
    M2 = 2 * m
    NDC, NMC = ri["NDC"], ri["NMC"]
    NLA = 2 * NDC + NMC
    svn = sv_list(G)
    o = []
    w = o.append
    # summary: {par_z, par_x, sig}
    w("// commit summary: sigma candidate m*(c, C - 1) per sector, parities of x (z) on the X (Z) observable line over the")
    w("// committed slices (t < C; final window: all slices 0..T)")
    w(f"module ptw_summ (input wire [{len(svn)-1}:0] sv, input wire fin, output wire [{M2+1}:0] sm);")
    idx = {nm: i for i, nm in enumerate(svn)}
    for sec in (0, 1):
        lo = [f"sv[{idx[nm]}]" for nm in svn if nm[0] == "d" and nm[3] == sec and nm[2] < C]
        hi = [f"sv[{idx[nm]}]" for nm in svn if nm[0] == "d" and nm[3] == sec and nm[2] >= C]
        w(f"    wire p{sec} = (^{{{', '.join(lo)}}}) ^ (fin & (^{{{', '.join(hi)}}}));")
    sig = [f"sv[{idx[('m', sec, c, C - 1)]}]" for sec in (0, 1) for c in range(m)]
    w("    assign sm = {p1, p0, " + ", ".join(reversed(sig)) + "};")
    w("endmodule")
    w("")
    # prep: start literals (all, logical order), LUTRAM words, start counts, start summary vector
    w("// start state: m(s, c, t) = syn(t, s, c) XOR sigma(s, c); final window: data chains on slice T for synp XOR sigma (the one-shot core's c0)")
    w("module ptw_prep (input wire clk, input wire fin, input wire [TR*M2-1:0] syn, input wire [M2-1:0] synp, "
      "input wire [M2-1:0] sig,")
    w("                 output reg [NLA-1:0] lit0, output reg [KS*NLD-1:0] lw, output reg [SVW-1:0] sv0, "
      "output reg [NDW-1:0] nd0, output reg [NMW-1:0] nm0);")
    w(f"    wire [NLA-1:0] l;")
    for i in range(NDC):
        t, q = divmod(i, n)
        if t < T:
            w(f"    assign l[{2*i}] = 1'b0; assign l[{2*i+1}] = 1'b0;")
    for sec in (0, 1):
        ch = wsf.chains[sec]
        for q in range(n):
            js = np.flatnonzero(ch[:, q])
            e = " ^ ".join(f"synp[{sec*m + int(c)}] ^ sig[{sec*m + int(c)}]" for c in js)
            w(f"    assign l[{2*(T*n + q) + sec}] = " + (f"fin & ({e});" if e else "1'b0;"))
    for sec in (0, 1):
        for t in range(T):
            for c in range(m):
                w(f"    assign l[{2*NDC + sec*m*T + t*m + c}] = syn[{t*M2 + sec*m + c}] ^ sig[{sec*m + c}];")
    # LUTRAM words
    if lram:
        WB = ri["NLD"]
        w(f"    wire [KS*NLD-1:0] lwv;")
        for j in range(k):
            for u in range(n * S):
                row, q = divmod(u, n)
                t = row + j * S
                for sec in (0, 1):
                    v = f"l[{2*(t*n + q) + sec}]" if t <= T else "1'b0"
                    w(f"    assign lwv[{j*WB + 2*u + sec}] = {v};")
            for u in range(2 * m * S):
                srow, c = divmod(u, m)
                sec, row = divmod(srow, S)
                t = row + j * S
                v = f"l[{2*NDC + sec*m*T + t*m + c}]" if t < T else "1'b0"
                w(f"    assign lwv[{j*WB + 2*n*S + u}] = {v};")
            if ri["xr"]:
                for q in range(n):
                    for sec in (0, 1):
                        v = f"l[{2*(T*n + q) + sec}]" if j == k - 1 else "1'b0"
                        w(f"    assign lwv[{j*WB + 2*n*S + 2*m*S + 2*q + sec}] = {v};")
    else:
        w("    wire [KS*NLD-1:0] lwv = l;")
    idx_l = []
    for nm in svn:
        if nm[0] == "d":
            idx_l.append(f"l[{2*(nm[2]*n + nm[1]) + nm[3]}]")
        else:
            idx_l.append(f"l[{2*NDC + nm[1]*m*T + nm[3]*m + nm[2]}]")
    # start counts: data clauses with x | z (slice T only), measurement literals; chunk popcounts then the sum
    dv = [f"(l[{2*(T*n+q)}] | l[{2*(T*n+q)+1}])" for q in range(n)]
    CH = 32
    mch = [(lo, min(NMC, lo + CH)) for lo in range(0, NMC, CH)]
    w(f"    wire [{n-1}:0] dvw = {{{', '.join(reversed(dv))}}};")
    w(f"    reg [{clog2(n+1)-1}:0] pdc; reg [5:0] pmc [0:{len(mch)-1}];")
    for i, (lo, hi) in enumerate(mch):
        w(f"    wire [5:0] pm{i}; tp_popcnt #(.N({hi-lo}), .W(6)) upm{i} (.in(lit0[{2*NDC+hi-1}:{2*NDC+lo}]), .cnt(pm{i}));")
    w(f"    wire [{clog2(n+1)-1}:0] pdw; reg [{n-1}:0] dvr;")
    w(f"    tp_popcnt #(.N({n})) upd (.in(dvr), .cnt(pdw));")
    w("    always @(posedge clk) begin")
    w("        lit0 <= l; lw <= lwv; dvr <= dvw;")
    w("        sv0 <= {" + ", ".join(reversed(idx_l)) + "};")
    w("        pdc <= pdw;")
    for i in range(len(mch)):
        w(f"        pmc[{i}] <= pm{i};")
    w("        nd0 <= NDW'(pdc);")
    w("        nm0 <= " + " + ".join(f"NMW'(pmc[{i}])" for i in range(len(mch))) + ";")
    w("    end")
    w("endmodule")
    return "\n".join(o) + "\n"


def gen_params(G, ri):
    hs, (wsn, wsf) = G["hs"], G["ws"]
    cfg = G["cfg"]
    R, B, K = G["R"], G["B"], G["K"]
    KSL = wsf.KSL
    ROWN = KSL + 12
    LW = clog2(R)
    NP = max(R - 1, 1)
    NDW = clog2(ri["NDC"] + 1)
    NMW = clog2(ri["NMC"] + 1)
    EW = clog2(((1 << cfg.EW) - 1) * (ri["NDC"] + ri["NMC"]) + 1)
    thr_init = 0
    for t in range(R - 1, -1, -1):
        for r in range(ROWN, 0, -1):
            thr_init = (thr_init << B) | int(wsf.thr[t, r])
    th_init = 0
    for t in range(NP - 1, -1, -1):
        th_init = (th_init << cfg.TW) | (int(wsf.theta[t]) if t < R - 1 else 0)
    tab = 0
    for c, tb in enumerate(hs["col_table"]):
        tab |= int(tb) << c
    k = hs["k"]
    RS = G["rs"]
    LDC = RS * k if hs["lram"] else 1
    prm = [f"localparam integer NS = {R}, NP = {NP}, LW = {LW}, NC = {hs['NC']}, KS = {k}, NB = {ri['NB']}, B = {B};",
           f"localparam integer KSL = {KSL}, ROWN = {ROWN}, ROWW = {ROWN * B}, NSLOT = {ri['NSLOT']}, AW_ = {ri['AW']}, AWS = {ri['AWS']};",
           f"localparam integer RS = {RS}, RB = {ri['RB']}, NE = {R // RS};",
           f"localparam integer TR = {G['T']}, CM = {G['C']}, M2 = {2 * ri['m']}, NLA = {2 * ri['NDC'] + ri['NMC']}, "
           f"NLD = {ri['NLD']}, LDC = {LDC}, SVW = {ri['SVW']};",
           f"localparam integer NDW = {NDW}, NMW = {NMW}, EW = {EW}, QW = {cfg.EW}, KSW = {K}, GAP_SW = {gap_sw(G['cs'], G.get('swp', False))}, CSL = {int(G['cs'])};",
           f"localparam integer USW = {cfg.USW}, TW = {cfg.TW}, SH = {cfg.LF + cfg.TF}, XIW = {cfg.XIW};",
           f"localparam [{cfg.XIW-1}:0] XIMAX = {cfg.XIW}'d{(1 << cfg.XIW) - 1};",
           f"localparam [NC-1:0] TAB = {hs['NC']}'d{tab};          // color table: 1 = T color",
           f"localparam [{R * ROWN * B - 1}:0] THR_INIT = {R * ROWN * B}'h{thr_init:x};",
           f"localparam [{NP * cfg.TW - 1}:0] THETA_INIT = {NP * cfg.TW}'h{th_init:x};",
           f"localparam [QW-1:0] WQ_INIT = {cfg.EW}'d{wsf.Wq}, WM_INIT = {cfg.EW}'d{wsf.wq};",
           # lanes per generator bank (the board wrapper reports them: the host pads the seed streams to bank_lanes x B)
           f"localparam integer BL0 = {int(ri['bank_lanes'][0])}, BL1 = {int(ri['bank_lanes'][1]) if ri['NB'] > 1 else 0};"]
    sub = dict(AWS=ri["AWS"], RS=RS, RB=ri["RB"], AW_=ri["AW"], NLD=ri["NLD"], NC=hs["NC"], KS=k, NB=ri["NB"], NSLOT=ri["NSLOT"], B=B, NDW=NDW, NMW=NMW,
               SVW=ri["SVW"], TR=G["T"], M2=2 * ri["m"], NLA=2 * ri["NDC"] + ri["NMC"])
    if G.get("swp"):
        prm.append("`define PTW_SWP          // ptw_top: swap decision registered (GAP_SW + 1)")
    return "`ifndef PTW_PARAMS_SVH\n`define PTW_PARAMS_SVH\n" + "\n".join(prm) + "\n`endif\n", dict(
        LW=LW, NP=NP, NDW=NDW, NMW=NMW, EW=EW, ROWN=ROWN, LDC=LDC, sub=sub)


def subst(txt, sub):
    """The generated modules carry their constants inline (ptw_params.svh is included by ptw_top only)."""
    import re
    return re.sub(r"\b(" + "|".join(sub) + r")\b", lambda mo: str(sub[mo.group(1)]), txt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True); ap.add_argument("--p", type=float, default=0.04)
    ap.add_argument("--R", type=int, default=16); ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--lram", action="store_true"); ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--B", type=int, default=16); ap.add_argument("--bmin", type=float, default=0.1)
    ap.add_argument("--C", type=int, default=None); ap.add_argument("--Bw", type=int, default=None)
    ap.add_argument("--rs", type=int, default=1); ap.add_argument("--srl", type=int, default=0)
    ap.add_argument("--cs", type=int, default=0)        # 1: C-slow 3-stage loop (LUTRAM, rs >= 3)
    ap.add_argument("--fo", type=int, default=1)        # with --swp: replicated S2 enables / generator advance (0: off)
    ap.add_argument("--swp", type=int, default=0)       # 1: registered swap decision (ptw_top PTW_SWP; GAP_SW + 1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    G = build(a.d, a.R, a.k, a.lram, a.p, a.C, a.Bw, a.K, a.B, a.bmin, a.rs, bool(a.srl), bool(a.cs), bool(a.swp), bool(a.fo))
    rep, ri = gen_rep(G)
    sh = gen_shared(G, ri)
    prm, pi = gen_params(G, ri)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "ptw_core.sv").write_text(subst(rep + "\n" + sh, pi["sub"]))
    (out / "ptw_params.svh").write_text(prm)
    pi = {k_: v for k_, v in pi.items() if k_ != "sub"}
    info = dict(ri, **pi, d=a.d, R=a.R, k=a.k, lram=a.lram, K=a.K, B=a.B, p=a.p, bmin=a.bmin, T=G["T"], C=G["C"],
                GAP_SW=gap_sw(bool(a.cs), bool(a.swp)), cs=a.cs, swp=a.swp, srl=a.srl, cycles_per_sweep=a.rs * a.k * ri["NC"])
    (out / "ptw_info.json").write_text(json.dumps(info, indent=1, default=int))
    print(json.dumps({k_: info[k_] for k_ in ("d", "k", "lram", "RS", "NC", "Lmax", "NB", "neurons", "comparators", "EW")}))
    return G, info


if __name__ == "__main__":
    main()

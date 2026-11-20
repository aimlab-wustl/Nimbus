// Real-time core: parallel tempering on one sliding window of the rotated surface code.
// Bit-exact with nimbus.decoding.real_time (hardware schedule, PTWConfig(k, lram)). Instance constants come from the generated
// ptw_core.sv (ptw_rep, ptw_prep, ptw_summ) and ptw_params.svh (hardware/gen/gen_ptw.py).
//
// Per window (syn_valid with the window's raw syndromes syn[t][sector][check], t < TR, the perfect round synp, fin = final
// window): start state m = s XOR sigma (+ final: the one-shot core's chains on slice TR), nsw sweeps of KS slabs x NC colors, a swap
// round (the one-shot core's unit) after every KSW-th sweep, global best after every sweep (lowest E over replicas, lowest index on ties,
// kept if strictly below the best so far), occupancy of the class of the beta = 1 replica per sweep block. Commit (done):
// non-final: sigma <= s(CM - 1) XOR m*(CM - 1), frame ^= parities over slices < CM; final: frame ^= parities over all
// slices, fr_ml = frame XOR (bestC XOR ML). The lane generators and swap generators are seeded per window (host ports, as in pt_top).
//
// Timing (t_s = take): start literals t_s + 1, counts t_s + 3, load L = t_s + 4 (LDC cycles), labels / thresholds L,
// first move L + max(4, LDC). The schedule item f_* at cycle a is the move of cycle a + 2 (lane generators advance at a,
// comparators at a + 1). After the last move c0 of a sweep: summary staged c0 + 1, class c0 + 2, nd / nm c0 + 3, energy
// c0 + 4 (swap energies), labels c0 + 5, threshold row c0 + 6 (next item; GAP_SW = 7 idle items after a swap sweep), counts
// and the sweep record (arr) c0 + 6, global best c0 + 5 + TL.
`include "ptw_params.svh"
module ptw_top (
    input  wire clk, input wire rst,
    input  wire sd_sh, input wire [NB-1:0] push_a, input wire [NB-1:0] push_b, input wire ss_sh,
    input  wire [NE*NB-1:0] sd_bit, input wire [NP-1:0] ss_bit, input wire [RB-1:0] psel,   // seeding (per engine, bank)
    input  wire fill, input wire recalc,
    input  wire tw_en, input wire [LW-1:0] tw_addr, input wire [ROWW-1:0] tw_data,   // threshold row (label tw_addr)
    input  wire th_en, input wire [LW-1:0] th_addr, input wire [TW-1:0] th_data,      // swap constant (pair th_addr)
    input  wire [QW-1:0] wq_d, input wire [QW-1:0] wq_m, input wire tmode,            // Wq, wq, rank mode (p < 1/28)
    input  wire [31:0] nsw,                                                            // sweeps per window (power of two)
    input  wire clr,                                                                   // shot start: sigma = 0, frame = 0
    input  wire syn_valid, input wire fin, input wire [TR*M2-1:0] syn, input wire [M2-1:0] synp,
    output wire ready, output reg busy, output reg done,
    output reg [EW-1:0] best_e, output reg [1:0] best_c, output reg [M2+1:0] best_s, output reg [1:0] dec_ml,
    output reg [19:0] cnt0, output reg [19:0] cnt1, output reg [19:0] cnt2, output reg [19:0] cnt3,
    output reg [M2-1:0] sigma, output reg [1:0] frame, output reg [1:0] fr_ml,
    output reg [31:0] cyc_win,
    output wire arr, output wire [NS*EW-1:0] E_all, output wire [2*NS-1:0] C_all, output wire [NS*LW-1:0] lab_all
);
`ifdef PTW_SWP
    localparam integer SWP = 1;    // swap decision registered (swb_r), labels / moves / counts one cycle later; GAP_SW + 1
`else
    localparam integer SWP = 0;
`endif
    localparam integer TL = (NS <= 1) ? 1 : $clog2(NS);
    localparam integer NT = 1 << TL;
    localparam integer G0 = (LDC > 4) ? LDC : 4;
    localparam integer D = CSL ? 2 : 0;                // C-slow: the last write of a sweep lands 2 cycles after its S1
    localparam integer CMI = TL + 6 + D;               // commit index in the sweep-end pipeline
    localparam integer SEL = CMI + 1;

    // ---------------------------------------------------------------- window input, start state
    reg [TR*M2-1:0] syn_r; reg [M2-1:0] synp_r; reg fin_r;
    wire [NLA-1:0] lit0; wire [KS*NLD-1:0] lw; wire [SVW-1:0] sv0; wire [NDW-1:0] nd0; wire [NMW-1:0] nm0;
    ptw_prep u_prep (.clk(clk), .fin(fin_r), .syn(syn_r), .synp(synp_r), .sig(sigma), .lit0(lit0), .lw(lw), .sv0(sv0),
                     .nd0(nd0), .nm0(nm0));
    wire [M2+1:0] sm0;
    ptw_summ u_sm0 (.sv(sv0), .fin(fin_r), .sm(sm0));

    // ---------------------------------------------------------------- schedule
    wire take = syn_valid & ready & ~busy;
    reg [7:0] pc;                                      // window preparation: 1 = t_s + 1, ...; 0 = off  (8 bits: LDC = rs k reaches 66 at d = 11, rs = 6)
    wire ldn = (pc >= 8'd3) && (pc < 8'd3 + LDC);      // next cycle is a load cycle
    wire [7:0] lci = pc - 8'd3;                        // load cycle: local replica lci / KS, slab lci % KS
    wire [AW_-1:0] lan = (RS > 1) ? AW_'(((lci / KS) << AWS) | (lci % KS)) : AW_'(lci);
    reg [7:0] wsel;                                    // LUTRAM word presented in this load cycle (KS:1 select of lw)
    wire [NLD-1:0] lbus = (LDC > 1) ? lw[wsel*NLD +: NLD] : lit0[NLD-1:0];
    reg ldL;                                           // first load cycle (labels, counters reset)
    reg f_run, f_last, f_swp, f_end, f_b0, f_par, f_pr;
    (* max_fanout = 32 *) reg f_mv;
    (* max_fanout = 64 *) reg f_tab;
    reg [31:0] f_s; reg [7:0] f_c, f_j, f_col, f_slab, f_rc, f_r; reg [7:0] f_gap;
    wire lastc = (f_c == NC - 1) && (f_j == KS - 1) && (f_rc == RS - 1);
    wire swsw = (((f_s + 1) % KSW) == 0);
    wire f_mv_n = ~rst & ~take & f_run & (f_gap == 0);
    reg m1_mv, m1_last, m1_swp, m1_end, m1_b0, m1_pr; reg [7:0] m1_col, m1_slab, m1_r; (* max_fanout = 64 *) reg m1_tab;
    reg m2_mv, m2_last, m2_swp, m2_end, m2_b0, m2_pr;
    always @(posedge clk) begin
        if (rst) begin f_run <= 0; f_mv <= 0; busy <= 0; pc <= 0; end
        else begin
            f_mv <= 0; f_last <= 0; f_end <= 0;
            pc <= (pc != 0 && pc < 8'd3 + LDC + 8'd2) ? pc + 8'd1 : 8'd0;
            if (take) begin
                syn_r <= syn; synp_r <= synp; fin_r <= fin; busy <= 1; pc <= 8'd1;
                f_run <= 1; f_s <= 0; f_c <= 0; f_j <= 0; f_rc <= 0; f_gap <= G0; f_par <= 0;
            end else if (f_run) begin
                if (f_gap != 0) f_gap <= f_gap - 1;
                else begin
                    f_mv <= 1; f_col <= f_c; f_slab <= f_j; f_r <= f_rc; f_tab <= TAB[f_c]; f_last <= lastc; f_swp <= swsw;
                    f_b0 <= ((f_s & (f_s - 1)) == 0); f_end <= lastc && (f_s == nsw - 1); f_pr <= f_par;
                    // order: local replica fastest (interleaved replicas; the C-slow loop needs it), then color, then slab
                    if (f_rc == RS - 1) begin
                        f_rc <= 0;
                        if (f_c == NC - 1) begin
                            f_c <= 0;
                            if (f_j == KS - 1) begin
                                f_j <= 0;
                                if (f_s == nsw - 1) f_run <= 0;
                                else begin
                                    f_s <= f_s + 1;
                                    if (swsw) begin f_gap <= GAP_SW; f_par <= ~f_par; end
                                end
                            end else f_j <= f_j + 1;
                        end else f_c <= f_c + 1;
                    end else f_rc <= f_rc + 1;
                end
            end
            if (done) busy <= 0;
        end
        wsel <= (pc == 8'd3 || wsel == KS - 1) ? 8'd0 : wsel + 8'd1;
        ldL <= (pc == 8'd3) & ~rst;
        m1_mv <= f_mv & ~rst; m1_col <= f_col; m1_slab <= f_slab; m1_r <= f_r; m1_tab <= f_tab; m1_last <= f_last; m1_swp <= f_swp; m1_end <= f_end;
        m1_b0 <= f_b0; m1_pr <= f_pr;
        m2_mv <= m1_mv & ~rst; m2_last <= m1_last; m2_swp <= m1_swp; m2_end <= m1_end; m2_b0 <= m1_b0; m2_pr <= m1_pr;
    end
    wire [NC-1:0] coln = {{(NC-1){1'b0}}, 1'b1} << m1_col;
    wire [KS-1:0] slabn = {{(KS-1){1'b0}}, 1'b1} << m1_slab;

    // sweep-end pipeline: se[i] = cycle c0 + 1 + i after the last move c0 of a sweep
    reg [SEL:0] se, se_sw, se_end, se_b0, se_pr;
    always @(posedge clk) begin
        se <= {se[SEL-1:0], m2_mv & m2_last & ~rst}; se_sw <= {se_sw[SEL-1:0], m2_swp}; se_end <= {se_end[SEL-1:0], m2_end};
        se_b0 <= {se_b0[SEL-1:0], m2_b0}; se_pr <= {se_pr[SEL-1:0], m2_pr};
    end
    assign arr = se[5 + D + SWP];

    // ---------------------------------------------------------------- replicas
    reg [NS-1:0] loh [0:NS-1];                        // one-hot temperature label per replica
    reg [LW-1:0] lab [0:NS-1];
    always_comb begin : enc
        integer t, b;
        for (t = 0; t < NS; t = t + 1) begin
            lab[t] = {LW{1'b0}};
            for (b = 0; b < NS; b = b + 1) if (loh[t][b]) lab[t] = lab[t] | b[LW-1:0];
        end
    end
    reg [LW-1:0] hold [0:NS-1];
    reg sw_dec, sw_par, sw_dec2;
    reg [NP-1:0] swb_r;
    (* max_fanout = 24 *) reg [NS-1:0] mvr;
    reg rnd_r;
    wire [NS-1:0] swx;
    wire [NDW-1:0] nd [0:NS-1]; wire [NMW-1:0] nm [0:NS-1]; wire [1:0] cls [0:NS-1];
    wire [M2+1:0] sm [0:NS-1]; wire [SVW-1:0] sv [0:NS-1];
    reg [EW-1:0] pd [0:NS-1], pm [0:NS-1];
    wire [EW-1:0] Ew [0:NS-1];
    reg [EW-1:0] Es [0:NS-1]; reg [1:0] cl_st [0:NS-1]; reg [M2+1:0] sm_st [0:NS-1];
    reg [ROWW-1:0] twd_r; reg [LW-1:0] twa_r; reg twe1;
    always @(posedge clk) begin
        twe1 <= tw_en & ~rst;
        if (tw_en) begin twd_r <= tw_data; twa_r <= tw_addr; end
    end
    genvar gi, ge;
    generate for (gi = 0; gi < NS; gi = gi + 1) begin : g_rep
        wire [ROWW-1:0] thr;
        (* keep = "true" *) reg twe_g, ldt_g; (* keep = "true" *) reg [LW-1:0] twa_g;
        always @(posedge clk) begin twe_g <= twe1 & ~rst; twa_g <= twa_r; ldt_g <= (pc == 8'd3) & ~rst; end
        pt_thr #(.NS(NS), .LW(LW), .W(ROWW), .RI(gi), .INIT(THR_INIT)) u_thr (.clk(clk), .we(twe_g), .wa(twa_g), .wd(twd_r),
                 .ld(ldt_g), .rnd(rnd_r), .mv(mvr[gi]), .thr_e(thr));
        ptw_summ u_sm (.sv(sv[gi]), .fin(fin_r), .sm(sm[gi]));
        always @(posedge clk) begin pd[gi] <= nd[gi] * wq_d; pm[gi] <= nm[gi] * wq_m; end
        assign Ew[gi] = pd[gi] + pm[gi];
        always @(posedge clk) begin
            if (se[D]) sm_st[gi] <= sm[gi];
            if (se[1 + D]) cl_st[gi] <= cls[gi];
            if (se[3 + D]) Es[gi] <= Ew[gi];
        end
        assign E_all[gi*EW +: EW] = Es[gi]; assign C_all[2*gi +: 2] = cl_st[gi]; assign lab_all[gi*LW +: LW] = lab[gi];
    end endgenerate
    // engines: replicas e RS .. e RS + RS - 1 share one set of clause units (LUTRAM words {r, slab}), neurons and lanes;
    // the sweep is replica-major inside an engine (replicas are independent within a sweep)
    generate for (ge = 0; ge < NE; ge = ge + 1) begin : g_eng
        // comparator row of the item's replica and color table (one select per engine, off the loop): SL slots = SL levels
        // 1..KSL, T slots 1..7 = T2 levels, 8..12 = T1 levels; unused slots 0 (never below)
        // combinational from the threshold store's row registers (as pt_top's comparators read pt_thr), selected by the table and
        // replica of the comparator cycle (m1_*): no register stage between a label swap and the comparators
        (* max_fanout = 24 *) reg [NSLOT*B-1:0] tr;      // comparator row: replicated (fan-out to every lane, d = 11 path)
        integer s;
        wire [ROWW-1:0] tt [0:RS-1];
        genvar gx;
        for (gx = 0; gx < RS; gx = gx + 1) begin : g_x
            assign tt[gx] = g_rep[ge * RS + gx].thr;
        end
        wire [ROWW-1:0] thx = (RS > 1) ? tt[m1_r[RB-1:0]] : tt[0];
        // registered (round 3: the combinational row of round 2 was the routed critical path at d = 5, mvr -> comparators):
        // valid in the comparator cycle; one cycle later with the C-slow loop (its comparators serve stage S2)
        wire tb_ = CSL ? m1_tab : f_tab;
        wire [7:0] rsel = CSL ? m1_r : f_r;
        wire [ROWW-1:0] thx2 = (RS > 1) ? tt[rsel[RB-1:0]] : tt[0];
        always @(posedge clk)
            for (s = 1; s <= NSLOT; s = s + 1)
                tr[(s-1)*B +: B] <= tb_ ? ((s <= 12) ? thx2[(KSL+s-1)*B +: B] : {B{1'b0}})
                                        : ((s <= KSL) ? thx2[(s-1)*B +: B] : {B{1'b0}});
        wire [RS*SVW-1:0] svw; wire [RS*NDW-1:0] ndw; wire [RS*NMW-1:0] nmw; wire [2*RS-1:0] clw;
        ptw_rep u_rep (.clk(clk), .ldn(ldn & ~rst), .lan(lan), .lit0(lbus), .mvn(m1_mv & ~rst), .coln(coln), .slabn(slabn),
                       .slabbn(AWS'(m1_slab)), .rn(RB'(m1_r)), .fin(fin_r), .tmode(tmode), .advn(f_mv_n), .ran(RB'(f_rc)),
                       .sd_sh(sd_sh), .sd_bit(sd_bit[ge*NB +: NB]), .push_a(push_a), .push_b(push_b), .psel(psel), .tr(tr),
                       .nd0(nd0), .nm0(nm0), .nd_o(ndw), .nm_o(nmw), .cls_o(clw), .sv(svw));
        genvar gy;
        for (gy = 0; gy < RS; gy = gy + 1) begin : g_y
            assign nd[ge*RS + gy] = ndw[gy*NDW +: NDW]; assign nm[ge*RS + gy] = nmw[gy*NMW +: NMW];
            assign cls[ge*RS + gy] = clw[2*gy +: 2]; assign sv[ge*RS + gy] = svw[gy*SVW +: SVW];
        end
    end endgenerate

    // ---------------------------------------------------------------- swap unit (pt_top's, energies E = Wq nd + wq nm)
    reg [XIW-1:0] xi [0:NP-1];
    reg [NP-1:0] need, infl;
    reg [EW-1:0] es [0:NS-1];
    reg [NP-1:0] par0;
    integer t0;
    always_comb for (t0 = 0; t0 < NP; t0 = t0 + 1) par0[t0] = (t0 % 2 == 0);
    wire [NP-1:0] act = sw_par ? ~par0 : par0;
    reg [NP-1:0] swb;
    wire [NP-1:0] swe = SWP ? swb_r : swb;                // decision applied (registered with SWP: es -> mvr was the d = 11 path)
    wire sw_ap = SWP ? sw_dec2 : sw_dec;                  // cycle in which labels move
    reg [LW-1:0] h2 [0:NS-1];
    reg [NS-1:0] o2 [0:NS-1];
    always_comb begin : dec
        integer t;
        reg signed [EW+1:0] D;
        for (t = 0; t < NS - 1; t = t + 1) begin
            D = $signed({2'b00, es[t + 1]}) - $signed({2'b00, es[t]});
            swb[t] = act[t] && (D <= $signed({2'b00, xi[t]}));
        end
        if (NS == 1) swb = 1'b0;
        for (t = 0; t < NS; t = t + 1)
            h2[t] = (t < NS - 1 && swe[t]) ? hold[t + 1] : (t > 0 && swe[t - 1]) ? hold[t - 1] : hold[t];
        for (t = 0; t < NS; t = t + 1) begin : oh
            integer b;
            for (b = 0; b < NS; b = b + 1)
                o2[t][b] = (loh[t][b] & ~(b < NS - 1 && swe[b]) & ~(b > 0 && swe[b - 1]))
                         | (b > 0 && loh[t][b - 1] && swe[b - 1]) | (b < NS - 1 && loh[t][b + 1] && swe[b]);
        end
    end
    assign swx = {1'b0, swe} | {swe, 1'b0};
    integer i;
    always @(posedge clk) begin
        sw_dec <= 0;
        sw_dec2 <= sw_dec & ~rst; swb_r <= swb;
        rnd_r <= sw_ap & ~rst;
        for (i = 0; i < NS; i = i + 1) mvr[i] <= sw_ap & ~rst & |(loh[i] & swx);
        if (se[3 + D] && se_sw[3 + D]) begin for (i = 0; i < NS; i = i + 1) es[i] <= Ew[hold[i]]; sw_dec <= 1; sw_par <= se_pr[3 + D]; end
        if (ldL) begin for (i = 0; i < NS; i = i + 1) begin loh[i] <= {{(NS-1){1'b0}}, 1'b1} << i; hold[i] <= i; end end
        else if (sw_ap) begin
            for (i = 0; i < NS; i = i + 1) begin hold[i] <= h2[i]; loh[i] <= o2[i]; end
        end
    end
    reg [TW-1:0] th_r [0:NP-1];
    integer ti;
    initial for (ti = 0; ti < NP; ti = ti + 1) th_r[ti] = THETA_INIT[ti*TW +: TW];
    always @(posedge clk) if (th_en) th_r[th_addr] <= th_data;
    reg [USW-1:0] ix [0:NP-1];
    reg [NP-1:0] rcm;
    wire [31:0] xo [0:NP-1];
    // registered top USW bits of every pair's output (round 4: the 13-level xoshiro multiply -> pair mux -> ix LUTRAM path was the
    // routed critical path at d = 9). A pair is never stepped in two consecutive cycles (infl blocks it for 6 cycles) and
    // fill follows the seeding by >= 1 cycle, so xq equals the current output whenever a pick reads it.
    reg [USW-1:0] xq [0:NP-1];
    integer tq;
    always @(posedge clk) for (tq = 0; tq < NP; tq = tq + 1) xq[tq] <= xo[tq][31 -: USW];
    reg [NP-1:0] stp, pkv;
    generate for (gi = 0; gi < NP; gi = gi + 1) begin : g_sw
        pt_xoshiro u_xo (.clk(clk), .seed_sh(ss_sh), .seed_bit(ss_bit[gi]), .step(stp[gi]), .out(xo[gi]));
    end endgenerate
    localparam integer RWC = 2 * (KS * NC * KSW + GAP_SW) - 7;
    localparam integer NH = (NP + 1) / 2;
    localparam integer NRF = (NH + RWC / 2 - 1) / (RWC / 2);      // both parities share a lane (as pt_top)
    reg [LW-1:0] p0t [0:NRF-1], p1t [0:NRF-1], p2t [0:NRF-1], p3t [0:NRF-1], p4t [0:NRF-1], p5t [0:NRF-1];
    reg [NRF-1:0] p0v, p0r, p1v, p2v, p3v, p4v, p5v;
    reg [LW-1:0] pick [0:NRF-1]; reg [NRF-1:0] pv;
    wire [NP-1:0] cand = need & ~infl;
    always_comb begin : pk
        integer t, l;
        pkv = {NP{1'b0}};
        for (l = 0; l < NRF; l = l + 1) begin
            pick[l] = 0; pv[l] = 0;
            for (t = NP - 1; t >= 0; t = t - 1) if (cand[t] && ((t / 2) % NRF) == l) begin pv[l] = 1; pick[l] = t; end
            if (pv[l] && !rst) pkv[pick[l]] = 1'b1;
        end
        stp = {NP{1'b0}};
        for (l = 0; l < NRF; l = l + 1) if (p0v[l] && !p0r[l] && !rst) stp[p0t[l]] = 1'b1;
    end
    wire [NP-1:0] act_now = sw_dec ? act : {NP{1'b0}};
    reg [19:0] a_r [0:NRF-1]; reg [TW-1:0] b_r [0:NRF-1]; reg [63:0] m_r [0:NRF-1]; reg [63:0] prod [0:NRF-1];
    reg [USW-1:0] addr [0:NRF-1];
    wire [19:0] Lq [0:NRF-1];
    generate for (gi = 0; gi < NRF; gi = gi + 1) begin : g_rf
        tp_lut_rom #(.U(USW), .LW(20)) u_lut (.clk(clk), .en(1'b1), .addr(addr[gi]), .dout(Lq[gi]));
    end endgenerate
    reg [NP-1:0] done3;
    always_comb begin : d3
        integer l;
        done3 = {NP{1'b0}};
        for (l = 0; l < NRF; l = l + 1) if (p5v[l]) done3[p5t[l]] = 1'b1;
    end
    always @(posedge clk) begin : rf
        integer l;
        if (rst) begin need <= 0; infl <= 0; rcm <= 0; p0v <= 0; p1v <= 0; p2v <= 0; p3v <= 0; p4v <= 0; p5v <= 0; end
        else begin
            need <= (need & ~pkv) | act_now | ((fill | recalc) ? {NP{1'b1}} : {NP{1'b0}});
            infl <= (infl | pkv) & ~done3;
            rcm <= recalc ? {NP{1'b1}} : (rcm & ~pkv);
            p0v <= pv; p1v <= p0v; p2v <= p1v; p3v <= p2v; p4v <= p3v; p5v <= p4v;
            for (l = 0; l < NRF; l = l + 1) begin
                p0t[l] <= pick[l]; p0r[l] <= rcm[pick[l]];
                p1t[l] <= p0t[l];
                if (p0v[l]) begin
                    addr[l] <= p0r[l] ? ix[p0t[l]] : xq[p0t[l]];
                    ix[p0t[l]] <= p0r[l] ? ix[p0t[l]] : xq[p0t[l]];
                end
                p2t[l] <= p1t[l];
                p3t[l] <= p2t[l]; a_r[l] <= Lq[l]; b_r[l] <= th_r[p2t[l]];
                p4t[l] <= p3t[l]; m_r[l] <= b_r[l] * a_r[l];
                p5t[l] <= p4t[l]; prod[l] <= m_r[l];
                if (p5v[l]) xi[p5t[l]] <= ((prod[l] >> SH) > XIMAX) ? XIMAX[XIW-1:0] : prod[l][SH +: XIW];
            end
        end
    end
    assign ready = ~busy && need == 0 && infl == 0 && p0v == 0 && p1v == 0 && p2v == 0 && p3v == 0 && p4v == 0 && p5v == 0;

    // ---------------------------------------------------------------- global best: argmin tree over the sweep-end energies
    // (ties: lowest replica), one register per level, then the strict comparison with the best so far
    wire [EW-1:0] te [0:TL][0:NT-1];
    wire [LW-1:0] tx [0:TL][0:NT-1];
    genvar gl, gx;
    generate
        for (gx = 0; gx < NT; gx = gx + 1) begin : g_leaf
            assign te[0][gx] = (gx < NS) ? Es[gx] : {EW{1'b1}};
            assign tx[0][gx] = (gx < NS) ? LW'(gx) : {LW{1'b0}};
        end
        for (gl = 1; gl <= TL; gl = gl + 1) begin : g_lv
            for (gx = 0; gx < (NT >> gl); gx = gx + 1) begin : g_nd
                wire s = te[gl-1][2*gx+1] < te[gl-1][2*gx];
                reg [EW-1:0] er; reg [LW-1:0] xr;
                always @(posedge clk) begin er <= s ? te[gl-1][2*gx+1] : te[gl-1][2*gx]; xr <= s ? tx[gl-1][2*gx+1] : tx[gl-1][2*gx]; end
                assign te[gl][gx] = er; assign tx[gl][gx] = xr;
            end
        end
    endgenerate
    reg [EW-1:0] pd0, pm0;
    wire [EW-1:0] e0w = pd0 + pm0;
    always @(posedge clk) begin pd0 <= nd0 * wq_d; pm0 <= nm0 * wq_m; end

    // ---------------------------------------------------------------- counts, commit
    wire [1:0] c0h = cl_st[hold[0]];
    wire [EW-1:0] bw = te[TL][0];
    wire [LW-1:0] bx = tx[TL][0];
    reg [1:0] mlc;
    always_comb begin
        if (cnt0 >= cnt1 && cnt0 >= cnt2 && cnt0 >= cnt3) mlc = 2'd0;
        else if (cnt1 >= cnt2 && cnt1 >= cnt3) mlc = 2'd1;
        else if (cnt2 >= cnt3) mlc = 2'd2;
        else mlc = 2'd3;
    end
    wire [1:0] frn = frame ^ best_s[M2+1:M2];
    always @(posedge clk) begin
        done <= 0;
        if (rst) begin sigma <= 0; frame <= 0; end
        else begin
            if (clr && !busy) begin sigma <= 0; frame <= 0; end
            if (take) cyc_win <= 0; else if (busy) cyc_win <= cyc_win + 1;
            if (ldL) begin cnt0 <= 0; cnt1 <= 0; cnt2 <= 0; cnt3 <= 0; end
            if (pc == 8'd5) begin best_e <= e0w; best_s <= sm0; best_c <= 2'd0; end
            if (se[5 + D + SWP]) begin
                cnt0 <= (se_b0[5 + D + SWP] ? 20'd0 : cnt0) + (c0h == 2'd0);
                cnt1 <= (se_b0[5 + D + SWP] ? 20'd0 : cnt1) + (c0h == 2'd1);
                cnt2 <= (se_b0[5 + D + SWP] ? 20'd0 : cnt2) + (c0h == 2'd2);
                cnt3 <= (se_b0[5 + D + SWP] ? 20'd0 : cnt3) + (c0h == 2'd3);
            end
            if (se[4 + TL + D] && bw < best_e) begin best_e <= bw; best_s <= sm_st[bx]; best_c <= cl_st[bx]; end
            if (se[CMI] && se_end[CMI]) begin
                dec_ml <= mlc;
                frame <= frn;
                if (fin_r) fr_ml <= frn ^ (best_c ^ mlc);
                else sigma <= syn_r[(CM-1)*M2 +: M2] ^ best_s[M2-1:0];
                done <= 1;
            end
        end
    end
endmodule

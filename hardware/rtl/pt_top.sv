// One-syndrome-per-shot top: parallel-tempering maximum-likelihood decoder of the rotated surface code.
// Bit-exact with nimbus.decoding.one_shot. Instance constants come from the generated pt_core.sv (pt_rep, pt_c0) and
// pt_params.svh (hardware/gen/gen_pt.py). The lane thresholds T[t][k] (pt_thr, one store per replica) and the swap constants theta_t
// are runtime registers (tw_* / th_* ports; write them only while idle). After changing theta, pulse recalc: it recomputes the
// pre-drawn swap thresholds from the same uniforms (no generator step), so the random streams stay those of the golden model.
//
// Timing (t_s = cycle in which the syndrome is taken): replicas load c0 at t_s + 1, thresholds of the reset labels at t_s + 2,
// first move cycle t_s + 4. The schedule register f_* describes cycle t + 2 so that the lane generators advance two cycles
// ahead of the move (u registered at t + 1, comparators at t + 2). A sweep is NCL move cycles. After a swap sweep (last move
// cycle c0) the energies of the post-move states arrive at c0 + 3 (es captured), the decision at c0 + 4 (labels), the label
// thresholds load into the registers and, bypassed, into the comparators at c0 + 5; the next move is at c0 + 6 (GAP = 5 idle
// cycles; 8 at an early-stop checkpoint, so that the two-stage stop test sees the last count). After the last sweep:
// last count at c_end + 3, ML decision at c_end + 4, minimum-energy scan over NS replicas, done.
`include "pt_params.svh"
module pt_top (
    input  wire clk, input wire rst,
    input  wire sd_sh, input wire push_a, input wire push_b, input wire ss_sh,      // generator seeding (see pt_rep)
    input  wire [NS-1:0] sd_bit, input wire [NP-1:0] ss_bit,
    input  wire fill,                                  // after seeding: compute the first swap thresholds
    input  wire recalc,                                // after a theta change: recompute them from the saved uniforms
    input  wire tw_en, input wire [LW-1:0] tw_addr, input wire [KMB-1:0] tw_data,   // threshold row write (label tw_addr)
    input  wire th_en, input wire [LW-1:0] th_addr, input wire [TW-1:0] th_data,    // swap constant write (pair th_addr)
    input  wire [31:0] nsw,                            // sweeps per shot (power of two, >= 2; the cap when stopping early)
    input  wire [31:0] es_gmin, input wire [5:0] es_th, // early stop: from es_gmin sweeps (power of two, multiple of KSW;
                                                       // 0 = off) stop when 32 * max(count) >= es_th * total of the block
    input  wire syn_valid, input wire [M-1:0] syn,
    output wire ready, output reg busy, output reg done,
    output reg [1:0] dec_ml, output reg [1:0] dec_me, output reg [EW-1:0] dec_e,
    output reg [19:0] cnt0, output reg [19:0] cnt1, output reg [19:0] cnt2, output reg [19:0] cnt3,
    output reg [31:0] cyc_start, output reg [31:0] cyc_run, output reg [31:0] sweeps_run,
    output wire arr, output wire [NS*EW-1:0] E_all, output wire [2*NS-1:0] C_all, output wire [NS*LW-1:0] lab_all
);
    // ---------------------------------------------------------------- syndrome, start state
    reg [M-1:0] sr;
    wire [N-1:0] c0x, c0z; wire [1:0] cpar;
    pt_c0 u_c0 (.sr(sr), .c0x(c0x), .c0z(c0z), .cpar(cpar));

    // ---------------------------------------------------------------- schedule
    reg ld, ld0;                        // ld0: cycle after take (start state computed); ld: replicas load (one cycle later)
    reg f_run; reg [31:0] f_s; reg [2:0] f_k; reg [3:0] f_gap; reg f_par;
    (* max_fanout = 24 *) reg f_mv;                                      // item for cycle t + 2 (fans out to every generator)
    reg [2:0] f_col; reg f_swp, f_b0, f_end, f_pr, chk, es_fin, stop_r;
    reg m1_mv; (* max_fanout = 24 *) reg m2_mv; reg [2:0] m1_col; reg m1_swp, m2_swp, m1_b0, m2_b0, m1_end, m2_end, m1_pr, m2_pr;
    wire [31:0] s1 = f_s + 1;
    wire ckp = (es_gmin != 0) && (s1 >= es_gmin) && ((s1 & f_s) == 0);
    (* max_fanout = 24 *) reg [NCL-1:0] col;                             // one-hot color of the move cycle (0 when idle)
    wire take = syn_valid & ready & ~busy;
    // next value of f_mv (= rng_adv), so that every replica can hold its own registered copy (fanout)
    wire f_mv_n = ~rst & ~take & f_run & (f_gap == 0) & ~(chk & stop_r);
    always @(posedge clk) begin
        if (rst) begin f_run <= 0; f_mv <= 0; f_swp <= 0; f_end <= 0; ld <= 0; ld0 <= 0; busy <= 0; chk <= 0; es_fin <= 0; end
        else begin
            ld0 <= take; ld <= ld0;
            f_mv <= 0; f_swp <= 0; f_end <= 0; es_fin <= 0;
            if (take) begin sr <= syn; busy <= 1; f_run <= 1; f_s <= 0; f_k <= 0; f_gap <= 1; f_par <= 0; chk <= 0; end   // one idle cycle: the start state is registered per replica
            else if (f_run) begin
                if (f_gap != 0) f_gap <= f_gap - 1;
                else if (chk && stop_r) begin f_run <= 0; chk <= 0; es_fin <= 1; sweeps_run <= f_s; end   // stop after the swap round
                else begin
                    chk <= 0;
                    f_mv <= 1; f_col <= f_k; f_b0 <= (f_k == 0) && ((f_s == (nsw >> 1)) || (((f_s & (f_s - 1)) == 0) && (f_s < (nsw >> 1)))); f_pr <= f_par;   // first cycle of a block: powers of two below nsw / 2, and nsw / 2 (the last block is the second half of the run)
                    if (f_k == NCL - 1) begin
                        f_k <= 0;
                        if (f_s == nsw - 1) begin f_end <= 1; f_run <= 0; sweeps_run <= nsw; end
                        else begin
                            f_s <= s1;
                            // checkpoint (s1 a power of two): three more gap cycles so that the stop test sees the final counts
                            // (two are not enough: the last count of the block would miss the test)
                            // (the gap at a checkpoint is GAP_CK cycles in total, fixed by the arrival of the last count)
                            if ((s1 % KSW) == 0) begin f_swp <= 1; f_gap <= ckp ? 4'(GAP_CK) : 4'(GAP); f_par <= ~f_par; end
                            if (ckp) chk <= 1;
                        end
                    end else f_k <= f_k + 1;
                end
            end
            if (done) busy <= 0;
        end
        m1_mv <= f_mv & ~rst; m1_col <= f_col; m1_swp <= f_swp; m1_b0 <= f_b0; m1_end <= f_end; m1_pr <= f_pr;
        m2_mv <= m1_mv & ~rst; col <= (m1_mv & ~rst) ? ({{(NCL-1){1'b0}}, 1'b1} << m1_col) : {NCL{1'b0}}; m2_swp <= m1_swp; m2_b0 <= m1_b0; m2_end <= m1_end; m2_pr <= m1_pr;
    end
    wire rng_adv = f_mv;
    wire mv = m2_mv;

    // ---------------------------------------------------------------- replicas
    reg [NS-1:0] loh [0:NS-1];          // temperature label of each replica, one-hot (binary lab only for observation)
    reg [LW-1:0] lab [0:NS-1];
    always_comb begin : enc
        integer t, b;
        for (t = 0; t < NS; t = t + 1) begin
            lab[t] = {LW{1'b0}};
            for (b = 0; b < NS; b = b + 1) if (loh[t][b]) lab[t] = lab[t] | b[LW-1:0];
        end
    end
    reg [LW-1:0] hold [0:NS-1];
    reg sw_dec, sw_par;
    (* max_fanout = 24 *) reg [NS-1:0] mvr;                   // per replica: label changed in this swap round (valid in the threshold-load cycle)
    reg rnd_r;                          // threshold-load cycle of every swap round
    wire [NS-1:0] swx;                  // per label: label t changes (pair t or pair t - 1 swaps)
    wire [EW-1:0] E [0:NS-1];
    wire [EW-1:0] Ew [0:NS-1];          // energy one cycle before E (combinational sum of the clause-count stage)
    wire [1:0] C [0:NS-1];
    // table write: row and address captured at tw_en, written two cycles later by per-replica strobe copies. twd_r is a
    // two-cycle path (set_multicycle_path in the board build): commits must be >= 2 cycles apart.
    reg [KMB-1:0] twd_r; reg [LW-1:0] twa_r; reg twe1;
    always @(posedge clk) begin
        twe1 <= tw_en & ~rst;
        if (tw_en) begin twd_r <= tw_data; twa_r <= tw_addr; end
    end
    genvar gi;
    generate for (gi = 0; gi < NS; gi = gi + 1) begin : g_rep
        wire [KMB-1:0] thr;
        (* keep = "true" *) reg twe_g; (* keep = "true" *) reg [LW-1:0] twa_g;   // local copy of the table write strobe
        always @(posedge clk) begin twe_g <= twe1 & ~rst; twa_g <= twa_r; end
        // per-replica copies of the load strobe (same timing as ld) and of the seeding strobes + seed bit (one cycle later,
        // all together): at large NS these nets reach every clause and generator of every replica
        (* keep = "true" *) reg ld_g, sdsh_g, pa_g, pb_g, sdb_g;
        always @(posedge clk) begin
            ld_g <= ld0 & ~rst; sdsh_g <= sd_sh; pa_g <= push_a; pb_g <= push_b; sdb_g <= sd_bit[gi];
        end
        pt_thr #(.NS(NS), .LW(LW), .W(KMB), .RI(gi), .INIT(THR_INIT)) u_thr (.clk(clk), .we(twe_g), .wa(twa_g), .wd(twd_r),
                 .ld(ld_g), .rnd(rnd_r), .mv(mvr[gi]), .thr_e(thr));
        // per-replica copies of the schedule enables (the shared ones were the routed critical path at d = 11)
        (* keep = "true" *) reg adv_g, mv_g; (* keep = "true" *) reg [NCL-1:0] col_g;
        always @(posedge clk) begin
            adv_g <= f_mv_n; mv_g <= m1_mv & ~rst;
            col_g <= (m1_mv & ~rst) ? ({{(NCL-1){1'b0}}, 1'b1} << m1_col) : {NCL{1'b0}};
        end
        // per-replica copy of the start state (a broadcast from the shared XOR network pulled every clause register toward it)
        (* keep = "true" *) reg [N-1:0] c0x_g, c0z_g; (* keep = "true" *) reg [1:0] cpar_g;
        always @(posedge clk) begin c0x_g <= c0x; c0z_g <= c0z; cpar_g <= cpar; end
        pt_rep u_rep (.clk(clk), .ld(ld_g), .c0x(c0x_g), .c0z(c0z_g), .cpar(cpar_g), .sd_sh(sdsh_g), .sd_bit(sdb_g), .push_a(pa_g),
                      .push_b(pb_g), .rng_adv(adv_g), .thr_e(thr), .mv(mv_g), .col(col_g),
                      .E_o(E[gi]), .E_w(Ew[gi]), .C_o(C[gi]));
        assign E_all[gi*EW +: EW] = E[gi]; assign C_all[2*gi +: 2] = C[gi]; assign lab_all[gi*LW +: LW] = lab[gi];
    end endgenerate

    // ---------------------------------------------------------------- arrival tags (the state after cycle c reaches E/C at c + 3)
    reg [2:0] t_mv, t_best, t_b0, t_swp, t_end, t_pr;
    always @(posedge clk) begin
        t_mv <= {t_mv[1:0], mv}; t_best <= {t_best[1:0], mv | ld}; t_b0 <= {t_b0[1:0], mv & m2_b0};
        t_swp <= {t_swp[1:0], mv & m2_swp}; t_end <= {t_end[1:0], mv & m2_end}; t_pr <= {t_pr[1:0], m2_pr};
    end
    assign arr = t_mv[2];

    // ---------------------------------------------------------------- per-replica best, occupancy of label 0 per block
    // block k = sweeps [2^(k-1), 2^k): the counts restart at the first move cycle of sweeps 0, 1, 2, 4, 8, ...
    reg [EW-1:0] bE [0:NS-1]; reg [1:0] bC [0:NS-1];
    integer i;
    always @(posedge clk) begin
        if (ld) begin for (i = 0; i < NS; i = i + 1) bE[i] <= {EW{1'b1}}; cnt0 <= 0; cnt1 <= 0; cnt2 <= 0; cnt3 <= 0; end
        else begin
            if (t_best[2]) for (i = 0; i < NS; i = i + 1) if (E[i] < bE[i]) begin bE[i] <= E[i]; bC[i] <= C[i]; end
            if (t_mv[2]) begin
                cnt0 <= (t_b0[2] ? 20'd0 : cnt0) + (C[hold[0]] == 2'd0);
                cnt1 <= (t_b0[2] ? 20'd0 : cnt1) + (C[hold[0]] == 2'd1);
                cnt2 <= (t_b0[2] ? 20'd0 : cnt2) + (C[hold[0]] == 2'd2);
                cnt3 <= (t_b0[2] ? 20'd0 : cnt3) + (C[hold[0]] == 2'd3);
            end
        end
    end

    // ---------------------------------------------------------------- swap unit
    reg [XIW-1:0] xi [0:NP-1];
    reg [NP-1:0] need, infl;
    reg [EW-1:0] es [0:NS-1];

    reg [NP-1:0] par0;
    integer t0;
    always_comb for (t0 = 0; t0 < NP; t0 = t0 + 1) par0[t0] = (t0 % 2 == 0);
    wire [NP-1:0] act = sw_par ? ~par0 : par0;
    // swap bit per pair, new holder per label, new one-hot label per replica (a shift by one: one LUT level after swb)
    reg [NP-1:0] swb;
    reg [LW-1:0] h2 [0:NS-1];
    reg [NS-1:0] o2 [0:NS-1];
    always_comb begin : dec
        integer t;
        reg signed [EW+1:0] D;
        for (t = 0; t < NS - 1; t = t + 1) begin
            D = $signed({2'b00, es[t + 1]}) - $signed({2'b00, es[t]});
            swb[t] = act[t] && (D <= $signed({2'b00, xi[t]}));
        end
        for (t = 0; t < NS; t = t + 1)
            h2[t] = (t < NS - 1 && swb[t]) ? hold[t + 1] : (t > 0 && swb[t - 1]) ? hold[t - 1] : hold[t];
        for (t = 0; t < NS; t = t + 1) begin : oh
            integer b;
            for (b = 0; b < NS; b = b + 1)
                o2[t][b] = (loh[t][b] & ~(b < NS - 1 && swb[b]) & ~(b > 0 && swb[b - 1]))
                         | (b > 0 && loh[t][b - 1] && swb[b - 1]) | (b < NS - 1 && loh[t][b + 1] && swb[b]);
        end
    end
    assign swx = {1'b0, swb} | {swb, 1'b0};
    always @(posedge clk) begin
        sw_dec <= 0;
        rnd_r <= sw_dec & ~rst;
        for (i = 0; i < NS; i = i + 1) mvr[i] <= sw_dec & ~rst & |(loh[i] & swx);
        if (t_swp[1]) begin for (i = 0; i < NS; i = i + 1) es[i] <= Ew[hold[i]]; sw_dec <= 1; sw_par <= t_pr[1]; end
        if (ld) begin for (i = 0; i < NS; i = i + 1) begin loh[i] <= {{(NS-1){1'b0}}, 1'b1} << i; hold[i] <= i; end end
        else if (sw_dec) begin
            for (i = 0; i < NS; i = i + 1) begin hold[i] <= h2[i]; loh[i] <= o2[i]; end
        end
    end
    // background refill of the swap thresholds: per pair, xi = (theta * L[u]) >> SH, saturated. The uniform index of every
    // pair is kept (ix), so that recalc can recompute xi after a theta change without stepping the generator.
    reg [TW-1:0] th_r [0:NP-1];
    integer ti;
    initial for (ti = 0; ti < NP; ti = ti + 1) th_r[ti] = THETA_INIT[ti*TW +: TW];
    always @(posedge clk) if (th_en) th_r[th_addr] <= th_data;
    reg [USW-1:0] ix [0:NP-1];
    reg [NP-1:0] rcm;                   // pairs still to recompute from ix
    wire [31:0] xo [0:NP-1];
    reg [NP-1:0] stp, pkv;
    generate for (gi = 0; gi < NP; gi = gi + 1) begin : g_sw
        pt_xoshiro u_xo (.clk(clk), .seed_sh(ss_sh), .seed_bit(ss_bit[gi]), .step(stp[gi]), .out(xo[gi]));
    end endgenerate
    // NRF refill lanes, so that a round's pairs are refilled before the next round of the same parity at any NS (one lane
    // serves pairs t with (t / 2) mod NRF = lane; each lane picks one pair per cycle, registers the pick (p0), then reads and
    // steps that pair's generator (p1), ROM, DSP: 6 stages). A lane serves pairs of both parities, so it gets up to
    // 2 ceil(NH / NRF) requests per RW cycles; with at most RW of them every request is served in time. (With
    // NRF = ceil(NH / RW) instead, one lane would serve 39 pairs at NS = 40, K = 2, and some swaps would use the previous xi.)
    localparam integer RW = 2 * (NCL * KSW + GAP) - 7;                     // cycles available per lane between same-parity rounds
    localparam integer NH = (NP + 1) / 2;
    localparam integer NRF = (NH + RW / 2 - 1) / (RW / 2);
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
                if (p0v[l]) begin                                           // the generator steps in this cycle (unless recomputing)
                    addr[l] <= p0r[l] ? ix[p0t[l]] : xo[p0t[l]][31 -: USW];
                    ix[p0t[l]] <= p0r[l] ? ix[p0t[l]] : xo[p0t[l]][31 -: USW];
                end
                p2t[l] <= p1t[l];                                           // Lq valid with p2
                p3t[l] <= p2t[l]; a_r[l] <= Lq[l]; b_r[l] <= th_r[p2t[l]];  // DSP input registers
                p4t[l] <= p3t[l]; m_r[l] <= b_r[l] * a_r[l];                // multiplier register
                p5t[l] <= p4t[l]; prod[l] <= m_r[l];                        // output register
                if (p5v[l]) xi[p5t[l]] <= ((prod[l] >> SH) > XIMAX) ? XIMAX[XIW-1:0] : prod[l][SH +: XIW];
            end
        end
    end
    assign ready = ~busy && need == 0 && infl == 0 && p0v == 0 && p1v == 0 && p2v == 0 && p3v == 0 && p4v == 0 && p5v == 0;

    // ---------------------------------------------------------------- early-stop test (valid when the swap gap ends)
    // two stages (max / total, then the comparison): the result reflects the counts two cycles earlier
    wire [19:0] mx01 = (cnt0 > cnt1) ? cnt0 : cnt1, mx23 = (cnt2 > cnt3) ? cnt2 : cnt3;
    reg [19:0] mx; reg [21:0] tot;
    always @(posedge clk) begin
        mx <= (mx01 > mx23) ? mx01 : mx23; tot <= cnt0 + cnt1 + cnt2 + cnt3;
        stop_r <= ({mx, 5'd0} >= es_th * tot);
    end

    // ---------------------------------------------------------------- minimum-energy decision: argmin tree over the per-replica
    // best energies (ties: lowest replica), registered every second level: TR = ceil(TL / 2) cycles instead of an NS-cycle scan
    localparam integer TL = (NS <= 1) ? 1 : $clog2(NS);
    localparam integer NT = 1 << TL;
    localparam integer TR = (TL + 1) / 2;
    wire [EW-1:0] te [0:TL][0:NT-1];
    wire [1:0] tc [0:TL][0:NT-1];
    genvar gl, gx;
    generate
        for (gx = 0; gx < NT; gx = gx + 1) begin : g_leaf
            assign te[0][gx] = (gx < NS) ? bE[gx] : {EW{1'b1}};
            assign tc[0][gx] = (gx < NS) ? bC[gx] : 2'd0;
        end
        for (gl = 1; gl <= TL; gl = gl + 1) begin : g_lv
            for (gx = 0; gx < (NT >> gl); gx = gx + 1) begin : g_nd
                localparam integer IL = 2 * gx, IR = 2 * gx + 1, LP = gl - 1;
                wire s = te[LP][IR] < te[LP][IL];
                wire [EW-1:0] ev = s ? te[LP][IR] : te[LP][IL];
                wire [1:0] cv = s ? tc[LP][IR] : tc[LP][IL];
                if ((gl % 2) == 0 || gl == TL) begin : g_r
                    reg [EW-1:0] er; reg [1:0] cr;
                    always @(posedge clk) begin er <= ev; cr <= cv; end
                    assign te[gl][gx] = er; assign tc[gl][gx] = cr;
                end else begin : g_c
                    assign te[gl][gx] = ev; assign tc[gl][gx] = cv;
                end
            end
        end
    endgenerate

    // ---------------------------------------------------------------- decisions, cycle counters
    reg [3:0] fin; reg [TR:0] tq; reg running;
    always @(posedge clk) begin
        done <= 0;
        if (rst) begin running <= 0; fin <= 0; tq <= 0; end
        else begin
            if (ld0) begin cyc_start <= 0; cyc_run <= 0; running <= 0; end
            else if (busy && !running && !mv) cyc_start <= cyc_start + 1;
            if (mv) running <= 1;
            if (running) cyc_run <= cyc_run + 1;
            fin <= {fin[2:0], t_end[2]};
            tq <= {tq[TR-1:0], fin[0] | es_fin};          // the best energies are final here; the tree needs TR cycles
            if (fin[0] | es_fin) begin
                if (cnt0 >= cnt1 && cnt0 >= cnt2 && cnt0 >= cnt3) dec_ml <= 0;     // argmax, ties: lowest class
                else if (cnt1 >= cnt2 && cnt1 >= cnt3) dec_ml <= 1;
                else if (cnt2 >= cnt3) dec_ml <= 2;
                else dec_ml <= 3;
            end
            if (tq[TR-1]) begin dec_me <= tc[TL][0]; dec_e <= te[TL][0]; done <= 1; running <= 0; end
        end
    end
endmodule

// Board top of the real-time decoder (hardware/README.md): AXI4-Lite slave,
// shot memory, programmable round timer, syndrome ring + window FSM (the streaming wrapper), result memory, ptw_top.
//
// Two clocks, as pt_board (one syndrome per shot): aclk (AXI, 100 MHz from the PS) runs the AXI4-Lite slave and port A of the two block
// RAMs; dclk (decoder, MMCM) runs everything else and every cycle counter. Crossings: dual-clock block RAMs and pt_board's
// request / acknowledge register tunnel (cdcA_* / cdcD_*, constrained by the build).
//
// Address map (byte offsets, AW = 20 bits = 1 MB):
//   0x00000-0x00FFF  registers (tunnel to dclk), see hardware/README.md
//   0x40000-0x7FFFF  result memory, 2^RES_AW 32-bit words, read only: window record j at words 16 j .. 16 j + 15
//   0x80000-0xFFFFF  syndrome memory, 2^SYN_AW words: round k of the batch (shot k / NR, round k % NR; NR = 2 TR + 1 rounds
//                    per shot, the last one the perfect round) at words k 2^WPL .. k 2^WPL + WPR - 1 (bit 32 w + j = bit j
//                    of the round, bit sector m + c: sector 0 = X, as ptw_top's syn bus)
//
// Streaming (one batch = NSHOT shots, their rounds back to back):
//   Round timer: with T_ROUND > 0, round k is due at T_START + (k + 1) T_ROUND + 1 decoder cycles; with T_ROUND = 0 every round
//   is due at once (throughput limit). A due round is released when the ring has room: its WPR words are read from the
//   shot memory and written as one row of the SYNDROME RING (RD = 3 CM + 1 rows: the window being assembled, 2 CM rounds,
//   the CM rounds arriving meanwhile and the perfect round). A round released after its due time (the ring was full: the
//   decoder fell behind) counts in LATE_ROUNDS.
//   Windows ((C, B) = (CM, CM), W = TR = 2 CM rounds; per shot two open windows and the final one, rounds [2 CM, 4 CM) +
//   the perfect round): window w of a shot is ready when its last round has been released (the trigger: round TR - 1,
//   TR + CM - 1, NR - 1). The FSM copies its rows from the ring (oldest row first; the oldest occupied row is always the
//   window's first round) into the window register, pulses clr before the first window of a shot (sigma = frame = 0), and
//   hands the window to ptw_top (syn_valid; the core latches it at take). At the take the window's committed rows are
//   freed (CM rows; all 2 CM + 1 rows of the final window), so the next window can be assembled while this one decodes.
//   Deadline: window w must commit before the trigger of window w + 1 (the next window needs its sigma). A trigger that
//   arrives while the previous window has not committed counts in MISSES (timed mode only).
//   Record per window (16 words): w0 {shot[15:0], 5'0, win[1:0], fin, best_c, dec_ml, frame, fr_ml (final windows; 0
//   otherwise)}, w1 best_e, w2-w5 cnt0-cnt3 (class counts of the last sweep block), w6-w9 sigma after the commit
//   (M2 <= 128 bits), w10-w11 trigger cycle (free-running counter), w12 take - trigger, w13 done - take, w14 cyc_win (core's
//   take-to-commit cycles), w15 trigger - due time of the trigger round (timed mode; 0 at T_ROUND = 0).
//   The generators are seeded by the host (serial ports, as in pt_board); they run on across windows, shots and batches until reseeded.
`include "ptw_params.svh"
module ptw_board #(parameter integer AW = 20, parameter integer SYN_AW = 17, parameter integer RES_AW = 16) (
    input  wire          aclk, input wire aresetn,
    input  wire [AW-1:0] s_axi_awaddr, input wire s_axi_awvalid, output reg s_axi_awready,
    input  wire [31:0]   s_axi_wdata, input wire [3:0] s_axi_wstrb, input wire s_axi_wvalid, output reg s_axi_wready,
    output reg  [1:0]    s_axi_bresp, output reg s_axi_bvalid, input wire s_axi_bready,
    input  wire [AW-1:0] s_axi_araddr, input wire s_axi_arvalid, output reg s_axi_arready,
    output reg  [31:0]   s_axi_rdata, output reg [1:0] s_axi_rresp, output reg s_axi_rvalid, input wire s_axi_rready,
    input  wire          dclk, input wire dresetn,
    output wire          irq
);
    localparam [31:0] ID = 32'h5054_5701;                      // 'PTW', board interface 1
    localparam integer WPR = (M2 + 31) / 32;                    // words per round
    localparam integer WPL = (WPR <= 1) ? 0 : $clog2(WPR);      // log2 of the per-round stride
    localparam integer NR = 2 * TR + 1;                         // rounds per shot (N = 2 TR noisy + the perfect round)
    localparam integer NWIN = 3;                                // windows per shot (two open, one final)
    localparam integer RD = 3 * CM + 1;                         // ring rows
    localparam integer RPW = (RD <= 2) ? 1 : $clog2(RD);
    localparam integer RECW = 16;                               // words per window record
    localparam integer MAXSH_S = ((1 << SYN_AW) >> WPL) / NR;
    localparam integer MAXSH_R = (1 << RES_AW) / (RECW * NWIN);
    localparam integer MAXSHOT = (MAXSH_S < MAXSH_R) ? MAXSH_S : MAXSH_R;
    localparam integer SDW = NE * NB;                           // lane seed bits per shift (engine, bank)
    // assumptions (checked by the driver from CFG0-8): TR = 2 CM, M2 <= 128, NE NB <= 32, NP <= 32

    // ====================================================================================== block RAMs
    // banked (4K words per bank, no cascades), per-bank registered port-B address / write, pipelined output mux (ptw_bmem)
    reg [SYN_AW-1:0] sa_addr; reg sa_we; reg [31:0] sa_wd; wire [31:0] sa_q;      // port A (aclk)
    reg [RES_AW-1:0] ra_addr; wire [31:0] ra_q;
    wire [SYN_AW-1:0] sb_next; wire [31:0] sb_p2;                              // port B (dclk): address of this cycle, data 4 cycles later
    reg [RES_AW-1:0] rb_addr; reg rb_we; reg [31:0] rb_wd;
    ptw_bmem #(.AWD(SYN_AW), .BWR(0)) u_smem (.aclk(aclk), .a_addr(sa_addr), .a_we(sa_we), .a_wd(sa_wd), .a_q(sa_q),
        .bclk(dclk), .b_addr(sb_next), .b_we(1'b0), .b_wd(32'd0), .b_q(sb_p2));
    ptw_bmem #(.AWD(RES_AW), .BWR(1)) u_rmem (.aclk(aclk), .a_addr(ra_addr), .a_we(1'b0), .a_wd(32'd0), .a_q(ra_q),
        .bclk(dclk), .b_addr(rb_addr), .b_we(rb_we), .b_wd(rb_wd), .b_q());

    // ====================================================================================== AXI4-Lite slave (aclk; as pt_board)
    wire arst = ~aresetn;
    reg [2:0] st;
    localparam [2:0] S_IDLE = 0, S_TW = 1, S_TR = 2, S_MR1 = 3, S_MR2 = 4, S_WA = 5;
    reg [AW-1:0] wa_q; reg [31:0] wd_q;
    reg cdcA_req, cdcA_wr; reg [9:0] cdcA_addr; reg [31:0] cdcA_data;
    reg cdcD_ack; reg [31:0] cdcD_rdata;
    (* ASYNC_REG = "TRUE" *) reg ak1, ak2;
    reg rsel;                                                 // memory read: 0 result, 1 syndrome
    always @(posedge aclk) begin ak1 <= cdcD_ack; ak2 <= ak1; end
    always @(posedge aclk) begin
        sa_we <= 1'b0;
        s_axi_awready <= 1'b0; s_axi_wready <= 1'b0; s_axi_arready <= 1'b0;
        if (arst) begin
            st <= S_IDLE; s_axi_bvalid <= 1'b0; s_axi_rvalid <= 1'b0; s_axi_bresp <= 2'b00; s_axi_rresp <= 2'b00;
            cdcA_req <= 1'b0; cdcA_wr <= 1'b0;
        end else begin
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;
            if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 1'b0;
            case (st)
            S_IDLE:
                if (s_axi_awvalid && s_axi_wvalid && !s_axi_bvalid) begin
                    s_axi_awready <= 1'b1; s_axi_wready <= 1'b1; s_axi_bresp <= 2'b00;
                    wa_q <= s_axi_awaddr; wd_q <= s_axi_wdata;
                    st <= S_WA;
                end else if (s_axi_arvalid && !s_axi_rvalid) begin
                    s_axi_arready <= 1'b1; s_axi_rresp <= 2'b00;
                    if (s_axi_araddr[AW-1:AW-2] == 2'b00) begin
                        cdcA_addr <= s_axi_araddr[11:2]; cdcA_wr <= 1'b0; cdcA_req <= ~cdcA_req;
                        st <= S_TR;
                    end else begin
                        rsel <= s_axi_araddr[AW-1];
                        sa_addr <= s_axi_araddr[SYN_AW+1:2]; ra_addr <= s_axi_araddr[RES_AW+1:2];
                        st <= S_MR1;
                    end
                end
            S_WA:
                if (wa_q[AW-1:AW-2] == 2'b00) begin                                  // register: tunnel to dclk
                    cdcA_addr <= wa_q[11:2]; cdcA_data <= wd_q; cdcA_wr <= 1'b1; cdcA_req <= ~cdcA_req;
                    st <= S_TW;
                end else begin
                    if (wa_q[AW-1]) begin                                            // syndrome memory
                        sa_addr <= wa_q[SYN_AW+1:2]; sa_wd <= wd_q; sa_we <= 1'b1;
                    end                                                              // result memory: read only
                    s_axi_bvalid <= 1'b1; st <= S_IDLE;
                end
            S_TW: if (ak2 == cdcA_req) begin s_axi_bvalid <= 1'b1; st <= S_IDLE; end
            S_TR: if (ak2 == cdcA_req) begin s_axi_rdata <= cdcD_rdata; s_axi_rvalid <= 1'b1; st <= S_IDLE; end
            S_MR1: st <= S_MR2;
            S_MR2: begin s_axi_rdata <= rsel ? sa_q : ra_q; s_axi_rvalid <= 1'b1; st <= S_IDLE; end
            default: st <= S_IDLE;
            endcase
        end
    end

    // ====================================================================================== register file (dclk)
    wire rst = ~dresetn;
    (* ASYNC_REG = "TRUE" *) reg rq1, rq2;
    reg rq3, cmd_d;
    always @(posedge dclk) begin rq1 <= cdcA_req; rq2 <= rq1; rq3 <= rq2; end
    wire cmd = rq2 ^ rq3;
    wire cw = cmd & cdcA_wr, cr = cmd & ~cdcA_wr;
    wire [9:0] ra = cdcA_addr;
    always @(posedge dclk) begin
        cmd_d <= cmd;
        if (rst) cdcD_ack <= 1'b0; else if (cmd_d) cdcD_ack <= rq3;
    end

    reg [31:0] nsw_r, tround_r, nshot_r;
    reg [QW-1:0] wqd_r, wqm_r; reg tmode_r;
    reg irq_en, fill_p, recalc_p, sd_sh, ss_sh, tw_en, th_en, start_p, abort_p;
    reg [NB-1:0] push_a, push_b; reg [RB-1:0] psel;
    reg [31:0] sd_w, ss_w;
    reg [ROWW-1:0] stg;                        // threshold row staging (word k = T[.][k + 1])
    reg [LW-1:0] tw_addr, th_addr; reg [TW-1:0] th_data;
    reg [63:0] frc; reg [31:0] frc_hi_l;
    reg [63:0] t_start, t_end;
    reg bbusy, bdone;
    reg [31:0] misses, late, released, ndone;
    wire core_ready, core_busy, core_done;
    wire [EW-1:0] best_e; wire [1:0] best_c, dec_ml, frame, fr_ml; wire [M2+1:0] best_s; wire [M2-1:0] sigma;
    wire [19:0] cnt0, cnt1, cnt2, cnt3; wire [31:0] cyc_win;
    integer k;
    always @(posedge dclk) begin
        fill_p <= 1'b0; recalc_p <= 1'b0; push_a <= 0; push_b <= 0; sd_sh <= 1'b0; ss_sh <= 1'b0; tw_en <= 1'b0;
        th_en <= 1'b0; start_p <= 1'b0; abort_p <= 1'b0;
        frc <= rst ? 64'd0 : frc + 64'd1;
        if (rst) begin
            nsw_r <= 32'd4096; tround_r <= 32'd0; nshot_r <= 32'd0; irq_en <= 1'b0; psel <= 0;
            wqd_r <= WQ_INIT; wqm_r <= WM_INIT; tmode_r <= 1'b0;
        end else if (cw) begin
            if (ra[9:8] == 2'b01) begin                                         // 0x400 + 4 t: THETA[t]
                th_en <= 1'b1; th_addr <= ra[LW-1:0]; th_data <= cdcA_data[TW-1:0];
            end else if (ra[9:7] == 3'b001) begin                               // 0x200 + 4 k: THR_STAGE[k]
                for (k = 0; k < ROWN; k = k + 1) if (ra[6:0] == k) stg[k*B +: B] <= cdcA_data[B-1:0];
            end else case (ra)
                10'h000: begin                                                  // CTRL
                    start_p <= cdcA_data[0]; fill_p <= cdcA_data[1]; recalc_p <= cdcA_data[2]; abort_p <= cdcA_data[3];
                    irq_en <= cdcA_data[8];
                end
                10'h002: nsw_r <= cdcA_data;
                10'h003: tround_r <= cdcA_data;
                10'h004: nshot_r <= cdcA_data;
                10'h005: wqd_r <= cdcA_data[QW-1:0];
                10'h006: wqm_r <= cdcA_data[QW-1:0];
                10'h007: begin push_a <= cdcA_data[NB-1:0]; push_b <= cdcA_data[8 +: NB]; psel <= cdcA_data[16 +: RB]; end
                10'h008: begin tw_en <= 1'b1; tw_addr <= cdcA_data[LW-1:0]; end                // THR_COMMIT (row = label)
                10'h01D: tmode_r <= cdcA_data[0];
                10'h020: begin sd_w <= cdcA_data; sd_sh <= 1'b1; end                         // SEED_LANE: set and shift
                10'h021: begin ss_w <= cdcA_data; ss_sh <= 1'b1; end                         // SEED_SWAP: set and shift
                default: ;
            endcase
        end
        if (cr) begin
            if (ra[9:7] == 3'b001) begin
                cdcD_rdata <= 32'd0;
                for (k = 0; k < ROWN; k = k + 1) if (ra[6:0] == k) cdcD_rdata <= {{(32-B){1'b0}}, stg[k*B +: B]};
            end else case (ra)
                10'h000: cdcD_rdata <= {23'd0, irq_en, 8'd0};
                10'h001: cdcD_rdata <= {27'd0, core_ready, core_busy, bdone, bbusy, core_ready & ~bbusy};
                10'h002: cdcD_rdata <= nsw_r;
                10'h003: cdcD_rdata <= tround_r;
                10'h004: cdcD_rdata <= nshot_r;
                10'h005: cdcD_rdata <= 32'(wqd_r);
                10'h006: cdcD_rdata <= 32'(wqm_r);
                10'h009: cdcD_rdata <= ID;
                10'h00A: cdcD_rdata <= {8'(NS), 8'(B), 8'(KSW), 8'(ROWN)};                   // CFG0
                10'h00B: cdcD_rdata <= {8'(CM), 8'(TR), 8'(M2), 8'(NC)};                     // CFG1
                10'h00C: cdcD_rdata <= {8'(TW), 8'(EW), 8'(LW), 8'(QW)};                     // CFG2
                10'h00D: cdcD_rdata <= {8'(SYN_AW), 8'(RES_AW), 8'(WPL), 8'(USW)};           // CFG3
                10'h00E: cdcD_rdata <= MAXSHOT;                                              // CFG4: shots per batch
                10'h00F: cdcD_rdata <= {8'(NE), 8'(RS), 8'(NB), 8'(KS)};                     // CFG5
                10'h010: begin cdcD_rdata <= frc[31:0]; frc_hi_l <= frc[63:32]; end         // FRC_LO (latches FRC_HI)
                10'h011: cdcD_rdata <= frc_hi_l;
                10'h012: cdcD_rdata <= t_start[31:0];
                10'h013: cdcD_rdata <= t_start[63:32];
                10'h014: cdcD_rdata <= t_end[31:0];
                10'h015: cdcD_rdata <= t_end[63:32];
                10'h016: cdcD_rdata <= misses;
                10'h017: cdcD_rdata <= late;
                10'h018: cdcD_rdata <= released;
                10'h019: cdcD_rdata <= ndone;
                10'h01A: cdcD_rdata <= {8'(CSL), 8'(GAP_SW), 8'(LDC), 8'(KSL)};             // CFG6
                10'h01B: cdcD_rdata <= {16'(BL1), 16'(BL0)};                                 // CFG7: lanes per bank
                10'h01C: cdcD_rdata <= {8'(RD), 8'(NP), 8'(RB), 8'(WPR)};                    // CFG8
                10'h01D: cdcD_rdata <= {31'd0, tmode_r};
                default: cdcD_rdata <= 32'd0;
            endcase
        end
    end

    // ====================================================================================== streaming (dclk)
    reg [M2-1:0] ring [0:RD-1];                // syndrome ring (distributed RAM)
    reg [RPW-1:0] wp, rp;                      // ring write pointer, oldest occupied row
    reg [RPW:0] occ;                           // occupied rows
    reg abort_r;
    reg [31:0] tot;                            // rounds in the batch
    // ---- round timer and release
    reg [31:0] tc, ndue, nst;                  // countdown to the next due round, rounds due so far, releases started
    reg [63:0] t_due;                          // due time of round nst (timed mode)
    reg rl_on, rl_v1, rl_v2, rl_v3, rl_v4; reg [WPL:0] rl_j, rl_j1, rl_j2, rl_j3, rl_j4; reg rl_wr;
    reg [32*WPR-1:0] rl_row;
    reg [31:0] rl_lag;                         // release start - due time of the round being released
    reg [7:0] r_w;                             // round index within its shot (write side)
    wire timed = (tround_r != 32'd0);
    wire tick = bbusy & timed & (tc == 32'd0) & (ndue < tot);
    wire room = (occ < RD);                    // a row is free (rows are freed at a take; the release writes one row)
    assign sb_next = SYN_AW'((nst - 1) << WPL | rl_j);                       // read while rl_on
    wire rl_go = bbusy & ~abort_r & ~rl_on & ~rl_v1 & ~rl_v2 & ~rl_v3 & ~rl_v4 & ~rl_wr & (nst < tot) & room & (timed ? (ndue > nst) : 1'b1);
    // ---- triggers (window ready), FIFO of trigger times
    reg [63:0] tq_t [0:3]; reg [31:0] tq_l [0:3]; reg [1:0] tq_w, tq_r;
    reg [31:0] ntrig, ntake;
    // ---- window FSM (read side)
    localparam [2:0] A_IDLE = 0, A_ASM = 1, A_WAIT = 2, A_VAL = 3;
    reg [2:0] as_st;
    reg [1:0] wi; reg [15:0] shot_r;
    reg [7:0] ai; reg [RPW-1:0] ap;
    reg [TR*M2-1:0] syn_w; reg [M2-1:0] synp_w;
    reg clr_p;
    wire [7:0] need = (wi == 2'd2) ? 8'(TR + 1) : 8'(TR);
    wire [7:0] freen = (wi == 2'd2) ? 8'(TR + 1) : 8'(CM);
    wire syn_valid = (as_st == A_VAL);
    wire take = syn_valid & core_ready & ~core_busy;             // = ptw_top's take
    // ---- window in flight, record
    reg [63:0] c_trig, c_take; reg [31:0] c_lag; reg [1:0] c_win; reg c_fin; reg [15:0] c_shot;
    reg [RECW*32-1:0] rec; reg [4:0] rec_n;
    wire [RPW-1:0] wp_n = (wp == RD - 1) ? {RPW{1'b0}} : wp + 1'b1;
    always @(posedge dclk) begin
        rb_we <= 1'b0; clr_p <= 1'b0;
        if (rst) begin
            bbusy <= 1'b0; bdone <= 1'b0; abort_r <= 1'b0; rl_on <= 1'b0; rl_v1 <= 1'b0; rl_v2 <= 1'b0; rl_v3 <= 1'b0; rl_v4 <= 1'b0; rl_wr <= 1'b0;
            as_st <= A_IDLE; rec_n <= 0; occ <= 0; wp <= 0; rp <= 0; tq_w <= 0; tq_r <= 0;
            misses <= 0; late <= 0; released <= 0; ndone <= 0; ntrig <= 0; ntake <= 0; nst <= 0; ndue <= 0;
        end else begin
            // ------------------------------------------------ batch start / abort
            if (start_p && !bbusy) begin
                bbusy <= 1'b1; bdone <= 1'b0; abort_r <= 1'b0; t_start <= frc;
                tot <= nshot_r * NR; nst <= 0; ndue <= 0; tc <= tround_r - 32'd1; t_due <= frc + 64'(tround_r) + 64'd1;
                occ <= 0; wp <= 0; rp <= 0; tq_w <= 0; tq_r <= 0; r_w <= 0;
                as_st <= A_IDLE; wi <= 0; shot_r <= 0;
                misses <= 0; late <= 0; released <= 0; ndone <= 0; ntrig <= 0; ntake <= 0;
            end
            if (abort_p && bbusy) abort_r <= 1'b1;
            // ------------------------------------------------ round timer: one round due every T_ROUND cycles
            // round k is due at T_START + (k + 1) T_ROUND + 1 (lag 0 = released on time)
            if (bbusy && timed) tc <= (tc == 32'd0) ? tround_r - 32'd1 : tc - 32'd1;
            if (tick) ndue <= ndue + 1;
            // ------------------------------------------------ release: WPR words -> one ring row
            rl_v1 <= rl_on; rl_j1 <= rl_j; rl_v2 <= rl_v1; rl_j2 <= rl_j1; rl_v3 <= rl_v2; rl_j3 <= rl_j2; rl_v4 <= rl_v3; rl_j4 <= rl_j3;
            if (rl_go) begin
                rl_on <= 1'b1; rl_j <= 0; nst <= nst + 1;
                rl_lag <= timed ? 32'(frc - t_due) : 32'd0;
                if (timed && frc != t_due) late <= late + 1;          // released after its due time (ring full: decoder behind)
            end
            if (rl_on) begin
                if (rl_j == WPR - 1) rl_on <= 1'b0;
                rl_j <= rl_j + 1'b1;
            end
            if (rl_v4) begin
                for (k = 0; k < WPR; k = k + 1) if (rl_j4 == k) rl_row[k*32 +: 32] <= sb_p2;
                if (rl_j4 == WPR - 1) rl_wr <= 1'b1;
            end
            if (rl_wr) begin                                          // row complete: write it, round bookkeeping
                rl_wr <= 1'b0;
                ring[wp] <= rl_row[M2-1:0];
                wp <= wp_n;
                released <= released + 1;
                t_due <= t_due + 64'(tround_r);
                r_w <= (r_w == NR - 1) ? 8'd0 : r_w + 8'd1;
                if (r_w == TR - 1 || r_w == TR + CM - 1 || r_w == NR - 1) begin     // trigger of a window
                    tq_t[tq_w] <= frc; tq_l[tq_w] <= rl_lag; tq_w <= tq_w + 1'b1;
                    ntrig <= ntrig + 1;
                    if (timed && ndone != ntrig) misses <= misses + 1;            // previous window not committed yet
                end
            end
            // ------------------------------------------------ window FSM
            case (as_st)
            A_IDLE:
                if (bbusy && !abort_r && occ >= need) begin as_st <= A_ASM; ai <= 0; ap <= rp; end
            A_ASM: begin
                if (ai < TR) syn_w <= {ring[ap], syn_w[TR*M2-1:M2]};
                else synp_w <= ring[ap];
                ap <= (ap == RD - 1) ? {RPW{1'b0}} : ap + 1'b1;
                ai <= ai + 1'b1;
                if (ai == need - 1) as_st <= A_WAIT;
            end
            A_WAIT:
                if (!core_busy && core_ready && !core_done) begin
                    if (wi == 2'd0) clr_p <= 1'b1;                    // first window of a shot: sigma = frame = 0
                    as_st <= A_VAL;
                end
            A_VAL:
                if (take) begin
                    as_st <= A_IDLE;
                    rp <= (rp + freen >= RD) ? RPW'(rp + freen - RD) : RPW'(rp + freen);
                    c_trig <= tq_t[tq_r]; c_lag <= tq_l[tq_r]; tq_r <= tq_r + 1'b1;
                    c_take <= frc; c_win <= wi; c_fin <= (wi == 2'd2); c_shot <= shot_r;
                    ntake <= ntake + 1;
                    wi <= (wi == 2'd2) ? 2'd0 : wi + 2'd1;
                    if (wi == 2'd2) shot_r <= shot_r + 1'b1;
                end
            default: as_st <= A_IDLE;
            endcase
            occ <= occ + (rl_wr ? 1'b1 : 1'b0) - ((as_st == A_VAL && take) ? freen : 8'd0);
            // ------------------------------------------------ commit record
            if (core_done) begin
                rec <= {32'(c_lag), cyc_win, 32'(frc - c_take), 32'(c_take - c_trig), c_trig,
                        128'(sigma), 12'd0, cnt3, 12'd0, cnt2, 12'd0, cnt1, 12'd0, cnt0, 32'(best_e),
                        c_shot, 5'd0, c_win, c_fin, best_c, dec_ml, frame, (c_fin ? fr_ml : 2'b00)};
                rec_n <= RECW; rb_addr <= RES_AW'(ndone * RECW);
                ndone <= ndone + 1;
            end else if (rec_n != 0) begin
                rb_we <= 1'b1; rb_wd <= rec[31:0]; rec <= rec >> 32; rec_n <= rec_n - 1'b1;
            end
            if (rb_we) rb_addr <= rb_addr + 1'b1;
            // ------------------------------------------------ batch end
            if (bbusy && !start_p && rec_n == 0 && !rb_we && !core_done && !core_busy &&
                ((ndone == nshot_r * NWIN) || (abort_r && ndone == ntake && as_st != A_VAL))) begin
                bbusy <= 1'b0; bdone <= 1'b1; t_end <= frc;
            end
        end
    end
    assign irq = irq_en & bdone;

    // ====================================================================================== decoder
    ptw_top u_pt (.clk(dclk), .rst(rst), .sd_sh(sd_sh), .push_a(push_a), .push_b(push_b), .ss_sh(ss_sh),
                  .sd_bit(sd_w[SDW-1:0]), .ss_bit(ss_w[NP-1:0]), .psel(psel), .fill(fill_p), .recalc(recalc_p),
                  .tw_en(tw_en), .tw_addr(tw_addr), .tw_data(stg), .th_en(th_en), .th_addr(th_addr), .th_data(th_data),
                  .wq_d(wqd_r), .wq_m(wqm_r), .tmode(tmode_r), .nsw(nsw_r), .clr(clr_p),
                  .syn_valid(syn_valid), .fin(wi == 2'd2), .syn(syn_w), .synp(synp_w),
                  .ready(core_ready), .busy(core_busy), .done(core_done), .best_e(best_e), .best_c(best_c), .best_s(best_s),
                  .dec_ml(dec_ml), .cnt0(cnt0), .cnt1(cnt1), .cnt2(cnt2), .cnt3(cnt3), .sigma(sigma), .frame(frame),
                  .fr_ml(fr_ml), .cyc_win(cyc_win), .arr(), .E_all(), .C_all(), .lab_all());
endmodule

// Dual-clock 32-bit memory of 2^AWD words in banks of 4K words (block RAM, no cascades). Port A (aclk): registered address
// a_addr, write a_we / a_wd, read data a_q one cycle after the address (combinational bank select). Port B (bclk): BWR = 0:
// read, b_addr presented in cycle c, b_q valid in cycle c + 4 (per-bank address register, RAM, two mux stages);
// BWR = 1: write (b_we, b_addr, b_wd registered per bank, written one cycle later).
module ptw_bmem #(parameter integer AWD = 17, parameter integer BWR = 0, parameter integer BKW = 12) (
    input  wire aclk, input wire [AWD-1:0] a_addr, input wire a_we, input wire [31:0] a_wd, output wire [31:0] a_q,
    input  wire bclk, input wire [AWD-1:0] b_addr, input wire b_we, input wire [31:0] b_wd, output reg [31:0] b_q
);
    localparam integer NBK = (AWD > BKW) ? (1 << (AWD - BKW)) : 1;
    localparam integer BW = (AWD > BKW) ? BKW : AWD;
    localparam integer SW = (NBK > 1) ? $clog2(NBK) : 1;
    localparam integer G = (NBK > 8) ? 8 : NBK;                 // first mux stage: groups of G banks
    localparam integer NG = NBK / G;
    wire [31:0] qa [0:NBK-1]; wire [31:0] qb [0:NBK-1];
    reg [SW-1:0] a_sel;
    always @(posedge aclk) a_sel <= (NBK > 1) ? a_addr[AWD-1 -: SW] : 1'b0;
    assign a_q = qa[a_sel];
    genvar gb;
    generate for (gb = 0; gb < NBK; gb = gb + 1) begin : g_bk
        (* ram_style = "block" *) reg [31:0] mem [0:(1 << BW) - 1];
        reg [31:0] ra, rb;
        (* keep = "true" *) reg [BW-1:0] ab; (* keep = "true" *) reg wb; reg [31:0] db;
        wire hit_a = (NBK == 1) || (a_addr[AWD-1 -: SW] == gb);
        always @(posedge aclk) begin
            if (a_we && hit_a) mem[a_addr[BW-1:0]] <= a_wd;
            ra <= mem[a_addr[BW-1:0]];
        end
        always @(posedge bclk) begin
            ab <= b_addr[BW-1:0];
            wb <= b_we && ((NBK == 1) || (b_addr[AWD-1 -: SW] == gb)); db <= b_wd;
            if (BWR) begin if (wb) mem[ab] <= db; end
            else rb <= mem[ab];
        end
        assign qa[gb] = ra; assign qb[gb] = rb;
    end endgenerate
    // port B read: bank select delayed with the data; stage 1 = G:1 within a group, stage 2 = NG:1
    reg [SW-1:0] s1, s2, s3;
    reg [31:0] m1 [0:NG-1];
    integer i;
    always @(posedge bclk) begin
        s1 <= (NBK > 1) ? b_addr[AWD-1 -: SW] : 1'b0; s2 <= s1; s3 <= s2;
        for (i = 0; i < NG; i = i + 1) m1[i] <= qb[i * G + ((NBK > 1) ? (s2 % G) : 0)];
        b_q <= m1[(NBK > 1) ? (s3 / G) : 0];
    end
endmodule


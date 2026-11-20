// Board top of the one-syndrome-per-shot decoder (hardware/README.md): AXI4-Lite slave + shot memories + batch controller + pt_top.
//
// Two clocks:
//  * aclk (AXI, e.g. 100 MHz from the PS): the AXI4-Lite slave and port A of the two block RAMs.
//  * dclk (decoder, e.g. 300-365 MHz from an MMCM): pt_top, the register file, the batch controller, port B of the RAMs and
//    every cycle counter.
// Crossings: the RAMs are dual-clock block RAMs; register accesses go through a request/acknowledge tunnel (toggle
// synchronizers, address and data held stable while a request is open). Only the cdcA_* / cdcD_* registers cross domains; the
// build constrains them with set_max_delay -datapath_only (pt_board_cdc.xdc, written by hardware/build/build_pt_board.tcl).
//
// Address map (byte offsets, AW = 18 bits = 256 KB):
//   0x00000-0x00FFF  registers (tunnel to dclk), see hardware/README.md
//   0x10000-0x1FFFF  syndrome memory, 2^SYN_AW 32-bit words: shot s, word w at s * 2^WPL + w (bit 32 w + j = syndrome bit j)
//   0x20000-0x3FFFF  result memory, 2^RES_AW words: shot s at 8 s .. 8 s + 7 (read only)
// The batch controller runs NSHOT shots from the syndrome memory back to back (the next syndrome is prefetched while the
// current one decodes) and writes one 8-word record per shot. Host traffic is outside the measured cycles.
`include "pt_params.svh"
module pt_board #(parameter integer AW = 18, parameter integer SYN_AW = 14, parameter integer RES_AW = 15) (
    input  wire          aclk, input wire aresetn,
    input  wire [AW-1:0] s_axi_awaddr, input wire s_axi_awvalid, output reg s_axi_awready,
    input  wire [31:0]   s_axi_wdata, input wire [3:0] s_axi_wstrb, input wire s_axi_wvalid, output reg s_axi_wready,
    output reg  [1:0]    s_axi_bresp, output reg s_axi_bvalid, input wire s_axi_bready,
    input  wire [AW-1:0] s_axi_araddr, input wire s_axi_arvalid, output reg s_axi_arready,
    output reg  [31:0]   s_axi_rdata, output reg [1:0] s_axi_rresp, output reg s_axi_rvalid, input wire s_axi_rready,
    input  wire          dclk, input wire dresetn,
    output wire          irq
);
    localparam [31:0] ID = 32'h5054_0200;                     // 'PT', board interface 2.0 (runtime tables, shot memories)
    localparam integer WPS = (M + 31) / 32;                    // syndrome words per shot
    localparam integer WPL = (WPS <= 1) ? 0 : $clog2(WPS);     // log2 of the per-shot stride
    localparam integer NWL = (NS + 31) / 32, NWS = (NP + 31) / 32;   // seed words per shift (lanes, swap pairs)
    localparam integer SHW = SYN_AW - WPL;                     // shot index bits of the syndrome memory
    localparam integer MAXSHOT = ((1 << SHW) < (1 << (RES_AW - 3))) ? (1 << SHW) : (1 << (RES_AW - 3));

    // ====================================================================================== block RAMs
    (* ram_style = "block" *) reg [31:0] smem [0:(1 << SYN_AW) - 1];
    (* ram_style = "block" *) reg [31:0] rmem [0:(1 << RES_AW) - 1];
    reg [SYN_AW-1:0] sa_addr; reg sa_we; reg [31:0] sa_wd, sa_q;      // port A (aclk)
    reg [RES_AW-1:0] ra_addr; reg [31:0] ra_q;
    reg [SYN_AW-1:0] sb_addr; reg [31:0] sb_q;                        // port B (dclk)
    reg [RES_AW-1:0] rb_addr; reg rb_we; reg [31:0] rb_wd;
    always @(posedge aclk) begin
        if (sa_we) smem[sa_addr] <= sa_wd;
        sa_q <= smem[sa_addr];
        ra_q <= rmem[ra_addr];
    end
    always @(posedge dclk) begin
        sb_q <= smem[sb_addr];
        if (rb_we) rmem[rb_addr] <= rb_wd;
    end

    // ====================================================================================== AXI4-Lite slave (aclk)
    wire arst = ~aresetn;
    reg [2:0] st;                                             // 0 idle, 1 tunnel write, 2 tunnel read, 3/4 memory read
    localparam [2:0] S_IDLE = 0, S_TW = 1, S_TR = 2, S_MR1 = 3, S_MR2 = 4, S_WA = 5;
    reg [AW-1:0] wa_q; reg [31:0] wd_q;
    reg cdcA_req, cdcA_wr; reg [9:0] cdcA_addr; reg [31:0] cdcA_data;
    reg cdcD_ack; reg [31:0] cdcD_rdata;
    (* ASYNC_REG = "TRUE" *) reg ak1, ak2;
    reg rsel;                                                 // memory read: 0 syndrome, 1 result
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
                    s_axi_awready <= 1'b1; s_axi_wready <= 1'b1; s_axi_bresp <= 2'b00;      // handshake at the end of S_WA
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
            S_WA:                                                                    // address and data accepted
                if (wa_q[AW-1:AW-2] == 2'b00) begin                                  // register: tunnel to dclk
                    cdcA_addr <= wa_q[11:2]; cdcA_data <= wd_q; cdcA_wr <= 1'b1; cdcA_req <= ~cdcA_req;
                    st <= S_TW;
                end else begin
                    if (wa_q[AW-1:AW-2] == 2'b01) begin                              // syndrome memory
                        sa_addr <= wa_q[SYN_AW+1:2]; sa_wd <= wd_q; sa_we <= 1'b1;
                    end                                                              // result memory: read only (ignored)
                    s_axi_bvalid <= 1'b1; st <= S_IDLE;
                end
            S_TW: if (ak2 == cdcA_req) begin s_axi_bvalid <= 1'b1; st <= S_IDLE; end
            S_TR: if (ak2 == cdcA_req) begin s_axi_rdata <= cdcD_rdata; s_axi_rvalid <= 1'b1; st <= S_IDLE; end
            S_MR1: st <= S_MR2;                                                      // RAM read in this cycle
            S_MR2: begin s_axi_rdata <= rsel ? ra_q : sa_q; s_axi_rvalid <= 1'b1; st <= S_IDLE; end
            default: st <= S_IDLE;
            endcase
        end
    end

    // ====================================================================================== register file (dclk)
    wire rst = ~dresetn;
    (* ASYNC_REG = "TRUE" *) reg rq1, rq2;
    reg rq3, cmd_d;
    always @(posedge dclk) begin rq1 <= cdcA_req; rq2 <= rq1; rq3 <= rq2; end
    wire cmd = rq2 ^ rq3;                     // a new request; cdcA_addr / cdcA_data are stable since before the toggle
    wire cw = cmd & cdcA_wr, cr = cmd & ~cdcA_wr;
    wire [9:0] ra = cdcA_addr;
    always @(posedge dclk) begin
        cmd_d <= cmd;
        if (rst) cdcD_ack <= 1'b0; else if (cmd_d) cdcD_ack <= rq3;   // acknowledge one cycle after the access
    end

    reg [31:0] nsw_r, gmin_r, th_r, nshot_r;
    reg irq_en, fill_p, recalc_p, push_a, push_b, sd_sh, ss_sh, tw_en, th_en;
    reg [NWL*32-1:0] sd_w; reg [NWS*32-1:0] ss_w;
    reg [KMB-1:0] stg;                         // threshold row staging (word k = T[.][k + 1])
    reg [LW-1:0] tw_addr, th_addr; reg [TW-1:0] th_data;
    reg start_p, abort_p;
    reg [63:0] frc; reg [31:0] frc_hi_l;
    reg [63:0] t_start, t_end;
    reg bbusy, bdone;
    reg [31:0] launched, completed;
    wire core_ready, core_busy, core_done;
    wire [1:0] dec_ml, dec_me; wire [EW-1:0] dec_e;
    wire [19:0] cnt0, cnt1, cnt2, cnt3;
    wire [31:0] cyc_start, cyc_run, sweeps_run;
    integer k;
    always @(posedge dclk) begin
        fill_p <= 1'b0; recalc_p <= 1'b0; push_a <= 1'b0; push_b <= 1'b0; sd_sh <= 1'b0; ss_sh <= 1'b0; tw_en <= 1'b0;
        th_en <= 1'b0; start_p <= 1'b0; abort_p <= 1'b0;
        frc <= rst ? 64'd0 : frc + 64'd1;
        if (rst) begin
            nsw_r <= 32'd1024; gmin_r <= 32'd0; th_r <= 32'd29; nshot_r <= 32'd0; irq_en <= 1'b0;
        end else if (cw) begin
            if (ra[9:8] == 2'b01) begin                                         // 0x400 + 4 t: THETA[t]
                th_en <= 1'b1; th_addr <= ra[LW-1:0]; th_data <= cdcA_data[TW-1:0];
            end else if (ra[9:6] == 4'b0010) begin                              // 0x200 + 4 k: THR_STAGE[k]
                for (k = 0; k < KM; k = k + 1) if (ra[5:0] == k) stg[k*B +: B] <= cdcA_data[B-1:0];
            end else if (ra[9:3] == 7'b0000100) begin                           // 0x080 + 4 j: SEED_LANE[j]
                for (k = 0; k < NWL; k = k + 1) if (ra[2:0] == k) sd_w[k*32 +: 32] <= cdcA_data;
                if (ra[2:0] == NWL - 1) sd_sh <= 1'b1;                          // the last word shifts
            end else if (ra[9:3] == 7'b0000101) begin                           // 0x0A0 + 4 j: SEED_SWAP[j]
                for (k = 0; k < NWS; k = k + 1) if (ra[2:0] == k) ss_w[k*32 +: 32] <= cdcA_data;
                if (ra[2:0] == NWS - 1) ss_sh <= 1'b1;
            end else case (ra)
                10'h000: begin                                                  // CTRL
                    start_p <= cdcA_data[0]; fill_p <= cdcA_data[1]; recalc_p <= cdcA_data[2]; abort_p <= cdcA_data[3];
                    irq_en <= cdcA_data[8];
                end
                10'h002: nsw_r <= cdcA_data;
                10'h003: gmin_r <= cdcA_data;
                10'h004: th_r <= cdcA_data;
                10'h005: nshot_r <= cdcA_data;
                10'h007: begin push_a <= cdcA_data[1]; push_b <= cdcA_data[2]; end                // SEED_CTRL
                10'h008: begin tw_en <= 1'b1; tw_addr <= cdcA_data[LW-1:0]; end                   // THR_COMMIT (row = label)
                default: ;
            endcase
        end
        if (cr) begin
            if (ra[9:6] == 4'b0010) begin
                cdcD_rdata <= 32'd0;
                for (k = 0; k < KM; k = k + 1) if (ra[5:0] == k) cdcD_rdata <= {{(32-B){1'b0}}, stg[k*B +: B]};
            end else case (ra)
                10'h000: cdcD_rdata <= {23'd0, irq_en, 8'd0};
                10'h001: cdcD_rdata <= {27'd0, core_ready, core_busy, bdone, bbusy, core_ready & ~bbusy};
                10'h002: cdcD_rdata <= nsw_r;
                10'h003: cdcD_rdata <= gmin_r;
                10'h004: cdcD_rdata <= th_r;
                10'h005: cdcD_rdata <= nshot_r;
                10'h006: cdcD_rdata <= completed;
                10'h009: cdcD_rdata <= ID;
                10'h00A: cdcD_rdata <= {8'(NS), 8'(B), 8'(KSW), 8'(KM)};                      // CFG0
                10'h00B: cdcD_rdata <= {16'(N), 16'(M)};                                     // CFG1
                10'h00C: cdcD_rdata <= {8'(TW), 8'(EW), 8'(NCL), 8'(LW)};                   // CFG2
                10'h00D: cdcD_rdata <= {8'(SYN_AW), 8'(RES_AW), 8'(WPL), 8'(USW)};           // CFG3
                10'h00E: cdcD_rdata <= MAXSHOT;                                              // CFG4: shots per batch
                10'h00F: cdcD_rdata <= {16'(NLB), 16'(LMAX)};                               // CFG5: lane bits per cycle, lanes
                10'h010: begin cdcD_rdata <= frc[31:0]; frc_hi_l <= frc[63:32]; end         // FRC_LO (latches FRC_HI)
                10'h011: cdcD_rdata <= frc_hi_l;
                10'h012: cdcD_rdata <= t_start[31:0];
                10'h013: cdcD_rdata <= t_start[63:32];
                10'h014: cdcD_rdata <= t_end[31:0];
                10'h015: cdcD_rdata <= t_end[63:32];
                default: cdcD_rdata <= 32'd0;
            endcase
        end
    end

    // ====================================================================================== batch controller (dclk)
    reg [WPS*32-1:0] syn_nx;                   // prefetched syndrome of the next shot
    reg nx_valid, pf_on, pf_v1, pf_v2; reg [WPL:0] pf_w, pf_w1, pf_w2; reg [31:0] pf_shot;
    reg [7:0] rec_w; reg [255:0] rec;          // record being written (8 words)
    reg abort_r;
    wire [31:0] lim = abort_r ? launched : nshot_r;
    wire syn_valid = nx_valid & bbusy & (launched < lim);
    wire take = syn_valid & core_ready & ~core_busy;          // = pt_top's take
    always @(posedge dclk) begin
        rb_we <= 1'b0;
        if (rst) begin
            bbusy <= 1'b0; bdone <= 1'b0; nx_valid <= 1'b0; pf_on <= 1'b0; pf_v1 <= 1'b0; pf_v2 <= 1'b0; rec_w <= 8'd0;
            abort_r <= 1'b0;
            launched <= 32'd0; completed <= 32'd0;
        end else begin
            if (start_p && !bbusy) begin
                bbusy <= 1'b1; bdone <= 1'b0; abort_r <= 1'b0; launched <= 32'd0; completed <= 32'd0; t_start <= frc;
                nx_valid <= 1'b0;
                if (nshot_r != 0) begin pf_on <= 1'b1; pf_shot <= 32'd0; pf_w <= 0; end
            end
            if (abort_p && bbusy) abort_r <= 1'b1;
            // prefetch: one word per cycle (address registered, RAM read, word captured: 2 cycles of latency)
            pf_v1 <= pf_on; pf_w1 <= pf_w; pf_v2 <= pf_v1; pf_w2 <= pf_w1;
            if (pf_on) begin
                sb_addr <= (pf_shot[SYN_AW-1:0] << WPL) | pf_w;
                if (pf_w == WPS - 1) pf_on <= 1'b0;
                pf_w <= pf_w + 1'b1;
            end
            if (pf_v2) begin
                for (k = 0; k < WPS; k = k + 1) if (pf_w2 == k) syn_nx[k*32 +: 32] <= sb_q;
                if (pf_w2 == WPS - 1) nx_valid <= 1'b1;
            end
            // launch
            if (take) begin
                nx_valid <= 1'b0;
                launched <= launched + 1;
                if (launched + 1 < lim) begin pf_on <= 1'b1; pf_shot <= launched + 1; pf_w <= 0; end
            end
            // record of a finished shot
            if (core_done) begin
                rec <= {frc[31:0], 16'd0, cnt3[19:4], cnt3[3:0], cnt2, cnt1[19:12], cnt1[11:0], cnt0, cyc_run, cyc_start,
                        sweeps_run, 16'(dec_e), 11'd0, (sweeps_run < nsw_r), dec_me, dec_ml};
                rec_w <= 8'd8; rb_addr <= completed[RES_AW-4:0] << 3;
                completed <= completed + 1;
            end else if (rec_w != 0) begin
                rb_we <= 1'b1; rb_wd <= rec[31:0]; rec <= rec >> 32; rec_w <= rec_w - 1'b1;
            end
            if (rb_we) rb_addr <= rb_addr + 1'b1;
            if (bbusy && !start_p && completed == lim && launched == lim && rec_w == 0 && !rb_we && !core_done) begin
                bbusy <= 1'b0; bdone <= 1'b1; t_end <= frc;
            end
        end
    end
    assign irq = irq_en & bdone;

    // ====================================================================================== decoder
    pt_top u_pt (.clk(dclk), .rst(rst), .sd_sh(sd_sh), .push_a(push_a), .push_b(push_b), .ss_sh(ss_sh), .sd_bit(sd_w[NS-1:0]),
                 .ss_bit(ss_w[NP-1:0]), .fill(fill_p), .recalc(recalc_p), .tw_en(tw_en), .tw_addr(tw_addr), .tw_data(stg),
                 .th_en(th_en), .th_addr(th_addr), .th_data(th_data), .nsw(nsw_r), .es_gmin(gmin_r), .es_th(th_r[5:0]),
                 .syn_valid(syn_valid), .syn(syn_nx[M-1:0]), .ready(core_ready), .busy(core_busy), .done(core_done),
                 .dec_ml(dec_ml), .dec_me(dec_me), .dec_e(dec_e), .cnt0(cnt0), .cnt1(cnt1), .cnt2(cnt2), .cnt3(cnt3),
                 .cyc_start(cyc_start), .cyc_run(cyc_run), .sweeps_run(sweeps_run), .arr(), .E_all(), .C_all(), .lab_all());
endmodule

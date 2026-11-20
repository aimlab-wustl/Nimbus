// Testbench for the real-time core (ptw_top): writes the runtime tables, then streams T_NSHOT shots through their T_NWIN
// windows. Before every window it seeds the lane generators (per bank) and the swap generators with the golden model's
// per-window seeds (hardware/sim/gen_ptw_case.py), pulses fill, takes the window; it compares every replica's energy, class and
// label after every sweep (arr) and the commit record at done (best energy, best class, summary, ML, counts of the last
// block, sigma, frame bits; fr_ml in final windows).
`timescale 1ns/1ps
`include "ptw_params.svh"
`include "tb_params.svh"
module tb_ptw;
    reg clk = 0, rst = 1;
    always #1.5 clk = ~clk;
    reg sd_sh = 0, ss_sh = 0, fill = 0, recalc = 0, syn_valid = 0, clr = 0, fin = 0;
    reg [NB-1:0] push_a = 0, push_b = 0;
    reg tw_en = 0, th_en = 0; reg [LW-1:0] tw_addr = 0, th_addr = 0; reg [ROWW-1:0] tw_data = 0; reg [TW-1:0] th_data = 0;
    reg [NE*NB-1:0] sd_bit = 0; reg [NP-1:0] ss_bit = 0; reg [RB-1:0] psel = 0;
    reg [TR*M2-1:0] syn = 0; reg [M2-1:0] synp = 0;
    wire ready, busy, done, arr;
    wire [EW-1:0] best_e; wire [1:0] best_c, dec_ml, frame, fr_ml; wire [M2+1:0] best_s; wire [M2-1:0] sigma;
    wire [19:0] cnt0, cnt1, cnt2, cnt3; wire [31:0] cyc_win;
    wire [NS*EW-1:0] E_all; wire [2*NS-1:0] C_all; wire [NS*LW-1:0] lab_all;
    ptw_top dut (.clk(clk), .rst(rst), .sd_sh(sd_sh), .push_a(push_a), .push_b(push_b), .ss_sh(ss_sh), .sd_bit(sd_bit),
                 .ss_bit(ss_bit), .psel(psel), .fill(fill), .recalc(recalc), .tw_en(tw_en), .tw_addr(tw_addr), .tw_data(tw_data),
                 .th_en(th_en), .th_addr(th_addr), .th_data(th_data), .wq_d(QW'(`T_WQ)), .wq_m(QW'(`T_WM)),
                 .tmode(1'(`T_MODE)), .nsw(`T_NSW), .clr(clr), .syn_valid(syn_valid), .fin(fin), .syn(syn), .synp(synp),
                 .ready(ready), .busy(busy), .done(done), .best_e(best_e), .best_c(best_c), .best_s(best_s),
                 .dec_ml(dec_ml), .cnt0(cnt0), .cnt1(cnt1), .cnt2(cnt2), .cnt3(cnt3), .sigma(sigma), .frame(frame),
                 .fr_ml(fr_ml), .cyc_win(cyc_win), .arr(arr), .E_all(E_all), .C_all(C_all), .lab_all(lab_all));
    localparam integer NWT = `T_NSHOT * `T_NWIN;
    reg [NS-1:0] sa0 [0:NWT*`T_PA0-1]; reg [NS-1:0] sb0 [0:NWT*`T_PB0-1];
`ifdef T_PA1
    reg [NS-1:0] sa1 [0:NWT*`T_PA1-1]; reg [NS-1:0] sb1 [0:NWT*`T_PB1-1];
`endif
    reg [NP-1:0] ss_mem [0:NWT*128-1];
    reg [`T_WB-1:0] rec_mem [0:`T_ROWS-1];
    reg [`T_WRW-1:0] win_mem [0:NWT-1];
    reg [ROWW-1:0] thr_mem [0:NS-1];
    reg [TW-1:0] th_mem [0:NP-1];
    integer k, row, nerr, shot, win, wi, fd, rc, fv, bnk;
    reg [`T_WB-1:0] got;
    reg [`T_WRW-1:0] e, gw;
    reg [TR*M2-1:0] sv; reg [M2-1:0] spv;

    task automatic seed_bank(input integer b, input integer which);
        integer kk, L, Nb, i;
        reg [NS-1:0] bits;
        begin
            L = (b == 0) ? (which ? `T_PB0 : `T_PA0) : 0;
            Nb = `T_N0;
`ifdef T_PA1
            if (b == 1) begin L = which ? `T_PB1 : `T_PA1; Nb = `T_N1; end
`endif
            for (kk = 0; kk < L; kk = kk + 1) begin
                bits = (b == 0) ? (which ? sb0[wi * L + kk] : sa0[wi * L + kk]) : {NS{1'b0}};
`ifdef T_PA1
                if (b == 1) bits = which ? sb1[wi * L + kk] : sa1[wi * L + kk];
`endif
                @(negedge clk); push_a = 0; push_b = 0; sd_sh = 1;
                for (i = 0; i < NE; i = i + 1) sd_bit[i * NB + b] = bits[i * RS + psel];
                if (kk % Nb == Nb - 1) begin
                    @(negedge clk); sd_sh = 0;
                    if (which) push_b[b] = 1; else push_a[b] = 1;
                end
            end
            @(negedge clk); push_a = 0; push_b = 0; sd_sh = 0;
        end
    endtask

    initial begin
        $readmemb("seed_a0.mem", sa0); $readmemb("seed_b0.mem", sb0);
`ifdef T_PA1
        $readmemb("seed_a1.mem", sa1); $readmemb("seed_b1.mem", sb1);
`endif
        $readmemb("seed_s.mem", ss_mem); $readmemh("exp_rec.mem", rec_mem); $readmemh("exp_win.mem", win_mem);
        $readmemh("thr.mem", thr_mem); $readmemh("theta.mem", th_mem);
        fd = $fopen("syn.mem", "r");
        nerr = 0; row = 0;
        repeat (4) @(posedge clk);
        rst <= 0;
        for (k = 0; k < NS; k = k + 1) begin                      // commits >= 2 cycles apart (two-cycle write path)
            @(negedge clk); tw_en = 1; tw_addr = k; tw_data = thr_mem[k];
            @(negedge clk); tw_en = 0;
        end
        for (k = 0; k < NP; k = k + 1) begin @(negedge clk); th_en = 1; th_addr = k; th_data = th_mem[k]; end
        @(negedge clk); th_en = 0;
        for (shot = 0; shot < `T_NSHOT; shot = shot + 1) begin
            while (!ready) @(negedge clk);
            @(negedge clk); clr = 1; @(negedge clk); clr = 0;
            for (win = 0; win < `T_NWIN; win = win + 1) begin
                wi = shot * `T_NWIN + win;
                while (!ready) @(negedge clk);
                if (`T_CARRY == 0 || win == 0) begin                  // T_CARRY: generators run on across the windows
                    for (k = 0; k < RS; k = k + 1) begin
                        psel = k;
                        for (bnk = 0; bnk < NB; bnk = bnk + 1) begin seed_bank(bnk, 0); seed_bank(bnk, 1); end
                    end
                    for (k = 0; k < 128; k = k + 1) begin @(negedge clk); ss_sh = 1; ss_bit = ss_mem[wi * 128 + k]; end
                    @(negedge clk); ss_sh = 0; fill = 1;
                    @(negedge clk); fill = 0;
                    @(negedge clk);
                    while (!ready) @(negedge clk);
                end
                rc = $fscanf(fd, "%d %h %h\n", fv, spv, sv);
                fin = fv[0]; synp = spv; syn = sv; syn_valid = 1;
                @(negedge clk); syn_valid = 0;
                while (!done) @(negedge clk);
                e = win_mem[wi];
                gw = {fin, fr_ml, frame, sigma, dec_ml, best_c, best_s, cnt3, cnt2, cnt1, cnt0, best_e};
                if (!fin) begin gw[`T_WRW-2 -: 2] = e[`T_WRW-2 -: 2]; end      // fr_ml: final windows only
                if (gw !== e) begin
                    nerr = nerr + 1;
                    $display("FAIL shot %0d window %0d: got %h expected %h", shot, win, gw, e);
                end
                $display("shot %0d window %0d fin %0d: bestE %0d bestC %0d ml %0d frame %0d fr_ml %0d sigma %h cycles %0d",
                         shot, win, fin, best_e, best_c, dec_ml, frame, fr_ml, sigma, cyc_win);
            end
        end
        if (row != `T_ROWS) begin nerr = nerr + 1; $display("FAIL rows %0d expected %0d", row, `T_ROWS); end
        if (nerr == 0) $display("PASS %0d shots x %0d windows, %0d sweep records x %0d replicas and every commit record bit-exact",
                                `T_NSHOT, `T_NWIN, row, NS);
        else $display("FAIL %0d errors", nerr);
        $finish;
    end
    always @(posedge clk) if (arr) begin
        got = {lab_all, C_all, E_all};
        if (row < `T_ROWS && got !== rec_mem[row]) begin
            nerr = nerr + 1;
            if (nerr < 10) $display("FAIL row %0d: got %h expected %h", row, got, rec_mem[row]);
        end
        row = row + 1;
    end
    initial begin #400000000; $display("FAIL timeout"); $finish; end
endmodule

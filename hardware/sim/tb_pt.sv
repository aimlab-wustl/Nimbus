// Testbench for the one-syndrome-per-shot top (pt_top): writes the runtime tables (thresholds per label, swap constants), seeds the
// generators, decodes T_NSHOT shots back to back and compares every replica's energy, class and label after every move cycle,
// the counts, both decisions and the minimum energy with the golden model. Before shot T_SWITCH the tables are replaced by a
// second set (idle write, then RECALC), as the board driver does when it changes the noise rate.
`timescale 1ns/1ps
`include "pt_params.svh"
`include "tb_params.svh"
module tb_pt;
    reg clk = 0, rst = 1;
    always #1.5 clk = ~clk;
    reg sd_sh = 0, push_a = 0, push_b = 0, ss_sh = 0, fill = 0, recalc = 0, syn_valid = 0;
    reg tw_en = 0, th_en = 0; reg [LW-1:0] tw_addr = 0, th_addr = 0; reg [KMB-1:0] tw_data = 0; reg [TW-1:0] th_data = 0;
    reg [NS-1:0] sd_bit; reg [NP-1:0] ss_bit;
    reg [M-1:0] syn;
    wire ready, busy, done, arr;
    wire [1:0] dec_ml, dec_me; wire [EW-1:0] dec_e;
    wire [19:0] cnt0, cnt1, cnt2, cnt3;
    wire [31:0] cyc_start, cyc_run, sweeps_run;
    wire [NS*EW-1:0] E_all; wire [2*NS-1:0] C_all; wire [NS*LW-1:0] lab_all;
    pt_top dut (.clk(clk), .rst(rst), .sd_sh(sd_sh), .push_a(push_a), .push_b(push_b), .ss_sh(ss_sh), .sd_bit(sd_bit), .ss_bit(ss_bit),
                .fill(fill), .recalc(recalc), .tw_en(tw_en), .tw_addr(tw_addr), .tw_data(tw_data), .th_en(th_en), .th_addr(th_addr),
                .th_data(th_data), .nsw(`T_NSW), .es_gmin(`T_GMIN), .es_th(`T_TH), .syn_valid(syn_valid), .syn(syn), .ready(ready),
                .busy(busy), .done(done), .dec_ml(dec_ml), .dec_me(dec_me), .dec_e(dec_e), .cnt0(cnt0), .cnt1(cnt1), .cnt2(cnt2),
                .cnt3(cnt3), .cyc_start(cyc_start), .cyc_run(cyc_run), .sweeps_run(sweeps_run), .arr(arr), .E_all(E_all),
                .C_all(C_all), .lab_all(lab_all));
    reg [NS-1:0] sa_mem [0:`T_PA-1];
    reg [NS-1:0] sb_mem [0:`T_PB-1];
    reg [NP-1:0] ss_mem [0:127];
    reg [M-1:0] shot_mem [0:`T_NSHOT-1];
    reg [`T_WB-1:0] rec_mem [0:`T_ROWS-1];
    reg [127:0] exp_mem [0:`T_NSHOT-1];
    reg [KMB-1:0] thr_mem [0:2*NS-1];
    reg [TW-1:0] th_mem [0:2*NP-1];
    integer k, row, nerr, shot, ncheck;
    reg [`T_WB-1:0] got;
    reg [127:0] e;
    task automatic tables(input integer set);
        integer t;
        begin
            for (t = 0; t < NS; t = t + 1) begin                  // commits >= 2 cycles apart (two-cycle write path)
                @(negedge clk); tw_en = 1; tw_addr = t; tw_data = thr_mem[set * NS + t];
                @(negedge clk); tw_en = 0;
            end
            @(negedge clk);
            for (t = 0; t < NP; t = t + 1) begin @(negedge clk); th_en = 1; th_addr = t; th_data = th_mem[set * NP + t]; end
            @(negedge clk); th_en = 0;
        end
    endtask
    initial begin
        $readmemb("seed_a.mem", sa_mem); $readmemb("seed_b.mem", sb_mem); $readmemb("seed_s.mem", ss_mem);
        $readmemh("shots.mem", shot_mem); $readmemh("exp_rec.mem", rec_mem); $readmemh("exp_shot.mem", exp_mem);
        $readmemh("thr.mem", thr_mem); $readmemh("theta.mem", th_mem);
        nerr = 0; row = 0; ncheck = 0;
        repeat (4) @(posedge clk);
        rst <= 0;
        tables(0);
        // padded seed streams, N bits staged then pushed (the last T_PA / T_PB bits pushed form the history)
        for (k = 0; k < `T_PA; k = k + 1) begin
            @(negedge clk); push_a = 0; sd_sh = 1; sd_bit = sa_mem[k];
            if (k % `T_N == `T_N - 1) begin @(negedge clk); sd_sh = 0; push_a = 1; end
        end
        @(negedge clk); push_a = 0;
        for (k = 0; k < `T_PB; k = k + 1) begin
            @(negedge clk); push_b = 0; sd_sh = 1; sd_bit = sb_mem[k];
            if (k % `T_N == `T_N - 1) begin @(negedge clk); sd_sh = 0; push_b = 1; end
        end
        @(negedge clk); push_b = 0; sd_sh = 0;
        for (k = 0; k < 128; k = k + 1) begin @(negedge clk); ss_sh = 1; ss_bit = ss_mem[k]; end
        @(negedge clk); ss_sh = 0; fill = 1;
        @(negedge clk); fill = 0;
        for (shot = 0; shot < `T_NSHOT; shot = shot + 1) begin
            while (!ready) @(negedge clk);
            if (shot == `T_SWITCH && shot > 0) begin
                tables(1);
                @(negedge clk); recalc = 1;
                @(negedge clk); recalc = 0;
                @(negedge clk);
                while (!ready) @(negedge clk);
            end
            syn = shot_mem[shot]; syn_valid = 1;
            @(negedge clk); syn_valid = 0;
            while (!done) @(negedge clk);
            e = exp_mem[shot];
            if ({sweeps_run, dec_me, dec_ml, cnt3, cnt2, cnt1, cnt0} !== e[115:0] || dec_e !== e[116 +: EW]) begin
                nerr = nerr + 1;
                $display("FAIL shot %0d: got me %0d ml %0d e %0d cnt %0d %0d %0d %0d, expected %h", shot, dec_me, dec_ml, dec_e,
                         cnt0, cnt1, cnt2, cnt3, e);
            end
            $display("shot %0d: ml %0d me %0d emin %0d cnt %0d %0d %0d %0d sweeps %0d  cycles load->start %0d start->decision %0d",
                     shot, dec_ml, dec_me, dec_e, cnt0, cnt1, cnt2, cnt3, sweeps_run, cyc_start, cyc_run);
        end
        if (row != `T_ROWS) begin nerr = nerr + 1; $display("FAIL rows %0d expected %0d", row, `T_ROWS); end
        if (nerr == 0) $display("PASS %0d shots, %0d move cycles x %0d replicas bit-exact (runtime tables, switch at shot %0d)",
                                `T_NSHOT, row, NS, `T_SWITCH);
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
    initial begin #50000000; $display("FAIL timeout"); $finish; end
endmodule

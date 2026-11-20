// Board-level testbench: replays the bus operations recorded from the PYNQ driver against the golden-model mock
// (hardware/sim/gen_ptw_board_case.py) through AXI4-Lite on ptw_board_top, with an AXI clock of 100 MHz and an unrelated decoder clock
// (about 323 MHz). op 0: write; op 1: read, compare under mask; op 2: poll until (read & mask) == data.
`timescale 1ns/1ps
`include "tb_board_params.svh"
module tb_ptw_board;
    reg aclk = 0, dclk = 0, arstn = 0, drstn = 0;
    always #5.0 aclk = ~aclk;
    always #1.55 dclk = ~dclk;
    reg [19:0] awaddr = 0, araddr = 0; reg awvalid = 0, wvalid = 0, bready = 1, arvalid = 0, rready = 1;
    reg [31:0] wdata = 0;
    wire awready, wready, bvalid, arready, rvalid, irq; wire [1:0] bresp, rresp; wire [31:0] rdata;
    ptw_board_top dut (.s_axi_aclk(aclk), .s_axi_aresetn(arstn), .s_axi_awaddr(awaddr), .s_axi_awprot(3'd0),
                      .s_axi_awvalid(awvalid), .s_axi_awready(awready), .s_axi_wdata(wdata), .s_axi_wstrb(4'hF),
                      .s_axi_wvalid(wvalid), .s_axi_wready(wready), .s_axi_bresp(bresp), .s_axi_bvalid(bvalid),
                      .s_axi_bready(bready), .s_axi_araddr(araddr), .s_axi_arprot(3'd0), .s_axi_arvalid(arvalid),
                      .s_axi_arready(arready), .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid),
                      .s_axi_rready(rready), .dclk(dclk), .dresetn(drstn), .irq(irq));
    // AXI protocol checks: no response before its address handshake
    reg aw_done = 0, ar_done = 0; integer perr = 0;
    always @(posedge aclk) begin
        if (awvalid && awready) aw_done <= 1;
        if (bvalid && bready) aw_done <= 0;
        if (arvalid && arready) ar_done <= 1;
        if (rvalid && rready) ar_done <= 0;
        if (bvalid && !aw_done && !(awvalid && awready)) begin perr = perr + 1; $display("PROTOCOL: BVALID before AW handshake"); end
        if (rvalid && !ar_done && !(arvalid && arready)) begin perr = perr + 1; $display("PROTOCOL: RVALID before AR handshake"); end
    end
    task automatic wr(input [19:0] a, input [31:0] d);
        begin
            @(negedge aclk); awaddr = a; wdata = d; awvalid = 1; wvalid = 1;
            while (!(awready && wready)) @(negedge aclk);
            @(negedge aclk); awvalid = 0; wvalid = 0;
            while (!bvalid) @(negedge aclk);
            @(negedge aclk);
        end
    endtask
    task automatic rd(input [19:0] a, output [31:0] d);
        begin
            @(negedge aclk); araddr = a; arvalid = 1;
            while (!arready) @(negedge aclk);
            @(negedge aclk); arvalid = 0;
            while (!rvalid) @(negedge aclk);
            d = rdata;
            @(negedge aclk);
        end
    endtask
    reg [87:0] ops [0:`T_NOPS-1];
    reg [3:0] op; reg [19:0] a; reg [31:0] dt, mk, v;
    integer i, nerr, nrd, npoll, nres;
    initial begin
        $readmemh("ops.mem", ops);
        nerr = 0; nrd = 0; npoll = 0; nres = 0;
        repeat (5) @(posedge aclk); arstn = 1;
        repeat (3) @(posedge dclk); drstn = 1;
        for (i = 0; i < `T_NOPS; i = i + 1) begin
            {op, a, dt, mk} = {ops[i][87:84], ops[i][83:64], ops[i][63:32], ops[i][31:0]};
            case (op)
                4'd0: wr(a, dt);
                4'd1: begin
                    rd(a, v); nrd = nrd + 1;
                    if (a >= 20'h40000 && a < 20'h80000) nres = nres + 1;
                    if (mk == 0 && (a == 20'h58 || a == 20'h5C || (a >= 20'h40000 && a < 20'h80000 && a[5:2] >= 4'd12))) $display("INFO op %0d read %h = %0d (not compared)", i, a, v);
                    if ((v & mk) !== (dt & mk)) begin
                        nerr = nerr + 1;
                        if (nerr < 20) $display("FAIL op %0d read %h: got %h expected %h (mask %h)", i, a, v, dt, mk);
                    end
                end
                4'd2: begin v = ~dt; while ((v & mk) !== (dt & mk)) rd(a, v); npoll = npoll + 1; end
                default: begin nerr = nerr + 1; $display("FAIL bad op %0d", op); end
            endcase
        end
        if (nerr == 0 && perr == 0) $display("PASS board %0d ops (%0d compared reads, %0d of them result words, %0d polls)", `T_NOPS, nrd,
                                             nres, npoll);
        else $display("FAIL %0d mismatches, %0d protocol errors", nerr, perr);
        $finish;
    end
    initial begin repeat (4) #1000000000; $display("FAIL timeout"); $finish; end
endmodule

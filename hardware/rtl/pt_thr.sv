// Runtime-loadable threshold store of one replica (one-syndrome-per-shot core).
//
// The table T[t][k] (label t, energy change k = 1..KM, B bits each; a row = KM*B bits, k = 1 at the LSB) is written over the
// register bus into a small distributed RAM per replica (every replica gets the same rows; broadcast write, only while the
// decoder is idle). The RAM is not in the label path. The replica keeps two registered rows, R0 and R1:
//   current row  = R[cur]  = T[lab],
//   other row    = R[~cur] = T[lab'], lab' = the label of this replica's partner in the next swap round
//                  (rounds alternate between even and odd pairs, starting with even pairs: lab' = lab + 1 if lab has the
//                  parity of the round, else lab - 1).
// In the threshold-load cycle of a round, mv = 1 if the replica's label changed (it can only become lab'), so the comparators
// see R[cur ^ mv]: a 2:1 select by two registers, no ROM and no bypass mux. The RAM then refills the other row for the next
// round (one read, at least NCL + GAP cycles before it is needed). At a shot start (ld) the labels reset to lab = RI: R0 is read
// in the cycle after ld (before the first comparator cycle), R1 one cycle later.
module pt_thr #(parameter integer NS = 16, parameter integer LW = 4, parameter integer W = 16, parameter integer RI = 0,
                parameter [NS*W-1:0] INIT = {NS*W{1'b0}}) (
    input  wire clk,
    input  wire we, input wire [LW-1:0] wa, input wire [W-1:0] wd,
    input  wire ld,                       // shot start (labels reset)
    input  wire rnd,                      // threshold-load cycle of a swap round (every round)
    input  wire mv,                       // this replica's label changed in that round
    output wire [W-1:0] thr_e
);
    localparam integer D = 1 << LW;
    (* ram_style = "distributed" *) reg [W-1:0] mem [0:D-1];
    integer r;
    initial for (r = 0; r < D; r = r + 1) mem[r] = (r < NS) ? INIT[r*W +: W] : {W{1'b0}};
    always @(posedge clk) if (we) mem[wa] <= wd;

    reg [LW-1:0] ra, lab;
    reg par; (* max_fanout = 24 *) reg cur;                         // parity of the next round; which row is current
    reg [1:0] ph;                         // 1: load the current row (after ld), 2: load the other row, 0: idle
    wire [W-1:0] rd = mem[ra];
    reg [W-1:0] r0, r1;
    wire sel = cur ^ mv;
    (* keep = "true" *) wire [W-1:0] thr_x = sel ? r1 : r0;      // one shared select (not folded into the comparators)
    assign thr_e = thr_x;
    wire [LW-1:0] alt = (lab[0] == par) ? lab + 1'b1 : lab - 1'b1;   // partner label in the next round
    wire [LW-1:0] labn = mv ? alt : lab;
    wire [LW-1:0] altn = (labn[0] == ~par) ? labn + 1'b1 : labn - 1'b1;
    always @(posedge clk) begin
        if (ph == 2'd1) begin if (cur) r1 <= rd; else r0 <= rd; end
        if (ph == 2'd2) begin if (cur) r0 <= rd; else r1 <= rd; end
        if (ld) begin
            lab <= RI[LW-1:0]; par <= 1'b0; cur <= 1'b0; ra <= RI[LW-1:0]; ph <= 2'd1;
        end else if (rnd) begin
            lab <= labn; par <= ~par; cur <= sel; ra <= altn; ph <= 2'd2;
        end else if (ph == 2'd1) begin
            ra <= alt; ph <= 2'd2;
        end else ph <= 2'd0;
    end
endmodule

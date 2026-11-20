// Balanced population count (recursive; leaves of up to 6 bits map to LUT6s).
module tp_popcnt #(
    parameter integer N = 8,
    parameter integer W = $clog2(N + 1)
) (
    input  wire [N-1:0] in,
    output wire [W-1:0] cnt
);
    generate
        if (N == 1) begin : g_one
            assign cnt = in;
        end else if (N <= 6) begin : g_leaf
            integer i;
            reg [W-1:0] s;
            always @* begin
                s = {W{1'b0}};
                for (i = 0; i < N; i = i + 1) s = s + in[i];
            end
            assign cnt = s;
        end else begin : g_split
            localparam integer NL = N / 2, NH = N - NL;
            localparam integer WL = $clog2(NL + 1), WH = $clog2(NH + 1);
            wire [WL-1:0] cl;
            wire [WH-1:0] ch;
            tp_popcnt #(.N(NL)) ul (.in(in[NL-1:0]), .cnt(cl));
            tp_popcnt #(.N(NH)) uh (.in(in[N-1:NL]), .cnt(ch));
            assign cnt = cl + ch;
        end
    endgenerate
endmodule

// Value of three weighted route terms in one LUT6: per route, a = switch open (alpha), b = clause bit (1 = -1):
// value(a, b) = (a & ~b) + ~(a & b) = a ? (b ? 0 : 2) : 1, so J*alpha*T = J*(value - 1).
module tp_cnt3 (
    input  wire [5:0] in,        // {b3, a3, b2, a2, b1, a1}
    output wire [2:0] s
);
    function automatic [1:0] val(input a, input b);
        val = a ? (b ? 2'd0 : 2'd2) : 2'd1;
    endfunction
    assign s = val(in[0], in[1]) + val(in[2], in[3]) + val(in[4], in[5]);
endmodule

// Constant multiplier c * A (modulo 2^WO), kept in LUTs.
(* use_dsp = "no" *)
module tp_cmul #(
    parameter integer A  = 3,
    parameter integer WC = 4,
    parameter integer WO = 18
) (
    input  wire [WC-1:0] c,
    output wire [WO-1:0] p
);
    localparam [WO-1:0] AA = A;
    assign p = AA * c;
endmodule

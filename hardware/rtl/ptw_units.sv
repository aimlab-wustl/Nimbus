// Architecture units of the real-time core. The data clause unit of the parallel
// and FF-slab layouts is the one-shot core's pt_clause (hardware/rtl/pt_units.sv), unchanged. The generator (hardware/gen/gen_ptw.py) wires the
// units through the switch fabric given by the move masks; bit-exact with nimbus.decoding.real_time.

// Measurement clause (weight wm): holds its literal m; output +1 if m = 0, -1 if m = 1 (the neuron reads m directly).
module ptw_mcl (
    input  wire clk, input wire ld, input wire m0,
    input  wire f,                            // flip request (OR of the connected neurons of the active color)
    output reg  m
);
    always @(posedge clk) m <= ld ? m0 : (m ^ f);
endmodule

// LUTRAM data clause unit (k >= 5 slabs): word a holds the literals (x, z) of slice row + a S. One address per cycle (the
// coloring guarantees one touching neuron per unit and color); the flip is written back into the same word in the cycle.
module ptw_dcl_ram #(parameter integer AW = 3) (
    input  wire clk, input wire ld, input wire [AW-1:0] a,
    input  wire x0, input wire z0, input wire fx, input wire fz,
    output wire x, output wire z, output wire ep, output wire exm, output wire ezm
);
    (* ram_style = "distributed" *) reg [1:0] mem [0:(1<<AW)-1];
    integer i;
    initial for (i = 0; i < (1 << AW); i = i + 1) mem[i] = 2'b00;
    wire [1:0] rd = mem[a];
    assign x = rd[0];
    assign z = rd[1];
    assign ep = ~x & ~z;
    assign exm = x & ~z;
    assign ezm = z & ~x;
    always @(posedge clk) mem[a] <= ld ? {z0, x0} : {z ^ fz, x ^ fx};
endmodule

// Data clause unit of the extra row (perfect slice T of a final window when k S = T, LUTRAM layouts): FF literals, one copy
// per local replica of the engine (r selects the copy that is read, flipped or loaded).
module ptw_dcl_x #(parameter integer NR = 1, parameter integer NRB = 1) (
    input  wire clk, input wire ld, input wire [NRB-1:0] r,
    input  wire x0, input wire z0, input wire fx, input wire fz,
    output wire x, output wire z, output wire ep, output wire exm, output wire ezm,
    output wire [NR-1:0] xa, output wire [NR-1:0] za
);
    reg [NR-1:0] xs = {NR{1'b0}}, zs = {NR{1'b0}};
    assign x = xs[r];
    assign z = zs[r];
    assign ep = ~x & ~z;
    assign exm = x & ~z;
    assign ezm = z & ~x;
    assign xa = xs;
    assign za = zs;
    always @(posedge clk) begin
        xs[r] <= ld ? x0 : (x ^ fx);
        zs[r] <= ld ? z0 : (z ^ fz);
    end
endmodule

// C-slow LUTRAM clause units (3-stage loop, replicas interleaved): read word ra in S1, write back word wa (the word read two
// cycles earlier) in S3 with its flip; dual-port distributed RAM (read port DPRA = ra, write / read port A = wa). Load: the
// word ra.
module ptw_dcl_ram2 #(parameter integer AW = 3) (
    input  wire clk, input wire ld, input wire [AW-1:0] ra, input wire [AW-1:0] wa,
    input  wire x0, input wire z0, input wire fx, input wire fz,
    output wire x, output wire z, output wire ep, output wire exm, output wire ezm
);
    (* ram_style = "distributed" *) reg [1:0] mem [0:(1<<AW)-1];
    integer i;
    initial for (i = 0; i < (1 << AW); i = i + 1) mem[i] = 2'b00;
    wire [AW-1:0] a = ld ? ra : wa;
    wire [1:0] rd = mem[ra];
    wire [1:0] old = mem[a];
    assign x = rd[0];
    assign z = rd[1];
    assign ep = ~x & ~z;
    assign exm = x & ~z;
    assign ezm = z & ~x;
    always @(posedge clk) mem[a] <= ld ? {z0, x0} : {old[1] ^ fz, old[0] ^ fx};
endmodule

module ptw_mcl_ram2 #(parameter integer AW = 3) (
    input  wire clk, input wire ld, input wire [AW-1:0] ra, input wire [AW-1:0] wa, input wire m0, input wire f,
    output wire m
);
    (* ram_style = "distributed" *) reg mem [0:(1<<AW)-1];
    integer i;
    initial for (i = 0; i < (1 << AW); i = i + 1) mem[i] = 1'b0;
    wire [AW-1:0] a = ld ? ra : wa;
    assign m = mem[ra];
    always @(posedge clk) mem[a] <= ld ? m0 : (mem[a] ^ f);
endmodule

// C-slow extra-row unit: read copy rr (S1), flip / load copy rw (S3).
module ptw_dcl_x2 #(parameter integer NR = 1, parameter integer NRB = 1) (
    input  wire clk, input wire ld, input wire [NRB-1:0] rr, input wire [NRB-1:0] rw,
    input  wire x0, input wire z0, input wire fx, input wire fz,
    output wire ep, output wire exm, output wire ezm, output wire [NR-1:0] xa, output wire [NR-1:0] za
);
    reg [NR-1:0] xs = {NR{1'b0}}, zs = {NR{1'b0}};
    wire x = xs[rr], z = zs[rr];
    assign ep = ~x & ~z;
    assign exm = x & ~z;
    assign ezm = z & ~x;
    assign xa = xs;
    assign za = zs;
    always @(posedge clk) begin
        xs[rw] <= ld ? x0 : (xs[rw] ^ fx);
        zs[rw] <= ld ? z0 : (zs[rw] ^ fz);
    end
endmodule

// LUTRAM measurement clause unit.
module ptw_mcl_ram #(parameter integer AW = 3) (
    input  wire clk, input wire ld, input wire [AW-1:0] a, input wire m0, input wire f,
    output wire m
);
    (* ram_style = "distributed" *) reg mem [0:(1<<AW)-1];
    integer i;
    initial for (i = 0; i < (1 << AW); i = i + 1) mem[i] = 1'b0;
    assign m = mem[a];
    always @(posedge clk) mem[a] <= ld ? m0 : (m ^ f);
endmodule

// SL transition neuron (stabilizer or logical line, b = 0): the one-shot core's pt_neuron rule, accept iff a = #up - #dn <= c, plus
// the accepted energy change da (0 when not accepted) for the incremental energy.
module ptw_sl #(parameter integer L = 4, parameter integer CB = 3, parameter integer DAW = 4) (
    input  wire en,
    input  wire [L-1:0] up, input wire [L-1:0] dn,
    input  wire [CB-1:0] c,
    output wire acc, output wire [DAW-1:0] da
);
    localparam integer CW = $clog2(L + 1);
    localparam integer WW = ((CW > CB) ? CW : CB) + 2;
    wire [CW-1:0] np, nm;
    tp_popcnt #(.N(L)) u_up (.in(up), .cnt(np));
    tp_popcnt #(.N(L)) u_dn (.in(dn), .cnt(nm));
    assign acc = en & (WW'(np) <= WW'(nm) + WW'(c));
    assign da = acc ? (DAW'(np) - DAW'(nm)) : {DAW{1'b0}};
endmodule

// T transition neuron (time-like move: ND data terms, NM = 1 or 2 measurement terms). dE = W a + wm b with a = #up - #dn,
// b = NM - 2 #m. Exact rank form: (a, b, mode) -> rank r of the positive level (0: dE <= 0);
// accept iff r <= c, c = the lane's thermometer count over the table's levels (T2: 7 levels, T1: 5 levels).
module ptw_tn #(parameter integer ND = 2, parameter integer NM = 2) (
    input  wire en,
    input  wire [ND-1:0] up, input wire [ND-1:0] dn, input wire [NM-1:0] m,
    input  wire mode,                         // 1: p < 1/28 (T2 levels 1 and 2 in the order 2W - 2wm, 2wm - W)
    input  wire [2:0] c,
    output wire acc, output wire [2:0] da, output wire [2:0] db
);
    integer i;
    reg signed [3:0] a, b;
    reg [2:0] r;
    always @* begin
        a = 0;
        for (i = 0; i < ND; i = i + 1) a = a + $signed({3'b000, up[i]}) - $signed({3'b000, dn[i]});
        b = NM;
        for (i = 0; i < NM; i = i + 1) b = b - $signed({2'b00, m[i], 1'b0});
        r = 3'd0;
        if (NM == 2) begin
            if      (a == 2  && b == -2) r = mode ? 3'd1 : 3'd2;
            else if (a == -1 && b == 2)  r = mode ? 3'd2 : 3'd1;
            else if (a == 1  && b == 0)  r = 3'd3;
            else if (a == 0  && b == 2)  r = 3'd4;
            else if (a == 2  && b == 0)  r = 3'd5;
            else if (a == 1  && b == 2)  r = 3'd6;
            else if (a == 2  && b == 2)  r = 3'd7;
        end else begin
            if      (a == 1  && b == -1) r = 3'd1;
            else if (a == 0  && b == 1)  r = 3'd2;
            else if (a == 2  && b == -1) r = 3'd3;
            else if (a == 1  && b == 1)  r = 3'd4;
            else if (a == 2  && b == 1)  r = 3'd5;
        end
    end
    assign acc = en & (r <= c);
    assign da = acc ? a[2:0] : 3'd0;
    assign db = acc ? b[2:0] : 3'd0;
endmodule

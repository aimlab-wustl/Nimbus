// Architecture units of the one-syndrome-per-shot replica: clause unit and transition neuron.
// The generator (hardware/gen/gen_pt.py) instantiates one pt_clause per qubit (cl_q<q>) and one pt_neuron per move (tn_r<r>) and
// wires them through the switch fabric given by the move masks (sw_up_r<r>, sw_dn_r<r>: clause -> neuron; fx / fz: neuron ->
// clause).

// Clause unit of qubit q: the clause x_q OR z_q (weight w3). It holds its own literals (the state lives in the clauses; each
// variable appears in exactly one clause in this problem). Its outputs are its value (for the energy) and its local energy
// change for a flip of x (ep = +1, exm = -1, else 0) and for a flip of z (ep = +1, ezm = -1, else 0): the clause changes value
// only when the other literal is 0. The transition neurons flip its literals directly over the switch connections (fx, fz);
// moves of one color never share a clause, so at most one neuron drives each port in a cycle.
module pt_clause (
    input  wire clk, input wire ld,
    input  wire c0x, input wire c0z,          // start state (loaded with the syndrome)
    input  wire fx, input wire fz,            // flip requests from the connected X / Z transition neurons
    output reg  x, output reg z,              // stored literals
    output wire v,                            // clause value x OR z (1 = unsatisfied, costs w3)
    output wire ep, output wire exm, output wire ezm
);
    assign v = x | z;
    assign ep = ~x & ~z;
    assign exm = x & ~z;
    assign ezm = z & ~x;
    always @(posedge clk) begin
        if (ld) begin x <= c0x; z <= c0z; end
        else begin x <= x ^ fx; z <= z ^ fz; end
    end
endmodule

// Transition neuron of move r: its input is the energy change of the move, dE = (#connected clauses at +1) - (#at -1), the
// sum of the clause outputs over its support. Acceptance (stochastic threshold rule, thermometer form): accept iff dE <= c,
// where c = #{k : u < T[lab][k]} is the registered comparator count of the neuron's lane (T non-increasing in k, so this
// equals "dE <= 0 or u < T[lab][dE]"). en = the neuron's color is active in this move cycle.
module pt_neuron #(parameter integer L = 4, parameter integer CB = 3) (
    input  wire en,
    input  wire [L-1:0] up, input wire [L-1:0] dn,
    input  wire [CB-1:0] c,
    output wire acc
);
    localparam integer CW = $clog2(L + 1);
    localparam integer WW = ((CW > CB) ? CW : CB) + 2;
    wire [CW-1:0] np, nm;
    tp_popcnt #(.N(L)) u_up (.in(up), .cnt(np));
    tp_popcnt #(.N(L)) u_dn (.in(dn), .cnt(nm));
    assign acc = en & (WW'(np) <= WW'(nm) + WW'(c));
endmodule

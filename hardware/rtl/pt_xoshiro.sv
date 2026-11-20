// xoshiro128** generator (32-bit output from the current state; the state steps when `step` = 1).
// Seed: 128 bits shifted in, first bit ends at s[0] bit 0 (state word order s0, s1, s2, s3 = bits 0-31, 32-63, ...).
// Bit-exact with nimbus.decoding.one_shot.xoshiro_next.
module pt_xoshiro (
    input  wire clk,
    input  wire seed_sh, input wire seed_bit,
    input  wire step,
    output wire [31:0] out
);
    reg [127:0] s;
    wire [31:0] s0 = s[31:0], s1 = s[63:32], s2 = s[95:64], s3 = s[127:96];
    wire [31:0] m5 = s1 * 32'd5;
    wire [31:0] r7 = {m5[24:0], m5[31:25]};
    assign out = r7 * 32'd9;
    wire [31:0] t = s1 << 9;
    wire [31:0] a2 = s2 ^ s0;
    wire [31:0] a3 = s3 ^ s1;
    wire [31:0] a1 = s1 ^ a2;
    wire [31:0] a0 = s0 ^ a3;
    wire [31:0] b2 = a2 ^ t;
    wire [31:0] b3 = {a3[20:0], a3[31:21]};
    always @(posedge clk) begin
        if (seed_sh) s <= {seed_bit, s[127:1]};
        else if (step) s <= {b3, b2, a1, a0};
    end
endmodule

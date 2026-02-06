//-----------------------------------------------------------------------------
// LFSR-Based Bitstream Generator for Stochastic Computing
//-----------------------------------------------------------------------------
// This module implements a Linear Feedback Shift Register (LFSR) based 
// bitstream generator as described in stochastic computing literature.
//
// The generator produces a stochastic bitstream where the probability of
// outputting a '1' equals k/L, where k is the input value and L is the
// bitstream length (2^N).
//
// Operation:
//   - LFSR generates pseudo-random numbers in range [0, L-1]
//   - Comparator outputs '1' if LFSR value < input value k
//   - This statistically produces k ones out of L bits
//
// Parameters:
//   N - Bit width (default 6 for L=64 bitstream length)
//
// Ports:
//   clk      - Clock input
//   rst_n    - Active-low synchronous reset
//   enable   - Enable signal for LFSR advancement
//   k        - Input value [0, 2^N - 1], represents probability k/(2^N)
//   seed     - Initial LFSR seed (must be non-zero)
//   load     - Load seed into LFSR when high
//   x_out    - Stochastic bitstream output
//   lfsr_val - Current LFSR value (for debugging/verification)
//-----------------------------------------------------------------------------

module lfsr_bitstream_gen #(
    parameter N = 8  // Bit width, L = 2^N (default: 128-bit bitstream)
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             enable,
    input  wire [N-1:0]     k,          // Input value to encode
    input  wire [N-1:0]     seed,       // LFSR seed (must be non-zero)
    input  wire             load,       // Load seed when high
    output wire             x_out,      // Stochastic bitstream output
    output wire [N-1:0]     lfsr_val    // Current LFSR value
);

    //-------------------------------------------------------------------------
    // LFSR Register
    //-------------------------------------------------------------------------
    reg [N-1:0] lfsr_reg;
    
    //-------------------------------------------------------------------------
    // Feedback tap selection based on LFSR width
    // Using Fibonacci LFSR (external XOR) with maximal-length polynomials
    // The feedback is XOR of tapped bits, shifted into MSB
    //-------------------------------------------------------------------------
    wire feedback;
    
    generate
        case (N)
            // Primitive polynomials for maximal-length sequences
            // Format: x^N + x^tap + 1 (XOR positions N-1 and tap-1 for 0-indexed)
            3:  assign feedback = lfsr_reg[0] ^ lfsr_reg[2];           // x^3 + x^2 + 1
            4:  assign feedback = lfsr_reg[0] ^ lfsr_reg[3];           // x^4 + x^3 + 1  
            5:  assign feedback = lfsr_reg[0] ^ lfsr_reg[2];           // x^5 + x^3 + 1
            6:  assign feedback = lfsr_reg[0] ^ lfsr_reg[5];           // x^6 + x^5 + 1
            7:  assign feedback = lfsr_reg[0] ^ lfsr_reg[6];           // x^7 + x^6 + 1
            8:  assign feedback = lfsr_reg[0] ^ lfsr_reg[2] ^ lfsr_reg[3] ^ lfsr_reg[4]; // x^8+x^6+x^5+x^4+1
            10: assign feedback = lfsr_reg[0] ^ lfsr_reg[6];           // x^10 + x^7 + 1
            12: assign feedback = lfsr_reg[0] ^ lfsr_reg[3] ^ lfsr_reg[5] ^ lfsr_reg[11]; // x^12+x^11+x^10+x^4+1
            16: assign feedback = lfsr_reg[0] ^ lfsr_reg[2] ^ lfsr_reg[3] ^ lfsr_reg[5];  // x^16+x^14+x^13+x^11+1
            default: assign feedback = lfsr_reg[0] ^ lfsr_reg[N-1];    // Generic fallback
        endcase
    endgenerate
    
    //-------------------------------------------------------------------------
    // LFSR State Machine - Fibonacci structure
    // Shift left, feedback enters LSB
    //-------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            // Reset to all 1s (valid non-zero state)
            lfsr_reg <= {N{1'b1}};
        end
        else if (load) begin
            // Load user-specified seed (ensure non-zero)
            lfsr_reg <= (seed == {N{1'b0}}) ? {N{1'b1}} : seed;
        end
        else if (enable) begin
            // Fibonacci LFSR: shift left, feedback enters bit 0
            lfsr_reg <= {lfsr_reg[N-2:0], feedback};
        end
    end

    wire [N-1:0] k_reg;
    register #(
        .WIDTH(N)
    ) k_register (
        .clk(clk),
        .rst_n(rst_n),
        .enable(load),
        .data_in(k),
        .data_out(k_reg)
    );
    
    //-------------------------------------------------------------------------
    // Stochastic Bit Generation
    // Output '1' if LFSR value < k, '0' otherwise
    // This produces a bitstream where P(x_out=1) ≈ k / 2^N
    //-------------------------------------------------------------------------
    assign x_out = (lfsr_reg < k_reg);
    
    //-------------------------------------------------------------------------
    // Debug output
    //-------------------------------------------------------------------------
    assign lfsr_val = lfsr_reg;

endmodule

module register #(
    parameter WIDTH = 8                           // Register width
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    enable,         // Load enable
    input  wire [WIDTH-1:0]        data_in,        // Data to store
    output reg  [WIDTH-1:0]        data_out        // Stored data
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= {WIDTH{1'b0}};
        end else if (enable) begin
            data_out <= data_in;
        end
    end
endmodule

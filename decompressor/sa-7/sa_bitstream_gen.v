//-----------------------------------------------------------------------------
// Streaming-Accurate (SA) Bitstream Generator for Stochastic Computing
//-----------------------------------------------------------------------------
// This module implements the SA bitstream generator as described in:
// "Streaming Accuracy: Characterizing Early Termination in Stochastic Computing"
// by Hsiao, San Miguel, and Anderson.
//
// The SA generator produces bitstreams that are optimal for early termination,
// achieving streaming accuracy of 1.0 at all partial bitstream lengths.
//
// Operation:
//   - Accumulator initialized to L/2 (midpoint)
//   - Each cycle: accumulator += k
//   - Output '1' on overflow, '0' otherwise
//   - This distributes 1s as evenly as possible across the bitstream
//
// Parameters:
//   N - Bit width (default 7 for L=128 bitstream length)
//
// Ports:
//   clk      - Clock input
//   rst_n    - Active-low synchronous reset
//   enable   - Enable signal for accumulator advancement
//   k        - Input value [0, 2^N - 1], represents probability k/(2^N)
//   start    - Initialize accumulator to L/2 (start new bitstream)
//   x_out    - Stochastic bitstream output
//   acc_val  - Current accumulator value (for debugging/verification)
//-----------------------------------------------------------------------------

module sa_bitstream_gen #(
    parameter N = 7  // Bit width, L = 2^N (default: 128-bit bitstream)
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             enable,
    input  wire [N-1:0]     k,          // Input value to encode
    input  wire             start,      // Start new bitstream (reset accumulator)
    output wire             x_out,      // Stochastic bitstream output
    output wire [N-1:0]     acc_val     // Current accumulator value
);

    //-------------------------------------------------------------------------
    // Accumulator Register
    //-------------------------------------------------------------------------
    reg [N-1:0] accumulator;
    
    //-------------------------------------------------------------------------
    // Adder with overflow detection
    // Sum is N+1 bits to capture overflow in MSB
    //-------------------------------------------------------------------------
    wire [N:0] sum;
    wire       overflow;

    wire [N-1:0] k_reg;
    register #(
        .WIDTH(N)
    ) k_register (
        .clk(clk),
        .rst_n(rst_n),
        .enable(1'b1),
        .data_in(k),
        .data_out(k_reg)
    );
    
    assign sum = {1'b0, accumulator} + {1'b0, k_reg};
    assign overflow = sum[N];  // Carry-out indicates overflow
    
    //-------------------------------------------------------------------------
    // Accumulator State Machine
    // - Reset/Start: Initialize to L/2 = 2^(N-1)
    // - Enable: Add k, keep lower N bits (automatic modulo 2^N)
    //-------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            // Reset to L/2 = 2^(N-1)
            accumulator <= {1'b1, {(N-1){1'b0}}};  // = L/2
        end
        else if (start) begin
            // Start new bitstream: reset to L/2
            accumulator <= {1'b1, {(N-1){1'b0}}};  // = L/2
        end
        else if (enable) begin
            // Add k, overflow wraps automatically (keep lower N bits)
            accumulator <= sum[N-1:0];
        end
    end
    
    //-------------------------------------------------------------------------
    // Output Generation
    // Output '1' when addition overflows, '0' otherwise
    //-------------------------------------------------------------------------
    assign x_out = enable & overflow;
    
    //-------------------------------------------------------------------------
    // Debug output
    //-------------------------------------------------------------------------
    assign acc_val = accumulator;

endmodule

module register #(
    parameter WIDTH = 7                           // Register width
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





module sipo #(
    parameter WIDTH = 4  // Default width of 4 bits, can be overridden
)(
    input wire clk,           // Clock signal
    input wire rst_n,         // Active low reset
    input wire shift_en,      // Shift enable signal
    input wire serial_in,     // Serial data input
    output reg [WIDTH-1:0] parallel_out  // Parallel data output
);

    // Shift register implementation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all bits to 0
            parallel_out <= {WIDTH{1'b0}};
        end else if (shift_en) begin
            // Shift left and insert new bit at LSB
            parallel_out <= {parallel_out[WIDTH-2:0], serial_in};
        end
        // If shift_en is low, hold current value
    end

endmodule

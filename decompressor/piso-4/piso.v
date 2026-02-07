module piso #(
    parameter WIDTH = 4  // Default width of 4 bits, can be overridden
)(
    input wire clk,           // Clock signal
    input wire rst_n,         // Active low reset
    input wire load_en,       // Load enable signal (load parallel data)
    input wire shift_en,      // Shift enable signal  
    input wire [WIDTH-1:0] parallel_in,  // Parallel data input
    output wire serial_out    // Serial data output (combinational)
);

    reg [WIDTH-1:0] shift_reg;  // Internal shift register

    // Serial output is directly from MSB of shift register (combinational)
    assign serial_out = shift_reg[WIDTH-1];

    // Shift register implementation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset shift register
            shift_reg <= {WIDTH{1'b0}};
        end else if (load_en) begin
            // Load parallel data into shift register
            shift_reg <= parallel_in;
        end else if (shift_en) begin
            // Shift left, fill with 0
            shift_reg <= {shift_reg[WIDTH-2:0], 1'b0};
        end
        // If neither load_en nor shift_en is high, hold current values
    end

endmodule

//============================================================================
// Adder module: Adds two inputs and produces sum
//============================================================================

module adder #(
    parameter A_WIDTH = 8,                         // Width of input A
    parameter B_WIDTH = 1,                         // Width of input B  
    parameter SUM_WIDTH = 8                        // Width of sum output
)(
    input  wire [A_WIDTH-1:0]      a,             // First operand
    input  wire [B_WIDTH-1:0]      b,             // Second operand
    output wire [SUM_WIDTH-1:0]    sum            // Sum output
);

    assign sum = a + {{(SUM_WIDTH-B_WIDTH){1'b0}}, b};

endmodule

//============================================================================
// Register module: Stores values with enable and clear functionality
//============================================================================

module register #(
    parameter WIDTH = 8                            // Register width
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

//============================================================================
// Accumulates input values: acc = acc + data_in
// Uses separate adder and register modules
//============================================================================

module accumulator #(
    parameter DATA_WIDTH = 1,                      // Input data width
    parameter ACC_WIDTH  = 8        // Accumulator width (extra bits for growth)
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    enable,         // Accumulate enable
    input  wire [DATA_WIDTH-1:0]   data_in,        // Data to accumulate
    output wire [ACC_WIDTH-1:0]    acc_out         // Accumulated result
);

    // Internal wire for adder output
    wire [ACC_WIDTH-1:0] sum;

    // Adder instance
    adder #(
        .A_WIDTH(ACC_WIDTH),
        .B_WIDTH(DATA_WIDTH),
        .SUM_WIDTH(ACC_WIDTH)
    ) u_adder (
        .a(acc_out),
        .b(data_in),
        .sum(sum)
    );

    // Register instance
    register #(
        .WIDTH(ACC_WIDTH)
    ) u_register (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .data_in(sum),
        .data_out(acc_out)
    );

endmodule
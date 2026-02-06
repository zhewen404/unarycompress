module histogram_compressor #(
    parameter STREAM_LENGTH = 128,
    parameter COUNTER_WIDTH = $clog2(STREAM_LENGTH+1)
) (
    input clk,
    input rst_n,
    input stream_a,
    input stream_b,
    input valid_in,
    
    // Compression outputs
    output reg [COUNTER_WIDTH-1:0] count_00,
    output reg [COUNTER_WIDTH-1:0] count_01,
    output reg [COUNTER_WIDTH-1:0] count_10,
    output reg [COUNTER_WIDTH-1:0] count_11
);

    // Compression Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count_00 <= 0;
            count_01 <= 0;
            count_10 <= 0;
            count_11 <= 0;
        end else begin
            if (valid_in) begin
                case ({stream_a, stream_b})
                    2'b00: count_00 <= count_00 + 1;
                    2'b01: count_01 <= count_01 + 1;
                    2'b10: count_10 <= count_10 + 1;
                    2'b11: count_11 <= count_11 + 1;
                endcase
            end
        end
    end

endmodule

module register #(
    parameter WIDTH = 32                            // Register width
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clear,          // Synchronous clear
    input  wire                    enable,         // Load enable
    input  wire [WIDTH-1:0]        data_in,        // Data to store
    output reg  [WIDTH-1:0]        data_out        // Stored data
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= {WIDTH{1'b0}};
        end else if (clear) begin
            data_out <= {WIDTH{1'b0}};
        end else if (enable) begin
            data_out <= data_in;
        end
    end
endmodule
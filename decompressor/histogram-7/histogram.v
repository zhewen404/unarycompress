// Linear Feedback Shift Register for random number generation
module lfsr_generator #(
    parameter LFSR_WIDTH = 3
) (
    input clk,
    input rst_n,
    input enable,
    output [LFSR_WIDTH-1:0] lfsr_out
);

    reg [LFSR_WIDTH-1:0] lfsr_state;
    
    // 3-bit LFSR polynomial: x^3 + x^2 + 1
    wire feedback;
    assign feedback = lfsr_state[2] ^ lfsr_state[1];
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr_state <= 3'b101; // Non-zero seed
        end else if (enable) begin
            lfsr_state <= {lfsr_state[1:0], feedback};
        end
    end
    
    assign lfsr_out = lfsr_state;

endmodule

module histogram_decompressor #(
    parameter STREAM_LENGTH = 128,
    parameter COUNTER_WIDTH = $clog2(STREAM_LENGTH+1),
    parameter LFSR_WIDTH = 3
) (
    input clk,
    input rst_n,
    input start_decompress,
    
    // Histogram inputs (compressed data)
    input [COUNTER_WIDTH-1:0] count_00,
    input [COUNTER_WIDTH-1:0] count_01,
    input [COUNTER_WIDTH-1:0] count_10,
    input [COUNTER_WIDTH-1:0] count_11,
    
    // Decompressed bitstream outputs
    output reg stream_a,
    output reg stream_b,
    output reg valid_out,
    output reg decompress_done
);

    // LFSR signals
    wire [LFSR_WIDTH-1:0] lfsr_value;
    
    // Working counters (count down from input values)
    reg [COUNTER_WIDTH-1:0] work_count_00;
    reg [COUNTER_WIDTH-1:0] work_count_01;
    reg [COUNTER_WIDTH-1:0] work_count_10;
    reg [COUNTER_WIDTH-1:0] work_count_11;
    
    // Output counter and active flag
    reg decompressing;
    
    // Random bin selection logic
    reg [1:0] selected_bin;
    
    // Instantiate LFSR
    lfsr_generator #(
        .LFSR_WIDTH(LFSR_WIDTH)
    ) lfsr_inst (
        .clk(clk),
        .rst_n(rst_n),
        .enable(decompressing),
        .lfsr_out(lfsr_value)
    );
    
    // Bin selection based on availability and random choice
    always @(*) begin
        selected_bin = 2'b00; // Default
        
        case (lfsr_value[1:0])
            2'b00: begin
                if (work_count_00 > 0) selected_bin = 2'b00;
                else if (work_count_01 > 0) selected_bin = 2'b01;
                else if (work_count_10 > 0) selected_bin = 2'b10;
                else if (work_count_11 > 0) selected_bin = 2'b11;
            end
            2'b01: begin
                if (work_count_01 > 0) selected_bin = 2'b01;
                else if (work_count_10 > 0) selected_bin = 2'b10;
                else if (work_count_11 > 0) selected_bin = 2'b11;
                else if (work_count_00 > 0) selected_bin = 2'b00;
            end
            2'b10: begin
                if (work_count_10 > 0) selected_bin = 2'b10;
                else if (work_count_11 > 0) selected_bin = 2'b11;
                else if (work_count_00 > 0) selected_bin = 2'b00;
                else if (work_count_01 > 0) selected_bin = 2'b01;
            end
            2'b11: begin
                if (work_count_11 > 0) selected_bin = 2'b11;
                else if (work_count_00 > 0) selected_bin = 2'b00;
                else if (work_count_01 > 0) selected_bin = 2'b01;
                else if (work_count_10 > 0) selected_bin = 2'b10;
            end
        endcase
    end
    
    // Main logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            work_count_00 <= 0;
            work_count_01 <= 0;
            work_count_10 <= 0;
            work_count_11 <= 0;
            stream_a <= 0;
            stream_b <= 0;
            valid_out <= 0;
            decompress_done <= 0;
            decompressing <= 0;
        end else begin
            // Start decompression
            if (start_decompress && !decompressing) begin
                work_count_00 <= count_00;
                work_count_01 <= count_01;
                work_count_10 <= count_10;
                work_count_11 <= count_11;
                decompress_done <= 0;
                decompressing <= 1;
                valid_out <= 0;
            end
            // Continue decompression
            else if (decompressing && (work_count_00 > 0 || work_count_01 > 0 || work_count_10 > 0 || work_count_11 > 0)) begin
                // Check if selected bin has remaining count and generate output
                case (selected_bin)
                    2'b00: begin
                        if (work_count_00 > 0) begin
                            {stream_a, stream_b} <= 2'b00;
                            valid_out <= 1;
                            work_count_00 <= work_count_00 - 1;
                        end else begin
                            valid_out <= 0;
                        end
                    end
                    2'b01: begin
                        if (work_count_01 > 0) begin
                            {stream_a, stream_b} <= 2'b01;
                            valid_out <= 1;
                            work_count_01 <= work_count_01 - 1;
                        end else begin
                            valid_out <= 0;
                        end
                    end
                    2'b10: begin
                        if (work_count_10 > 0) begin
                            {stream_a, stream_b} <= 2'b10;
                            valid_out <= 1;
                            work_count_10 <= work_count_10 - 1;
                        end else begin
                            valid_out <= 0;
                        end
                    end
                    2'b11: begin
                        if (work_count_11 > 0) begin
                            {stream_a, stream_b} <= 2'b11;
                            valid_out <= 1;
                            work_count_11 <= work_count_11 - 1;
                        end else begin
                            valid_out <= 0;
                        end
                    end
                endcase
                
                // Check if done (all counters are zero)
                if (work_count_00 == 1 && selected_bin == 2'b00 && work_count_01 == 0 && work_count_10 == 0 && work_count_11 == 0) begin
                    decompressing <= 0;
                    decompress_done <= 1;
                end else if (work_count_01 == 1 && selected_bin == 2'b01 && work_count_00 == 0 && work_count_10 == 0 && work_count_11 == 0) begin
                    decompressing <= 0;
                    decompress_done <= 1;
                end else if (work_count_10 == 1 && selected_bin == 2'b10 && work_count_00 == 0 && work_count_01 == 0 && work_count_11 == 0) begin
                    decompressing <= 0;
                    decompress_done <= 1;
                end else if (work_count_11 == 1 && selected_bin == 2'b11 && work_count_00 == 0 && work_count_01 == 0 && work_count_10 == 0) begin
                    decompressing <= 0;
                    decompress_done <= 1;
                end
            end
            // Finished decompression (all counters zero)
            else if (decompressing) begin
                decompressing <= 0;
                decompress_done <= 1;
                valid_out <= 0;
            end
            // Finished decompression
            else if (!decompressing) begin
                valid_out <= 0;
                if (!start_decompress) begin
                    decompress_done <= 0;
                end
            end
        end
    end

endmodule

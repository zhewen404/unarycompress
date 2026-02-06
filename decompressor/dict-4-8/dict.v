
// 8:1 Multiplexer
module mux_8to1 #(
    parameter CHUNK_SIZE = 4,
    parameter INDEX_BITS = 3
)(
    input wire [INDEX_BITS-1:0] select,
    // Flattened inputs for Verilog-2001 compatibility  
    input wire [CHUNK_SIZE-1:0] data0, data1, data2, data3,
    input wire [CHUNK_SIZE-1:0] data4, data5, data6, data7,
    output reg [CHUNK_SIZE-1:0] data_output
);

    // 8:1 MUX with case statement
    always @(*) begin
        case (select)
            3'd0: data_output = data0;
            3'd1: data_output = data1;
            3'd2: data_output = data2;
            3'd3: data_output = data3;
            3'd4: data_output = data4;
            3'd5: data_output = data5;
            3'd6: data_output = data6;
            3'd7: data_output = data7;
            default: data_output = {CHUNK_SIZE{1'b0}};
        endcase
    end
endmodule

// Modular decompressor using MUX component with parallel-to-serial output
module dict_decompressor #(
    parameter CHUNK_SIZE = 4,
    parameter CODEBOOK_SIZE = 8,
    parameter INDEX_BITS = $clog2(CODEBOOK_SIZE)
)(
    input wire clk,
    input wire rst_n,
    input wire [INDEX_BITS-1:0] compressed_index,
    input wire load,           // Load new chunk into shift register
    input wire shift_enable,   // Enable serial shifting
    output wire [CHUNK_SIZE-1:0] decompressed_chunk, // Parallel output
    output wire serial_out,    // Serial output (MSB first)
    output wire shift_done     // Indicates all bits have been shifted out
);

    // Hardwired codebook entries (same as compressor)
    wire [CHUNK_SIZE-1:0] cb0 = 4'b0000; // [0 0 0 0]
    wire [CHUNK_SIZE-1:0] cb1 = 4'b0010; // [0 0 1 0]
    wire [CHUNK_SIZE-1:0] cb2 = 4'b1001; // [1 0 0 1]
    wire [CHUNK_SIZE-1:0] cb3 = 4'b1011; // [1 0 1 1]
    wire [CHUNK_SIZE-1:0] cb4 = 4'b1111; // [1 1 1 1]
    wire [CHUNK_SIZE-1:0] cb5 = 4'b1000; // [1 0 0 0]
    wire [CHUNK_SIZE-1:0] cb6 = 4'b1100; // [1 1 0 0]
    wire [CHUNK_SIZE-1:0] cb7 = 4'b0111; // [0 1 1 1]

    // 8:1 MUX for decompression
    mux_8to1 #(
        .CHUNK_SIZE(CHUNK_SIZE),
        .INDEX_BITS(INDEX_BITS)
    ) mux_inst (
        .select(compressed_index),
        .data0(cb0), .data1(cb1), .data2(cb2), .data3(cb3),
        .data4(cb4), .data5(cb5), .data6(cb6), .data7(cb7),
        .data_output(decompressed_chunk)
    );

    // Parallel-to-serial shift register
    reg [CHUNK_SIZE-1:0] shift_reg;
    reg [$clog2(CHUNK_SIZE+1)-1:0] bit_count;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg <= {CHUNK_SIZE{1'b0}};
            bit_count <= 0;
        end else begin
            if (load) begin
                // Load decompressed chunk into shift register
                shift_reg <= decompressed_chunk;
                bit_count <= 0;
            end else if (shift_enable && bit_count < CHUNK_SIZE) begin
                // Shift left to output MSB first
                shift_reg <= {shift_reg[CHUNK_SIZE-2:0], 1'b0};
                bit_count <= bit_count + 1;
            end
        end
    end
    
    // Output assignments - output MSB
    assign serial_out = shift_reg[CHUNK_SIZE-1];  // MSB output first
    assign shift_done = (bit_count >= CHUNK_SIZE); // All bits shifted out

endmodule
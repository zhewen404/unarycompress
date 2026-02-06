
// 16:1 Multiplexer
module mux_16to1 #(
    parameter CHUNK_SIZE = 8,
    parameter INDEX_BITS = 4
)(
    input wire [INDEX_BITS-1:0] select,
    // Flattened inputs for Verilog-2001 compatibility  
    input wire [CHUNK_SIZE-1:0] data0, data1, data2, data3,
    input wire [CHUNK_SIZE-1:0] data4, data5, data6, data7,
    input wire [CHUNK_SIZE-1:0] data8, data9, data10, data11,
    input wire [CHUNK_SIZE-1:0] data12, data13, data14, data15,
    output reg [CHUNK_SIZE-1:0] data_output
);

    // 16:1 MUX with case statement
    always @(*) begin
        case (select)
            4'd0: data_output = data0;
            4'd1: data_output = data1;
            4'd2: data_output = data2;
            4'd3: data_output = data3;
            4'd4: data_output = data4;
            4'd5: data_output = data5;
            4'd6: data_output = data6;
            4'd7: data_output = data7;
            4'd8: data_output = data8;
            4'd9: data_output = data9;
            4'd10: data_output = data10;
            4'd11: data_output = data11;
            4'd12: data_output = data12;
            4'd13: data_output = data13;
            4'd14: data_output = data14;
            4'd15: data_output = data15;
            default: data_output = {CHUNK_SIZE{1'b0}};
        endcase
    end
endmodule

// Modular decompressor using MUX component with parallel-to-serial output
module dict_decompressor #(
    parameter CHUNK_SIZE = 8,
    parameter CODEBOOK_SIZE = 16,
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
    wire [CHUNK_SIZE-1:0] cb0 = 8'b00000000; // [0 0 0 0 0 0 0 0]
    wire [CHUNK_SIZE-1:0] cb1 = 8'b00100010; // [0 0 1 0 0 0 1 0]
    wire [CHUNK_SIZE-1:0] cb2 = 8'b10011001; // [1 0 0 1 1 0 0 1]
    wire [CHUNK_SIZE-1:0] cb3 = 8'b10111011; // [1 0 1 1 1 0 1 1]
    wire [CHUNK_SIZE-1:0] cb4 = 8'b11111111; // [1 1 1 1 1 1 1 1]
    wire [CHUNK_SIZE-1:0] cb5 = 8'b10001000; // [1 0 0 0 1 0 0 0]
    wire [CHUNK_SIZE-1:0] cb6 = 8'b11001100; // [1 1 0 0 1 1 0 0]
    wire [CHUNK_SIZE-1:0] cb7 = 8'b01110111; // [0 1 1 1 0 1 1 1]
    wire [CHUNK_SIZE-1:0] cb8 = 8'b00001111; // [0 0 0 0 1 1 1 1]
    wire [CHUNK_SIZE-1:0] cb9 = 8'b11110000; // [1 1 1 1 0 0 0 0]
    wire [CHUNK_SIZE-1:0] cb10 = 8'b01010101; // [0 1 0 1 0 1 0 1]
    wire [CHUNK_SIZE-1:0] cb11 = 8'b10101010; // [1 0 1 0 1 0 1 0]
    wire [CHUNK_SIZE-1:0] cb12 = 8'b00110011; // [0 0 1 1 0 0 1 1]
    wire [CHUNK_SIZE-1:0] cb13 = 8'b11001100; // [1 1 0 0 1 1 0 0]
    wire [CHUNK_SIZE-1:0] cb14 = 8'b11100011; // [1 1 1 0 0 0 1 1]
    wire [CHUNK_SIZE-1:0] cb15 = 8'b00011100; // [0 0 0 1 1 1 0 0]

    // 16:1 MUX for decompression
    mux_16to1 #(
        .CHUNK_SIZE(CHUNK_SIZE),
        .INDEX_BITS(INDEX_BITS)
    ) mux_inst (
        .select(compressed_index),
        .data0(cb0), .data1(cb1), .data2(cb2), .data3(cb3),
        .data4(cb4), .data5(cb5), .data6(cb6), .data7(cb7),
        .data8(cb8), .data9(cb9), .data10(cb10), .data11(cb11),
        .data12(cb12), .data13(cb13), .data14(cb14), .data15(cb15),
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
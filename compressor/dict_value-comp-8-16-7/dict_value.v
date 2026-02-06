// Modular compressor with integrated shift register for serial input
module dict_value_compressor #(
    parameter CHUNK_SIZE = 8,
    parameter CODEBOOK_SIZE = 16,
    parameter INDEX_BITS = $clog2(CODEBOOK_SIZE)
)(
    input wire clk,
    input wire rst_n,
    
    // Serial input interface
    input wire data_in,
    input wire data_valid,
    
    // Compression output
    output reg [INDEX_BITS-1:0] compressed_index,
    output reg compressed_valid
);

    // Shift register to accumulate input bits
    reg [CHUNK_SIZE-1:0] shift_reg;
    reg [$clog2(CHUNK_SIZE+1)-1:0] bit_count;

    // Shift register logic with integrated lookup
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg <= {CHUNK_SIZE{1'b0}};
            bit_count <= 0;
            compressed_index <= 0;
            compressed_valid <= 1'b0;
        end else begin
            compressed_valid <= 1'b0; // Default: no output
            
            if (data_valid) begin
                // Shift in new bit (MSB first)
                shift_reg <= {shift_reg[CHUNK_SIZE-2:0], data_in};
                
                if (bit_count == CHUNK_SIZE-1) begin
                    // We have a complete chunk, process it
                    bit_count <= 0;
                    compressed_valid <= 1'b1;
                    
                    // Direct lookup
                    case ({shift_reg[CHUNK_SIZE-2:0], data_in})
                        8'b00000000: compressed_index <= 4'd0;
                        8'b00000001: compressed_index <= 4'd1;
                        8'b00000010: compressed_index <= 4'd2;
                        8'b00000011: compressed_index <= 4'd3;
                        8'b00000100: compressed_index <= 4'd7;
                        8'b00000101: compressed_index <= 4'd5;
                        8'b00000110: compressed_index <= 4'd6;
                        8'b00000111: compressed_index <= 4'd7;
                        8'b00001000: compressed_index <= 4'd8;
                        8'b00001001: compressed_index <= 4'd9;
                        8'b00001010: compressed_index <= 4'd10;
                        8'b00001011: compressed_index <= 4'd11;
                        8'b00001100: compressed_index <= 4'd12;
                        8'b00001101: compressed_index <= 4'd13;
                        8'b00001110: compressed_index <= 4'd14;
                        8'b00001111: compressed_index <= 4'd15;
                        8'b00010000: compressed_index <= 4'd15;
                        8'b00010001: compressed_index <= 4'd14;
                        8'b00010010: compressed_index <= 4'd13;
                        8'b00010011: compressed_index <= 4'd12;
                        8'b00010100: compressed_index <= 4'd11;
                        8'b00010101: compressed_index <= 4'd10;
                        8'b00010110: compressed_index <= 4'd9;
                        8'b00010111: compressed_index <= 4'd8;
                        8'b00011000: compressed_index <= 4'd7;
                        8'b00011001: compressed_index <= 4'd6;
                        8'b00011010: compressed_index <= 4'd5;
                        8'b00011011: compressed_index <= 4'd4;
                        8'b00011100: compressed_index <= 4'd3;
                        8'b00011101: compressed_index <= 4'd2;
                        8'b00011110: compressed_index <= 4'd1;
                        8'b00011111: compressed_index <= 4'd5;
                        8'b00100000: compressed_index <= 4'd9;
                        8'b00100001: compressed_index <= 4'd1;
                        8'b00100010: compressed_index <= 4'd2;
                        8'b00100011: compressed_index <= 4'd3;
                        8'b00100100: compressed_index <= 4'd4;
                        8'b00100101: compressed_index <= 4'd5;
                        8'b00100110: compressed_index <= 4'd6;
                        8'b00100111: compressed_index <= 4'd7;
                        8'b00101000: compressed_index <= 4'd8;
                        8'b00101001: compressed_index <= 4'd9;
                        8'b00101010: compressed_index <= 4'd10;
                        8'b00101011: compressed_index <= 4'd11;
                        8'b00101100: compressed_index <= 4'd12;
                        8'b00101101: compressed_index <= 4'd13;
                        8'b00101110: compressed_index <= 4'd14;
                        8'b00101111: compressed_index <= 4'd15;
                        8'b00110000: compressed_index <= 4'd15;
                        8'b00110001: compressed_index <= 4'd14;
                        8'b00110010: compressed_index <= 4'd13;
                        8'b00110011: compressed_index <= 4'd12;
                        8'b00110100: compressed_index <= 4'd11;
                        8'b00110101: compressed_index <= 4'd10;
                        8'b00110110: compressed_index <= 4'd0;
                        8'b00110111: compressed_index <= 4'd8;
                        8'b00111000: compressed_index <= 4'd7;
                        8'b00111001: compressed_index <= 4'd6;
                        8'b00111010: compressed_index <= 4'd5;
                        8'b00111011: compressed_index <= 4'd4;
                        8'b00111100: compressed_index <= 4'd3;
                        8'b00111101: compressed_index <= 4'd2;
                        8'b00111110: compressed_index <= 4'd1;
                        8'b00111111: compressed_index <= 4'd0;
                        8'b01000000: compressed_index <= 4'd0;
                        8'b01000001: compressed_index <= 4'd1;
                        8'b01000010: compressed_index <= 4'd2;
                        8'b01000011: compressed_index <= 4'd3;
                        8'b01000100: compressed_index <= 4'd4;
                        8'b01000101: compressed_index <= 4'd5;
                        8'b01000110: compressed_index <= 4'd6;
                        8'b01000111: compressed_index <= 4'd7;
                        8'b01001000: compressed_index <= 4'd8;
                        8'b01001001: compressed_index <= 4'd9;
                        8'b01001010: compressed_index <= 4'd10;
                        8'b01001011: compressed_index <= 4'd11;
                        8'b01001100: compressed_index <= 4'd12;
                        8'b01001101: compressed_index <= 4'd13;
                        8'b01001110: compressed_index <= 4'd14;
                        8'b01001111: compressed_index <= 4'd15;
                        8'b01010000: compressed_index <= 4'd15;
                        8'b01010001: compressed_index <= 4'd14;
                        8'b01010010: compressed_index <= 4'd13;
                        8'b01010011: compressed_index <= 4'd12;
                        8'b01010100: compressed_index <= 4'd11;
                        8'b01010101: compressed_index <= 4'd10;
                        8'b01010110: compressed_index <= 4'd9;
                        8'b01010111: compressed_index <= 4'd8;
                        8'b01011000: compressed_index <= 4'd7;
                        8'b01011001: compressed_index <= 4'd6;
                        8'b01011010: compressed_index <= 4'd5;
                        8'b01011011: compressed_index <= 4'd4;
                        8'b01011100: compressed_index <= 4'd3;
                        8'b01011101: compressed_index <= 4'd2;
                        8'b01011110: compressed_index <= 4'd1;
                        8'b01011111: compressed_index <= 4'd0;
                        8'b01100000: compressed_index <= 4'd0;
                        8'b01100001: compressed_index <= 4'd1;
                        8'b01100010: compressed_index <= 4'd2;
                        8'b01100011: compressed_index <= 4'd3;
                        8'b01100100: compressed_index <= 4'd4;
                        8'b01100101: compressed_index <= 4'd5;
                        8'b01100110: compressed_index <= 4'd6;
                        8'b01100111: compressed_index <= 4'd7;
                        8'b01101000: compressed_index <= 4'd8;
                        8'b01101001: compressed_index <= 4'd9;
                        8'b01101010: compressed_index <= 4'd10;
                        8'b01101011: compressed_index <= 4'd11;
                        8'b01101100: compressed_index <= 4'd12;
                        8'b01101101: compressed_index <= 4'd13;
                        8'b01101110: compressed_index <= 4'd14;
                        8'b01101111: compressed_index <= 4'd15;
                        8'b01110000: compressed_index <= 4'd15;
                        8'b01110001: compressed_index <= 4'd14;
                        8'b01110010: compressed_index <= 4'd13;
                        8'b01110011: compressed_index <= 4'd12;
                        8'b01110100: compressed_index <= 4'd11;
                        8'b01110101: compressed_index <= 4'd10;
                        8'b01110110: compressed_index <= 4'd9;
                        8'b01110111: compressed_index <= 4'd8;
                        8'b01111000: compressed_index <= 4'd7;
                        8'b01111001: compressed_index <= 4'd6;
                        8'b01111010: compressed_index <= 4'd5;
                        8'b01111011: compressed_index <= 4'd4;
                        8'b01111100: compressed_index <= 4'd3;
                        8'b01111101: compressed_index <= 4'd2;
                        8'b01111110: compressed_index <= 4'd1;
                        8'b01111111: compressed_index <= 4'd0;
                        8'b10000000: compressed_index <= 4'd0;
                        8'b10000001: compressed_index <= 4'd1;
                        8'b10000010: compressed_index <= 4'd2;
                        8'b10000011: compressed_index <= 4'd3;
                        8'b10000100: compressed_index <= 4'd4;
                        8'b10000101: compressed_index <= 4'd0;
                        8'b10000110: compressed_index <= 4'd6;
                        8'b10000111: compressed_index <= 4'd7;
                        8'b10001000: compressed_index <= 4'd8;
                        8'b10001001: compressed_index <= 4'd9;
                        8'b10001010: compressed_index <= 4'd10;
                        8'b10001011: compressed_index <= 4'd11;
                        8'b10001100: compressed_index <= 4'd12;
                        8'b10001101: compressed_index <= 4'd13;
                        8'b10001110: compressed_index <= 4'd14;
                        8'b10001111: compressed_index <= 4'd15;
                        8'b10010000: compressed_index <= 4'd15;
                        8'b10010001: compressed_index <= 4'd14;
                        8'b10010010: compressed_index <= 4'd13;
                        8'b10010011: compressed_index <= 4'd12;
                        8'b10010100: compressed_index <= 4'd11;
                        8'b10010101: compressed_index <= 4'd10;
                        8'b10010110: compressed_index <= 4'd11;
                        8'b10010111: compressed_index <= 4'd8;
                        8'b10011000: compressed_index <= 4'd7;
                        8'b10011001: compressed_index <= 4'd6;
                        8'b10011010: compressed_index <= 4'd5;
                        8'b10011011: compressed_index <= 4'd4;
                        8'b10011100: compressed_index <= 4'd3;
                        8'b10011101: compressed_index <= 4'd2;
                        8'b10011110: compressed_index <= 4'd1;
                        8'b10011111: compressed_index <= 4'd0;
                        8'b10100000: compressed_index <= 4'd0;
                        8'b10100001: compressed_index <= 4'd1;
                        8'b10100010: compressed_index <= 4'd2;
                        8'b10100011: compressed_index <= 4'd3;
                        8'b10100100: compressed_index <= 4'd4;
                        8'b10100101: compressed_index <= 4'd5;
                        8'b10100110: compressed_index <= 4'd6;
                        8'b10100111: compressed_index <= 4'd7;
                        8'b10101000: compressed_index <= 4'd8;
                        8'b10101001: compressed_index <= 4'd9;
                        8'b10101010: compressed_index <= 4'd10;
                        8'b10101011: compressed_index <= 4'd11;
                        8'b10101100: compressed_index <= 4'd12;
                        8'b10101101: compressed_index <= 4'd13;
                        8'b10101110: compressed_index <= 4'd14;
                        8'b10101111: compressed_index <= 4'd15;
                        8'b10110000: compressed_index <= 4'd15;
                        8'b10110001: compressed_index <= 4'd14;
                        8'b10110010: compressed_index <= 4'd13;
                        8'b10110011: compressed_index <= 4'd12;
                        8'b10110100: compressed_index <= 4'd11;
                        8'b10110101: compressed_index <= 4'd10;
                        8'b10110110: compressed_index <= 4'd9;
                        8'b10110111: compressed_index <= 4'd8;
                        8'b10111000: compressed_index <= 4'd7;
                        8'b10111001: compressed_index <= 4'd6;
                        8'b10111010: compressed_index <= 4'd5;
                        8'b10111011: compressed_index <= 4'd4;
                        8'b10111100: compressed_index <= 4'd3;
                        8'b10111101: compressed_index <= 4'd2;
                        8'b10111110: compressed_index <= 4'd1;
                        8'b10111111: compressed_index <= 4'd0;
                        8'b11000000: compressed_index <= 4'd0;
                        8'b11000001: compressed_index <= 4'd1;
                        8'b11000010: compressed_index <= 4'd2;
                        8'b11000011: compressed_index <= 4'd3;
                        8'b11000100: compressed_index <= 4'd4;
                        8'b11000101: compressed_index <= 4'd5;
                        8'b11000110: compressed_index <= 4'd6;
                        8'b11000111: compressed_index <= 4'd7;
                        8'b11001000: compressed_index <= 4'd8;
                        8'b11001001: compressed_index <= 4'd9;
                        8'b11001010: compressed_index <= 4'd10;
                        8'b11001011: compressed_index <= 4'd11;
                        8'b11001100: compressed_index <= 4'd12;
                        8'b11001101: compressed_index <= 4'd13;
                        8'b11001110: compressed_index <= 4'd14;
                        8'b11001111: compressed_index <= 4'd15;
                        8'b11010000: compressed_index <= 4'd15;
                        8'b11010001: compressed_index <= 4'd14;
                        8'b11010010: compressed_index <= 4'd13;
                        8'b11010011: compressed_index <= 4'd12;
                        8'b11010100: compressed_index <= 4'd11;
                        8'b11010101: compressed_index <= 4'd10;
                        8'b11010110: compressed_index <= 4'd9;
                        8'b11010111: compressed_index <= 4'd8;
                        8'b11011000: compressed_index <= 4'd7;
                        8'b11011001: compressed_index <= 4'd6;
                        8'b11011010: compressed_index <= 4'd5;
                        8'b11011011: compressed_index <= 4'd4;
                        8'b11011100: compressed_index <= 4'd3;
                        8'b11011101: compressed_index <= 4'd2;
                        8'b11011110: compressed_index <= 4'd1;
                        8'b11011111: compressed_index <= 4'd0;
                        8'b11100000: compressed_index <= 4'd0;
                        8'b11100001: compressed_index <= 4'd1;
                        8'b11100010: compressed_index <= 4'd2;
                        8'b11100011: compressed_index <= 4'd3;
                        8'b11100100: compressed_index <= 4'd4;
                        8'b11100101: compressed_index <= 4'd5;
                        8'b11100110: compressed_index <= 4'd6;
                        8'b11100111: compressed_index <= 4'd4;
                        8'b11101000: compressed_index <= 4'd8;
                        8'b11101001: compressed_index <= 4'd9;
                        8'b11101010: compressed_index <= 4'd10;
                        8'b11101011: compressed_index <= 4'd9;
                        8'b11101100: compressed_index <= 4'd12;
                        8'b11101101: compressed_index <= 4'd13;
                        8'b11101110: compressed_index <= 4'd14;
                        8'b11101111: compressed_index <= 4'd15;
                        8'b11110000: compressed_index <= 4'd15;
                        8'b11110001: compressed_index <= 4'd14;
                        8'b11110010: compressed_index <= 4'd13;
                        8'b11110011: compressed_index <= 4'd12;
                        8'b11110100: compressed_index <= 4'd11;
                        8'b11110101: compressed_index <= 4'd10;
                        8'b11110110: compressed_index <= 4'd9;
                        8'b11110111: compressed_index <= 4'd8;
                        8'b11111000: compressed_index <= 4'd7;
                        8'b11111001: compressed_index <= 4'd6;
                        8'b11111010: compressed_index <= 4'd5;
                        8'b11111011: compressed_index <= 4'd4;
                        8'b11111100: compressed_index <= 4'd3;
                        8'b11111101: compressed_index <= 4'd2;
                        8'b11111110: compressed_index <= 4'd1;
                        8'b11111111: compressed_index <= 4'd0;
                        default: compressed_index <= 4'd0;
                    endcase
                end else begin
                    bit_count <= bit_count + 1;
                end
            end
        end
    end

endmodule

module dict_value_compressor_with_reg #(
    parameter CHUNK_SIZE = 4,
    parameter CODEBOOK_SIZE = 8,
    parameter INDEX_BITS = $clog2(CODEBOOK_SIZE),
    parameter NUM_CHUNKS = 32
)(
    input wire clk,
    input wire rst_n,
    
    // Serial input interface
    input wire data_in,
    input wire data_valid,
    
    // Compression output
    output wire [(NUM_CHUNKS * INDEX_BITS)-1:0] compressed_output,
    output reg compression_done
);
    // Internal signals
    wire [INDEX_BITS-1:0] compressed_index;
    wire compressed_valid;
    
    // Instantiate the compressor module
    dict_value_compressor #(
        .CHUNK_SIZE(CHUNK_SIZE),
        .CODEBOOK_SIZE(CODEBOOK_SIZE)
    ) compressor_inst (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .data_valid(data_valid),
        .compressed_index(compressed_index),
        .compressed_valid(compressed_valid)
    );

    // NUM_CHUNKS Registers to store compressed indices
    reg [INDEX_BITS-1:0] stored_indices [0:NUM_CHUNKS-1];
    reg [$clog2(NUM_CHUNKS+1)-1:0] chunk_counter;
    
    // Store compressed indices as they arrive
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            chunk_counter <= 0;
            compression_done <= 1'b0;
        end else begin
            if (compressed_valid && chunk_counter < NUM_CHUNKS) begin
                stored_indices[chunk_counter] <= compressed_index;
                chunk_counter <= chunk_counter + 1;
                
                // Signal completion when all chunks are stored
                if (chunk_counter == NUM_CHUNKS - 1) begin
                    compression_done <= 1'b1;
                end
            end
        end
    end
    
    // Pack stored indices into output vector
    genvar i;
    generate
        for (i = 0; i < NUM_CHUNKS; i = i + 1) begin : gen_output
            assign compressed_output[(i+1)*INDEX_BITS-1 : i*INDEX_BITS] = stored_indices[i];
        end
    endgenerate
endmodule

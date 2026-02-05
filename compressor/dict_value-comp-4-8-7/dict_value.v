// Modular compressor with integrated shift register for serial input
module dict_value_compressor #(
    parameter CHUNK_SIZE = 4,
    parameter CODEBOOK_SIZE = 8,
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
    
    // Internal compression signals
    wire [CHUNK_SIZE-1:0] chunk_to_compress;
    reg [INDEX_BITS-1:0] compression_result;

    // Lookup table for direct mapping from 4-bit input to best codebook index
    // Pre-calculated based on: weight priority + hamming distance tiebreaker
    // Codebook: cb0=0000, cb1=0010, cb2=1001, cb3=1011, cb4=1111, cb5=1000, cb6=1100, cb7=0111
    always @(*) begin
        case (chunk_to_compress)
            4'b0000: compression_result = 3'd0; 
            4'b0001: compression_result = 3'd1; 
            4'b0010: compression_result = 3'd1; 
            4'b0011: compression_result = 3'd2; 
            4'b0100: compression_result = 3'd5; 
            4'b0101: compression_result = 3'd2; 
            4'b0110: compression_result = 3'd6; 
            4'b0111: compression_result = 3'd7; 
            4'b1000: compression_result = 3'd5; 
            4'b1001: compression_result = 3'd2; 
            4'b1010: compression_result = 3'd2; 
            4'b1011: compression_result = 3'd3; 
            4'b1100: compression_result = 3'd6; 
            4'b1101: compression_result = 3'd3; 
            4'b1110: compression_result = 3'd3; 
            4'b1111: compression_result = 3'd4; 
            default: compression_result = 3'd0;
        endcase
    end

    // Shift register logic
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
                    compressed_index <= compression_result;
                    compressed_valid <= 1'b1;
                end else begin
                    bit_count <= bit_count + 1;
                end
            end
        end
    end
    
    // Assign chunk for compression (current shift register contents)
    assign chunk_to_compress = {shift_reg[CHUNK_SIZE-2:0], data_in};

endmodule

module register #(
    parameter WIDTH = 8                            // Register width
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

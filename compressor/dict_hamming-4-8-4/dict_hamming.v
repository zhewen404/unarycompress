// Dictionary Compression using Hamming Distance - Modular Design
// Separate parameterized modules for each component (Verilog-2001 compatible)

// Single Hamming distance calculator
module hamming_distance_calc #(
    parameter CHUNK_SIZE = 4
)(
    input wire [CHUNK_SIZE-1:0] input_chunk,
    input wire [CHUNK_SIZE-1:0] codebook_entry,
    output wire [$clog2(CHUNK_SIZE+1)-1:0] hamming_distance
);

    wire [CHUNK_SIZE-1:0] xor_result;
    
    // XOR gates (CHUNK_SIZE per calculator)
    assign xor_result = input_chunk ^ codebook_entry;
    
    // Population count (sum XOR outputs) - parameterized
    wire [$clog2(CHUNK_SIZE+1)-1:0] bit_sums [0:CHUNK_SIZE-1];
    
    genvar i;
    generate
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin : gen_popcount
            if (i == 0) begin
                assign bit_sums[i] = xor_result[i];
            end else begin
                assign bit_sums[i] = bit_sums[i-1] + xor_result[i];
            end
        end
    endgenerate
    
    assign hamming_distance = bit_sums[CHUNK_SIZE-1];

endmodule

// Comparator tree + Priority encoder (finds minimum index)
module min_finder #(
    parameter CODEBOOK_SIZE = 8,
    parameter INDEX_BITS = $clog2(CODEBOOK_SIZE),
    parameter DISTANCE_BITS = 3
)(
    // Flattened inputs for Verilog-2001 compatibility
    input wire [DISTANCE_BITS-1:0] dist0, dist1, dist2, dist3,
    input wire [DISTANCE_BITS-1:0] dist4, dist5, dist6, dist7,
    output reg [INDEX_BITS-1:0] min_index
);

    // Combinational logic to find minimum distance index
    always @(*) begin
        // Initialize with first entry
        min_index = 0;
        
        // Compare all distances to find minimum (priority encoder)
        if (dist1 < dist0 && (min_index == 0)) min_index = 1;
        else if (dist2 < ((min_index == 0) ? dist0 : dist1) && (min_index <= 1)) min_index = 2;
        else if (dist3 < ((min_index == 0) ? dist0 : (min_index == 1) ? dist1 : dist2) && (min_index <= 2)) min_index = 3;
        
        // Simplified sequential comparison
        if (dist1 < dist0) min_index = 1; else min_index = 0;
        if (dist2 < ((min_index == 0) ? dist0 : dist1)) min_index = 2;
        if (dist3 < ((min_index == 0) ? dist0 : (min_index == 1) ? dist1 : dist2)) min_index = 3;
        if (dist4 < ((min_index == 0) ? dist0 : (min_index == 1) ? dist1 : (min_index == 2) ? dist2 : dist3)) min_index = 4;
        if (dist5 < ((min_index == 0) ? dist0 : (min_index == 1) ? dist1 : (min_index == 2) ? dist2 : (min_index == 3) ? dist3 : dist4)) min_index = 5;
        if (dist6 < ((min_index == 0) ? dist0 : (min_index == 1) ? dist1 : (min_index == 2) ? dist2 : (min_index == 3) ? dist3 : (min_index == 4) ? dist4 : dist5)) min_index = 6;
        if (dist7 < ((min_index == 0) ? dist0 : (min_index == 1) ? dist1 : (min_index == 2) ? dist2 : (min_index == 3) ? dist3 : (min_index == 4) ? dist4 : (min_index == 5) ? dist5 : dist6)) min_index = 7;
    end

endmodule


// Modular compressor with integrated shift register for serial input
module dict_hamming_compressor #(
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
    wire [INDEX_BITS-1:0] compression_result;

    // Hardwired codebook entries
    wire [CHUNK_SIZE-1:0] cb0 = 4'b0000; // Weight 0
    wire [CHUNK_SIZE-1:0] cb1 = 4'b0001; // Weight 1
    wire [CHUNK_SIZE-1:0] cb2 = 4'b1000; // Weight 1
    wire [CHUNK_SIZE-1:0] cb3 = 4'b0011; // Weight 2
    wire [CHUNK_SIZE-1:0] cb4 = 4'b1100; // Weight 2
    wire [CHUNK_SIZE-1:0] cb5 = 4'b0111; // Weight 3
    wire [CHUNK_SIZE-1:0] cb6 = 4'b1110; // Weight 3
    wire [CHUNK_SIZE-1:0] cb7 = 4'b1111; // Weight 4

    // Hamming distances from separate calculator modules
    wire [$clog2(CHUNK_SIZE+1)-1:0] hd0, hd1, hd2, hd3, hd4, hd5, hd6, hd7;
    
    // 8 separate Hamming distance calculators
    hamming_distance_calc #(.CHUNK_SIZE(CHUNK_SIZE)) calc0 (.input_chunk(chunk_to_compress), .codebook_entry(cb0), .hamming_distance(hd0));
    hamming_distance_calc #(.CHUNK_SIZE(CHUNK_SIZE)) calc1 (.input_chunk(chunk_to_compress), .codebook_entry(cb1), .hamming_distance(hd1));
    hamming_distance_calc #(.CHUNK_SIZE(CHUNK_SIZE)) calc2 (.input_chunk(chunk_to_compress), .codebook_entry(cb2), .hamming_distance(hd2));
    hamming_distance_calc #(.CHUNK_SIZE(CHUNK_SIZE)) calc3 (.input_chunk(chunk_to_compress), .codebook_entry(cb3), .hamming_distance(hd3));
    hamming_distance_calc #(.CHUNK_SIZE(CHUNK_SIZE)) calc4 (.input_chunk(chunk_to_compress), .codebook_entry(cb4), .hamming_distance(hd4));
    hamming_distance_calc #(.CHUNK_SIZE(CHUNK_SIZE)) calc5 (.input_chunk(chunk_to_compress), .codebook_entry(cb5), .hamming_distance(hd5));
    hamming_distance_calc #(.CHUNK_SIZE(CHUNK_SIZE)) calc6 (.input_chunk(chunk_to_compress), .codebook_entry(cb6), .hamming_distance(hd6));
    hamming_distance_calc #(.CHUNK_SIZE(CHUNK_SIZE)) calc7 (.input_chunk(chunk_to_compress), .codebook_entry(cb7), .hamming_distance(hd7));

    // Minimum finder (comparator tree + priority encoder)
    min_finder #(
        .CODEBOOK_SIZE(CODEBOOK_SIZE),
        .INDEX_BITS(INDEX_BITS),
        .DISTANCE_BITS($clog2(CHUNK_SIZE+1))
    ) min_finder_inst (
        .dist0(hd0), .dist1(hd1), .dist2(hd2), .dist3(hd3),
        .dist4(hd4), .dist5(hd5), .dist6(hd6), .dist7(hd7),
        .min_index(compression_result)
    );

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

module dict_hamming_compressor_with_reg #(
    parameter CHUNK_SIZE = 4,
    parameter CODEBOOK_SIZE = 8,
    parameter INDEX_BITS = $clog2(CODEBOOK_SIZE),
    parameter NUM_CHUNKS = 4
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
    dict_hamming_compressor #(
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


// Updated testbench for dict_value_compressor_with_reg
module dict_value_tb;
    parameter CHUNK_SIZE = 8;
    parameter CODEBOOK_SIZE = 16;
    parameter INDEX_BITS = $clog2(CODEBOOK_SIZE);
    parameter NUM_CHUNKS = 16;
    
    reg clk, rst_n;
    reg data_in, data_valid;
    
    wire [(NUM_CHUNKS * INDEX_BITS)-1:0] compressed_output;
    wire compression_done;
    
    dict_value_compressor_with_reg #(
        .CHUNK_SIZE(CHUNK_SIZE),
        .CODEBOOK_SIZE(CODEBOOK_SIZE),
        .NUM_CHUNKS(NUM_CHUNKS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .data_valid(data_valid),
        .compressed_output(compressed_output),
        .compression_done(compression_done)
    );
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Test bitstream
    parameter STREAM_LENGTH = NUM_CHUNKS * CHUNK_SIZE;
    // Random test bitstream generation according to STREAM_LENGTH
    reg [STREAM_LENGTH-1:0] test_bitstream;
    integer bit_idx, i;
    
    initial begin
        // Initialize
        clk = 0;
        rst_n = 0;
        data_in = 0;
        data_valid = 0;
        
        // Generate random test bitstream
        test_bitstream = {$random, $random, $random, $random};
        
        $display("=== Dictionary Compression with Register Bank ===");
        $display("Input: 1 bit per clock, accumulates %d chunks of %d bits each", NUM_CHUNKS, CHUNK_SIZE);
        $display("Original bitstream: %b", test_bitstream);
        $display("Expected chunks:");
        for (i = 0; i < NUM_CHUNKS; i = i + 1) begin
            $display("  Chunk %d: %b", i, test_bitstream[STREAM_LENGTH-1 - (i*CHUNK_SIZE) -: CHUNK_SIZE]);
        end

        // Reset
        #10 rst_n = 1;
        #10;
        
        // Send bits serially (MSB first)
        $display("\n=== Compression Phase ===");
        for (bit_idx = STREAM_LENGTH-1; bit_idx >= 0; bit_idx = bit_idx - 1) begin
            @(posedge clk);
            data_in = test_bitstream[bit_idx];
            data_valid = 1'b1;
            $display("Sending bit %d: %b", STREAM_LENGTH-1-bit_idx, data_in);
        end
        
        // Stop input
        @(posedge clk);
        data_valid = 1'b0;
        
        // Wait for compression to complete
        $display("\nWaiting for compression to complete...");
        wait (compression_done);
        
        // Display compressed results
        $display("\n=== Compression Results ===");
        $display("Compression done: %b", compression_done);
        $display("Compressed output: %b", compressed_output);
        $display("Individual indices:");
        for (i = 0; i < NUM_CHUNKS; i = i + 1) begin
            $display("  Index %d: %b", i, compressed_output[(NUM_CHUNKS*INDEX_BITS)-1 - (i*INDEX_BITS) -: INDEX_BITS]);
        end
        
        $display("\n=== Test Complete ===");
        
        repeat(5) @(posedge clk);
        $finish;
    end
    
endmodule
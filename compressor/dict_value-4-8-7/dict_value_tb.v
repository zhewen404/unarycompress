
// Updated testbench for dict_value_compressor_with_reg
module dict_value_tb;
    parameter CHUNK_SIZE = 4;
    parameter CODEBOOK_SIZE = 8;
    parameter INDEX_BITS = $clog2(CODEBOOK_SIZE);
    parameter NUM_CHUNKS = 32;
    
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
        $display("  Chunk 0: %b", test_bitstream[15:12]);
        $display("  Chunk 1: %b", test_bitstream[11:8]);
        $display("  Chunk 2: %b", test_bitstream[7:4]);
        $display("  Chunk 3: %b", test_bitstream[3:0]);
        
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
        $display("  Index 0: %d", compressed_output[2:0]);
        $display("  Index 1: %d", compressed_output[5:3]);
        $display("  Index 2: %d", compressed_output[8:6]);
        $display("  Index 3: %d", compressed_output[11:9]);
        $display("  Index 4: %d", compressed_output[14:12]);
        $display("  Index 5: %d", compressed_output[17:15]);
        $display("  Index 6: %d", compressed_output[20:18]);
        $display("  Index 7: %d", compressed_output[23:21]);
        $display("  Index 8: %d", compressed_output[26:24]);
        $display("  Index 9: %d", compressed_output[29:27]);
        $display("  Index 10: %d", compressed_output[32:30]);
        $display("  Index 11: %d", compressed_output[35:33]);
        $display("  Index 12: %d", compressed_output[38:36]);
        $display("  Index 13: %d", compressed_output[41:39]);
        $display("  Index 14: %d", compressed_output[44:42]);
        $display("  Index 15: %d", compressed_output[47:45]);
        $display("  Index 16: %d", compressed_output[50:48]);
        $display("  Index 17: %d", compressed_output[53:51]);
        $display("  Index 18: %d", compressed_output[56:54]);
        $display("  Index 19: %d", compressed_output[59:57]);
        $display("  Index 20: %d", compressed_output[62:60]);
        $display("  Index 21: %d", compressed_output[65:63]);
        $display("  Index 22: %d", compressed_output[68:66]);
        $display("  Index 23: %d", compressed_output[71:69]);
        $display("  Index 24: %d", compressed_output[74:72]);
        $display("  Index 25: %d", compressed_output[77:75]);
        $display("  Index 26: %d", compressed_output[80:78]);
        $display("  Index 27: %d", compressed_output[83:81]);
        $display("  Index 28: %d", compressed_output[86:84]);
        $display("  Index 29: %d", compressed_output[89:87]);
        $display("  Index 30: %d", compressed_output[92:90]);
        $display("  Index 31: %d", compressed_output[95:93]);
        
        $display("\n=== Test Complete ===");
        
        repeat(5) @(posedge clk);
        $finish;
    end
    
endmodule
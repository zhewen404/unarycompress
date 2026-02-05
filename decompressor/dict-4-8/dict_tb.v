`timescale 1ns / 1ps

module dict_tb();

    // Parameters
    parameter CHUNK_SIZE = 4;
    parameter CODEBOOK_SIZE = 8;
    parameter INDEX_BITS = $clog2(CODEBOOK_SIZE);
    
    // Signals
    reg [INDEX_BITS-1:0] compressed_index;
    wire [CHUNK_SIZE-1:0] decompressed_chunk;
    
    // Test variables
    integer i;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Expected codebook values (same as in the decompressor)
    reg [CHUNK_SIZE-1:0] expected_codebook [0:CODEBOOK_SIZE-1];
    
    // Instantiate DUT
    dict_decompressor #(
        .CHUNK_SIZE(CHUNK_SIZE),
        .CODEBOOK_SIZE(CODEBOOK_SIZE),
        .INDEX_BITS(INDEX_BITS)
    ) dut (
        .compressed_index(compressed_index),
        .decompressed_chunk(decompressed_chunk)
    );
    
    // Task to initialize expected codebook
    task init_expected_codebook;
        begin
            expected_codebook[0] = 4'b0000; // [0 0 0 0]
            expected_codebook[1] = 4'b0010; // [0 0 1 0]
            expected_codebook[2] = 4'b1001; // [1 0 0 1]
            expected_codebook[3] = 4'b1011; // [1 0 1 1]
            expected_codebook[4] = 4'b1111; // [1 1 1 1]
            expected_codebook[5] = 4'b1000; // [1 0 0 0]
            expected_codebook[6] = 4'b1100; // [1 1 0 0]
            expected_codebook[7] = 4'b0111; // [0 1 1 1]
        end
    endtask
    
    // Task to test a specific index
    task test_index;
        input [INDEX_BITS-1:0] index;
        input [CHUNK_SIZE-1:0] expected_output;
        input [80*8-1:0] test_name;
        begin
            compressed_index = index;
            #10; // Small delay for combinational logic to settle
            
            test_count = test_count + 1;
            
            $display("Test %0d - %0s:", test_count, test_name);
            $display("  Index: %0d (3'b%03b)", index, index);
            $display("  Expected: 4'b%04b (%0d)", expected_output, expected_output);
            $display("  Got:      4'b%04b (%0d)", decompressed_chunk, decompressed_chunk);
            
            if (decompressed_chunk == expected_output) begin
                $display("  PASSED");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAILED");
                $display("  ERROR: Output mismatch!");
            end
            $display("");
        end
    endtask
    
    // Task to test all codebook entries
    task test_all_entries;
        begin
            $display("=== Testing All Codebook Entries ===");
            for (i = 0; i < CODEBOOK_SIZE; i = i + 1) begin
                case (i)
                    0: test_index(i, expected_codebook[i], "Entry 0 - All Zeros");
                    1: test_index(i, expected_codebook[i], "Entry 1 - Single One");
                    2: test_index(i, expected_codebook[i], "Entry 2 - Two Ones");
                    3: test_index(i, expected_codebook[i], "Entry 3 - Three Ones");
                    4: test_index(i, expected_codebook[i], "Entry 4 - All Ones");
                    5: test_index(i, expected_codebook[i], "Entry 5 - Single One MSB");
                    6: test_index(i, expected_codebook[i], "Entry 6 - Two MSB Ones");
                    7: test_index(i, expected_codebook[i], "Entry 7 - Three LSB Ones");
                endcase
            end
        end
    endtask
    
    // Task to test edge cases
    task test_edge_cases;
        begin
            $display("=== Testing Edge Cases ===");
            
            // Test index 0 (minimum)
            test_index(3'b000, expected_codebook[0], "Edge Case - Index 0");
            
            // Test index 7 (maximum)
            test_index(3'b111, expected_codebook[7], "Edge Case - Index 7");
            
            // Test some intermediate values
            test_index(3'b011, expected_codebook[3], "Edge Case - Index 3");
            test_index(3'b100, expected_codebook[4], "Edge Case - Index 4");
        end
    endtask
    
    // Task to test rapid index changes
    task test_rapid_changes;
        integer j;
        begin
            $display("=== Testing Rapid Index Changes ===");
            
            for (j = 0; j < 3; j = j + 1) begin
                compressed_index = 0;
                #1;
                if (decompressed_chunk != expected_codebook[0]) begin
                    $display("✗ FAILED: Rapid change test %0d at index 0", j);
                    test_count = test_count + 1;
                end else begin
                    pass_count = pass_count + 1;
                    test_count = test_count + 1;
                end
                
                compressed_index = 7;
                #1;
                if (decompressed_chunk != expected_codebook[7]) begin
                    $display("✗ FAILED: Rapid change test %0d at index 7", j);
                    test_count = test_count + 1;
                end else begin
                    pass_count = pass_count + 1;
                    test_count = test_count + 1;
                end
            end
            
            $display("Rapid change tests completed");
            $display("");
        end
    endtask
    
    // Task to display codebook
    task display_codebook;
        integer k;
        begin
            $display("=== Dictionary Codebook ===");
            $display("Index | Binary   | Decimal | Pattern");
            $display("------|----------|---------|--------");
            for (k = 0; k < CODEBOOK_SIZE; k = k + 1) begin
                $display("  %0d   | %04b     |    %0d    | %0s", 
                         k, expected_codebook[k], expected_codebook[k],
                         k == 0 ? "[0 0 0 0]" :
                         k == 1 ? "[0 0 1 0]" :
                         k == 2 ? "[1 0 0 1]" :
                         k == 3 ? "[1 0 1 1]" :
                         k == 4 ? "[1 1 1 1]" :
                         k == 5 ? "[1 0 0 0]" :
                         k == 6 ? "[1 1 0 0]" : "[0 1 1 1]");
            end
            $display("");
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting Dictionary Decompressor Testbench");
        $display("Chunk Size: %0d bits", CHUNK_SIZE);
        $display("Codebook Size: %0d entries", CODEBOOK_SIZE);
        $display("Index Bits: %0d bits", INDEX_BITS);
        $display("");
        
        // Initialize
        compressed_index = 0;
        init_expected_codebook();
        
        #100; // Wait for initial settling
        
        // Display the codebook being tested
        display_codebook();
        
        // Run comprehensive tests
        test_all_entries();
        test_edge_cases();
        test_rapid_changes();
        
        // Final results
        $display("========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Total Tests: %0d", test_count);
        $display("Passed:      %0d", pass_count);
        $display("Failed:      %0d", test_count - pass_count);
        $display("Success Rate: %0.1f%%", (pass_count * 100.0) / test_count);
        
        if (pass_count == test_count) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("Some tests failed.");
        end
        
        $display("Testbench completed.");
        $finish;
    end

endmodule

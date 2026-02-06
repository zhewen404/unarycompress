`timescale 1ns / 1ps

module histogram_tb();

    // Parameters
    parameter STREAM_LENGTH = 128;
    parameter COUNTER_WIDTH = $clog2(STREAM_LENGTH+1);
    parameter LFSR_WIDTH = 3;
    parameter CLK_PERIOD = 10;
    
    // Signals
    reg clk;
    reg rst_n;
    reg start_decompress;
    
    // Input histogram counts
    reg [COUNTER_WIDTH-1:0] count_00;
    reg [COUNTER_WIDTH-1:0] count_01;
    reg [COUNTER_WIDTH-1:0] count_10;
    reg [COUNTER_WIDTH-1:0] count_11;
    
    // Output signals
    wire stream_a;
    wire stream_b;
    wire valid_out;
    wire decompress_done;
    
    // Test variables
    integer test_count = 0;
    integer pass_count = 0;
    integer i;
    
    // Counters to verify output histogram
    integer output_count_00, output_count_01, output_count_10, output_count_11;
    integer total_output_pairs;
    integer timeout_counter;
    
    // Expected values
    integer expected_00, expected_01, expected_10, expected_11;
    integer expected_total;
    
    // Instantiate DUT
    histogram_decompressor #(
        .STREAM_LENGTH(STREAM_LENGTH),
        .COUNTER_WIDTH(COUNTER_WIDTH),
        .LFSR_WIDTH(LFSR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_decompress(start_decompress),
        .count_00(count_00),
        .count_01(count_01),
        .count_10(count_10),
        .count_11(count_11),
        .stream_a(stream_a),
        .stream_b(stream_b),
        .valid_out(valid_out),
        .decompress_done(decompress_done)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Task to reset DUT
    task reset_dut;
        begin
            rst_n = 0;
            start_decompress = 0;
            count_00 = 0;
            count_01 = 0;
            count_10 = 0;
            count_11 = 0;
            @(posedge clk);
            @(posedge clk);
            rst_n = 1;
            @(posedge clk);
        end
    endtask
    
    // Task to set histogram inputs
    task set_histogram;
        input integer c00, c01, c10, c11;
        begin
            count_00 = c00;
            count_01 = c01;
            count_10 = c10;
            count_11 = c11;
            expected_00 = c00;
            expected_01 = c01;
            expected_10 = c10;
            expected_11 = c11;
            expected_total = c00 + c01 + c10 + c11;
        end
    endtask
    
    // Task to start decompression and collect output
    task run_decompression;
        begin
            // Initialize counters
            output_count_00 = 0;
            output_count_01 = 0;
            output_count_10 = 0;
            output_count_11 = 0;
            total_output_pairs = 0;
            timeout_counter = 0;
            
            // Start decompression
            start_decompress = 1;
            @(posedge clk);
            start_decompress = 0;
            
            // Collect all output pairs
            while (!decompress_done && timeout_counter < STREAM_LENGTH * 2) begin
                @(posedge clk);
                
                if (valid_out) begin
                    case ({stream_a, stream_b})
                        2'b00: output_count_00 = output_count_00 + 1;
                        2'b01: output_count_01 = output_count_01 + 1;
                        2'b10: output_count_10 = output_count_10 + 1;
                        2'b11: output_count_11 = output_count_11 + 1;
                    endcase
                    total_output_pairs = total_output_pairs + 1;
                end
                
                timeout_counter = timeout_counter + 1;
            end
            
            // Wait a few more cycles
            repeat(5) @(posedge clk);
        end
    endtask
    
    // Task to check results
    task check_results;
        input [80*8-1:0] test_name;
        begin
            test_count = test_count + 1;
            
            $display("\n--- Test %0d: %0s ---", test_count, test_name);
            $display("Input histogram:  00=%0d, 01=%0d, 10=%0d, 11=%0d (Total=%0d)", 
                     expected_00, expected_01, expected_10, expected_11, expected_total);
            $display("Output histogram: 00=%0d, 01=%0d, 10=%0d, 11=%0d (Total=%0d)", 
                     output_count_00, output_count_01, output_count_10, output_count_11, total_output_pairs);
            
            if (output_count_00 == expected_00 && 
                output_count_01 == expected_01 && 
                output_count_10 == expected_10 && 
                output_count_11 == expected_11 &&
                total_output_pairs == expected_total) begin
                $display(" PASSED - Histogram preserved correctly");
                pass_count = pass_count + 1;
            end else begin
                $display(" FAILED - Histogram mismatch");
                if (total_output_pairs != expected_total) begin
                    $display("  ERROR: Total pairs mismatch (expected %0d, got %0d)", 
                             expected_total, total_output_pairs);
                end
                if (output_count_00 != expected_00) begin
                    $display("  ERROR: 00 count mismatch (expected %0d, got %0d)", 
                             expected_00, output_count_00);
                end
                if (output_count_01 != expected_01) begin
                    $display("  ERROR: 01 count mismatch (expected %0d, got %0d)", 
                             expected_01, output_count_01);
                end
                if (output_count_10 != expected_10) begin
                    $display("  ERROR: 10 count mismatch (expected %0d, got %0d)", 
                             expected_10, output_count_10);
                end
                if (output_count_11 != expected_11) begin
                    $display("  ERROR: 11 count mismatch (expected %0d, got %0d)", 
                             expected_11, output_count_11);
                end
            end
            
            if (timeout_counter >= STREAM_LENGTH * 2) begin
                $display("  WARNING: Test timed out");
            end
        end
    endtask
    
    // Task to run a complete test
    task run_test;
        input integer c00, c01, c10, c11;
        input [80*8-1:0] test_name;
        begin
            reset_dut();
            set_histogram(c00, c01, c10, c11);
            run_decompression();
            check_results(test_name);
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting Histogram Decompressor Testbench");
        $display("Stream Length: %0d pairs", STREAM_LENGTH);
        $display("LFSR Width: %0d bits", LFSR_WIDTH);
        
        // Initialize
        clk = 0;
        rst_n = 0;
        start_decompress = 0;
        count_00 = 0;
        count_01 = 0;
        count_10 = 0;
        count_11 = 0;
        
        #100; // Wait for initial settling
        
        // Test 1: All zeros
        run_test(10, 0, 0, 0, "All 00 pairs");
        
        // Test 2: All ones
        run_test(0, 0, 0, 12, "All 11 pairs");
        
        // Test 3: Single non-zero bin
        run_test(0, 15, 0, 0, "All 01 pairs");
        run_test(0, 0, 8, 0, "All 10 pairs");
        
        // Test 4: Two bins
        run_test(5, 7, 0, 0, "00 and 01 pairs");
        run_test(0, 0, 6, 4, "10 and 11 pairs");
        run_test(8, 0, 0, 3, "00 and 11 pairs");
        
        // Test 5: Three bins
        run_test(4, 6, 2, 0, "00, 01, and 10 pairs");
        run_test(0, 3, 5, 7, "01, 10, and 11 pairs");
        
        // Test 6: All four bins (balanced)
        run_test(5, 5, 5, 5, "Balanced histogram");
        
        // Test 7: All four bins (unbalanced)
        run_test(10, 3, 1, 2, "Unbalanced histogram");
        
        // Test 8: Large counts
        run_test(20, 15, 12, 8, "Large counts");
        
        // Test 9: Edge case - single pair
        run_test(1, 0, 0, 0, "Single 00 pair");
        run_test(0, 1, 0, 0, "Single 01 pair");
        run_test(0, 0, 1, 0, "Single 10 pair");
        run_test(0, 0, 0, 1, "Single 11 pair");
        
        // Test 10: Empty histogram (edge case)
        run_test(0, 0, 0, 0, "Empty histogram");
        
        // Final results
        $display("\n========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Total Tests: %0d", test_count);
        $display("Passed:      %0d", pass_count);
        $display("Failed:      %0d", test_count - pass_count);
        
        if (pass_count == test_count) begin
            $display(" ALL TESTS PASSED!");
        end else begin
            $display(" Some tests failed.");
        end
        
        $display("Testbench completed.");
        $finish;
    end
    
    // Monitor for debugging (uncomment if needed)
    /*
    initial begin
        $monitor("Time=%0t: start=%b, valid_out=%b, stream_a=%b, stream_b=%b, done=%b", 
                 $time, start_decompress, valid_out, stream_a, stream_b, decompress_done);
    end
    */

endmodule

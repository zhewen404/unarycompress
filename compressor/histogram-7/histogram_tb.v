`timescale 1ns / 1ps

module histogram_tb();

    // Parameters
    parameter STREAM_LENGTH = 128;
    parameter COUNTER_WIDTH = $clog2(STREAM_LENGTH+1);
    parameter CLK_PERIOD = 10;
    
    // Signals
    reg clk;
    reg rst_n;
    reg stream_a;
    reg stream_b;
    reg valid_in;
    
    wire [COUNTER_WIDTH-1:0] count_00;
    wire [COUNTER_WIDTH-1:0] count_01;
    wire [COUNTER_WIDTH-1:0] count_10;
    wire [COUNTER_WIDTH-1:0] count_11;
    
    // Test variables
    reg [STREAM_LENGTH-1:0] test_stream_a;
    reg [STREAM_LENGTH-1:0] test_stream_b;
    integer expected_count_00, expected_count_01, expected_count_10, expected_count_11;
    integer i, j;
    integer seed = 12345;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Instantiate DUT
    histogram_compressor #(
        .STREAM_LENGTH(STREAM_LENGTH),
        .COUNTER_WIDTH(COUNTER_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .stream_a(stream_a),
        .stream_b(stream_b),
        .valid_in(valid_in),
        .count_00(count_00),
        .count_01(count_01),
        .count_10(count_10),
        .count_11(count_11)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Task to generate random bitstreams
    task generate_random_streams;
        input integer random_seed;
        integer seed_a, seed_b;
        begin
            seed_a = random_seed;
            seed_b = random_seed + 1000;
            
            // Generate random stream A
            for (i = 0; i < STREAM_LENGTH; i = i + 1) begin
                test_stream_a[i] = $random(seed_a) % 2;
            end
            
            // Generate random stream B
            for (i = 0; i < STREAM_LENGTH; i = i + 1) begin
                test_stream_b[i] = $random(seed_b) % 2;
            end
        end
    endtask
    
    // Task to calculate expected histogram counts
    task calculate_expected_counts;
        begin
            expected_count_00 = 0;
            expected_count_01 = 0;
            expected_count_10 = 0;
            expected_count_11 = 0;
            
            for (i = 0; i < STREAM_LENGTH; i = i + 1) begin
                case ({test_stream_a[i], test_stream_b[i]})
                    2'b00: expected_count_00 = expected_count_00 + 1;
                    2'b01: expected_count_01 = expected_count_01 + 1;
                    2'b10: expected_count_10 = expected_count_10 + 1;
                    2'b11: expected_count_11 = expected_count_11 + 1;
                endcase
            end
            
            $display("Expected counts: 00=%0d, 01=%0d, 10=%0d, 11=%0d", 
                     expected_count_00, expected_count_01, expected_count_10, expected_count_11);
        end
    endtask
    
    // Task to feed streams to DUT
    task feed_streams_to_dut;
        begin
            valid_in = 0;
            @(posedge clk);
            
            for (i = 0; i < STREAM_LENGTH; i = i + 1) begin
                stream_a = test_stream_a[i];
                stream_b = test_stream_b[i];
                valid_in = 1;
                @(posedge clk);
            end
            
            valid_in = 0;
            
            // Wait one extra cycle for final update
            @(posedge clk);
        end
    endtask
    
    // Task to check results
    task check_results;
        begin
            test_count = test_count + 1;
            
            $display("Hardware counts: 00=%0d, 01=%0d, 10=%0d, 11=%0d", 
                     count_00, count_01, count_10, count_11);
            
            if (count_00 == expected_count_00 && 
                count_01 == expected_count_01 && 
                count_10 == expected_count_10 && 
                count_11 == expected_count_11) begin
                $display("Test %0d PASSED", test_count);
                pass_count = pass_count + 1;
            end else begin
                $display("Test %0d FAILED", test_count);
                $display("  Expected: 00=%0d, 01=%0d, 10=%0d, 11=%0d", 
                         expected_count_00, expected_count_01, expected_count_10, expected_count_11);
                $display("  Got:      00=%0d, 01=%0d, 10=%0d, 11=%0d", 
                         count_00, count_01, count_10, count_11);
            end
        end
    endtask
    
    // Task to reset DUT
    task reset_dut;
        begin
            rst_n = 0;
            stream_a = 0;
            stream_b = 0;
            valid_in = 0;
            @(posedge clk);
            @(posedge clk);
            rst_n = 1;
            @(posedge clk);
        end
    endtask
    
    // Test specific patterns
    task test_specific_pattern;
        input [STREAM_LENGTH-1:0] pattern_a;
        input [STREAM_LENGTH-1:0] pattern_b;
        input [80*8-1:0] test_name;
        begin
            $display("\n--- Running %0s ---", test_name);
            
            test_stream_a = pattern_a;
            test_stream_b = pattern_b;
            
            calculate_expected_counts();
            reset_dut();
            feed_streams_to_dut();
            check_results();
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting Histogram Compressor Testbench");
        $display("Stream Length: %0d bits", STREAM_LENGTH);
        
        // Initialize
        clk = 0;
        rst_n = 0;
        stream_a = 0;
        stream_b = 0;
        valid_in = 0;
        
        #100;  // Wait for initial settling
        
        // Test 1: All zeros
        test_specific_pattern({STREAM_LENGTH{1'b0}}, {STREAM_LENGTH{1'b0}}, "All Zeros");
        
        // Test 2: All ones
        test_specific_pattern({STREAM_LENGTH{1'b1}}, {STREAM_LENGTH{1'b1}}, "All Ones");
        
        // Test 3: Stream A all 0, Stream B all 1
        test_specific_pattern({STREAM_LENGTH{1'b0}}, {STREAM_LENGTH{1'b1}}, "A=0, B=1");
        
        // Test 4: Stream A all 1, Stream B all 0
        test_specific_pattern({STREAM_LENGTH{1'b1}}, {STREAM_LENGTH{1'b0}}, "A=1, B=0");
        
        // Test 5: Alternating patterns
        for (i = 0; i < STREAM_LENGTH; i = i + 1) begin
            test_stream_a[i] = i % 2;
            test_stream_b[i] = (i + 1) % 2;
        end
        test_specific_pattern(test_stream_a, test_stream_b, "Alternating Patterns");
        
        // Test 6-10: Random patterns
        for (j = 1; j <= 5; j = j + 1) begin
            $display("\n--- Running Random Test %0d ---", j);
            
            generate_random_streams(seed + j * 100);
            calculate_expected_counts();
            reset_dut();
            feed_streams_to_dut();
            check_results();
        end
        
        // Final results
        $display("\n========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Total Tests: %0d", test_count);
        $display("Passed:      %0d", pass_count);
        $display("Failed:      %0d", test_count - pass_count);
        
        if (pass_count == test_count) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("Some tests failed.");
        end
        
        $display("Testbench completed.");
        $finish;
    end
    
    // Monitor for debugging (uncomment if needed)
    /*
    initial begin
        $monitor("Time=%0t: stream_a=%b, stream_b=%b, valid_in=%b, count_00=%0d, count_01=%0d, count_10=%0d, count_11=%0d, done=%b", 
                 $time, stream_a, stream_b, valid_in, count_00, count_01, count_10, count_11, compress_done);
    end
    */

endmodule
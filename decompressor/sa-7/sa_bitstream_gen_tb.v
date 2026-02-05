
//-----------------------------------------------------------------------------
// Testbench for SA Bitstream Generator
//-----------------------------------------------------------------------------
`timescale 1ns/1ps

module sa_bitstream_gen_tb;

    parameter N = 7;
    parameter L = 1 << N;  // 128
    
    reg             clk;
    reg             rst_n;
    reg             enable;
    reg  [N-1:0]    k;
    reg             start;
    wire            x_out;
    wire [N-1:0]    acc_val;
    
    // Instantiate DUT
    sa_bitstream_gen #(.N(N)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .k(k),
        .start(start),
        .x_out(x_out),
        .acc_val(acc_val)
    );
    
    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz clock
    
    // Test variables
    integer i;
    integer one_count;
    real expected_prob;
    real actual_prob;
    reg [L-1:0] bitstream;  // Store generated bitstream
    
    initial begin
        $display("==============================================");
        $display("SA (Streaming-Accurate) Bitstream Generator");
        $display("==============================================");
        $display("N = %0d, L = %0d", N, L);
        $display("");
        
        // Initialize
        rst_n = 0;
        enable = 0;
        k = 0;
        start = 0;
        
        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        
        // Test various k values
        $display("--- Single Bitstream Generation (L=%0d cycles) ---", L);
        test_value(64);   // 50% = 64/128
        test_value(32);   // 25% = 32/128
        test_value(96);   // 75% = 96/128
        test_value(16);   // 12.5% = 16/128
        test_value(2);    // ~1.6% = 2/128
        test_value(127);  // ~99.2% = 127/128
        
        // Show bitstream patterns for small values
        $display("");
        $display("--- Bitstream Patterns (showing bit distribution) ---");
        show_bitstream(16);    // 16/128 = 1/8
        show_bitstream(32);   // 32/128 = 1/4
        show_bitstream(64);   // 64/128 = 1/2
        show_bitstream(96);   // 96/128 = 3/4
        
        // Test streaming accuracy - error at each partial length
        $display("");
        $display("--- Streaming Accuracy Analysis (k=20) ---");
        test_streaming_accuracy(20);
        
        $display("");
        $display("==============================================");
        $display("Testbench Complete");
        $display("==============================================");
        $finish;
    end
    
    // Task to test a specific k value
    task test_value;
        input [N-1:0] test_k;
        begin
            // Start new bitstream
            start = 1;
            @(posedge clk);
            start = 0;
            
            k = test_k;
            one_count = 0;
            enable = 1;
            
            // Generate L bits and count ones
            for (i = 0; i < L; i = i + 1) begin
                @(posedge clk);
                if (x_out) one_count = one_count + 1;
            end
            
            enable = 0;
            @(posedge clk);
            
            expected_prob = $itor(test_k) / $itor(L);
            actual_prob = $itor(one_count) / $itor(L);
            
            $display("k = %2d | Expected P(1) = %0.4f | Actual P(1) = %0.4f | Ones = %2d/%0d | %s",
                     test_k, expected_prob, actual_prob, one_count, L,
                     (one_count == test_k) ? "EXACT" : "MISMATCH");
        end
    endtask
    
    // Task to show the actual bitstream pattern
    task show_bitstream;
        input [N-1:0] test_k;
        integer j;
        begin
            // Start new bitstream
            start = 1;
            @(posedge clk);
            start = 0;
            
            k = test_k;
            enable = 1;
            bitstream = 0;
            
            // Generate and capture bitstream
            for (i = 0; i < L; i = i + 1) begin
                @(posedge clk);
                bitstream[i] = x_out;
            end
            
            enable = 0;
            @(posedge clk);
            
            // Display pattern (first 32 bits for readability)
            $write("k = %2d | Pattern: ", test_k);
            for (j = 0; j < 32; j = j + 1) begin
                $write("%b", bitstream[j]);
                if ((j + 1) % 8 == 0) $write(" ");
            end
            $display("...");
        end
    endtask
    
    // Task to analyze streaming accuracy at each partial length
    task test_streaming_accuracy;
        input [N-1:0] test_k;
        integer partial_ones;
        real target_value;
        real partial_value;
        real error;
        real max_error;
        real sum_error;
        begin
            // Start new bitstream
            start = 1;
            @(posedge clk);
            start = 0;
            
            k = test_k;
            enable = 1;
            partial_ones = 0;
            max_error = 0;
            sum_error = 0;
            target_value = $itor(test_k) / $itor(L);
            
            $display("Target value: %0.4f (k=%0d, L=%0d)", target_value, test_k, L);
            $display("Partial Len |  Ones | Partial Val |   Error   | Acc Error");
            $display("------------|-------|-------------|-----------|----------");
            
            // Generate bits and track error at each partial length
            for (i = 1; i <= L; i = i + 1) begin
                @(posedge clk);
                if (x_out) partial_ones = partial_ones + 1;
                
                partial_value = $itor(partial_ones) / $itor(i);
                error = partial_value - target_value;
                if (error < 0) error = -error;
                sum_error = sum_error + error;
                if (error > max_error) max_error = error;
                
                // Print every 8th cycle and the last one
                if (i % 8 == 0 || i == L) begin
                    $display("     %3d    |  %3d  |   %0.4f    |  %0.4f   |  %0.4f",
                             i, partial_ones, partial_value, error, sum_error/i);
                end
            end
            
            enable = 0;
            @(posedge clk);
            
            $display("------------|-------|-------------|-----------|----------");
            $display("Max error at any point: %0.4f", max_error);
            $display("Average error: %0.4f", sum_error / L);
        end
    endtask
    
    // Dump waveforms
    initial begin
        $dumpfile("sa_bitstream_gen.vcd");
        $dumpvars(0, sa_bitstream_gen_tb);
    end

endmodule
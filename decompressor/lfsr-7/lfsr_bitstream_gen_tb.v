
//-----------------------------------------------------------------------------
// Testbench for LFSR Bitstream Generator
//-----------------------------------------------------------------------------
`timescale 1ns/1ps

module lfsr_bitstream_gen_tb;

    parameter N = 7;
    parameter L = 1 << N;  // 64
    
    reg             clk;
    reg             rst_n;
    reg             enable;
    reg  [N-1:0]    k;
    reg  [N-1:0]    seed;
    reg             load;
    wire            x_out;
    wire [N-1:0]    lfsr_val;
    
    // Instantiate DUT
    lfsr_bitstream_gen #(.N(N)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .k(k),
        .seed(seed),
        .load(load),
        .x_out(x_out),
        .lfsr_val(lfsr_val)
    );
    
    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz clock
    
    // Test variables
    integer i;
    integer one_count;
    real expected_prob;
    real actual_prob;
    
    initial begin
        $display("==============================================");
        $display("LFSR Bitstream Generator Testbench");
        $display("==============================================");
        $display("N = %0d, L = %0d", N, L);
        $display("");
        
        // Initialize
        rst_n = 0;
        enable = 0;
        k = 0;
        seed = 6'b101010;
        load = 0;
        
        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        
        // Load seed
        load = 1;
        @(posedge clk);
        load = 0;
        @(posedge clk);
        
        // Diagnostic: Check LFSR value distribution
        $display("");
        $display("--- LFSR Value Distribution Analysis ---");
        check_lfsr_distribution();
        
        // Test with exact LFSR period (2^N - 1 = 63 cycles)
        // Over one full period, LFSR produces each value 1-63 exactly once
        $display("");
        $display("--- Full LFSR Period (%0d cycles) ---", L-1);
        $display("Note: Over full period, LFSR hits each value 1-%0d exactly once", L-1);
        test_value(32, L-1);   // Expected: 31/63 = 0.492
        test_value(16, L-1);   // Expected: 15/63 = 0.238
        test_value(48, L-1);   // Expected: 47/63 = 0.746
        test_value(8, L-1);    // Expected: 7/63 = 0.111
        
        // Multiple periods for better statistics
        $display("");
        $display("--- Multiple Periods (%0d cycles = %0d periods) ---", (L-1)*16, 16);
        test_value(32, (L-1)*16);
        test_value(16, (L-1)*16);
        test_value(48, (L-1)*16);
        test_value(8, (L-1)*16);
        
        $display("");
        $display("==============================================");
        $display("Testbench Complete");
        $display("==============================================");
        $finish;
    end
    
    // Task to test a specific k value
    task test_value;
        input [N-1:0] test_k;
        input integer num_cycles;
        begin
            k = test_k;
            load = 1;
            @(posedge clk);
            load = 0;
            one_count = 0;
            enable = 1;
            
            // Generate bits and count ones
            for (i = 0; i < num_cycles; i = i + 1) begin
                @(posedge clk);
                if (x_out) one_count = one_count + 1;
            end
            
            enable = 0;
            @(posedge clk);
            
            expected_prob = $itor(test_k) / $itor(L);
            actual_prob = $itor(one_count) / $itor(num_cycles);
            
            $display("k = %2d | Expected P(1) = %0.4f | Actual P(1) = %0.4f | Ones = %4d/%0d",
                     test_k, expected_prob, actual_prob, one_count, num_cycles);
        end
    endtask
    
    // Task to analyze LFSR value distribution
    integer min_val, max_val, val_count;
    reg [L-1:0] seen_values;  // Bitmap of seen values
    integer unique_count;
    task check_lfsr_distribution;
        begin
            min_val = L;
            max_val = 0;
            val_count = 0;
            seen_values = 0;
            unique_count = 0;
            enable = 1;
            
            // Run for full period (2^N - 1 for maximal LFSR)
            for (i = 0; i < L-1; i = i + 1) begin
                @(posedge clk);
                if (lfsr_val < min_val) min_val = lfsr_val;
                if (lfsr_val > max_val) max_val = lfsr_val;
                if (!seen_values[lfsr_val]) begin
                    seen_values[lfsr_val] = 1;
                    unique_count = unique_count + 1;
                end
                val_count = val_count + 1;
            end
            
            enable = 0;
            @(posedge clk);
            
            $display("Over %0d cycles: values range [%0d, %0d], %0d unique values seen", 
                     val_count, min_val, max_val, unique_count);
            $display("6-bit maximal LFSR period = 2^6-1 = 63 (all values 1-63 appear once)");
        end
    endtask
    
    // Optional: Dump waveforms
    initial begin
        $dumpfile("lfsr_bitstream_gen.vcd");
        $dumpvars(0, lfsr_bitstream_gen_tb);
    end

endmodule

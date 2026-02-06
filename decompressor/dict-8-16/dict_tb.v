`timescale 1ns / 1ps

// Testbench for dictionary decompressor with serial output
module dict_tb();

    // Parameters
    parameter CHUNK_SIZE = 8;
    parameter CODEBOOK_SIZE = 16;
    parameter INDEX_BITS = $clog2(CODEBOOK_SIZE);
    
    // Testbench signals
    reg clk, rst_n;
    reg [INDEX_BITS-1:0] compressed_index;
    reg load, shift_enable;
    
    wire [CHUNK_SIZE-1:0] decompressed_chunk;
    wire serial_out;
    wire shift_done;
    
    // Instantiate DUT
    dict_decompressor #(
        .CHUNK_SIZE(CHUNK_SIZE),
        .CODEBOOK_SIZE(CODEBOOK_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .compressed_index(compressed_index),
        .load(load),
        .shift_enable(shift_enable),
        .decompressed_chunk(decompressed_chunk),
        .serial_out(serial_out),
        .shift_done(shift_done)
    );
    
    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz clock
    
    // Test variables
    integer i, bit_idx;
    reg [CHUNK_SIZE-1:0] received_bits;
    
    // Test sequence
    initial begin
        $display("=== Dictionary Decompressor with Serial Output Test ===");
        $display("Codebook:");
        $display("  Index 0: 00000000,  Index 1: 00100010,  Index 2: 10011001,  Index 3: 10111011");
        $display("  Index 4: 11111111,  Index 5: 10001000,  Index 6: 11001100,  Index 7: 01110111");
        $display("  Index 8: 00001111,  Index 9: 11110000,  Index 10: 01010101,  Index 11: 10101010");
        $display("  Index 12: 00110011,  Index 13: 11001100,  Index 14: 11100011,  Index 15: 00011100");
        $display("");
        
        // Initialize
        rst_n = 0;
        compressed_index = 0;
        load = 0;
        shift_enable = 0;
        
        // Reset
        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        
        // Test all codebook entries
        for (i = 0; i < CODEBOOK_SIZE; i = i + 1) begin
            $display("--- Testing Index %0d ---", i);
            
            // Set index and load
            compressed_index = i;
            @(posedge clk);
            
            $display("Parallel output: %b", decompressed_chunk);
            
            // Load into shift register
            load = 1;
            @(posedge clk);
            load = 0;
            
            $display("Serial output sequence (MSB first):");
            
            // Shift out all bits
            received_bits = 8'b00000000;
            shift_enable = 1;
            
            for (bit_idx = 0; bit_idx < CHUNK_SIZE; bit_idx = bit_idx + 1) begin
                $display("  Bit %0d: %b", bit_idx, serial_out);
                // Build received_bits by shifting in from LSB (since we get MSB first)
                received_bits = {received_bits[CHUNK_SIZE-2:0], serial_out};
                @(posedge clk);  // Clock edge happens after reading
            end
            
            shift_enable = 0;
            @(posedge clk);
            
            $display("Reconstructed: %b", received_bits);
            $display("Shift done: %b", shift_done);
            
            // Check if reconstructed matches parallel output
            if (received_bits == decompressed_chunk) begin
                $display("✓ PASS: Serial output matches parallel output");
            end else begin
                $display("✗ FAIL: Mismatch - Expected %b, Got %b", 
                        decompressed_chunk, received_bits);
            end
            
            $display("");
        end
        
        $display("=== Test Complete ===");
        $finish;
    end
    
    // Optional: dump waveforms
    initial begin
        $dumpfile("dict_tb.vcd");
        $dumpvars(0, dict_tb);
    end
    
endmodule
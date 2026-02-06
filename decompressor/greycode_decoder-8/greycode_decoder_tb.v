`timescale 1ns / 1ps

module greycode_decoder_tb;

    // Parameters
    parameter WIDTH = 8;
    parameter TEST_CYCLES = 2**WIDTH; // Test all possible values
    
    // Testbench signals
    reg [WIDTH-1:0] grey_in;
    wire [WIDTH-1:0] binary_out;
    
    // Expected binary value for verification
    reg [WIDTH-1:0] expected_binary;
    reg [WIDTH-1:0] test_binary;
    
    // Test control
    integer i;
    integer errors = 0;
    
    // Instantiate the greycode_decoder
    greycode_decoder #(
        .WIDTH(WIDTH)
    ) uut (
        .grey_in(grey_in),
        .binary_out(binary_out)
    );
    
    // Function to convert binary to grey code (for reference)
    function [WIDTH-1:0] binary_to_grey;
        input [WIDTH-1:0] binary;
        integer j;
        begin
            binary_to_grey[WIDTH-1] = binary[WIDTH-1];
            for (j = WIDTH-2; j >= 0; j = j - 1) begin
                binary_to_grey[j] = binary[j+1] ^ binary[j];
            end
        end
    endfunction
    
    // Function to convert grey code to binary (reference implementation)
    function [WIDTH-1:0] grey_to_binary_ref;
        input [WIDTH-1:0] grey;
        integer j;
        begin
            grey_to_binary_ref[WIDTH-1] = grey[WIDTH-1];
            for (j = WIDTH-2; j >= 0; j = j - 1) begin
                grey_to_binary_ref[j] = grey[j] ^ grey_to_binary_ref[j+1];
            end
        end
    endfunction
    
    // Main test sequence
    initial begin
        $display("Starting greycode_decoder testbench...");
        $display("Testing %d-bit grey code decoder", WIDTH);
        $display("Time\tGrey\t\tBinary\t\tExpected\tResult");
        $display("----\t----\t\t------\t\t--------\t------");
        
        // Test all possible grey code values
        for (i = 0; i < TEST_CYCLES; i = i + 1) begin
            // Generate a binary value and convert it to grey code
            test_binary = i;
            grey_in = binary_to_grey(test_binary);
            expected_binary = test_binary;
            
            #10; // Wait for combinational logic to settle
            
            // Check if the decoder output matches expected
            if (binary_out !== expected_binary) begin
                errors = errors + 1;
                $display("%4dns\t%08b\t%08b\t%08b\tFAIL", $time, grey_in, binary_out, expected_binary);
            end else begin
                $display("%4dns\t%08b\t%08b\t%08b\tPASS", $time, grey_in, binary_out, expected_binary);
            end
        end
        
        // Test some specific known cases
        $display("\nTesting specific known cases:");
        
        // Test case 1: All zeros
        grey_in = 8'b00000000;
        expected_binary = 8'b00000000;
        #10;
        if (binary_out !== expected_binary) begin
            errors = errors + 1;
            $display("FAIL: All zeros - Got %08b, Expected %08b", binary_out, expected_binary);
        end else begin
            $display("PASS: All zeros");
        end
        
        // Test case 2: All ones (grey code 10000000 = binary 11111111)
        grey_in = 8'b10000000;
        expected_binary = 8'b11111111;
        #10;
        if (binary_out !== expected_binary) begin
            errors = errors + 1;
            $display("FAIL: All ones - Got %08b, Expected %08b", binary_out, expected_binary);
        end else begin
            $display("PASS: All ones");
        end
        
        // Test case 3: Alternating pattern
        grey_in = 8'b11000000;
        expected_binary = grey_to_binary_ref(grey_in);
        #10;
        if (binary_out !== expected_binary) begin
            errors = errors + 1;
            $display("FAIL: Pattern test - Got %08b, Expected %08b", binary_out, expected_binary);
        end else begin
            $display("PASS: Pattern test");
        end
        
        // Final results
        $display("\n=== Test Summary ===");
        $display("Total tests run: %d", TEST_CYCLES + 3);
        $display("Errors found: %d", errors);
        
        if (errors == 0) begin
            $display("*** ALL TESTS PASSED! ***");
        end else begin
            $display("*** %d TESTS FAILED ***", errors);
        end
        
        $display("Testbench completed.");
        $finish;
    end
    
    // Optional: Generate waveform dump
    initial begin
        $dumpfile("greycode_decoder_tb.vcd");
        $dumpvars(0, greycode_decoder_tb);
    end

endmodule

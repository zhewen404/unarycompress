//============================================================================
// Random Increment Memory (RIM)
//
// Stores a skew number and supports efficient increment.
// Key property: At most 3 bits change per increment (no carry propagation!)
//
// Skew number properties:
//   - Each digit can be 0, 1, or 2 (stored as 2 bits)
//   - At most ONE '2' can exist in the entire number
//   - Digit weights: position i has weight (2^(i+1) - 1) = 1, 3, 7, 15, ...
//
// Increment algorithm:
//   - If no '2' exists: increment digit 0 (LSB)
//   - If '2' exists at position p: clear digit p, increment digit p+1
//============================================================================

module rim #(
    parameter NUM_DIGITS = 8    // Number of skew digits (supports values up to ~2^(NUM_DIGITS+1))
)(
    input  wire                           clk,
    input  wire                           rst_n,
    
    // Control
    input  wire                           clear,          // Reset to zero
    input  wire                           increment,      // Increment by 1
    
    // Status
    output wire                           has_two,        // A '2' exists in the number
    output wire [$clog2(NUM_DIGITS)-1:0]  two_position,   // Position of the '2' (valid if has_two)
    
    // Read interface (for converter)
    output wire [2*NUM_DIGITS-1:0]        skew_out        // Raw skew storage: {d[N-1], ..., d[1], d[0]}
);

    //------------------------------------------------------------------------
    // Skew number storage
    // Each digit stored as 2 bits: 00=0, 01=1, 10=2, 11=invalid
    //------------------------------------------------------------------------
    reg [1:0] digits [0:NUM_DIGITS-1];
    
    // Pack digits into output
    genvar g;
    generate
        for (g = 0; g < NUM_DIGITS; g = g + 1) begin : pack_output
            assign skew_out[2*g +: 2] = digits[g];
        end
    endgenerate
    
    //------------------------------------------------------------------------
    // Track the '2' position
    // Since at most one '2' exists, we can use a simple register
    //------------------------------------------------------------------------
    reg                          two_valid;      // Is there a '2'?
    reg [$clog2(NUM_DIGITS)-1:0] two_pos;        // Where is the '2'?
    
    assign has_two      = two_valid;
    assign two_position = two_pos;
    
    //------------------------------------------------------------------------
    // Determine which digit to increment
    //------------------------------------------------------------------------
    wire [$clog2(NUM_DIGITS)-1:0] inc_position;
    
    // If '2' exists, increment the digit AFTER the '2'
    // Otherwise, increment digit 0 (LSB)
    assign inc_position = two_valid ? (two_pos + 1'b1) : {$clog2(NUM_DIGITS){1'b0}};
    
    //------------------------------------------------------------------------
    // Increment logic
    //------------------------------------------------------------------------
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all digits to 0
            for (i = 0; i < NUM_DIGITS; i = i + 1) begin
                digits[i] <= 2'b00;
            end
            two_valid <= 1'b0;
            two_pos   <= {$clog2(NUM_DIGITS){1'b0}};
            
        end else if (clear) begin
            // Clear all digits to 0
            for (i = 0; i < NUM_DIGITS; i = i + 1) begin
                digits[i] <= 2'b00;
            end
            two_valid <= 1'b0;
            two_pos   <= {$clog2(NUM_DIGITS){1'b0}};
            
        end else if (increment) begin
            // Step 1: If '2' exists, clear it to '0'
            if (two_valid) begin
                digits[two_pos] <= 2'b00;
            end
            
            // Step 2: Increment the target digit
            // Target is digit 0 if no '2', or digit (two_pos + 1) if '2' exists
            case (digits[inc_position])
                2'b00: begin    // 0 -> 1
                    digits[inc_position] <= 2'b01;
                    // '2' is cleared (if existed), no new '2' created
                    two_valid <= 1'b0;
                end
                
                2'b01: begin    // 1 -> 2
                    digits[inc_position] <= 2'b10;
                    // New '2' created at inc_position
                    two_valid <= 1'b1;
                    two_pos   <= inc_position;
                end
                
                2'b10: begin    // 2 -> should not happen (can't increment a '2')
                    // This case shouldn't occur due to algorithm design
                    // If '2' exists, we increment the NEXT digit, not the '2' itself
                    digits[inc_position] <= 2'b10;  // Keep as is (error case)
                end
                
                default: begin
                    digits[inc_position] <= 2'b00;  // Invalid -> reset
                end
            endcase
        end
    end

endmodule
//============================================================================
// Skew-to-Binary Converter
//
// Converts a skew number to binary representation.
//
// Mathematical basis:
//   Skew digit d_i at position i has weight (2^(i+1) - 1)
//   Value = Σ d_i * (2^(i+1) - 1)
//         = Σ d_i * 2^(i+1) - Σ d_i
//         = 2 * Σ d_i * 2^i - Σ d_i
//
// Implementation:
//   If each digit d_i is stored as [upper_i, lower_i] where d_i = 2*upper_i + lower_i:
//   - U = upper bits as binary number = Σ upper_i * 2^i
//   - L = lower bits as binary number = Σ lower_i * 2^i
//   - digit_sum = Σ d_i = 2*popcount(U) + popcount(L)
//   
//   Value = 4*U + 2*L - digit_sum
//============================================================================

module skew_to_binary_converter #(
    parameter NUM_DIGITS   = 8,
    parameter OUTPUT_WIDTH = 16
)(
    input  wire                        clk,
    input  wire                        rst_n,
    
    // Skew number input
    input  wire [2*NUM_DIGITS-1:0]     skew_in,
    input  wire                        convert_en,
    
    // Binary output
    output reg  [OUTPUT_WIDTH-1:0]     binary_out,
    output reg                         valid_out
);

    //------------------------------------------------------------------------
    // Extract upper and lower bits from skew digits
    //------------------------------------------------------------------------
    wire [NUM_DIGITS-1:0] upper_bits;   // High bits (indicate '2's)
    wire [NUM_DIGITS-1:0] lower_bits;   // Low bits (indicate '1's)
    
    genvar g;
    generate
        for (g = 0; g < NUM_DIGITS; g = g + 1) begin : extract_bits
            assign upper_bits[g] = skew_in[2*g + 1];    // High bit of each digit
            assign lower_bits[g] = skew_in[2*g];        // Low bit of each digit
        end
    endgenerate
    
    //------------------------------------------------------------------------
    // Compute 4*U + 2*L
    // U and L are treated as binary numbers
    //------------------------------------------------------------------------
    wire [OUTPUT_WIDTH-1:0] U_extended;
    wire [OUTPUT_WIDTH-1:0] L_extended;
    
    assign U_extended = {{(OUTPUT_WIDTH-NUM_DIGITS){1'b0}}, upper_bits};
    assign L_extended = {{(OUTPUT_WIDTH-NUM_DIGITS){1'b0}}, lower_bits};
    
    wire [OUTPUT_WIDTH-1:0] four_U;     // 4 * U
    wire [OUTPUT_WIDTH-1:0] two_L;      // 2 * L
    wire [OUTPUT_WIDTH-1:0] weighted_sum;
    
    assign four_U      = U_extended << 2;
    assign two_L       = L_extended << 1;
    assign weighted_sum = four_U + two_L;
    
    //------------------------------------------------------------------------
    // Popcount: count 1s in upper and lower bits
    //------------------------------------------------------------------------
    function automatic [$clog2(NUM_DIGITS+1)-1:0] popcount;
        input [NUM_DIGITS-1:0] bits;
        integer j;
        begin
            popcount = 0;
            for (j = 0; j < NUM_DIGITS; j = j + 1) begin
                popcount = popcount + bits[j];
            end
        end
    endfunction
    
    wire [$clog2(NUM_DIGITS+1)-1:0] pop_upper;
    wire [$clog2(NUM_DIGITS+1)-1:0] pop_lower;
    
    assign pop_upper = popcount(upper_bits);
    assign pop_lower = popcount(lower_bits);
    
    //------------------------------------------------------------------------
    // Compute digit sum: Σ d_i = 2*popcount(upper) + popcount(lower)
    //------------------------------------------------------------------------
    wire [OUTPUT_WIDTH-1:0] digit_sum;
    
    assign digit_sum = {pop_upper, 1'b0} + {{(OUTPUT_WIDTH-$clog2(NUM_DIGITS+1)){1'b0}}, pop_lower};
    // Note: {pop_upper, 1'b0} is pop_upper << 1 = 2 * pop_upper
    
    //------------------------------------------------------------------------
    // Final value: 4*U + 2*L - digit_sum
    //------------------------------------------------------------------------
    wire [OUTPUT_WIDTH-1:0] binary_value;
    
    assign binary_value = weighted_sum - digit_sum;
    
    //------------------------------------------------------------------------
    // Output register
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            binary_out <= {OUTPUT_WIDTH{1'b0}};
            valid_out  <= 1'b0;
        end else begin
            valid_out <= convert_en;
            if (convert_en) begin
                binary_out <= binary_value;
            end
        end
    end

endmodule
//============================================================================
// Complete RIM Accumulator System
//
// Top-level module that combines RIM and Converter.
// Accepts a unary bitstream and outputs the binary count.
//============================================================================

module rim_accumulator_system #(
    parameter NUM_DIGITS   = 8,
    parameter OUTPUT_WIDTH = 16
)(
    input  wire                        clk,
    input  wire                        rst_n,
    
    // Unary input
    input  wire                        unary_bit,      // 0 or 1 from unary stream
    input  wire                        bit_valid,      // Bit is valid this cycle
    
    // Control
    input  wire                        start,          // Start new accumulation
    input  wire                        finish,         // Finish and get result
    
    // Output
    output wire [OUTPUT_WIDTH-1:0]     binary_result,
    output wire                        result_valid,
    
    // Debug/Status
    output wire                        has_two,
    output wire [$clog2(NUM_DIGITS)-1:0] two_position
);

    //------------------------------------------------------------------------
    // Internal signals
    //------------------------------------------------------------------------
    wire                       rim_clear;
    wire                       rim_increment;
    wire [2*NUM_DIGITS-1:0]    skew_bits;
    
    //------------------------------------------------------------------------
    // Control logic
    //------------------------------------------------------------------------
    assign rim_clear     = start;
    assign rim_increment = bit_valid && unary_bit;  // Increment only when bit is '1'
    
    //------------------------------------------------------------------------
    // RIM instance
    //------------------------------------------------------------------------
    rim #(
        .NUM_DIGITS(NUM_DIGITS)
    ) u_rim (
        .clk(clk),
        .rst_n(rst_n),
        .clear(rim_clear),
        .increment(rim_increment),
        .has_two(has_two),
        .two_position(two_position),
        .skew_out(skew_bits)
    );
    
    //------------------------------------------------------------------------
    // Converter instance
    //------------------------------------------------------------------------
    skew_to_binary_converter #(
        .NUM_DIGITS(NUM_DIGITS),
        .OUTPUT_WIDTH(OUTPUT_WIDTH)
    ) u_converter (
        .clk(clk),
        .rst_n(rst_n),
        .skew_in(skew_bits),
        .convert_en(finish),
        .binary_out(binary_result),
        .valid_out(result_valid)
    );

endmodule
//============================================================================
// Testbench for RIM Accumulator System
//============================================================================

module tb_rim_accumulator;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter NUM_DIGITS   = 8;
    parameter OUTPUT_WIDTH = 16;
    
    //------------------------------------------------------------------------
    // Signals
    //------------------------------------------------------------------------
    reg                        clk;
    reg                        rst_n;
    reg                        unary_bit;
    reg                        bit_valid;
    reg                        start;
    reg                        finish;
    
    wire [OUTPUT_WIDTH-1:0]    binary_result;
    wire                       result_valid;
    wire                       has_two;
    wire [$clog2(NUM_DIGITS)-1:0] two_position;
    
    // For comparison - traditional accumulator
    reg [OUTPUT_WIDTH-1:0]     expected_count;
    
    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    rim_accumulator_system #(
        .NUM_DIGITS(NUM_DIGITS),
        .OUTPUT_WIDTH(OUTPUT_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .unary_bit(unary_bit),
        .bit_valid(bit_valid),
        .start(start),
        .finish(finish),
        .binary_result(binary_result),
        .result_valid(result_valid),
        .has_two(has_two),
        .two_position(two_position)
    );
    
    //------------------------------------------------------------------------
    // Clock
    //------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;
    
    //------------------------------------------------------------------------
    // Helper task: Send N ones
    //------------------------------------------------------------------------
    task send_ones;
        input integer count;
        integer i;
        begin
            for (i = 0; i < count; i = i + 1) begin
                @(posedge clk);
                bit_valid = 1;
                unary_bit = 1;
                expected_count = expected_count + 1;
            end
            @(posedge clk);
            bit_valid = 0;
            unary_bit = 0;
        end
    endtask
    
    //------------------------------------------------------------------------
    // Helper task: Send N zeros
    //------------------------------------------------------------------------
    task send_zeros;
        input integer count;
        integer i;
        begin
            for (i = 0; i < count; i = i + 1) begin
                @(posedge clk);
                bit_valid = 1;
                unary_bit = 0;
            end
            @(posedge clk);
            bit_valid = 0;
        end
    endtask
    
    //------------------------------------------------------------------------
    // Helper task: Send random bitstream
    //------------------------------------------------------------------------
    task send_random_stream;
        input integer length;
        input integer ones_count;
        integer i, ones_sent, zeros_sent;
        integer remaining_ones, remaining_zeros;
        begin
            ones_sent  = 0;
            zeros_sent = 0;
            remaining_ones  = ones_count;
            remaining_zeros = length - ones_count;
            
            for (i = 0; i < length; i = i + 1) begin
                @(posedge clk);
                bit_valid = 1;
                
                // Randomly decide to send 0 or 1 (weighted by remaining)
                if (remaining_ones == 0) begin
                    unary_bit = 0;
                    remaining_zeros = remaining_zeros - 1;
                end else if (remaining_zeros == 0) begin
                    unary_bit = 1;
                    remaining_ones = remaining_ones - 1;
                    expected_count = expected_count + 1;
                end else if ($random % (remaining_ones + remaining_zeros) < remaining_ones) begin
                    unary_bit = 1;
                    remaining_ones = remaining_ones - 1;
                    expected_count = expected_count + 1;
                end else begin
                    unary_bit = 0;
                    remaining_zeros = remaining_zeros - 1;
                end
            end
            @(posedge clk);
            bit_valid = 0;
            unary_bit = 0;
        end
    endtask
    
    //------------------------------------------------------------------------
    // Test sequence
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("rim_accumulator.vcd");
        $dumpvars(0, tb_rim_accumulator);
        
        // Initialize
        rst_n          = 0;
        unary_bit      = 0;
        bit_valid      = 0;
        start          = 0;
        finish         = 0;
        expected_count = 0;
        
        #25 rst_n = 1;
        #10;
        
        //--------------------------------------------------------------------
        // Test 1: Simple count of 10 ones
        //--------------------------------------------------------------------
        $display("\n========================================");
        $display("Test 1: Count 10 ones");
        $display("========================================");
        
        @(posedge clk);
        start = 1;
        expected_count = 0;
        @(posedge clk);
        start = 0;
        
        send_ones(10);
        
        @(posedge clk);
        finish = 1;
        @(posedge clk);
        finish = 0;
        
        @(posedge clk);
        $display("Result: %d, Expected: %d, %s", 
                 binary_result, expected_count,
                 (binary_result == expected_count) ? "PASS" : "FAIL");
        
        //--------------------------------------------------------------------
        // Test 2: Count with zeros interspersed
        //--------------------------------------------------------------------
        $display("\n========================================");
        $display("Test 2: 15 ones with zeros");
        $display("========================================");
        
        #20;
        @(posedge clk);
        start = 1;
        expected_count = 0;
        @(posedge clk);
        start = 0;
        
        send_ones(5);
        send_zeros(3);
        send_ones(5);
        send_zeros(2);
        send_ones(5);
        
        @(posedge clk);
        finish = 1;
        @(posedge clk);
        finish = 0;
        
        @(posedge clk);
        $display("Result: %d, Expected: %d, %s", 
                 binary_result, expected_count,
                 (binary_result == expected_count) ? "PASS" : "FAIL");
        
        //--------------------------------------------------------------------
        // Test 3: Larger count (100 ones)
        //--------------------------------------------------------------------
        $display("\n========================================");
        $display("Test 3: Count 100 ones");
        $display("========================================");
        
        #20;
        @(posedge clk);
        start = 1;
        expected_count = 0;
        @(posedge clk);
        start = 0;
        
        send_ones(100);
        
        @(posedge clk);
        finish = 1;
        @(posedge clk);
        finish = 0;
        
        @(posedge clk);
        $display("Result: %d, Expected: %d, %s", 
                 binary_result, expected_count,
                 (binary_result == expected_count) ? "PASS" : "FAIL");
        
        //--------------------------------------------------------------------
        // Test 4: Watch skew number evolve (demonstrate 3-bit flipping)
        //--------------------------------------------------------------------
        $display("\n========================================");
        $display("Test 4: Observe skew number evolution");
        $display("========================================");
        $display("Showing first 20 increments:");
        $display("Count | Skew Digits (d7..d0) | Has 2 | 2-Pos");
        $display("------|----------------------|-------|------");
        
        #20;
        @(posedge clk);
        start = 1;
        expected_count = 0;
        @(posedge clk);
        start = 0;
        
        repeat(20) begin
            @(posedge clk);
            bit_valid = 1;
            unary_bit = 1;
            expected_count = expected_count + 1;
            @(posedge clk);
            bit_valid = 0;
            
            // Display skew state
            $display("  %2d  |  %d %d %d %d %d %d %d %d  |   %b   |   %d",
                     expected_count,
                     dut.u_rim.digits[7],
                     dut.u_rim.digits[6],
                     dut.u_rim.digits[5],
                     dut.u_rim.digits[4],
                     dut.u_rim.digits[3],
                     dut.u_rim.digits[2],
                     dut.u_rim.digits[1],
                     dut.u_rim.digits[0],
                     has_two,
                     two_position);
        end
        
        @(posedge clk);
        finish = 1;
        @(posedge clk);
        finish = 0;
        
        @(posedge clk);
        $display("\nFinal Result: %d, Expected: %d, %s", 
                 binary_result, expected_count,
                 (binary_result == expected_count) ? "PASS" : "FAIL");
        
        //--------------------------------------------------------------------
        // Test 5: Random bitstream
        //--------------------------------------------------------------------
        $display("\n========================================");
        $display("Test 5: Random bitstream (50 ones in 100 bits)");
        $display("========================================");
        
        #20;
        @(posedge clk);
        start = 1;
        expected_count = 0;
        @(posedge clk);
        start = 0;
        
        send_random_stream(100, 50);
        
        @(posedge clk);
        finish = 1;
        @(posedge clk);
        finish = 0;
        
        @(posedge clk);
        $display("Result: %d, Expected: %d, %s", 
                 binary_result, expected_count,
                 (binary_result == expected_count) ? "PASS" : "FAIL");
        
        //--------------------------------------------------------------------
        // Test 6: Edge case - zero ones
        //--------------------------------------------------------------------
        $display("\n========================================");
        $display("Test 6: Zero ones (empty stream)");
        $display("========================================");
        
        #20;
        @(posedge clk);
        start = 1;
        expected_count = 0;
        @(posedge clk);
        start = 0;
        
        send_zeros(10);
        
        @(posedge clk);
        finish = 1;
        @(posedge clk);
        finish = 0;
        
        @(posedge clk);
        $display("Result: %d, Expected: %d, %s", 
                 binary_result, expected_count,
                 (binary_result == expected_count) ? "PASS" : "FAIL");
        
        //--------------------------------------------------------------------
        // Test 7: Maximum value test
        //--------------------------------------------------------------------
        $display("\n========================================");
        $display("Test 7: Large count (255 ones)");
        $display("========================================");
        
        #20;
        @(posedge clk);
        start = 1;
        expected_count = 0;
        @(posedge clk);
        start = 0;
        
        send_ones(255);
        
        @(posedge clk);
        finish = 1;
        @(posedge clk);
        finish = 0;
        
        @(posedge clk);
        $display("Result: %d, Expected: %d, %s", 
                 binary_result, expected_count,
                 (binary_result == expected_count) ? "PASS" : "FAIL");
        
        //--------------------------------------------------------------------
        // Done
        //--------------------------------------------------------------------
        #50;
        $display("\n========================================");
        $display("All tests complete!");
        $display("========================================\n");
        $finish;
    end

endmodule

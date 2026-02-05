module greycode_counter #(
    parameter WIDTH = 8
) (
    input clk,
    input rst_n,
    input enable,
    output reg [WIDTH-1:0] grey_count
);

    // Direct Gray code counter logic
    wire [WIDTH-1:0] grey_next;
    
    // Generate next Gray code value
    genvar i;
    generate
        // Bit 0 (LSB) always flips
        assign grey_next[0] = ~grey_count[0];
        
        // Higher bits flip based on lower bit pattern
        for (i = 1; i < WIDTH; i = i + 1) begin : gen_grey_logic
            wire [i-1:0] lower_bits_pattern;
            
            // Create the trigger pattern: ~bit[0] & bit[1] & bit[2] & ... & bit[i-1] & ~bit[i]
            assign lower_bits_pattern[0] = ~grey_count[0];
            if (i > 1) begin : gen_pattern
                integer j;
                for (j = 1; j < i; j = j + 1) begin : gen_pattern_bits
                    assign lower_bits_pattern[j] = lower_bits_pattern[j-1] & grey_count[j];
                end
            end
            
            // Bit i flips when the pattern matches and current bit i is 0
            assign grey_next[i] = grey_count[i] ^ (lower_bits_pattern[i-1] & ~grey_count[i]);
        end
    endgenerate
    
    // Gray code counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            grey_count <= {WIDTH{1'b0}};
        end else if (enable) begin
            grey_count <= grey_next;
        end
    end

endmodule

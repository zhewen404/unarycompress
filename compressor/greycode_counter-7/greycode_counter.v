module greycode_counter (
    input clk,
    input rst_n,
    input enable,
    output reg [7:0] grey_count
);

    // Direct Gray code counter logic
    wire [7:0] grey_next;
    
    // Generate next Gray code value
    // The logic determines which bit to flip based on current state
    assign grey_next[0] = ~grey_count[0];
    assign grey_next[1] = grey_count[1] ^ (~grey_count[0] & ~grey_count[1]);
    assign grey_next[2] = grey_count[2] ^ (~grey_count[0] & grey_count[1] & ~grey_count[2]);
    assign grey_next[3] = grey_count[3] ^ (~grey_count[0] & grey_count[1] & grey_count[2] & ~grey_count[3]);
    assign grey_next[4] = grey_count[4] ^ (~grey_count[0] & grey_count[1] & grey_count[2] & grey_count[3] & ~grey_count[4]);
    assign grey_next[5] = grey_count[5] ^ (~grey_count[0] & grey_count[1] & grey_count[2] & grey_count[3] & grey_count[4] & ~grey_count[5]);
    assign grey_next[6] = grey_count[6] ^ (~grey_count[0] & grey_count[1] & grey_count[2] & grey_count[3] & grey_count[4] & grey_count[5] & ~grey_count[6]);
    assign grey_next[7] = grey_count[7] ^ (~grey_count[0] & grey_count[1] & grey_count[2] & grey_count[3] & grey_count[4] & grey_count[5] & grey_count[6] & ~grey_count[7]);
    
    // Gray code counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            grey_count <= 8'b0;
        end else if (enable) begin
            grey_count <= grey_next;
        end
    end

endmodule

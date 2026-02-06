module greycode_decoder #(
    parameter WIDTH = 8
) (
    input  [WIDTH-1:0] grey_in,
    output [WIDTH-1:0] binary_out
);

    // Grey code to binary conversion
    // The MSB of binary is the same as MSB of grey
    // Each subsequent binary bit is XOR of current grey bit and previous binary bit
    
    genvar i;
    generate
        // MSB of binary is same as MSB of grey
        assign binary_out[WIDTH-1] = grey_in[WIDTH-1];
        
        // For remaining bits: binary[i] = grey[i] XOR binary[i+1]
        for (i = WIDTH-2; i >= 0; i = i - 1) begin : grey_to_binary
            assign binary_out[i] = grey_in[i] ^ binary_out[i+1];
        end
    endgenerate

endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: tb_layer_norm
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

`timescale 1ns/100ps

module tb_layer_norm;

  parameter data_width=16;
  parameter length = 256;
  parameter p_ln = 8;
  parameter CLK_PERIOD = 10;

  reg Clock, Reset_n, Up_vld, Dn_rdy;
  wire [data_width*p_ln-1 : 0] Up_dat;
  wire Dn_vld, Up_rdy;
  wire [data_width*p_ln-1 : 0] Dn_dat;
  reg [16-1 : 0] Bias_LN;
  reg [16-1 : 0] Length;
  real input_array[length-1:0];
  real output_array[length-1:0];
  int indx;
  int fd_r, fd_w;
  bit [data_width-1 : 0] outputs[p_ln][$];
  reg [data_width-1:0] error_half;
  real error_real;
  real output_real;
  reg [15:0] output_half;
  reg [63:0] output_ext;
  reg [63:0] input_real;
  reg [15:0] input_half;
  reg [63:0] input_ext;

  real golden_real;
  reg [15:0] golden_half;
  reg [63:0] golden_ext;

  string line;

  reg [data_width-1 : 0] Up_dats[p_ln-1 : 0];
  wire [data_width-1 : 0] Dn_dats[p_ln-1 : 0];

  task realtohalf (input [63:0] real_num, output [15:0] half_num);
    begin
        half_num = {real_num[63:62], real_num[55:52], real_num[51:42]}; 
        if (real_num[41]) half_num = half_num+1;// neareat
    end
  endtask
 
  task halftoreal (input [15:0] half_num, output [63:0] real_num);
    begin
        real_num = 64'b0;
        real_num[63:62] = half_num[15:14];
        real_num[55:52] = half_num[13:10];
        if (real_num[62] == 1'b0) real_num[61:56] = {6{1'b1}};
        real_num[51:42] = half_num[9:0];
    end
  endtask
  
  typedef real real_1d [length-1:0];
  function real_1d read1darray (output real_1d array, input string name);
    begin
      fd_r = $fopen (name, "r"); 
      $display("function file open result: %0d", fd_r);
      for (int i=0; i < length; i++) begin
        $fgets(line, fd_r);
        array[i] = line.atoreal();
        // $display("read element: %f", array[i]);
      end  
      $fclose(fd_r);
    end
  endfunction


  // Clock generator
  always #CLK_PERIOD Clock = ~Clock; 
  genvar g;
  generate
  for(g=0 ; g<p_ln ; g=g+1)
  begin : ASSIGN_WEIGHT
      assign Up_dat[(data_width*g + data_width-1) : (data_width*g)] = Up_dats[g];
    end
  endgenerate

  layer_norm #(
    .data_width(data_width),
    .p_ln(p_ln)
  )DUT1
  (
    .rst_n(Reset_n),
    .clk(Clock),

    .up_vld(Up_vld),
    .up_dat(Up_dat),
    .up_rdy(Up_rdy),
    
    .bias_ln(Bias_LN),
    .length(Length),

    .dn_vld(Dn_vld),
    .dn_rdy(Dn_rdy),
    .dn_dat(Dn_dat)
  );

  // Test stimulus
  initial
  begin
    Clock = 1;
    Reset_n = 0;
    Bias_LN = 0;
    Dn_rdy = 0;
    Up_vld = 0;
    Length = length;
    for (int j = 0; j < p_ln; j++) begin
        Up_dats[j] = 0;
    end
    #(CLK_PERIOD*2); // Should be reset
    Reset_n = 1;
    #(CLK_PERIOD*20);
    //Read inputs from file 
    $display("Reading from %s", "/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_bfly_ln_sc256/output_bfly.txt");
    read1darray(input_array, "/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_bfly_ln_sc256/output_bfly.txt");
    for (int i = 0; i < length/p_ln; i++) begin
        Up_vld = 1;
        for (int j = 0; j < p_ln; j++) begin
            indx = i*p_ln+j;
            input_real = $realtobits(input_array[indx]);
            realtohalf(input_real, input_half);
            Up_dats[j] = input_half;
            halftoreal(input_half, input_ext);
            $display("======Input[%0d] = %f , b:(%b), h:(%h) (%h)======", indx, input_array[indx], input_half, input_half, input_ext);
        end
        #(CLK_PERIOD*2);
    end
    Up_vld = 0;

    #(CLK_PERIOD*500);

      //Read outputs from file for comparison
      $display("Reading from %s", "/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_bfly_ln_sc256/output_ln.txt");
      read1darray(output_array, "/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_bfly_ln_sc256/output_ln.txt");

    for (int i = 0; i < length/p_ln; i++) begin
        for (int j = 0; j < p_ln; j++) begin
            indx = i*p_ln+j;
            golden_ext = $realtobits(output_array[indx]);
            realtohalf(golden_ext, golden_half); // Get Half Golden

            output_half = outputs[j][i];
            halftoreal(output_half, output_ext);

            golden_real = $bitstoreal(golden_ext);
            output_real = $bitstoreal(output_ext);

            if (output_real > golden_real) error_real = output_real - golden_real;
            else error_real = golden_real - output_real;
            $display("Output[%0d] = %h (%.3f), Golden:%h (%.3f),  Absolute Error:%.3f", indx, output_half, $bitstoreal(output_ext), golden_half, $bitstoreal(golden_ext), error_real);
        end
    end

    $finish;
  end

  generate
  for(g=0 ; g<p_ln ; g=g+1)
  begin : GET_OUTPUTS
        always @ (posedge Clock) begin
            if (Dn_vld) begin
                outputs[g].push_back(Dn_dat[(data_width*g + data_width-1) : (data_width*g)]);
            end
        end
    end
  endgenerate


endmodule

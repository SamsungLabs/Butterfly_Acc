`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: float16_add_test
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


module tb_weight_buffer;

  parameter row = 3;
  parameter col = 3;
  parameter length = 32;
  parameter CLK_PERIOD = 10;
  parameter bu_parallelism = 4;
  parameter depth = (length*2)/(bu_parallelism*4);
  parameter data_width = 16;
  real output_array[row-1:0][col-1:0];
  
  reg [63:0] output_real;
  reg [15:0] output_half;
  bit [16-1:0] output_half_q[$];
  bit [data_width*(4*bu_parallelism)-1:0] outputs[$];
  bit [data_width*(4*bu_parallelism)-1:0] inputs[$];
  reg [63:0] output_ext;
  

  wire                       up_vld;
  wire   [data_width*(4*bu_parallelism)-1:0]       up_dat;
  wire                      up_rdy;
  wire                      dn_vld;
  wire  [data_width*(4*bu_parallelism)-1:0]       dn_dat;
  reg                       dn_rdy;
  
  reg   [16-1:0]       Length;
  reg                  Butterfly_start;
  
  typedef real real_2d [depth-1:0][bu_parallelism*4-1:0];
  
  int fd_r, fd_w;
  string line;
  genvar k;
  
  int indx;
  
  reg Clock, Reset_n;
        
  // Clock generator
  always #CLK_PERIOD Clock = ~Clock;  
  
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

  function real_2d read2darray (output real_2d array, input string name);
    begin
      fd_r = $fopen (name, "r"); 
      $display("function file open result: %0d", fd_r);
      for (int i=0; i < depth; i++) begin
        for (int j=0; j < 4*bu_parallelism; j++) begin
            $fgets(line, fd_r);
            array[i][j] = line.atoreal();
            $display("read element: %f", array[i][j]);
        end
      end  
      $fclose(fd_r);
    end
  endfunction
  reg weight_vld;
  
  weight_buffer u_weight_buffer (
    .clk(Clock),
    .rst_n(Reset_n),
    .length(Length),
    .butterfly_start(Butterfly_start),
    .up_vld(weight_vld),
    .up_dat(up_dat),
    .up_rdy(up_rdy),
    .dn_vld(dn_vld),
    .dn_dat(dn_dat),
    .dn_rdy(dn_rdy)
  );
  
  ////////////************** Stimulus, Driver **************/////////////
  string file_name;
  real weight_array[depth-1:0][bu_parallelism*4-1:0];
  reg [data_width-1:0] weights[bu_parallelism*4-1:0];
  reg [63:0] weight_real;
  reg [15:0] weight_half;
  
  assign up_vld = weight_vld;
  
  genvar g;
  
  generate
  for(g=0 ; g<4*bu_parallelism ; g=g+1)
  begin : ASSIGN_WEIGHT
      assign up_dat[(data_width*g + data_width-1) : (data_width*g)] = weights[g];
    end
  endgenerate
  
  
  initial 
    begin
      /////////////// Reading inputs and outputs /////////////// 
      // Reading data from file
      Clock = 1;
      dn_rdy = 1;
      for (int k=0; k < (4*bu_parallelism); k++) begin
        weights[k] = 0;
      end  
      weight_vld = 1'b0;
      Butterfly_start = 0;
      Length = length;
      #(CLK_PERIOD*2);
      Reset_n = 0;
      #(CLK_PERIOD*2);
      Reset_n = 1;
      #(CLK_PERIOD*4);
      weight_vld = 1'b1;
      for (int i = 0; i < $clog2(length); i++) begin
        //Read weights from file
        file_name = $sformatf("/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_bfly32/weight_stage%0d.txt", i);
        $display("Reading from %s", file_name);
        read2darray(weight_array, file_name);
          for (int j=0; j < depth; j++) begin
            for (int k=0; k < (4*bu_parallelism); k++) begin
              weight_real = $realtobits(weight_array[j][k]);
              realtohalf(weight_real, weight_half);
              weights[k] = weight_half;
            end
            #(CLK_PERIOD*2);
            inputs.push_back(up_dat);
          end
      end
      weight_vld = 1'b0;
      #(CLK_PERIOD*2);
      Butterfly_start = 1;
      #(CLK_PERIOD*2);
      Butterfly_start = 0;
      #(CLK_PERIOD*1000);
      $display("************Scoreboard Calculating************");
      ////////////************** Scoreboard **************/////////////
      
      for (int i = 0; i < $clog2(length); i++) begin
        for (int j=0; j < depth; j++) begin
            indx = i*row + j;
            $display("Output[%0d] = %h, Golden:%h ,  Error:%b", indx, outputs[indx], inputs[indx], outputs[indx]-inputs[indx]);
        end
      end
      //foreach(outputs[i]) $display("Output[%0d] = %h", i, outputs[i]);
      $finish;
    end
  ////////////************** Monitor **************/////////////

  always @ (posedge Clock) begin
    if (dn_vld) begin
        outputs.push_back(dn_dat);
    end
  end
  
  
endmodule

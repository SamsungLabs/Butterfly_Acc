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


module tb_butterfly_processor_be32_length256;

  parameter row = 3;
  parameter col = 3;
  parameter length = 256;
  parameter CLK_PERIOD = 10;
  parameter bu_parallelism = 4;
  parameter be_parallelism = 32;
  parameter depth = (length*2)/(bu_parallelism*4);
  parameter data_width = 16;
  parameter input_axi_chnl = 8;
  parameter output_axi_chnl = 8;
  real output_array[row-1:0][col-1:0];
  

  bit [16-1:0] output_half_q[$];
  bit [data_width-1:0] outputs[be_parallelism][$];
  bit [data_width*(4*bu_parallelism)-1:0] inputs[$];
  reg [63:0] input_ext;
  reg [data_width-1:0] error_half;
  real error_real;

  real output_real;
  reg [15:0] output_half;
  reg [63:0] output_ext;
  real golden_real;
  reg [15:0] golden_half;
  reg [63:0] golden_ext;


  wire                       up_vld;
  wire   [data_width*(4*bu_parallelism)-1:0]       up_dat;
  wire                      up_rdy;
  wire                      dn_vld;
  wire  [data_width*(4*bu_parallelism)-1:0]       dn_dat;
  reg                       dn_rdy;
  
  reg   [16-1:0]       Length;
  reg                  Butterfly_start;
  
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
  
  typedef real real_2d [depth-1:0][bu_parallelism*4-1:0];
  function real_2d read2darray (output real_2d array, input string name);
    begin
      fd_r = $fopen (name, "r"); 
      $display("function file open result: %0d", fd_r);
      for (int i=0; i < depth; i++) begin
        for (int j=0; j < 4*bu_parallelism; j++) begin
            $fgets(line, fd_r);
            array[i][j] = line.atoreal();
            // $display("read element: %f", array[i][j]);
        end
      end  
      $fclose(fd_r);
    end
  endfunction
  


  reg               Is_FFT;
  reg               Is_SC_Add;
  reg               Is_SC_Cache;
  reg               Is_LN;
  wire [data_width*(4*bu_parallelism)-1:0]    Up_Weight_Dat;
  wire               Up_Weight_Vld;
  reg [data_width*(4*bu_parallelism)-1:0]    Up_Weight_Dat_r;
  reg               Up_Weight_Vld_r;
  reg               Is_Bypass_P2S;
  reg [input_axi_chnl-1:0]  Up_Vld;
  reg [(2*data_width)*be_parallelism-1:0]    Up_Dat;
  reg [input_axi_chnl-1:0]  Up_Vld_r;
  reg [(2*data_width)*be_parallelism-1:0]    Up_Dat_r;
  wire              Up_Rdy;

  // Port A
  // down stream data output for FFT
  wire  [output_axi_chnl-1:0]                Dn_Serial_Vld_A;
  wire  [data_width*be_parallelism-1:0]      Dn_Serial_Dat_A; // real
  reg                         Dn_Serial_Rdy_A;
  
  wire  [output_axi_chnl-1:0]                       Dn_Parallel_Vld_A;
  wire  [(2*bu_parallelism)*data_width*be_parallelism-1:0]      Dn_Parallel_Dat_A; // real
  reg                         Dn_Parallel_Rdy_A;

  // Port B
  // down stream data output for FFT
  wire  [output_axi_chnl-1:0]                Dn_Serial_Vld_B;
  wire  [data_width*be_parallelism-1:0]      Dn_Serial_Dat_B; // complex
  reg                         Dn_Serial_Rdy_B;
  
  wire  [output_axi_chnl-1:0]                 Dn_Parallel_Vld_B; 
  wire  [(2*bu_parallelism)*data_width*be_parallelism-1:0]      Dn_Parallel_Dat_B; // complex
  reg                         Dn_Parallel_Rdy_B;


  butterfly_processor # (
      .INPUT_AXI_CHNL(input_axi_chnl),
      .OUTPUT_AXI_CHNL(output_axi_chnl),
      .data_width(data_width),
      .be_parallelism(be_parallelism),
      .bu_parallelism(bu_parallelism),
      .latency_add(1),
      .latency_mul(1)
  ) u_butterfly_processor (
    .clk(Clock),
    .rst_n(Reset_n),
    .is_fft(Is_FFT),
    .is_ln(Is_LN),
    .is_sc_add(Is_SC_Add),
    .is_sc_cache(Is_SC_Cache),
    .length(Length),
    .is_bypass_p2s(Is_Bypass_P2S),
    .up_weight_dat(Up_Weight_Dat_r),
    .up_weight_vld(Up_Weight_Vld_r),
    // Receive input one by one 
    .up_vld(Up_Vld_r),
    .up_dat(Up_Dat_r), // real + complex
    .up_rdy(Up_Rdy),

    // Port A
    // down stream data output for FFT
    .dn_serial_vld_A(Dn_Serial_Vld_A), 
    .dn_serial_dat_A(Dn_Serial_Dat_A), // real
    .dn_serial_rdy_A(Dn_Serial_Rdy_A),
    
    .dn_parallel_vld_A(Dn_Parallel_Vld_A), 
    .dn_parallel_dat_A(Dn_Parallel_Dat_A), // real
    .dn_parallel_rdy_A(Dn_Parallel_Rdy_A),

    // Port B
    // down stream data output for FFT
    .dn_serial_vld_B(Dn_Serial_Vld_B), 
    .dn_serial_dat_B(Dn_Serial_Dat_B), // complex
    .dn_serial_rdy_B(Dn_Serial_Rdy_B),
    
    .dn_parallel_vld_B(Dn_Parallel_Vld_B), 
    .dn_parallel_dat_B(Dn_Parallel_Dat_B), // complex
    .dn_parallel_rdy_B(Dn_Parallel_Rdy_B)
  );
  
  ////////////************** Stimulus, Driver **************/////////////
  string data_file_name;
  string weight_file_name;
  real input_array[length-1:0];
  real imresult_array[length-1:0];
  real weight_array[depth-1:0][bu_parallelism*4-1:0];
  reg [data_width-1:0] weights[bu_parallelism*4-1:0];
  reg [2*data_width*be_parallelism-1:0] input_dat;
  reg [63:0] weight_real;
  reg [15:0] weight_half;
  reg [63:0] input_real;
  reg [15:0] input_half;
  reg [63:0] imresult_real;
  reg [15:0] imresult_half;
  reg weight_vld;
  
  genvar g;


  ///// Assign weight data
  generate
  for(g=0 ; g<4*bu_parallelism ; g=g+1)
  begin : ASSIGN_WEIGHT
      assign Up_Weight_Dat[(data_width*g + data_width-1) : (data_width*g)] = weights[g];
    end
  endgenerate
  assign Up_Weight_Vld = weight_vld;

  /////////////// Loading input ///////////////
  initial 
    begin
      Up_Vld = 0;
      Up_Dat = 0;
      #(CLK_PERIOD*6);
      //Read inputs from file 
      $display("Reading from %s", "/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_bfly256/input_bfly.txt");
      read1darray(input_array, "/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_bfly256/input_bfly.txt");
      for (int i = 0; i < length; i++) begin
          input_real = $realtobits(input_array[i]);
          realtohalf(input_real, input_half);
          Up_Vld = {8{1'b1}};
          Up_Dat = {be_parallelism{16'h0000, input_half}}; // Pad higher bit as zeros
          halftoreal(input_half, input_ext);
          $display("======Input[%0d] = %f , b:(%b), h:(%h) (%h)======", i, input_array[i], input_half, input_half, input_ext);
          #(CLK_PERIOD*2);
      end
      Up_Vld = 0;
    end

  always @ (posedge Clock) begin
    begin
        Up_Vld_r = Up_Vld;
        Up_Dat_r = Up_Dat;
        Up_Weight_Dat_r = Up_Weight_Dat;
        Up_Weight_Vld_r = Up_Weight_Vld;
    end
  end


  /////////////// Loading intermediate results for comparing ///////////////
  initial 
    begin
      //Read intermediate results from file 
      for (int i = 0; i < $clog2(length); i++) begin
        //Read weights from file
        data_file_name = $sformatf("/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_bfly256/data_stage%0d.txt", i);
        $display("Reading from %s, Stage %0d", data_file_name, i);
        read1darray(imresult_array, data_file_name);
          for (int j=0; j < length; j++) begin
            imresult_real = $realtobits(imresult_array[j]);
            realtohalf(imresult_real, imresult_half);
            $display("======Interm Results, Stage %0d, imresults[%0d] = %f, b:(%b), h:(%h))======", i, j, imresult_array[j], imresult_half, imresult_half);
          end
      end
    end

  /////////////// Loading weight /////////////// 
  initial 
    begin
      Clock = 1;
      Is_Bypass_P2S = 0;
      Is_FFT = 0;
      Is_SC_Add = 0;
      Is_SC_Cache = 0;
      Is_LN = 0;
      dn_rdy = 1;
      for (int k=0; k < (4*bu_parallelism); k++) begin
        weights[k] = 0;
      end  
      weight_vld = 1'b0;
      Butterfly_start = 0;
      Length = length;
      Dn_Serial_Rdy_A = 1;
      Dn_Parallel_Rdy_A = 1;
      Dn_Serial_Rdy_B = 1;
      Dn_Parallel_Rdy_B = 1;      
      Reset_n = 0;
      #(CLK_PERIOD*2);
      Reset_n = 1;
      #(CLK_PERIOD*4);
      for (int i = 0; i < $clog2(length); i++) begin
        //Read weights from file
        weight_file_name = $sformatf("/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_bfly256/weight_stage%0d.txt", i);
        $display("Reading from %s", weight_file_name);
        read2darray(weight_array, weight_file_name);
          for (int j=0; j < depth; j++) begin
          weight_vld = 1'b1;
            for (int k=0; k < (4*bu_parallelism); k++) begin
              weight_real = $realtobits(weight_array[j][k]);
              realtohalf(weight_real, weight_half);
              weights[k] = weight_half;
              $display("======Weights, Stage %0d, imresults[%0d] = %f, b:(%b), h:(%h))======", i, j, weight_array[j][k], weight_half, weight_half);
            end
            #(CLK_PERIOD*2);
          end
      end
      weight_vld = 1'b0;
      #(CLK_PERIOD*2);
      Butterfly_start = 1;
      #(CLK_PERIOD*2);
      Butterfly_start = 0;
      #(CLK_PERIOD*2000);
      $display("************Scoreboard Calculating************");
      ////////////************** Scoreboard **************/////////////

      for (int i = 0; i < be_parallelism; i++) begin
        for (int j=0; j < length; j++) begin
            golden_ext = $realtobits(imresult_array[j]);
            realtohalf(golden_ext, golden_half); // Get Half Golden
            golden_real = imresult_array[j];

            output_half = outputs[i][j];
            halftoreal(output_half, output_ext);
            output_real = $bitstoreal(output_ext);

            if (output_real > golden_real) error_real = output_real - golden_real;
            else error_real = golden_real - output_real;

            $display("BE[%0d] Output[%0d] = %h (%.3f), Golden:%h (%.3f),  Absolute Error:%.3f", i, j, output_half, output_real, golden_half, golden_real, error_real);
        end
      end

      //foreach(outputs[i]) $display("Output[%0d] = %h", i, outputs[i]);
      $finish;
    end
  ////////////************** Monitor **************/////////////

  generate
  for(g=0 ; g<be_parallelism ; g=g+1)
  begin : GET_OUTPUTS
      always @ (posedge Clock) begin
        if (Dn_Serial_Vld_A) begin
            outputs[g].push_back(Dn_Serial_Dat_A[(data_width*g + data_width-1) : (data_width*g)]);
        end
      end
    end
  endgenerate
  
endmodule

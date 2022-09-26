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


module tb_fft_processor_be32_length256;

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
  bit [data_width-1:0] outputs_real[be_parallelism][$];
  bit [data_width-1:0] outputs_image[be_parallelism][$];
  bit [data_width*(4*bu_parallelism)-1:0] inputs[$];
  reg [63:0] output_ext;
  reg [63:0] input_ext;
  real error_real_double;
  real error_image_double;

  real output_real_double;
  reg [15:0] output_real_half;
  reg [63:0] output_real_ext;
  real golden_real_double;
  reg [15:0] golden_real_half;
  reg [63:0] golden_real_ext;

  real output_image_double;
  reg [15:0] output_image_half;
  reg [63:0] output_image_ext;
  real golden_image_double;
  reg [15:0] golden_image_half;
  reg [63:0] golden_image_ext;


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
    .is_sc_add(Is_SC_Add),
    .is_sc_cache(Is_SC_Cache),
    .is_ln(Is_LN),
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
  real imresult_array_real[length-1:0];
  real imresult_array_image[length-1:0];
  real weight_array_real[depth-1:0][bu_parallelism*4-1:0];
  real weight_array_image[depth-1:0][bu_parallelism*4-1:0];
  reg [data_width-1:0] weights[bu_parallelism*4-1:0];
  reg [2*data_width*be_parallelism-1:0] input_dat;
  reg [63:0] weight_real_double;
  reg [15:0] weight_real_half;
  reg [63:0] weight_image_double;
  reg [15:0] weight_image_half;
  reg [63:0] input_real;
  reg [15:0] input_half;
  reg [63:0] imresult_real_double;
  reg [63:0] imresult_image_double;
  reg [15:0] imresult_half_real;
  reg [15:0] imresult_half_image;
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
      $display("Reading from %s", "/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_fft256/input_fft.txt");
      read1darray(input_array, "/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_fft256/input_fft.txt");
      for (int i = 0; i < length; i++) begin
          input_real = $realtobits(input_array[i]);
          realtohalf(input_real, input_half);
          Up_Vld = {8{1'b1}};
          Up_Dat = {be_parallelism{16'h0000, input_half}}; // Pad higher bit as zeros, real at low, image at high
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
        //Read intermediate real from file
        data_file_name = $sformatf("/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_fft256/data_stage%0d_real.txt", i);
        $display("Reading from %s, Stage %0d", data_file_name, i);
        read1darray(imresult_array_real, data_file_name);
        //Read intermediate image from file
        data_file_name = $sformatf("/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_fft256/data_stage%0d_image.txt", i);
        $display("Reading from %s, Stage %0d", data_file_name, i);
        read1darray(imresult_array_image, data_file_name);
          for (int j=0; j < length; j++) begin
            imresult_real_double = $realtobits(imresult_array_real[j]);
            realtohalf(imresult_real_double, imresult_half_real);
            imresult_image_double = $realtobits(imresult_array_image[j]);
            realtohalf(imresult_image_double, imresult_half_image);
            $display("======Interm Results, Stage %0d, imresults[%0d] = %f + %f j, real:(%h), image:(%h)======", i, j, imresult_array_real[j], imresult_array_image[j], imresult_half_real, imresult_half_image);
          end
      end
    end

  /////////////// Loading weight /////////////// 
  initial 
    begin
      Clock = 1;
      Is_Bypass_P2S = 0;
      Is_FFT = 1;
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
        //Read real weights from file
        weight_file_name = $sformatf("/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_fft256/weight%0d_real.txt", i);
        $display("Reading from %s", weight_file_name);
        read2darray(weight_array_real, weight_file_name);
        //Read image weights from file
        weight_file_name = $sformatf("/mnt/ccnas2/bdp/hf17/Transformer/benchmarks/float16_fft256/weight%0d_image.txt", i);
        $display("Reading from %s", weight_file_name);
        read2darray(weight_array_image, weight_file_name);
          for (int j=0; j < depth; j++) begin
          weight_vld = 1'b1;

            for (int k=0; k < (bu_parallelism); k++) begin
              weight_real_double = $realtobits(weight_array_real[j][4*k+1]); // Get the second one
              realtohalf(weight_real_double, weight_real_half);

              weight_image_double = $realtobits(weight_array_image[j][4*k+1]); // Get the second one
              realtohalf(weight_image_double, weight_image_half);

              weights[4*k] = weight_real_half;
              weights[4*k+1] = weight_image_half;
              weights[4*k+2] = {16{1'b0}};
              weights[4*k+3] = {16{1'b0}};

              $display("======Weights, Stage %0d, imresults[%0d] = %f + %f j, real:(%h), image:(%h)======", i, j, weight_array_real[j][4*k+1], weight_array_image[j][4*k+1], weight_real_half, weight_image_half);
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
            // Real part
            golden_real_ext = $realtobits(imresult_array_real[j]);
            realtohalf(golden_real_ext, golden_real_half); // Get Half Golden
            golden_real_double = imresult_array_real[j]; // double data type of real part

            output_real_half = outputs_real[i][j];
            halftoreal(output_real_half, output_real_ext);
            output_real_double = $bitstoreal(output_real_ext);

            // Image part
            golden_image_ext = $realtobits(imresult_array_image[j]);
            realtohalf(golden_image_ext, golden_image_half); // Get Half Golden
            golden_image_double = imresult_array_image[j];

            output_image_half = outputs_image[i][j];
            halftoreal(output_image_half, output_image_ext);
            output_image_double = $bitstoreal(output_image_ext);

            // Calculate Error
            // Real part
            if (output_real_double > golden_real_double) error_real_double = output_real_double - golden_real_double;
            else error_real_double = golden_real_double - output_real_double;

            // Image part
            if (output_image_double > golden_image_double) error_image_double = output_image_double - golden_image_double;
            else error_image_double = golden_image_double - output_image_double;

            $display("Real Part:  BE[%0d] Output[%0d] = %h (%.3f), Golden:%h (%.3f),  Absolute Error:%.3f", i, j, output_real_half, output_real_double, golden_real_half, golden_real_double, error_real_double);
            $display("Image Part:  BE[%0d] Output[%0d] = %h (%.3f), Golden:%h (%.3f),  Absolute Error:%.3f", i, j, output_image_half, output_image_double, golden_image_half, golden_image_double, error_image_double);
        end
      end

      //foreach(outputs_real[i]) $display("Output[%0d] = %h", i, outputs_real[i]);
      $finish;
    end
  ////////////************** Monitor **************/////////////

  generate
  for(g=0 ; g<be_parallelism ; g=g+1)
  begin : GET_OUTPUTS_REAL
      always @ (posedge Clock) begin
        if (Dn_Serial_Vld_A) begin
            outputs_real[g].push_back(Dn_Serial_Dat_A[(data_width*g + data_width-1) : (data_width*g)]);
        end
      end
    end
  endgenerate

  generate
  for(g=0 ; g<be_parallelism ; g=g+1)
  begin : GET_OUTPUTS_IMAGE
      always @ (posedge Clock) begin
        if (Dn_Serial_Vld_B) begin
            outputs_image[g].push_back(Dn_Serial_Dat_B[(data_width*g + data_width-1) : (data_width*g)]);
        end
      end
    end
  endgenerate
  
endmodule

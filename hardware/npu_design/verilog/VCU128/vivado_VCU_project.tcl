# Create Project
create_project -force Butterfly_Acc_VCU ./Butterfly_Acc_VCU -part xcvu37p-fsvh2892-1-e

# Add design source
add_files -norecurse -scan_for_includes {./design/crossbar_comb.v ./design/mux_comb.v ./design/hbm_dummy_read.v ./design/hbm_dummy_write.v ./design/sif_mult_half_fp.v ./design/butterfly_s2p_opt.v ./design/sif_sub_half_fp.v ./design/butterfly_engine_opt.v ./design/butterfly_engine_opt_control.v ./design/ram_simple_dual.v ./design/reg_timing.v ./design/hmb_custom_write.v ./design/sif_add_complex.v ./design/butterfly_indx_generator.v ./design/tree_fanout_opt.v ./design/butterfly_p2s.v ./design/acc_top.v ./design/hbm.v ./design/hbm_control.v ./design/sif_addsub_half_fp.v ./design/sif_add_complex_half_fp.v ./design/pingpong_ram_2d.v ./design/weight_buffer.v ./design/butterfly_processor.v ./design/hbm_auto_read.v ./design/bu_read_addr_generator_opt.v ./design/bu_write_addr_generator_opt.v ./design/sif_fifo.v ./design/hmb_auto_write.v ./design/hbm_top.v ./design/data_pack.v ./design/sif_add_bfe_fixedp.v ./design/butterfly_engine_opt_comp.v ./design/butterfly_engine_opt_top.v ./design/butterfly_s2p.v ./design/tree_fanout_double.v ./design/butterfly_unit_opt.v ./design/ram_2d.v ./design/butterfly_p2s_opt.v}
import_files -norecurse {./design/crossbar_comb.v ./design/mux_comb.v ./design/hbm_dummy_read.v ./design/hbm_dummy_write.v ./design/sif_mult_half_fp.v ./design/butterfly_s2p_opt.v ./design/sif_sub_half_fp.v ./design/butterfly_engine_opt.v ./design/butterfly_engine_opt_control.v ./design/ram_simple_dual.v ./design/reg_timing.v ./design/hmb_custom_write.v ./design/sif_add_complex.v ./design/butterfly_indx_generator.v ./design/tree_fanout_opt.v ./design/butterfly_p2s.v ./design/acc_top.v ./design/hbm.v ./design/hbm_control.v ./design/sif_addsub_half_fp.v ./design/sif_add_complex_half_fp.v ./design/pingpong_ram_2d.v ./design/weight_buffer.v ./design/butterfly_processor.v ./design/hbm_auto_read.v ./design/bu_read_addr_generator_opt.v ./design/bu_write_addr_generator_opt.v ./design/sif_fifo.v ./design/hmb_auto_write.v ./design/hbm_top.v ./design/data_pack.v ./design/sif_add_bfe_fixedp.v ./design/butterfly_engine_opt_comp.v ./design/butterfly_engine_opt_top.v ./design/butterfly_s2p.v ./design/tree_fanout_double.v ./design/butterfly_unit_opt.v ./design/ram_2d.v ./design/butterfly_p2s_opt.v}
update_compile_order -fileset sources_1
set_property SOURCE_SET sources_1 [get_filesets sim_1]


# Add constraints file
add_files -fileset constrs_1 -norecurse ./constraint/acc_top.xdc



# Create IPs

# RAM
create_ip -name blk_mem_gen -vendor xilinx.com -library ip -version 8.4 -module_name ram_naive_16_1024_1r1w
set_property -dict [list CONFIG.Component_Name {ram_naive_16_1024_1r1w} CONFIG.Memory_Type {Simple_Dual_Port_RAM} CONFIG.Assume_Synchronous_Clk {true} CONFIG.Write_Depth_A {1024} CONFIG.Operating_Mode_A {READ_FIRST} CONFIG.Operating_Mode_B {READ_FIRST} CONFIG.Enable_B {Use_ENB_Pin} CONFIG.Register_PortA_Output_of_Memory_Primitives {false} CONFIG.Register_PortB_Output_of_Memory_Primitives {false} CONFIG.Port_B_Clock {100} CONFIG.Port_B_Enable_Rate {100}] [get_ips ram_naive_16_1024_1r1w]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/ram_naive_16_1024_1r1w/ram_naive_16_1024_1r1w.xci]
update_compile_order -fileset sources_1
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/ram_naive_16_1024_1r1w/ram_naive_16_1024_1r1w.xci]
catch { config_ip_cache -export [get_ips -all ram_naive_16_1024_1r1w] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/ram_naive_16_1024_1r1w/ram_naive_16_1024_1r1w.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/ram_naive_16_1024_1r1w/ram_naive_16_1024_1r1w.xci]
launch_runs -jobs 72 ram_naive_16_1024_1r1w_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/ram_naive_16_1024_1r1w/ram_naive_16_1024_1r1w.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

create_ip -name blk_mem_gen -vendor xilinx.com -library ip -version 8.4 -module_name ram_naive_16_512_1r1w
set_property -dict [list CONFIG.Component_Name {ram_naive_16_512_1r1w} CONFIG.Memory_Type {Simple_Dual_Port_RAM} CONFIG.Assume_Synchronous_Clk {true} CONFIG.Write_Depth_A {512} CONFIG.Operating_Mode_A {READ_FIRST} CONFIG.Operating_Mode_B {READ_FIRST} CONFIG.Enable_B {Use_ENB_Pin} CONFIG.Register_PortA_Output_of_Memory_Primitives {false} CONFIG.Register_PortB_Output_of_Memory_Primitives {false} CONFIG.Port_B_Clock {100} CONFIG.Port_B_Enable_Rate {100}] [get_ips ram_naive_16_512_1r1w]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/ram_naive_16_512_1r1w/ram_naive_16_512_1r1w.xci]
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/ram_naive_16_512_1r1w/ram_naive_16_512_1r1w.xci]
catch { config_ip_cache -export [get_ips -all ram_naive_16_512_1r1w] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/ram_naive_16_512_1r1w/ram_naive_16_512_1r1w.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/ram_naive_16_512_1r1w/ram_naive_16_512_1r1w.xci]
launch_runs -jobs 72 ram_naive_16_512_1r1w_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/ram_naive_16_512_1r1w/ram_naive_16_512_1r1w.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

# FIFO
create_ip -name fifo_generator -vendor xilinx.com -library ip -version 13.2 -module_name naive_fifo_16b_1024
set_property -dict [list CONFIG.Component_Name {naive_fifo_16b_1024} CONFIG.Fifo_Implementation {Common_Clock_Block_RAM} CONFIG.Input_Data_Width {16} CONFIG.Output_Data_Width {16} CONFIG.Use_Embedded_Registers {false} CONFIG.Reset_Type {Asynchronous_Reset} CONFIG.Full_Flags_Reset_Value {1} CONFIG.Almost_Full_Flag {true} CONFIG.Full_Threshold_Assert_Value {1022} CONFIG.Full_Threshold_Negate_Value {1021} CONFIG.Enable_Safety_Circuit {true}] [get_ips naive_fifo_16b_1024]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/naive_fifo_16b_1024/naive_fifo_16b_1024.xci]
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/naive_fifo_16b_1024/naive_fifo_16b_1024.xci]
catch { config_ip_cache -export [get_ips -all naive_fifo_16b_1024] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/naive_fifo_16b_1024/naive_fifo_16b_1024.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/naive_fifo_16b_1024/naive_fifo_16b_1024.xci]
launch_runs -jobs 72 naive_fifo_16b_1024_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/naive_fifo_16b_1024/naive_fifo_16b_1024.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

# Arithmic Half_Precision IPs

create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name half_fp_add
# BE40
set_property -dict [list CONFIG.Component_Name {half_fp_add} CONFIG.Add_Sub_Value {Add} CONFIG.A_Precision_Type {Half} CONFIG.C_Mult_Usage {No_Usage} CONFIG.C_A_Exponent_Width {5} CONFIG.C_A_Fraction_Width {11} CONFIG.Result_Precision_Type {Half} CONFIG.C_Result_Exponent_Width {5} CONFIG.C_Result_Fraction_Width {11} CONFIG.C_Accum_Msb {32} CONFIG.C_Accum_Lsb {-24} CONFIG.C_Accum_Input_Msb {15} CONFIG.C_Latency {12} CONFIG.C_Rate {1}] [get_ips half_fp_add]
# BE120
# set_property -dict [list CONFIG.Component_Name {half_fp_add} CONFIG.Add_Sub_Value {Add} CONFIG.A_Precision_Type {Half} CONFIG.C_Mult_Usage {Medium_Usage} CONFIG.C_A_Exponent_Width {5} CONFIG.C_A_Fraction_Width {11} CONFIG.Result_Precision_Type {Half} CONFIG.C_Result_Exponent_Width {5} CONFIG.C_Result_Fraction_Width {11} CONFIG.C_Accum_Msb {32} CONFIG.C_Accum_Lsb {-24} CONFIG.C_Accum_Input_Msb {15} CONFIG.C_Latency {12} CONFIG.C_Rate {1}] [get_ips half_fp_add]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_add/half_fp_add.xci]
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_add/half_fp_add.xci]
catch { config_ip_cache -export [get_ips -all half_fp_add] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_add/half_fp_add.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_add/half_fp_add.xci]
launch_runs -jobs 72 half_fp_add_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_add/half_fp_add.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name half_fp_addsub
set_property -dict [list CONFIG.Component_Name {half_fp_addsub} CONFIG.A_Precision_Type {Half} CONFIG.C_Mult_Usage {No_Usage} CONFIG.C_A_Exponent_Width {5} CONFIG.C_A_Fraction_Width {11} CONFIG.Result_Precision_Type {Half} CONFIG.C_Result_Exponent_Width {5} CONFIG.C_Result_Fraction_Width {11} CONFIG.C_Accum_Msb {32} CONFIG.C_Accum_Lsb {-24} CONFIG.C_Accum_Input_Msb {15} CONFIG.C_Latency {9} CONFIG.C_Rate {1}] [get_ips half_fp_addsub]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_addsub/half_fp_addsub.xci]
update_compile_order -fileset sources_1
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_addsub/half_fp_addsub.xci]
catch { config_ip_cache -export [get_ips -all half_fp_addsub] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_addsub/half_fp_addsub.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_addsub/half_fp_addsub.xci]
launch_runs -jobs 72 half_fp_addsub_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_addsub/half_fp_addsub.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name half_fp_mult
set_property -dict [list CONFIG.Component_Name {half_fp_mult} CONFIG.Operation_Type {Multiply} CONFIG.A_Precision_Type {Half} CONFIG.C_A_Exponent_Width {5} CONFIG.C_A_Fraction_Width {11} CONFIG.Result_Precision_Type {Half} CONFIG.C_Result_Exponent_Width {5} CONFIG.C_Result_Fraction_Width {11} CONFIG.C_Accum_Msb {32} CONFIG.C_Accum_Lsb {-24} CONFIG.C_Accum_Input_Msb {15} CONFIG.C_Mult_Usage {Full_Usage} CONFIG.C_Latency {7} CONFIG.C_Rate {1}] [get_ips half_fp_mult]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_mult/half_fp_mult.xci]
update_compile_order -fileset sources_1
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_mult/half_fp_mult.xci]
catch { config_ip_cache -export [get_ips -all half_fp_mult] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_mult/half_fp_mult.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_mult/half_fp_mult.xci]
launch_runs -jobs 72 half_fp_mult_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_mult/half_fp_mult.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name half_fp_sub
# BE40
# set_property -dict [list CONFIG.Component_Name {half_fp_sub} CONFIG.Add_Sub_Value {Subtract} CONFIG.A_Precision_Type {Half} CONFIG.C_Mult_Usage {No_Usage} CONFIG.C_A_Exponent_Width {5} CONFIG.C_A_Fraction_Width {11} CONFIG.Result_Precision_Type {Half} CONFIG.C_Result_Exponent_Width {5} CONFIG.C_Result_Fraction_Width {11} CONFIG.C_Accum_Msb {32} CONFIG.C_Accum_Lsb {-24} CONFIG.C_Accum_Input_Msb {15} CONFIG.C_Latency {12} CONFIG.C_Rate {1}] [get_ips half_fp_sub]
# BE120
# set_property -dict [list CONFIG.Component_Name {half_fp_sub} CONFIG.Add_Sub_Value {Subtract} CONFIG.A_Precision_Type {Half} CONFIG.C_Mult_Usage {Medium_Usage} CONFIG.C_A_Exponent_Width {5} CONFIG.C_A_Fraction_Width {11} CONFIG.Result_Precision_Type {Half} CONFIG.C_Result_Exponent_Width {5} CONFIG.C_Result_Fraction_Width {11} CONFIG.C_Accum_Msb {32} CONFIG.C_Accum_Lsb {-24} CONFIG.C_Accum_Input_Msb {15} CONFIG.C_Latency {12} CONFIG.C_Rate {1}] [get_ips half_fp_sub]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_sub/half_fp_sub.xci]
update_compile_order -fileset sources_1
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_sub/half_fp_sub.xci]
catch { config_ip_cache -export [get_ips -all half_fp_sub] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_sub/half_fp_sub.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_sub/half_fp_sub.xci]
launch_runs -jobs 72 half_fp_sub_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_sub/half_fp_sub.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet


create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name half_fp_div
set_property -dict [list CONFIG.Component_Name {half_fp_div} CONFIG.Operation_Type {Divide} CONFIG.A_Precision_Type {Half} CONFIG.C_A_Exponent_Width {5} CONFIG.C_A_Fraction_Width {11} CONFIG.Result_Precision_Type {Half} CONFIG.C_Result_Exponent_Width {5} CONFIG.C_Result_Fraction_Width {11} CONFIG.C_Accum_Msb {32} CONFIG.C_Accum_Lsb {-24} CONFIG.C_Accum_Input_Msb {15} CONFIG.C_Mult_Usage {No_Usage} CONFIG.C_Latency {16} CONFIG.C_Rate {1}] [get_ips half_fp_div]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_div/half_fp_div.xci]
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_div/half_fp_div.xci]
catch { config_ip_cache -export [get_ips -all half_fp_div] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_div/half_fp_div.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_div/half_fp_div.xci]
launch_runs -jobs 72 half_fp_div_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_div/half_fp_div.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name half_fp_recip_square
set_property -dict [list CONFIG.Component_Name {half_fp_recip_square} CONFIG.Operation_Type {Rec_Square_Root} CONFIG.A_Precision_Type {Half} CONFIG.C_A_Exponent_Width {5} CONFIG.C_A_Fraction_Width {11} CONFIG.Result_Precision_Type {Half} CONFIG.C_Result_Exponent_Width {5} CONFIG.C_Result_Fraction_Width {11} CONFIG.C_Accum_Msb {32} CONFIG.C_Accum_Lsb {-24} CONFIG.C_Accum_Input_Msb {15} CONFIG.C_Mult_Usage {No_Usage} CONFIG.C_BRAM_Usage {Full_Usage} CONFIG.C_Latency {5} CONFIG.C_Rate {1}] [get_ips half_fp_recip_square]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_recip_square/half_fp_recip_square.xci]
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_recip_square/half_fp_recip_square.xci]
catch { config_ip_cache -export [get_ips -all half_fp_recip_square] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_recip_square/half_fp_recip_square.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_recip_square/half_fp_recip_square.xci]
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/half_fp_recip_square/half_fp_recip_square.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
write_project_tcl -help


# Register Slice for timing

create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axi_register_slice_read_256d_33a
set_property -dict [list CONFIG.READ_WRITE_MODE {READ_ONLY} CONFIG.ADDR_WIDTH {33} CONFIG.DATA_WIDTH {256} CONFIG.ID_WIDTH {6} CONFIG.USE_AUTOPIPELINING {1} CONFIG.NUM_WRITE_OUTSTANDING {0} CONFIG.Component_Name {axi_register_slice_read_256d_33a}] [get_ips axi_register_slice_read_256d_33a]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/axi_register_slice_read_256d_33a/axi_register_slice_read_256d_33a.xci]
update_compile_order -fileset sources_1
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/axi_register_slice_read_256d_33a/axi_register_slice_read_256d_33a.xci]
catch { config_ip_cache -export [get_ips -all axi_register_slice_read_256d_33a] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/axi_register_slice_read_256d_33a/axi_register_slice_read_256d_33a.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/axi_register_slice_read_256d_33a/axi_register_slice_read_256d_33a.xci]
launch_runs -jobs 72 axi_register_slice_read_256d_33a_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/axi_register_slice_read_256d_33a/axi_register_slice_read_256d_33a.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axi_register_slice_write_256d_33a
set_property -dict [list CONFIG.READ_WRITE_MODE {WRITE_ONLY} CONFIG.ADDR_WIDTH {33} CONFIG.DATA_WIDTH {256} CONFIG.ID_WIDTH {6} CONFIG.USE_AUTOPIPELINING {1} CONFIG.NUM_READ_OUTSTANDING {0} CONFIG.Component_Name {axi_register_slice_write_256d_33a}] [get_ips axi_register_slice_write_256d_33a]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/axi_register_slice_write_256d_33a/axi_register_slice_write_256d_33a.xci]
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/axi_register_slice_write_256d_33a/axi_register_slice_write_256d_33a.xci]
catch { config_ip_cache -export [get_ips -all axi_register_slice_write_256d_33a] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/axi_register_slice_write_256d_33a/axi_register_slice_write_256d_33a.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/axi_register_slice_write_256d_33a/axi_register_slice_write_256d_33a.xci]
launch_runs -jobs 72 axi_register_slice_write_256d_33a_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/axi_register_slice_write_256d_33a/axi_register_slice_write_256d_33a.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

# HBM

create_ip -name clk_wiz -vendor xilinx.com -library ip -version 6.0 -module_name clk_wiz_0
set_property -dict [list CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT3_USED {true} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {200.000} CONFIG.MMCM_CLKOUT1_DIVIDE {6} CONFIG.MMCM_CLKOUT2_DIVIDE {12} CONFIG.NUM_OUT_CLKS {3} CONFIG.CLKOUT2_JITTER {102.086} CONFIG.CLKOUT2_PHASE_ERROR {87.180} CONFIG.CLKOUT3_JITTER {115.831} CONFIG.CLKOUT3_PHASE_ERROR {87.180}] [get_ips clk_wiz_0]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/clk_wiz_0/clk_wiz_0.xci]
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/clk_wiz_0/clk_wiz_0.xci]
catch { config_ip_cache -export [get_ips -all clk_wiz_0] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/clk_wiz_0/clk_wiz_0.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/clk_wiz_0/clk_wiz_0.xci]
launch_runs -jobs 72 clk_wiz_0_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/clk_wiz_0/clk_wiz_0.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet


create_ip -name hbm -vendor xilinx.com -library ip -version 1.0 -module_name hbm_0
set_property -dict [list CONFIG.USER_HBM_HEX_FBDIV_CLKOUTDIV_0 {0x00000804} CONFIG.USER_HBM_TCK_0 {400} CONFIG.USER_HBM_TCK_0_PERIOD {2.5} CONFIG.USER_tRC_0 {0x13} CONFIG.USER_tRAS_0 {0xE} CONFIG.USER_tRCDRD_0 {0x6} CONFIG.USER_tRCDWR_0 {0x4} CONFIG.USER_tRRDL_0 {0x2} CONFIG.USER_tRRDS_0 {0x2} CONFIG.USER_tFAW_0 {0x8} CONFIG.USER_tRP_0 {0x6} CONFIG.USER_tWR_0 {0x7} CONFIG.USER_tWTRL_0 {0x5} CONFIG.USER_tXP_0 {0x4} CONFIG.USER_tRFC_0 {0x68} CONFIG.USER_tRFCSB_0 {0x40} CONFIG.USER_tRREFD_0 {0x4} CONFIG.USER_HBM_REF_OUT_CLK_0 {800} CONFIG.USER_MC0_REF_CMD_PERIOD {0x0618} CONFIG.USER_MC1_REF_CMD_PERIOD {0x0618} CONFIG.USER_MC2_REF_CMD_PERIOD {0x0618} CONFIG.USER_MC3_REF_CMD_PERIOD {0x0618} CONFIG.USER_MC4_REF_CMD_PERIOD {0x0618} CONFIG.USER_MC5_REF_CMD_PERIOD {0x0618} CONFIG.USER_MC6_REF_CMD_PERIOD {0x0618} CONFIG.USER_MC7_REF_CMD_PERIOD {0x0618} CONFIG.USER_DFI_CLK0_FREQ {200.000}] [get_ips hbm_0]
generate_target {instantiation_template} [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/hbm_0/hbm_0.xci]
generate_target all [get_files  ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/hbm_0/hbm_0.xci]
catch { config_ip_cache -export [get_ips -all hbm_0] }
export_ip_user_files -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/hbm_0/hbm_0.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/hbm_0/hbm_0.xci]
launch_runs -jobs 72 hbm_0_synth_1
export_simulation -of_objects [get_files ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.srcs/sources_1/ip/hbm_0/hbm_0.xci] -directory ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/sim_scripts -ip_user_files_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files -ipstatic_source_dir ./Butterfly_Acc_VCU/Butterfly_Acc_VCU.ip_user_files/ipstatic -lib_map_path [list {modelsim=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/modelsim} {questa=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/questa} {ies=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/ies} {xcelium=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/xcelium} {vcs=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/vcs} {riviera=./Butterfly_Acc_VCU/Butterfly_Acc_VCU.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

# Set top function for synthsis and implementation

# Disabling source management mode.  This is to allow the top design properties to be set without GUI intervention.
set_property source_mgmt_mode None [current_project]
set_property top bfly_acc_top [current_fileset]
# Re-enabling previously disabled source management mode.
set_property source_mgmt_mode All [current_project]
update_compile_order -fileset sources_1

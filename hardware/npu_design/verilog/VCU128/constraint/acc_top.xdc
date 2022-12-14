create_clock -period 5.000 -name sys_clk -waveform {0.000 2.500} [get_ports -filter { NAME =~  "*sys_clk*" && DIRECTION == "IN" }]
create_clock -period 10.000 -name ddr0_clk -waveform {0.000 5.000} [get_ports -filter { NAME =~  "*ddr0_clk*" && DIRECTION == "IN" }]
connect_debug_port dbg_hub/clk [get_nets u_hbm_0/u_hbm_top/inst_clk_wiz_0/inst/clk_out3]
set_property C_USER_SCAN_CHAIN 1 [get_debug_cores dbg_hub]

set_multicycle_path 6 -setup -from [get_clocks ddr0_clk] -to [get_clocks sys_clk]
set_multicycle_path 5 -hold -end -from [get_clocks ddr0_clk] -to [get_clocks sys_clk]
set_multicycle_path 6 -setup -from [get_cells u_hbm_0/u_hbm_control/is_fft_r*] -to [get_cells u_data_pack/is_pad_r*]
set_multicycle_path 5 -hold -end -from [get_cells u_hbm_0/u_hbm_control/is_fft_r*] -to [get_cells u_data_pack/is_pad_r*]
set_multicycle_path 6 -setup -from [get_cells u_hbm_0/u_hbm_control/is_bypass_p2s_r*] -to [get_cells u_bp/is_bypass_p2s_timing*]
set_multicycle_path 5 -hold -end -from [get_cells u_hbm_0/u_hbm_control/is_bypass_p2s_r*] -to [get_cells u_bp/is_bypass_p2s_timing*]
set_multicycle_path 6 -setup -from [get_cells u_hbm_0/u_hbm_control/is_fft_r*] -to [get_cells u_bp/is_fft_timing*]
set_multicycle_path 5 -hold -end -from [get_cells u_hbm_0/u_hbm_control/is_fft_r*] -to [get_cells u_bp/is_fft_timing*]
set_multicycle_path 6 -setup -from [get_cells u_hbm_0/u_hbm_control/length_r*] -to [get_cells u_bp/lengths_timing*]
set_multicycle_path 5 -hold -end -from [get_cells u_hbm_0/u_hbm_control/length_r*] -to [get_cells u_bp/lengths_timing*]

set_multicycle_path 3 -setup -from [get_cells u_bp/lengths_timing*] -to [get_cells u_bp/*u_butterfly_engine_opt/u_butterfly_engine_opt_control/length_r*]
set_multicycle_path 2 -hold -from [get_cells u_bp/lengths_timing*] -to [get_cells u_bp/*u_butterfly_engine_opt/u_butterfly_engine_opt_control/length_r*]
set_multicycle_path 3 -setup -from [get_cells u_bp/*u_butterfly_engine_opt/u_butterfly_engine_opt_control/length_r*] -to [get_cells u_bp/*u_butterfly_engine_opt/u_butterfly_engine_opt_control/u_butterfly_s2p/length_r*]
set_multicycle_path 2 -hold -from [get_cells u_bp/*u_butterfly_engine_opt/u_butterfly_engine_opt_control/length_r*] -to [get_cells u_bp/*u_butterfly_engine_opt/u_butterfly_engine_opt_control/u_butterfly_s2p/length_r*]

set_multicycle_path 3 -setup -from [get_cells u_bp/lengths_timing*] -to [get_cells u_bp/*u_butterfly_engine_opt/*u_butterfly_engine_opt_comp/length_r*]
set_multicycle_path 2 -hold -from [get_cells u_bp/lengths_timing*] -to [get_cells u_bp/*u_butterfly_engine_opt/*u_butterfly_engine_opt_comp/length_r*]
set_multicycle_path 3 -setup -from [get_cells u_bp/*u_butterfly_engine_opt/*u_butterfly_engine_opt_comp/length_r*] -to [get_cells u_bp/*u_butterfly_engine_opt/*u_butterfly_engine_opt_comp/u_butterfly_s2p/length_r*]
set_multicycle_path 2 -hold -from [get_cells u_bp/*u_butterfly_engine_opt/*u_butterfly_engine_opt_comp/length_r*] -to [get_cells u_bp/*u_butterfly_engine_opt/*u_butterfly_engine_opt_comp/u_butterfly_s2p/length_r*]

set_multicycle_path 3 -setup -from [get_cells u_bp/is_bypass_p2s_timing*] -to [get_cells u_bp/*u_butterfly_engine_opt/u_butterfly_engine_opt_control/is_bypass_p2s_r*]
set_multicycle_path 2 -hold -from [get_cells u_bp/is_bypass_p2s_timing*] -to [get_cells u_bp/*u_butterfly_engine_opt/u_butterfly_engine_opt_control/is_bypass_p2s_r*]
set_multicycle_path 3 -setup -from [get_cells u_bp/is_fft_timing*] -to [get_cells u_bp/*u_butterfly_engine_opt/*u_butterfly_engine_opt_comp/is_fft_r*]
set_multicycle_path 2 -hold -from [get_cells u_bp/is_fft_timing*] -to [get_cells u_bp/*u_butterfly_engine_opt/*u_butterfly_engine_opt_comp/is_fft_r*]

from multi_head_engine import Multi_Head_Engine
from bfly_accelerator import Butterfly_Accelerator
import argparse
import logging

logger = logging.getLogger()


def simulation(args):
    if args.debug:
        logger.setLevel(logging.DEBUG) 
    else:
        logger.setLevel(logging.INFO)
     
    logging.info("####################Start######################")
    # Setting the configurations of the design
    if args.version == "base":
        num_layer = 12
        hidden_dim = 768
        ffn_inner_dim = 3072
    elif args.version == "large":
        num_layer = 24
        hidden_dim = 1024
        ffn_inner_dim = 4096
    else:
        raise NotImplementedError("Not supported version.")

    if args.offchip_mem == "hbm":
        indata_dram_bw=2048
        coef_dram_bw=256
        outdata_dram_bw=2048
    elif args.offchip_mem == "ddr3":
        indata_dram_bw=64
        coef_dram_bw=64
        outdata_dram_bw=128
    else:
        raise NotImplementedError("Not supported off-chip memory.")

    if args.fpga_board == "zcu128":
        parallesm_bu=4
        parallesm_be=128 if args.parallesm_be == 0 else args.parallesm_be
    elif args.fpga_board == "zynq7045":
        parallesm_bu=4
        parallesm_be=32 if args.parallesm_be == 0 else args.parallesm_be
    else:
        raise NotImplementedError("Not supported FPGA board")

    # Instantiate Design
    design = Butterfly_Accelerator(args.head_dim, hidden_dim, args.num_len, ffn_inner_dim, 
                                    parallesm_bu=parallesm_bu, parallesm_be=parallesm_be,
                                    indata_dram_bw=indata_dram_bw, coef_dram_bw=coef_dram_bw, outdata_dram_bw=outdata_dram_bw)

    # Run Fourier Layer
    design.run_fft(complex_input=False, complex_output=True) # 1st dimension FFT
    design.run_fft(complex_input=True, complex_output=False) # 2nd dimension FFT
    
    # Run Butterfly Layer for FFN
    design.run_bfly(design.num_len, design.hidden_dim, design.ffn_inner_dim) 
    design.run_bfly(design.num_len, design.ffn_inner_dim, design.hidden_dim)
    network_run_cost = num_layer * design.run_cycles
    ms_per_clock = (1.0/args.frequency/1000) / args.efficiency
    print ("The overall latecy is:", network_run_cost*ms_per_clock) 
    logging.info("####################Finish######################")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--head_dim", default=32, type=int, help="Dimension per head")
    # parser.add_argument("--hidden_dim", default=1024, type=int, help="Hidden dimension")
    parser.add_argument("--num_len", default=512, type=int, help="Lengh of input sequence")
    parser.add_argument("--frequency", default=200, type=int, help="The frequency of the design")
    parser.add_argument("--version", default="base", type=str, help="base of large")
    # parser.add_argument("--ffn_inner_dim", default=4096, type=int, help="Inner dimension of FFN")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--efficiency", default=0.85, type=float, help="The hardware implementation efficiency")
    parser.add_argument("--fpga_board", default="zcu128", type=str, help="The FPGA board used for implementation")
    # parser.add_argument("--parallesm_bu", default=4, type=int, help="parallesm of butterfly unit per butterfly engine")
    parser.add_argument("--parallesm_be", default=0, type=int, help="parallesm of butterfly engine in the whole design")
    parser.add_argument("--offchip_mem", default="hbm", type=str, help="The off-chip memory installed in the design")

    args = parser.parse_args()

    simulation(args)
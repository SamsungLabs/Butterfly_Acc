import enum
from multi_head_engine import Multi_Head_Engine
from bfly_accelerator import Butterfly_Accelerator
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger()

BROWN = "#AD8C97"
BROWN_DARKER = "#7d3a46"
GREEN = "#2FC1D3"
BLUE = "#076FA1"
GREY = "#C7C9CB"
GREY_DARKER = "#5C5B5D"
RED = "#E3120B"


def collect_data(args):
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

    num_lens = [128, 256, 512, 1024]
    percentage_list = []
    parallesm_be = 128
    bw = 2048
    # Get the bandwidhth
    for i, num_len in enumerate(num_lens):
        percentage = []
        # Instantiate Design
        design = Butterfly_Accelerator(args.head_dim, hidden_dim, num_len, ffn_inner_dim, 
                                        parallesm_bu=4, parallesm_be=parallesm_be,
                                        indata_dram_bw=bw, coef_dram_bw=bw, outdata_dram_bw=bw)

        # Run Fourier Layer
        fft_time = 0
        fft_time += design.run_fft(complex_input=False, complex_output=True) # 1st dimension FFT
        fft_time += design.run_fft(complex_input=True, complex_output=False) # 2nd dimension FFT
        fft_time *= num_layer

        # Run Butterfly Layer for FFN
        bfly_time = 0
        bfly_time += design.run_bfly(design.num_len, design.hidden_dim, design.ffn_inner_dim) 
        bfly_time +=  design.run_bfly(design.num_len, design.ffn_inner_dim, design.hidden_dim)
        bfly_time *= num_layer

        network_run_cost = num_layer * design.run_cycles
        ms_per_clock = (1/args.frequency/1000)

        fft_time *= ms_per_clock
        bfly_time *= ms_per_clock
        whole_time = network_run_cost*ms_per_clock / args.efficiency

        percentage.append(fft_time/whole_time)
        percentage.append(bfly_time/whole_time)
        percentage.append(1 - fft_time/whole_time - bfly_time/whole_time)
        percentage_list.append(percentage)
    print (percentage_list)
    return np.array(percentage_list)

def draw_figs(percentage_list):
    fig, ax = plt.subplots(figsize=(8, 6))
    COLORS = [BLUE, GREEN, BROWN_DARKER]
    bw_list = [6.4, 12.8, 25.6, 51.2, 102.4, 204.8]
    barWidth = 0.85
    r = ["128", "256", "512", "1024"]
    print (percentage_list[:, 0])
    ax.bar(r, percentage_list[:, 0], color=BLUE, edgecolor='black', width=barWidth)
    
    ax.bar(r, percentage_list[:, 1], bottom=percentage_list[:, 0], color=GREEN, edgecolor='black', width=barWidth)

    ax.bar(r, percentage_list[:, 2], bottom=percentage_list[:, 0] + percentage_list[:, 1], color=BROWN_DARKER, edgecolor='black', width=barWidth)

    # ax.set_ylim([0, 50])
    plt.xlabel('Models', fontsize = 13)
    plt.ylabel('Percentage (%)', fontsize = 13)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.grid()
    plt.show()
    fig.savefig("Latency Breakdown.pdf", bbox_inches='tight')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--head_dim", default=32, type=int, help="Dimension per head")
    # parser.add_argument("--hidden_dim", default=1024, type=int, help="Hidden dimension")
    parser.add_argument("--frequency", default=200, type=int, help="The frequency of the design")
    parser.add_argument("--version", default="large", type=str, help="base of large") # Set large as default for bandwdith analysis
    # parser.add_argument("--ffn_inner_dim", default=4096, type=int, help="Inner dimension of FFN")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--efficiency", default=0.85, type=float, help="The hardware implementation efficiency")


    args = parser.parse_args()

    percentage_list = collect_data(args)
    draw_figs(percentage_list)
    
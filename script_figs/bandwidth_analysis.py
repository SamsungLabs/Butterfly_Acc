import enum
import sys
from turtle import title 
sys.path.insert(0,'..')
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

num_lens = [128, 1024, 4096]

def collect_data(args, num_len):
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

    parallesms_be = [16, 32, 64, 96, 128]
    bw_list = [64, 128, 256, 512, 1024, 2048]
    latency_list = []
    # Get the bandwidhth
    for i, parallesm_be in enumerate(parallesms_be):
        latency = []
        for j, bw in enumerate(bw_list):
            # Instantiate Design
            design = Butterfly_Accelerator(args.head_dim, hidden_dim, num_len, ffn_inner_dim, 
                                            parallesm_bu=4, parallesm_be=parallesm_be,
                                            indata_dram_bw=bw, coef_dram_bw=bw, outdata_dram_bw=bw)

            # Run Fourier Layer
            design.run_fft(complex_input=False, complex_output=True) # 1st dimension FFT
            design.run_fft(complex_input=True, complex_output=False) # 2nd dimension FFT
            
            # Run Butterfly Layer for FFN
            design.run_bfly(design.num_len, design.hidden_dim, design.ffn_inner_dim) 
            design.run_bfly(design.num_len, design.ffn_inner_dim, design.hidden_dim)
            network_run_cost = num_layer * design.run_cycles
            ms_per_clock = (1/args.frequency/1000) / args.efficiency
            latency.append(network_run_cost*ms_per_clock)
        print ("Parallesm of Butterfly Engines:", parallesm_be," with latency:", latency)
        latency_list.append(latency)
    return latency_list

def draw_figs(latency_list):
    fig, axs = plt.subplots(nrows=1, ncols=len(num_lens), figsize=(18, 2.0)) #, gridspec_kw={'height_ratios': [0.3], 'width_ratios': [4, 4, 4]}
    # fig, ax = plt.subplots(figsize=(8, 6))
    COLORS = [BLUE, GREEN, BROWN_DARKER, GREY, RED]
    label_name = ["16 BEs", "32 BEs", "64 BEs", "96 BEs", "128 BEs"]
    bw_list = [6.4, 12.8, 25.6, 51.2, 102.4, 204.8]
    tick_font_size = 9
    label_font_size = 13
    title_name = []
    title_num = ['(a) ', '(b) ', '(c) ']
    for i, num_len in enumerate(num_lens):
        title_name.append(title_num[i] + "FABNet-Large with " + str(num_len) +  " input sequence")
    for i, latencys in enumerate(latency_list): 
        for latency, color, label in zip(latencys, COLORS, label_name):
            axs[i].plot(bw_list, latency, color=color, lw=1.5, label=label)
            axs[i].scatter(bw_list, latency, fc=color, s=50, lw=1., ec="white", zorder=12)
            axs[i].set_xlabel('Bandwidth (GB/s)', fontsize = label_font_size)
            axs[i].set_ylabel('Latency (ms)', fontsize = label_font_size)
            axs[i].tick_params(axis='both', labelsize=tick_font_size)
            axs[i].grid()
            axs[i].set_title(title_name[i],y=0, pad=-53, fontsize = label_font_size)#, fontweight="bold"
        if i==0: 
            axs[0].legend(ncol=len(label_name), loc='lower left', bbox_to_anchor=(0.0, 1.0), prop={'size': label_font_size-1})
    # ax.set_ylim([0, 50])
    # plt.xlabel('Bitwidth (GB/s)', fontsize = 13)
    # plt.ylabel('Latency (ms)', fontsize = 13)
    # plt.xticks(fontsize = 13)
    # plt.yticks(fontsize = 13)
    # plt.grid()
    plt.show()
    fig.savefig("Bandwidth_v2.pdf", bbox_inches='tight')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--head_dim", default=32, type=int, help="Dimension per head")
    # parser.add_argument("--hidden_dim", default=1024, type=int, help="Hidden dimension")
    # parser.add_argument("--num_len", default=128, type=int, help="Lengh of input sequence")
    parser.add_argument("--frequency", default=200, type=int, help="The frequency of the design")
    parser.add_argument("--version", default="large", type=str, help="base of large") # Set large as default for bandwdith analysis
    # parser.add_argument("--ffn_inner_dim", default=4096, type=int, help="Inner dimension of FFN")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--efficiency", default=0.85, type=float, help="The hardware implementation efficiency")

    args = parser.parse_args()
    latency_list = []
    for num_len in num_lens:
        latency = collect_data(args, num_len)
        latency_list.append(latency)
    draw_figs(latency_list)
    

import enum
import sys
from turtle import title 
sys.path.insert(0,'../hardware/npu_design/simulator/')
from multi_head_engine import Multi_Head_Engine
from bfly_accelerator import Butterfly_Accelerator
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

logger = logging.getLogger()

BROWN = "#AD8C97"
BROWN_DARKER = "#7d3a46"
GREEN = "#2FC1D3"
BLUE = "#076FA1"
GREY = "#C7C9CB"
GREY_DARKER = "#5C5B5D"
RED = "#E3120B"


file_names = ["acc_explore_text.log", "acc_explore_image.log"]
title_names = ["(a) LRA-Text", "(b) LRA-Image"]

def collect_data(args, file_name):
    with open(file_name) as f:
        f = f.readlines()
    acc = []
    for line in f:
        if '\'test\'' in line:
            words = line.split()
            test_acc = float(words[9][:-1])
            acc.append(test_acc*100)
    return acc

def draw_figs(args, latency_list):
    fig, axs = plt.subplots(nrows=1, ncols=len(file_names), figsize=(8, 2.0)) #, gridspec_kw={'height_ratios': [0.3], 'width_ratios': [4, 4, 4]}
    # fig, ax = plt.subplots(figsize=(8, 6))
    COLORS = [BROWN_DARKER, GREY_DARKER]
    label_name = ["16 BEs", "32 BEs", "64 BEs", "96 BEs", "128 BEs"]
    bw_list = [0, 1, 2, 3, 4, 5, 6]
    tick_font_size = 9
    label_font_size = 10
    # title_name = []
    # title_num = ['(a) ', '(b) ', '(c) ']
    # for i, num_len in enumerate(file_names):
    #     title_name.append(title_num[i] + "FABNet-Large with " + str(num_len) +  " input sequence")
    for i, latencys in enumerate(latency_list): 
        color = COLORS[i]
        # for latency, color, label in zip(latencys, COLORS, label_name):
        #axs[i].plot(bw_list, latencys, color=color, lw=1.5, label=label)
        axs[i].plot(bw_list, latencys, color=color, lw=1.5)
        axs[i].scatter(bw_list, latencys, fc=color, s=50, lw=1., ec="white", zorder=12)
        axs[i].set_xlabel('# of Compressed Layers', fontsize = label_font_size)
        if i == 0:
            axs[i].set_ylabel('Accuracy (%)', fontsize = label_font_size)
        axs[i].tick_params(axis='both', labelsize=tick_font_size)
        axs[i].grid()
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs[i].set_title(title_names[i],y=0, pad=-46, fontsize = label_font_size)#, fontweight="bold"
        # if i==0: 
        #     axs[0].legend(ncol=len(label_name), loc='lower left', bbox_to_anchor=(0.0, 1.0), prop={'size': label_font_size-1})
    # ax.set_ylim([0, 50])
    # plt.xlabel('Bitwidth (GB/s)', fontsize = 13)
    # plt.ylabel('Latency (ms)', fontsize = 13)
    # plt.xticks(fontsize = 13)
    # plt.yticks(fontsize = 13)
    # plt.grid()
    fig.tight_layout()
    plt.show()
    fig.savefig("Acc_Explore.pdf", bbox_inches='tight')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--head_dim", default=32, type=int, help="Dimension per head")
    # parser.add_argument("--hidden_dim", default=1024, type=int, help="Hidden dimension")
    # parser.add_argument("--num_len", default=128, type=int, help="Lengh of input sequence")
    #parser.add_argument("--file_name", default="acc_explore_text.log", type=str, help="The accuracy result file to read") # Set large as default for bandwdith analysis
    # parser.add_argument("--ffn_inner_dim", default=4096, type=int, help="Inner dimension of FFN")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    latency_list = []
    for file_name in file_names:
        latency_list.append(collect_data(args, file_name))
    # Waiting results for other datasets
    #for i in range(len(latency_list[1])):
    #    latency_list[1][i] = 70.9
    print (latency_list)
    draw_figs(args, latency_list)
    

from cProfile import label
import enum
import sys
sys.path.insert(0,'..')
from multi_head_engine import Multi_Head_Engine
from bfly_accelerator import Butterfly_Accelerator
from att_accelerator import Att_Accelerator
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes
from matplotlib.ticker import MaxNLocator
logger = logging.getLogger()

BROWN = "#AD8C97"
BROWN_DARKER = "#7d3a46"
BLUE = "#076FA1"
GREY = "#C7C9CB"
GREY_DARKER = "#5C5B5D"
RED = "#E3120B"

PINK = "#FD4659"
PINK_DARK = "#4A0100"
GREEN = "#017374"
GREEN_DARK = "#1F3B4D"

Energy_cloud = {
  "Over_GPU": [128,  138,   62,    40,    30,   149,   122,   55,   36,  30],
  "Over_CPU": [4478, 6868, 6101, 5913, 6007, 4727, 7634, 7013, 6853, 6083]
}


COLORS = ["#B7C9E2", "#5684AE", GREEN, GREEN_DARK]


def draw_figs(args):

    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    # subfigs = GridSpec(1,len(ops), wspace=0.6)
    barWidth = 0.7
    fig, ax = plt.subplots(figsize=(9, 2.5))
    # fig = plt.figure(figsize=(9, 3))
    energy_over_gpu = Energy_cloud["Over_GPU"]
    energy_over_cpu = Energy_cloud["Over_CPU"]
    label_name = ["FABNet-Base\n128", "FABNet-Large\n128", "FABNet-Base\n256", "FABNet-Large\n256",
                  "FABNet-Base\n512", "FABNet-Large\n512", "FABNet-Base\n768", "FABNet-Large\n768",
                  "FABNet-Base\n1024", "FABNet-Large\n1024"]

    dataset_name = ["Listops_transformer","Listops_fnet", "Text_transformer", "Text_fnet", 
                    "Retrieval_transformer", "Retrieval_fnet", "Image_transformer", "Image_fnet", "Path_transformer", "Path_fnet"]
    xs = []
    x_init = 1
    for i in range(len(label_name)):
        xs.append(x_init - 0.45)
        xs.append(x_init + 0.45)
        x_init +=2.2
    ys = []
    for i in range(len(energy_over_gpu)):
        ys.append(energy_over_gpu[i])
        ys.append(energy_over_cpu[i])
    xtick_name = []
    for x in label_name:
        xtick_name.append(x)

    xtick = []
    x_init = 1.38
    for i in range(len(label_name)):
        xtick.append(x_init)
        x_init+=2.2

    font_size = 8
    bars = ax.bar(xs[0], ys[0], color=COLORS[0], edgecolor='black', width=barWidth, align='edge', label="Over RTX 2080 Ti")
    bars = ax.bar(xs[0], ys[0], color="none", edgecolor='white', width=barWidth, align='edge', hatch="//")
    bars = ax.bar(xs[0], ys[0], color="none", edgecolor='black', width=barWidth, align='edge')
    ax.bar_label(bars, fontsize=font_size)
    bars = ax.bar(xs[1], ys[1], color=COLORS[1], edgecolor='black', width=barWidth, align='edge', label="Over Intel Golden 6154 CPU")
    bars = ax.bar(xs[1], ys[1], color="none", edgecolor='white', width=barWidth, align='edge', hatch="//")
    bars = ax.bar(xs[1], ys[1], color="none", edgecolor='black', width=barWidth, align='edge')
    ax.bar_label(bars, fontsize=font_size)
    bars  = ax.bar(xs[2:], ys[2:], color=[COLORS[0], COLORS[1]] * int(len(ys)/2-1), edgecolor='black', width=barWidth, align='edge') # Draw colow
    bars  = ax.bar(xs[2:], ys[2:], color="none", edgecolor="white", width=barWidth, align='edge', hatch="//") # Draw hatch
    bars  = ax.bar(xs[2:], ys[2:], color="none", edgecolor="black", width=barWidth, align='edge') # Draw edge
    # bars  = ax.bar(xs[2:], ys[2:], color="none", edgecolor='black', width=barWidth, align='edge')
    ax.bar_label(bars, fontsize=font_size)

    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_name)
    ax.tick_params('x', labelrotation=15, labelsize=font_size)
    ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': font_size+1})
    ax.set_ylabel("Reduction in FLOPs")
    ax.grid(axis='y', which='major', ls='-')
    ax.set_axisbelow(True)
    ax.set_yscale('log', basey=2)
    ax.set_ylabel('Energy Eff.(GOPs/Watt)', fontsize=font_size+1)
    ax.set_ylim(top=12000)
    plt.rcParams['hatch.linewidth'] = 3
    

    # axs[1].bar(dataset_name[0], params_compress_rate[0], color=COLORS[2], edgecolor='black', width=barWidth, align='edge', label="Over Transformer")
    # axs[1].bar(dataset_name[1], params_compress_rate[1], color=COLORS[3], edgecolor='black', width=barWidth, align='edge', label="Over FNet")
    # axs[1].bar(dataset_name[2:], params_compress_rate[2:], color=[COLORS[2], COLORS[3]]*4, edgecolor='black', width=barWidth, align='edge')
    # axs[1].set_xticklabels(label_name)
    # axs[1].tick_params('x', labelrotation=15)
    # axs[1].legend(ncol=2, loc='lower left', bbox_to_anchor=(-0.08, 1.0))
    # axs[1].set_ylabel("Reduction in Model size")
    # axs[1].grid(axis='y', which='major', ls='-')
    # axs[1].set_axisbelow(True)
    
    # if args.version == "base": colors = COLORS[:2]
    # else: colors = COLORS[2:4]
    # for i, subfig in enumerate(subfigs):
    #     att_fft_time = []
    #     ffn_bfly_time = []
    #     model_name = ["Bert_"+args.version+"_"+str(num_lens[i]), "FABNet_"+args.version+"_"+str(num_lens[i])]
    #     att_fft_time.append(att_times[i])
    #     att_fft_time.append(fft_times[i])
    #     ffn_bfly_time.append(ffn_times[i])
    #     ffn_bfly_time.append(bfly_times[i]) 

    #     y1_break_high = math.ceil(fft_times[i] + bfly_times[i])
    #     y2_break_low = math.floor(att_times[i])
    #     y2_break_high = math.ceil(att_times[i]) + math.ceil(bfly_times[i])
    #     y3_break_low = math.floor(att_times[i] + ffn_times[i]) - math.ceil(bfly_times[i]/2)
    #     y3_break_high = math.ceil(att_times[i] + ffn_times[i])

    #     bax = brokenaxes(ylims=((0.0, y1_break_high), (y2_break_low, y2_break_high), 
    #                       (y3_break_low, y3_break_high)), hspace=.15, subplot_spec=subfig)
    #     bax.bar(model_name, att_fft_time, color=colors[0], edgecolor='black', width=barWidth, tick_label=model_name, align='edge', label='Attentiion/FFT')
    #     bax.bar(model_name, ffn_bfly_time, bottom=att_fft_time, color=colors[1], edgecolor='black', width=barWidth, tick_label=model_name, align='edge', label='FFN')
    #     bax.tick_params(labelsize=9)
    #     bax.tick_params('x', labelrotation=15)
    #     bax.grid(axis='y', which='major', ls='-')
    #     bax.set_axisbelow(True)
    #     if i==0: 
    #         bax.set_ylabel('Latency (ms)')
    #         bax.legend(ncol=2, loc='lower left', bbox_to_anchor=(0.0, 1.0))
    #     print ("Sequance-", num_lens[i], " att speedup:", att_times[i]/fft_times[i], " ffn speedup", ffn_times[i]/bfly_times[i])
    #     # for ax in bax.axs:
    #     #   ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # print (len(bax.axs[0].xaxis))
    # plt.xlabel('Models', fontsize = 13)
    # plt.ylabel('Latency (ms)', fontsize = 13)
    # plt.xticks(fontsize = 13)
    # plt.yticks(fontsize = 13)
    # plt.yscale('log')
    # plt.grid()
    plt.show()
    fig.savefig("CPU_GPU_Energy_Cloud.pdf", bbox_inches='tight')

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

    draw_figs(args)
    

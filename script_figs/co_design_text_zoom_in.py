from ast import Num
from cProfile import label
import sys
sys.path.insert(0,'..')
import enum
from multi_head_engine import Multi_Head_Engine
from bfly_accelerator import Butterfly_Accelerator
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

#### Download co-design1.log and co-design2.log from https://drive.google.com/drive/folders/1twc8KPlZKbCqgIywBq8r-Xm9lk83EXB4
#### Put them under the same folder "figs"

logger = logging.getLogger()

read_names = ["co_design1_text.log", "co_design2_text.log"] 

BROWN = "#AD8C97"
BROWN_DARKER = "#7d3a46"
GREEN = "#2FC1D3"
BLUE = "#1E488F"
GREY = "#C7C9CB"
GREY_DARKER = "#5C5B5D"
RED = "#E3120B"
num_lens = [256, 512, 1024, 2048, 4096, 8192]
hidden_dims = [512, 768, 1024, 1600]

FFN_intern_ratio = [1.0, 2.0, 3.0]
Hidden_dim = [64, 128, 256, 512, 1024]
Num_layer = [1, 2]
num_len = 4000
bw = 2048
num_be = [128, 64, 32, 16]

# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    print (len(costs))
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        prev_point_index = next_point_index
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        #print (next_point_index)
        #print (prev_point_index)
        #print (next_point_index//100)
        #print (prev_point_index//100)
        #print ((prev_point_index//100) != (next_point_index//100))
        if ((next_point_index//1000) != (prev_point_index//1000)):
            print ("%d / %d"%(next_point_index, len(costs)))
    print (next_point_index, len(costs))
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def collect_data(args, file_name):
    with open(file_name) as f:
        f = f.readlines()
    acc = []
    for line in f:
        if '\'test\'' in line:
            words = line.split()
            test_acc = float(words[9][:-1])
            for i in range(len(num_be)):
                acc.append(test_acc*100)
    lat = []
    ms_per_clock = (1/args.frequency/1000) / args.efficiency
    for ratio in FFN_intern_ratio:
        for dim in Hidden_dim:
            for layer in Num_layer:
                if (layer == 1) and (file_name == "co_design2_text.log"): continue
                for be in num_be:
                    bu = 4*128/be
                    print ("##############simulator using %d be and %d bu######################"%(be, bu))
                    ############ Get Latency breakdown from butterfly accelerator ####################
                    bfly_design = Butterfly_Accelerator(args.head_dim, dim, num_len, int(dim*ratio), 
                                                    parallesm_bu=4*128/be, parallesm_be=be,
                                                    indata_dram_bw=bw, coef_dram_bw=bw, outdata_dram_bw=bw)
                    # Run Fourier Layer
                    bfly_design.run_fft(complex_input=False, complex_output=True) # 1st dimension FFT
                    bfly_design.run_fft(complex_input=True, complex_output=False) # 2nd dimension FFT

                    # Run Butterfly Layer for FFN
                    bfly_design.run_bfly(bfly_design.num_len, bfly_design.hidden_dim, bfly_design.ffn_inner_dim) 
                    bfly_design.run_bfly(bfly_design.num_len, bfly_design.ffn_inner_dim, bfly_design.hidden_dim)
                    lat.append(bfly_design.run_cycles * layer * ms_per_clock)
    return acc, lat

def draw_figs(acc, lat):

    COLORS = [GREY_DARKER, RED, BROWN_DARKER, BLUE]


    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(lat, acc, fc=BLUE, s=100, ec="white", alpha=0.9)
    ax.set_ylim([62.75, 63.05])
    ax.set_xlim([0, 5])
    np_cost = np.array(list(zip([-x for x in acc],lat)))
    print (np.shape(np_cost))
    ptf_mask = is_pareto_efficient(np_cost)
    paratos = np.array(list(zip(acc,lat)))[ptf_mask]
    paratos = paratos[paratos[:,1].argsort()]
    non_paratos = np_cost[np.invert(ptf_mask)]
    print (paratos[:,0])
    print (paratos[:,1])
    ax.plot(paratos[:,1], paratos[:,0], color=BROWN_DARKER, lw=1.5, label="Pareto Frontier")
    ax.scatter(paratos[:,1], paratos[:,0], fc=BROWN_DARKER, s=70, ec="white", alpha=0.9)
    # for i, att_percents in enumerate(att_percent_list):
    #     ffn_percents = ffn_percent_list[i]
    #     ax.plot(num_lens, [x*100 for x in ffn_percents], color=COLORS[i], lw=3, label=models[i])
    #     ax.scatter(num_lens, [x*100 for x in ffn_percents], fc=COLORS[i], s=100, lw=1.5, ec="white", zorder=12)
    # ax.legend(ncol=1, loc='lower left', fontsize = 13) #, bbox_to_anchor=(0.0, 1.0)
    # ax.set_ylim([0, 100])
    ax.set_xlabel('Latency (ms)', fontsize = 11, labelpad=16)
    ax.set_ylabel('Accuracy', fontsize = 11, labelpad=32)
    ax.axhline(y=63.7, color=BLUE, linestyle='-')
    ax.axhline(y=62.7, color=BROWN_DARKER, linestyle='-')
    # plt.xticks([256, 1024, 2048, 4096, 8192], fontsize = 13)
    # plt.yticks(fontsize = 13)
    # import matplotlib.ticker as mtick
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    # ax2 = ax.twinx()
    # ax2.set_ylabel('FLOP Percent (Attention)', fontsize = 13)
    # ax2.set_yticks([0, 80, 60, 40, 20, 100])
    # ax2.set_ylim([0, 100])
    # ax2.set_ylim(ax2.get_ylim()[::-1])
    ax.tick_params('y', labelsize = 10)
    # ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid()
    plt.show()
    fig.savefig("Codesign_Text_Zoomin.pdf", bbox_inches='tight')

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
    acc_list = []
    lat_list = []
    for i in range(len(read_names)):
        acc, lat = collect_data(args, read_names[i])
        acc_list = acc_list + acc
        lat_list = lat_list + lat
    print ((acc_list))
    print ((lat_list))
        # sys.exit()
    draw_figs(acc_list, lat_list)
    # draw_figs(att_percent_list, ffn_percent_list)
    

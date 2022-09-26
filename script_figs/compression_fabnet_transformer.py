import enum
import sys
sys.path.insert(0,'../hardware/npu_design/simulator/')
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


Transformer_configs = {"Listops": {"num_layer":2, "att_dim":64, "hid_dim":128, "num_len": 2000},
                      "Text": {"num_layer":2, "att_dim":64, "hid_dim":128, "num_len": 4000},
                      "Retrieval": {"num_layer":2, "att_dim":64, "hid_dim":128, "num_len": 4000},
                      "Image": {"num_layer":2, "att_dim":64, "hid_dim":128, "num_len": 1024},
                      "Path": {"num_layer":2, "att_dim":64, "hid_dim":128, "num_len": 1024},
                      }

FNet_configs = {"Listops": {"num_layer":2, "att_dim":64, "hid_dim":128, "num_len": 2000},
                      "Text": {"num_layer":2, "att_dim":64, "hid_dim":128, "num_len": 4000},
                      "Retrieval": {"num_layer":2, "att_dim":1024, "hid_dim":256, "num_len": 4000},
                      "Image": {"num_layer":2, "att_dim":64, "hid_dim":128, "num_len": 1024},
                      "Path": {"num_layer":2, "att_dim":64, "hid_dim":128, "num_len": 1024},
                      }

FABNet_configs = {"Listops": {"num_layer":2, "att_dim":128, "hid_dim":256, "num_len": 2000}, # acc:0.378
                      "Text": {"num_layer":2, "att_dim":64, "hid_dim":256, "num_len": 4000}, # acc: 0.6295
                      "Retrieval": {"num_layer":2, "att_dim":1024, "hid_dim":256, "num_len": 4000}, # acc: 0.799
                      "Image": {"num_layer":1, "att_dim":64, "hid_dim":128, "num_len": 1024},# 0.399
                      "Path": {"num_layer":6, "att_dim":64, "hid_dim":128, "num_len": 1024}, # 0.674
                      }

num_lens = [128, 256, 512, 1024]
COLORS = [PINK, PINK_DARK, GREEN, GREEN_DARK]

def ceil_power2(x):
    shift_bits = (x-1).bit_length() - 1
    return 2 << shift_bits

def collect_data(args):
    params = []
    ops = []
    # Collect transformer ops and params
    params_transformer = []
    ops_transformer = []
    params.append(params_transformer)
    ops.append(ops_transformer)

    for dataset, config in Transformer_configs.items():
        num_layer = config["num_layer"]
        att_dim = config["att_dim"]
        hid_dim = config["hid_dim"]
        num_len = config["num_len"]
        op_count = 0
        param_count = 0
        op_count += num_layer * (2 * num_len * num_len * att_dim) # Attention Matrix Multiplication
        op_count += num_layer * (4 * num_len * att_dim * att_dim + 2 * num_len * att_dim * (hid_dim)) # Linear Layers
        
        # Linear Layers, no parameters are reuqired by attention matrix mult (Q*K and S*V)
        param_count +=  num_layer * (4 * att_dim * att_dim + 2 * att_dim * (hid_dim))  
        
        ops_transformer.append(op_count)
        params_transformer.append(param_count)
    print (ops_transformer)
    print (params_transformer)

    # Collect fnet ops and params
    params_fnet = []
    ops_fnet = []
    params.append(params_fnet)
    ops.append(ops_fnet)

    for dataset, config in FNet_configs.items():
        num_layer = config["num_layer"]
        att_dim = ceil_power2(config["att_dim"])
        hid_dim = ceil_power2(config["hid_dim"])
        num_len = config["num_len"]
        op_count = 0
        param_count = 0
        op_count += num_layer * (num_len * (2 * att_dim) * (att_dim.bit_length()-1)) # first 1D FFT: len * 2N * stage
        expanded_num_len = ceil_power2(num_len)
        op_count += num_layer * (att_dim * (2 * expanded_num_len) * (expanded_num_len.bit_length()-1)) # second 1D FFT

        op_count += num_layer * (2 * num_len * att_dim * (hid_dim)) # Linear Layers
        print ("fnet:", num_layer * (2 * num_len * att_dim * (hid_dim)))
        # Linear Layers, no parameters are reuqired by attention matrix mult (Q*K and S*V)
        param_count +=  num_layer * (2 * att_dim * (hid_dim))
        
        ops_fnet.append(op_count)
        params_fnet.append(param_count)

    # Collect fabnet ops and params
    params_fabnet = []
    ops_fabnet = []
    params.append(params_fabnet)
    ops.append(ops_fabnet)

    for dataset, config in FABNet_configs.items():
        num_layer = config["num_layer"]
        att_dim = ceil_power2(config["att_dim"])
        hid_dim = ceil_power2(config["hid_dim"])
        num_len = config["num_len"]
        op_count = 0
        param_count = 0
        op_count += num_layer * (num_len * (2 * att_dim) * (att_dim.bit_length()-1)) # first 1D FFT: len * 2N * stage
        expanded_num_len = ceil_power2(num_len)
        op_count += num_layer * (att_dim * (2 * expanded_num_len) * (expanded_num_len.bit_length()-1)) # second 1D FFT

        op_count += num_layer * (num_len * (2 * att_dim) * (att_dim.bit_length()-1) * (hid_dim/att_dim)) # first Butterfly Linear Layers
        op_count += num_layer * (num_len * (2 * hid_dim) * (hid_dim.bit_length()-1) ) # second Butterfly Linear Layers
        print ("fabnet:", num_layer * (num_len * (2 * att_dim) * (att_dim.bit_length()-1) * (hid_dim/att_dim)) + num_layer * (num_len * (2 * hid_dim) * (hid_dim.bit_length()-1)))
        # Butterfly Linear Layers
        param_count +=  num_layer * (2 * att_dim)  # first 1D FFT, 2N
        param_count +=  num_layer * (2 * expanded_num_len)  # second 1D FFT, N
        param_count +=  num_layer * (4 * att_dim)  # first 1D FFT, 2N, according to page 6 of https://arxiv.org/pdf/1903.05895.pdf
        param_count +=  num_layer * (4 * hid_dim)  # first 1D FFT, 2N
        
        ops_fabnet.append(op_count)
        params_fabnet.append(param_count)
    for i, v in enumerate(ops_fabnet):
        print ("ops:", ops_transformer[i], " vs ", ops_fnet[i], " vs ", ops_fabnet[i], " compression rate:", ops_transformer[i]/ops_fabnet[i], " and ", ops_fnet[i]/ops_fabnet[i])
        print ("param:", params_transformer[i], " vs ", params_fnet[i], params_fabnet[i], " compression rate:", params_transformer[i]/params_fabnet[i], " and ", params_fnet[i]/params_fabnet[i])
    return ops, params

def draw_figs(ops, params):
    ops_transformer, ops_fnet, ops_fabnet = ops[0], ops[1], ops[2]
    params_transformer, params_fnet, params_fabnet = params[0], params[1], params[2]
    ops_compress_rate = []
    params_compress_rate = []

    for i, v in enumerate(ops_fabnet):
        ops_compress_rate.append(ops_transformer[i]/ops_fabnet[i])
        ops_compress_rate.append(ops_fnet[i]/ops_fabnet[i])
        params_compress_rate.append(params_transformer[i]/params_fabnet[i])
        params_compress_rate.append(params_fnet[i]/params_fabnet[i])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    # subfigs = GridSpec(1,len(ops), wspace=0.6)
    barWidth = 0.7
    # fig, ax = plt.subplots(figsize=(8, 6))
    # fig = plt.figure(figsize=(9, 3))
    dataset_name = ["Listops_transformer","Listops_fnet", "Text_transformer", "Text_fnet", 
                    "Retrieval_transformer", "Retrieval_fnet", "Image_transformer", "Image_fnet", "Path_transformer", "Path_fnet"]
    label_name = ["", "Listops", "", "Text", "", "Retrieval", "", "Image", "", "Path"]
    axs[0].bar(dataset_name[0], ops_compress_rate[0], color=COLORS[0], edgecolor='black', width=barWidth, align='edge', label="Over Transformer")
    axs[0].bar(dataset_name[1], ops_compress_rate[1], color=COLORS[1], edgecolor='black', width=barWidth, align='edge', label="Over FNet")
    axs[0].bar(dataset_name[2:], ops_compress_rate[2:], color=[COLORS[0], COLORS[1]]*4, edgecolor='black', width=barWidth, align='edge')
    axs[0].set_xticklabels(label_name)
    axs[0].tick_params('x', labelrotation=15)
    axs[0].legend(ncol=2, loc='lower left', bbox_to_anchor=(-0.08, 1.0))
    axs[0].set_ylabel("Reduction in FLOPs")
    axs[0].grid(axis='y', which='major', ls='-')
    axs[0].set_axisbelow(True)

    axs[1].bar(dataset_name[0], params_compress_rate[0], color=COLORS[2], edgecolor='black', width=barWidth, align='edge', label="Over Transformer")
    axs[1].bar(dataset_name[1], params_compress_rate[1], color=COLORS[3], edgecolor='black', width=barWidth, align='edge', label="Over FNet")
    axs[1].bar(dataset_name[2:], params_compress_rate[2:], color=[COLORS[2], COLORS[3]]*4, edgecolor='black', width=barWidth, align='edge')
    axs[1].set_xticklabels(label_name)
    axs[1].tick_params('x', labelrotation=15)
    axs[1].legend(ncol=2, loc='lower left', bbox_to_anchor=(-0.08, 1.0))
    axs[1].set_ylabel("Reduction in Model size")
    axs[1].grid(axis='y', which='major', ls='-')
    axs[1].set_axisbelow(True)
    
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
    fig.savefig("Compression_Rate.pdf", bbox_inches='tight')

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

    ops, params = collect_data(args)
    draw_figs(ops, params)
    

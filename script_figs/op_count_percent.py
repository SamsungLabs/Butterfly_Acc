from cProfile import label
import sys
sys.path.insert(0,'../hardware/npu_design/simulator/')
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
BLUE = "#1E488F"
GREY = "#C7C9CB"
GREY_DARKER = "#5C5B5D"
RED = "#E3120B"
num_lens = [256, 512, 1024, 2048, 4096, 8192]
hidden_dims = [512, 768, 1024, 1600]

def collect_data(args):
    att_percent_list = []
    ffn_percent_list = []
    for hidden_dim in hidden_dims:
        att_percents = []
        ffn_percents = []
        for num_len in num_lens:
            att_ops = 2 * num_len * num_len * hidden_dim
            ffn_ops =  4 * num_len * hidden_dim * hidden_dim + 2 * num_len * hidden_dim * (4 * hidden_dim)
            total_ops = att_ops + ffn_ops
            att_percent = att_ops / total_ops
            ffn_percent = ffn_ops / total_ops
            att_percents.append(att_percent)
            ffn_percents.append(ffn_percent)
        att_percent_list.append(att_percents)
        ffn_percent_list.append(ffn_percents)
    return att_percent_list, ffn_percent_list

def draw_figs(att_percent_list, ffn_percent_list):
    barWidth = 0.5
    r = [str(l) for l in num_lens]
    models = ["T5-Small", "BERT-Base", "RoBERTa-Large", "GPT2-XL"]
    x = []
    for model in models:
        for i in r:
            x.append(model + i)
    COLORS = [GREY_DARKER, RED, BROWN_DARKER, BLUE]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, att_percents in enumerate(att_percent_list):
        ffn_percents = ffn_percent_list[i]
        ax.plot(num_lens, [x*100 for x in ffn_percents], color=COLORS[i], lw=3, label=models[i])
        ax.scatter(num_lens, [x*100 for x in ffn_percents], fc=COLORS[i], s=100, lw=1.5, ec="white", zorder=12)
    ax.legend(ncol=1, loc='lower left', fontsize = 13) #, bbox_to_anchor=(0.0, 1.0)
    ax.set_ylim([0, 100])
    plt.xlabel('Length of Sequence', fontsize = 13)
    plt.ylabel('FLOP Percent (Linear Layers)', fontsize = 13)
    plt.xticks([256, 1024, 2048, 4096, 8192], fontsize = 13)
    plt.yticks(fontsize = 13)
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2 = ax.twinx()
    ax2.set_ylabel('FLOP Percent (Attention)', fontsize = 13)
    # ax2.set_yticks([0, 80, 60, 40, 20, 100])
    ax2.set_ylim([0, 100])
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.tick_params('y', labelsize = 13)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid()
    plt.show()
    fig.savefig("Op_Count.pdf", bbox_inches='tight')

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

    att_percent_list, ffn_percent_list = collect_data(args)
    draw_figs(att_percent_list, ffn_percent_list)
    

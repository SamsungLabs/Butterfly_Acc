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
GREEN = "#2FC1D3"
BLUE = "#076FA1"
GREY = "#C7C9CB"
GREY_DARKER = "#5C5B5D"
RED = "#E3120B"

num_lens = [128, 256, 512, 1024]
COLORS = [BLUE, GREEN, GREY_DARKER, BROWN]

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
    bw = 2048
    ms_per_clock = (1/args.frequency/1000) / args.efficiency
    fft_times_opt = []
    bfly_times_opt = []
    fft_times_unopt = []
    bfly_times_unopt = []
    att_times = []
    ffn_times = []

    for i, num_len in enumerate(num_lens):
        ############ Running FABNet on butterfly accelerator, Both SW Opt + HW Opt ####################
        bfly_design = Butterfly_Accelerator(args.head_dim, hidden_dim, num_len, ffn_inner_dim, 
                                        parallesm_bu=4, parallesm_be=128,
                                        indata_dram_bw=bw, coef_dram_bw=bw, outdata_dram_bw=bw)
        # Run Fourier Layer
        fft_time_opt = 0
        fft_time_opt += bfly_design.run_fft(complex_input=False, complex_output=True) # 1st dimension FFT
        fft_time_opt += bfly_design.run_fft(complex_input=True, complex_output=False) # 2nd dimension FFT
        fft_time_opt *= num_layer
        fft_time_opt *= ms_per_clock
        fft_times_opt.append(fft_time_opt)

        # Run Butterfly Layer for FFN
        bfly_time_opt = 0
        bfly_time_opt += bfly_design.run_bfly(bfly_design.num_len, bfly_design.hidden_dim, bfly_design.ffn_inner_dim) 
        bfly_time_opt +=  bfly_design.run_bfly(bfly_design.num_len, bfly_design.ffn_inner_dim, bfly_design.hidden_dim)
        bfly_time_opt *= num_layer
        bfly_time_opt *= ms_per_clock
        bfly_times_opt.append(bfly_time_opt)

        ############ Running FABNet on attention accelerator, Only SW Opt ####################
        # Each head engine is 64*4, totally 8 head engines, so the parallelsm is 64 * 4 * 8 = 2048
        att_design = Att_Accelerator(args.head_dim, hidden_dim, num_len, ffn_inner_dim, dram_bw=bw, pv_lt=64, p_head = 8) 
        # Run Fourier Layer. Since the hardware does not support, the time equivalent to run standard FC.
        fft_time_unopt = 0
        fft_time_unopt += att_design.run_fft(complex_input=False, complex_output=True)
        fft_time_unopt += att_design.run_fft(complex_input=True, complex_output=False)
        fft_time_unopt *= num_layer
        fft_time_unopt *= ms_per_clock
        fft_times_unopt.append(fft_time_unopt)

        # Run Butterfly Layer for FFN. Since the hardware does not support, the time equivalent to run standard FC.
        bfly_time_unopt = 0
        bfly_time_unopt += att_design.run_fc1()
        bfly_time_unopt += att_design.run_fc2()
        bfly_time_unopt *= num_layer
        bfly_time_unopt *= ms_per_clock
        bfly_times_unopt.append(bfly_time_unopt)

        ############# Runnig Bert on attention accelerator, Baseline ###################
        # Each head engine is 64*4, totally 8 head engines, so the parallelsm is 64 * 4 * 8 = 2048
        att_design = Att_Accelerator(args.head_dim, hidden_dim, num_len, ffn_inner_dim, dram_bw=bw, pv_lt=64, p_head = 8) 
        # Run Attention
        att_time = 0
        att_time += att_design.run_att()
        att_time += att_design.run_lp()
        att_time *= num_layer
        att_time *= ms_per_clock
        att_times.append(att_time)

        # Run FFN
        ffn_time = 0
        ffn_time += att_design.run_fc1()
        ffn_time += att_design.run_fc2()
        ffn_time *= num_layer
        ffn_time *= ms_per_clock
        ffn_times.append(ffn_time)
    print ([sum(s) for s in zip(fft_times_opt, bfly_times_opt)])
    print ([sum(s) for s in zip(bfly_times_unopt, fft_times_unopt)])
    print ([sum(s) for s in zip(ffn_times, att_times)])
    return [fft_times_opt, bfly_times_opt, fft_times_unopt, bfly_times_unopt, att_times, ffn_times]

def draw_figs(latency, args):
    fft_times_opt, bfly_times_opt, fft_times_unopt, bfly_times_unopt, att_times, ffn_times = [lat for lat in latency]
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    subfigs = GridSpec(1,len(fft_times_opt), wspace=0.6)
    barWidth = 0.5
    # fig, ax = plt.subplots(figsize=(8, 6))
    fig = plt.figure(figsize=(9, 3))
    
    if args.version == "base": 
        colors = COLORS[:2]
        title_char = 'a'
    else: 
        colors = COLORS[2:4]
        title_char = chr(ord('a')+4)
    for i, subfig in enumerate(subfigs):
        att_fft_time = []
        ffn_bfly_time = []
        model_name = ["Bert", "FABNet_SW", "FABNet_SW&HW"]
        att_fft_time.append(att_times[i])
        att_fft_time.append(fft_times_unopt[i])
        att_fft_time.append(fft_times_opt[i])
        
        ffn_bfly_time.append(ffn_times[i])
        ffn_bfly_time.append(bfly_times_unopt[i])
        ffn_bfly_time.append(bfly_times_opt[i]) 

        y1_break_high = math.ceil(fft_times_opt[i] + bfly_times_opt[i])
        if (args.version == "large") and (i > 1): 
            y2_break_low = math.floor(fft_times_unopt[i]-0.02*fft_times_unopt[i])# - math.ceil(bfly_times_opt[i]/2)
        else:
            y2_break_low = math.floor(fft_times_unopt[i]-0.02*fft_times_unopt[i]) 
        y2_break_high = math.ceil(fft_times_unopt[i]+0.02*fft_times_unopt[i])

        y3_break_low = math.floor(min(att_times[i], fft_times_unopt[i]+bfly_times_unopt[i])-0.02*fft_times_unopt[i])
        y3_break_high = math.ceil(min(att_times[i], fft_times_unopt[i]+bfly_times_unopt[i])+0.02*fft_times_unopt[i])

        y4_break_low = math.floor(max(att_times[i], fft_times_unopt[i]+bfly_times_unopt[i])-0.02*fft_times_unopt[i])
        y4_break_high = math.ceil(max(att_times[i], fft_times_unopt[i]+bfly_times_unopt[i])+0.5)

        y5_break_low = math.floor(att_times[i] + ffn_times[i]-0.02*ffn_times[i])
        y5_break_high = math.ceil(att_times[i] + ffn_times[i]+0.5)
        '''
        y3_break_low = math.floor(att_times[i] + ffn_times[i]) - math.ceil(bfly_times_opt[i]/2)
        if (args.version == "large") and (i > 2):
            y3_break_high = math.ceil(att_times[i] + ffn_times[i]) + 1
        else:
            y3_break_high = math.ceil(att_times[i] + ffn_times[i])
        '''
        bax = brokenaxes(ylims=((0.0, y1_break_high), (y2_break_low, y2_break_high), 
                          (y3_break_low, y3_break_high), (y4_break_low, y4_break_high), (y5_break_low, y5_break_high)), hspace=.3, subplot_spec=subfig)
        #bax = brokenaxes(ylims=((0.0, y1_break_high), (y2_break_low, y5_break_high)), hspace=.15, subplot_spec=subfig)    
        bax.bar(model_name, att_fft_time, color=colors[0], edgecolor='black', width=barWidth, tick_label=model_name, align='edge', label='Attention/FFT')
        bax.bar(model_name, ffn_bfly_time, bottom=att_fft_time, color=colors[1], edgecolor='black', width=barWidth, tick_label=model_name, align='edge', label='FFN')
        bax.tick_params(labelsize=9)
        bax.tick_params('x', labelrotation=15)
        bax.grid(axis='y', which='major', ls='-')
        bax.set_axisbelow(True)
        if i==0: 
            bax.set_ylabel('Latency (ms)')
            bax.legend(ncol=2, loc='lower left', bbox_to_anchor=(0.0, 1.0))
        bax.set_title('('+ chr(ord(title_char) + i) + ') ' + args.version.capitalize()+"_"+str(num_lens[i]), y=-0.3, fontsize=11)
        #bax.text(0.5, 100, args.version.capitalize()+"_"+str(num_lens[i]), fontsize=11)#, verticalalignment='center', horizontalalignment='right')
        # print (list((bax.get_yticks())))
        # extend_yticks = list(bax.get_yticks())
        # extend_yticks.append(np.array([0.0, 2.5, 5.0, 7.5]))
        # print (extend_yticks)
        # bax.set_yticks(list(bax.get_yticks()))
        # bax.tick_params('y', length=0)
        # print ("Sequance-", num_lens[i], " sw speedup on attention:", att_times[i]/fft_times_unopt[i], " sw speedup on ffn:", ffn_times[i]/bfly_times_unopt[i])
        # print ("Sequance-", num_lens[i], " sw&hw speedup on attention:", fft_times_unopt[i]/fft_times_opt[i], " sw&hw speedup on ffn:", bfly_times_unopt[i]/bfly_times_opt[i])
        print ("Sequance-", num_lens[i], " sw speedup:", (att_times[i] + ffn_times[i])/(fft_times_unopt[i] + bfly_times_unopt[i]))
        print ("Sequance-", num_lens[i], " hw speedup:", (fft_times_unopt[i] + bfly_times_unopt[i])/(bfly_times_opt[i]+fft_times_opt[i]))
        print ("Sequance-", num_lens[i], " overall speedup:", (att_times[i] + ffn_times[i])/(bfly_times_opt[i]+fft_times_opt[i]))
        # for ax in bax.axs:
        #   ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # print (len(bax.axs[0].xaxis))
    # plt.xlabel('Models', fontsize = 13)
    # plt.ylabel('Latency (ms)', fontsize = 13)
    # plt.xticks(fontsize = 13)
    # plt.yticks(fontsize = 13)
    # plt.yscale('log')
    # plt.grid()
    plt.show()
    fig.savefig("SW_HW_Opt_"+args.version+".pdf", bbox_inches='tight')

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

    latency = collect_data(args)
    draw_figs(latency, args)
    

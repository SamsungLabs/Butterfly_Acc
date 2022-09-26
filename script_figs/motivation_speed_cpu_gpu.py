import matplotlib.pyplot as plt
import numpy as np
import os

class AttrDict(dict):
    def __init__(self, d={}):
        super(AttrDict, self).__init__()
        for k, v in d.items():
            self.__setitem__(k, v)

    def __setitem__(self, k, v):
        if isinstance(v, dict):
            v = AttrDict(v)
        super(AttrDict, self).__setitem__(k, v)

    def __getattr__(self, k):
        try:
            return self.__getitem__(k)
        except KeyError:
            raise AttributeError(k)

    __setattr__ = __setitem__

# Data from excel sheet: https://docs.google.com/spreadsheets/d/13Rl6OzdCsBzaLrmRGXpJn1BtQkbpWlIr/edit?usp=sharing&ouid=101073722577456698647&rtpof=true&sd=true
# The data sheet is generated from the script: .software/profile_cpu.sh and .software/profile_gpu.sh
# You can also run it on your machines, which may lead to slightly different percentages
lengths = [128,256,512,1024]
devices = ['gpu', 'cpu']

data = AttrDict({})
for d in devices:
    setattr(data, d, AttrDict({}))
    for l in lengths:
        getattr(data, d)[l] = AttrDict({})

# GPU        
data.gpu[128].total = 1.02
data.gpu[128].attn = 0.268
data.gpu[128].linear = 0.418
data.gpu[128].other = 0.334

data.gpu[256].total = 1.24
data.gpu[256].attn = 0.39
data.gpu[256].linear = 0.47
data.gpu[256].other = 0.38

data.gpu[512].total = 2.6
data.gpu[512].attn = 1.2
data.gpu[512].linear = 0.78
data.gpu[512].other = 0.62

data.gpu[1024].total = 1.0
data.gpu[1024].attn = 0.33
data.gpu[1024].linear = 0.33
data.gpu[1024].other = 0.33

# CPU
data.cpu[128].total = 0.319
data.cpu[128].attn = 0.036
data.cpu[128].linear = 0.226
data.cpu[128].other = 0.057

data.cpu[256].total = 0.465
data.cpu[256].attn = 0.074
data.cpu[256].linear = 0.333
data.cpu[256].other = 0.058

data.cpu[512].total = 0.692
data.cpu[512].attn = 0.204
data.cpu[512].linear = 0.427
data.cpu[512].other = 0.061

data.cpu[1024].total = 1.0
data.cpu[1024].attn = 0.33
data.cpu[1024].linear = 0.33
data.cpu[1024].other = 0.33



width = 0.5
fig, ax = plt.subplots(ncols=2,figsize=(10,4))

trunc_lengths = [str(l) for l in lengths[:3]]

all_colors = AttrDict({})
for d in devices:
    setattr(all_colors, d, AttrDict({}))
    
all_colors.gpu.attn = '#03254c'
all_colors.gpu.linear = '#2a9df4'
all_colors.gpu.other = '#d0efff'

all_colors.cpu.attn = '#FF0000'
all_colors.cpu.linear = '#FF6666'
all_colors.cpu.other = '#FFCCCC'

data.gpu.title = 'On NVIDIA GeForce RTX 2080 Ti'
data.cpu.title = 'On Intel Xeon'

for idx, d in enumerate(devices):
    ax[idx].set_ylabel('Execution Time (%)', fontsize=15)
    ax[idx].set_yticks(list(range(0,101,20)))
    ax[idx].tick_params(axis='both', labelsize=15)
    ax[idx].set_title(getattr(data, d).title, y=-0.35, fontsize=15)
    
    attn_y = np.array([])
    linear_y = np.array([])
    other_y = np.array([])
    for l in trunc_lengths:
        v = getattr(data, d)[int(l)]
        attn_y = np.append(attn_y, v.attn / v.total * 100)
        linear_y = np.append(linear_y, v.linear / v.total * 100)
        other_y = np.append(other_y ,v.other / v.total * 100)

    for line in range(20,81,20):
        ax[idx].axhline(y=line, color='black', linestyle = '--', zorder=0., label='_nolegend_')
        
    ax[idx].bar(trunc_lengths, attn_y, width=width, color=getattr(all_colors, d).attn, edgecolor='black')
    ax[idx].bar(trunc_lengths, linear_y, bottom=attn_y, width=width, color=getattr(all_colors, d).linear, edgecolor='black')
    ax[idx].bar(trunc_lengths, other_y, bottom=attn_y+linear_y, width=width, color=getattr(all_colors, d).other, edgecolor='black')

    legend = ax[idx].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=3, labels=['Attention', 'Linear', 'Others'],
        handlelength=0.75, fontsize=15, columnspacing=1.0, handletextpad=0.3)
    legend.get_frame().set_linewidth(0)

plt.tight_layout()
plt.show()
plt.savefig('motivation_fig3.pdf')

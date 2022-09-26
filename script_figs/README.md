## Description

This folder contains the scripts to draw all the figures and tables in our paper. Note that, both accuracy and power/resource may be slightly different depends on your running machine and software version.

### 1. Tables and Figures in Introduction and Motivation Parts
#### *Figure 1*

In introduction part, Figure1 shows the operation counts of attention and linear layers in different models, which is obtained by running:
```
bash gen_fig1.sh
```

#### *Figure 3*

In motivation part, Figure3 profiles the latency consumption of attention, linear and other layers in *BERT-Base* and *BERT-Large*.
To get the performance on CPU, run `bash ../software/speed/profile_cpu.sh`
To get the performance on GPU, run `bash ../software/speed/profile_gpu.sh`

We upload the latency breakdown we got in the google drive [link](https://docs.google.com/spreadsheets/d/13Rl6OzdCsBzaLrmRGXpJn1BtQkbpWlIr/edit?usp=sharing&ouid=101073722577456698647&rtpof=true&sd=true). A python code `motivation_speed_cpu_gpu.py` is used to draw Figure3 using the latency percent in the link.

You can simply draw the Figure3 using:
```
bash gen_fig3.sh
```

### 2. Figures and Tables in Experimental Parts

#### *Figure 16*

Figure16 evaluates the accuracy impact of the *FBfly*  on *LRA-text* and *LRA-image*.
You first need to get the log file by running (It takes ~100 GPU hours to get the log depending on your machine)
```
bash ../software/accuracy/code/acc_impact/acc_impact_text.sh 
bash ../software/accuracy/code/acc_impact/acc_impact_image.sh 
```
You can also download our training log files from the google drive [link](https://drive.google.com/file/d/1ldCFtxejhyXHLnd_Nuz3uUia2UeClpqf/view?usp=sharing).
Put the log file under this directory, then run
```
bash gen_fig16.sh
```

####  *Figure 17*
Figure 17 first performs a grid search to obtain the optimized *FABNet* configurations.
For instant, to perform the grid search on *LRA-image*, run
```
bash ../software/accuracy/code/grid_search/grid_search_image.sh
```
Running all all the datasets in *LRA* takes up to hunders of GPU hours.
You can also download our training log files from the google drive [link](https://drive.google.com/file/d/1uZW0Pm8H-on3ctD6fhWreDxHLc32-YI3/view?usp=sharing).

The optimized configurations are summarized in `../software/accuracy/code/grid_search/README.md`.
Using the optimzied configurations,  you can generate Figure17 by running:
```
bash gen_fig17.sh
```

#### *Table 3*
The accuracy of *Transformer* and *FNet* is generate by running:
```
bash ../software/accuracy/code/benchmark/benchmarking_regression.sh
```
The accuracy of FABNet is generate by running:
```
bash ../software/accuracy/code/fabnet_quant/half_precision_fabnet_all.sh
```
You can also check our training log files in the google drive [link](https://drive.google.com/file/d/1prZ-sse4RVPPkp9FSDdugHcnp0dTioL7/view?usp=sharing)

####  *Figure 18*
Figure 18 draws the figure of our algorithm and hardware co-design. To get the accuracy performance of all the design points in Figure18, you need to run
```
bash ../software/accuracy/code/co_design/co_design1_text.sh 
bash ../software/accuracy/code/co_design/co_design2_text.sh 
```
It takes around 10 hours on our GPU server, which might take longer on your GPU machines. Alternatively, you can download our training log files in the google drive [link](https://drive.google.com/file/d/15nysdleZeJP5dkBgIPLjzS0pPVMs0dxA/view?usp=sharing).

Put the log files in this directory, and run the following script to generate Figure 18:
```
bash gen_fig18.sh
```

####  *Figure 19*
Using the optimized hardware and software configurations, Figure 19 breakdown the effect of hardware and software optimizations.
Run the following script to generate Figure 19: (The configurations )
```
bash bash gen_fig19.sh
```

####  *Figure 19*
Using the optimized hardware and software configurations, Figure 19 breakdown the effect of hardware and software optimizations.
Run the following script to generate Figure 19: (The configurations )
```
bash bash gen_fig19.sh
```

####  *Figure 20*
We run the scripts `../hardware/npu_design/simulator/speed_benchmark_sim.sh` to get FPGA performance.
We run the scripts under `../software/speed` to get CPUS, GPUs performance on their corresponding platforms (Jeston Nano, Raspberry P, V100, TITAN Xp).
The performance we obtained can be found [here](https://docs.google.com/spreadsheets/d/1Bduw_xdDAbpbfjMe0b0LlrmwuzAZb4el/edit?usp=sharing&ouid=101073722577456698647&rtpof=true&sd=true).


####  *Table 5*
We provide a detailed document on how we obtained the performance of other SOTA accelerators in [here](https://docs.google.com/document/d/1E8n2OtfpZ5wpfxai7mScIRGTKQBE_4Xu/edit?usp=sharing&ouid=101073722577456698647&rtpof=true&sd=true).
We obtained our latency and throughput using the python simulator under `../hardware/npu_design/simulator`. The power is obtained by running experiments under `../hardware/npu_design/verilog` with `be_parallelsim=40`. You can also find our Vivado report [here](https://drive.google.com/file/d/1CbHchAz-TM0XpVYArH6Jqb-hL9I-cfhj/view?usp=sharing).

####  *Table 6 and 7*

You will need vivado to run synthsis and implementation follow the instruction in `../hardware/npu_design/verilog`. It may takes ~ 50 hours to generate all.  
Note that, to get the resource and power of *BE-120*, the DSP usage of *half_fp_sub* and *half_fp_add* IPs need to be set as *Medium_Usage*.
We also provide the post-impmentation reports of all our designs [here](https://drive.google.com/file/d/1CbHchAz-TM0XpVYArH6Jqb-hL9I-cfhj/view?usp=sharing).



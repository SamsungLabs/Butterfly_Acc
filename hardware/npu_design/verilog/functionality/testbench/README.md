## Description

This folder contains the testbench to validate the functionality of our hardware, including butterfly matrix multiplication, fast fourier transform, layer nomalizationa and shortcut addition. All these layers are run on one unifed hardware engine as mentioned in our paper.

## Test Data Generation

Before running our testbench, you will need to generate test data, including inputs, weights and golden outputs. The training script is in the subfolder `data_gen`. You will need to install the same environment of running our `software`.

### Butterfly Test Data
You can generate the test data for **butterfly matrix multiplication** by running:

```
python data_gen/torch_float16_bfly.py --length 512
```

You can specify your length from {128, 256, 512, 1024} depending on the testbench you wan to run.

To evalute butterfly with **short-cut addition** enabled, run:

```
python data_gen/torch_float16_bfly_sc.py --length 512
```

To evalute butterfly with **layer normalization and short-cut addition** enabled, run:

```
python data_gen/torch_float16_bfly_ln_sc.py --length 512
```

### FFT Test Data

```
python data_gen/torch_float16_fft.py --length 512
```
You can specify your length from {128, 256, 512, 1024} accordingly, depending on the testbench you wan to run.

To evalute fft with **short-cut addition** enabled, run:

```
python data_gen/torch_float16_fft_sc.py --length 512
```

To evalute fft with **layer normalization and short-cut addition** enabled, run:

```
python data_gen/torch_float16_fft_ln_sc.py --length 512
```


## Run Test Testbench on Vivado Behavior Simulation

To run our test tenbench, you will need to 

1. Change the file path of input, weight and output in the system verilog testbench.
1. Set the testbench you want to run as the top module in 'sim' file (Right click and select `Set As Top`).
1. Click the `Run All` button to kick off the simulation. 

### Butterfly Layer
For instance, to evaluate the Butterfly Matrix Multiplication layer, you could select the following system verilog files as top module in `sim` folder.

```
tb_butterfly_be32_length128.sv # Sequence Length = 128, Number of BE = 32
tb_butterfly_be32_length256.sv # Sequence Length = 256, Number of BE = 32
tb_butterfly_be32_length512.sv # Sequence Length = 512, Number of BE = 32
tb_butterfly_be32_length1024.sv # Sequence Length = 1024, Number of BE = 32
```
### FFT Layer
Similarly, to evaluate FFT layer, you could select `tb_fft_be\*_length\*.sv`

```
tb_fft_be32_length256.sv # Sequence Length = 256, Number of BE = 32
tb_fft_be32_length1024.sv # Sequence Length = 1024, Number of BE = 32
```

You could also create your own testbench to evaluate different lengths and numbers of BEs. In this case, the test data will need to be generated accordingly.

### Butterfly and FFT with post processing (LN and SC)

We also include the Test bench to evaluate the functionality of layer normalization (LN) and shortcut addition (SC).

For instance, you can run butterfly layer with **SC** enabled using the system verilog file.
```
tb_butterfly_sc_be32_length256.sv
```
To run butterfly layer with **LN and SC** enabled using:
```
tb_butterfly_ln_sc_be32_length256.sv
```

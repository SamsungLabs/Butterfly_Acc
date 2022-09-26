# NPU Transformer
Hardware Architecture/Accelerator for Transformer. The latest codes and updates are maintainted in this github [repo](https://github.com/os-hxfan/Butterfly_Acc).
## 1. Structure

```
.
├── README.md
├── figs                 # All the scripts to generate figures in the paper
├── hardware             # All code related to hardware implementation on an FPGA
    ├── cpu_gpu_perf     # Code for evaluate hardware performance on cpu and gpu
    ├── npu_design       # Code for evaluating our butterfly accelerator
        ├── simulator    # Our custom simulator/performance model to get latency
        ├── verilog      # Verilog code of our hardware implementation
            ├── functionality   # Design and Testbench to evaluate the functionality of hardware
            ├── VCU128          # Design and Constraints for synthesis, place&route on VCU128
            ├── Zynq7045        # Design and Constraints for synthsis, place&route on Zynq7045
├── requirements.txt        
└── software          # All code related to software experiments   
    ├── speed         # Code for speed testing
    ├── accuracy      # Code for training/inference to get accuracy
```

## 2. Environment Setup
### 2.1 Install Dependencies
We use conda to manage the required enviroment
```
conda create -n npu_transformer python=3.8 scipy
source activate npu_transformer
pip3 install -r requirements.txt
```

Then, install the butterfly operation(hand-written CUDA optimization has included in their repo).
```
git clone https://github.com/HazyResearch/butterfly.git
cd butterfly/
python setup.py install
```

To compare the speed of dense linear and butterfly linear, run:
```
cd NPU_Transformer/software
python -m src.bflylr
```

### 2.2 Prepare Dataset
Download [lra_release.gz](https://storage.googleapis.com/long-range-arena/lra_release.gz) released by LRA repo and place the unzipped folder in folder: 
```
NPU_Transformer/software/src/LRA/datasets
```
Then, run `sh create_datasets.sh` in `NPU_Transformer/software/src/LRA/datasets` and it will create train, dev, and test dataset pickle files for each task.

To run experiment, go to `NPU_Transformer/software/src/LRA/code`. For instance, running the command running listop is in `benchmarking_listop.sh`.

## 3. Artifact Evaluation

### 3.1 Functionality

#### 3.1.1. Verilog Design
All the testbenches (Butterfly matrix multiplication, FFT, Layer normalization, Shortcut addition) are in `hardware/npu_design/verilog/functionality/testbench/`. Pls read [instruction](./hardware/npu_design/verilog/functionality/testbench/README.md) before running, where you need to generate test data first.

#### 3.1.2. Training and Evaluation of FABNet on CPU/GPU
The code and scripts of evaluting speed are put under `software/speed/`.
The code and scripts of evaluating accuracy are put under `software/accuracy/`.


### 3.2 Reproducibility

All the scripts are put under the folder `./script_figs`, we refer reviewers/users to read [instruction](./script_figs/README.md) before running it. Some training may take over hunderds of GPU hours to finish, so we also attach our log files in google drive [link](https://drive.google.com/drive/folders/1zn38AjjQvqHZh-xsmeeIFK2BA-poIRAn?usp=sharing).

## Citation 

Our paper is online now ([link](https://arxiv.org/pdf/2209.09570.pdf))! If you found it helpful, pls cite us using:


``` 
@inproceedings{fan2022adaptable,
  title={Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design},
  author={Fan, Hongxiang and Chau, Thomas and Venieris, Stylianos I and Lee, Royson and Kouris, Alexandros and Luk, Wayne and Lane, Nicholas D and Abdelfattah, Mohamed S},
  booktitle={MICRO-55: 55th Annual IEEE/ACM International Symposium on Microarchitecture},
  year={2022}
}

```

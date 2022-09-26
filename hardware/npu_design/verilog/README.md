## Description

This folder contains all the Verilog RTL design, Sytemverilog testbenches and Constraints to evluate the functionality and reproducibility of our hardware design. To facilitate the Artifact Evluation, we provide vivado tcl file in each sub-directory to quickly create project and setup IPs.

## Functionality

1. Go to the subfolder `functionality` by typing `cd ./functionality`.
1. Open vivado, type `vivado_functionality_project.tcl` in the vivado tcl console to create the project. All the necessary files, IPs and testbenches will be imported and generated authomatically.
1. Generate the testdata according to the instruction in *functionality/testbench/README.md*. 
1. Change the path to the test data in your Systemverilog testbench.
1. In the *Sources* window (next to *Flow Navigator*) of your vivado, select the testbench you want to run under `Simulation Sources/sim_1` as the top module by right clicking and choosing *set as top*. 


## Power and Resource Utilization

The follwoing examples run on *Zynq7045* and *VCU128*. You can also change target platforms, and modify the `act_be_pararallism` in the top module `acc_top.v` according to the available resources.

### Zynq Board
1. Go to the subfolder `Zynq7045` by typing `cd ./Zynq7045`.
1. Open vivado, type `source ./vivado_Zyqn_project.tcl` in the vivado tcl console to create the project. All the necessary files, IPs and testbenches will be imported and generated authomatically.
1. Wait until all the synthsis of IP blocks finish.
1. Due to the [issue]() of Vivado tool, it will report error "Validation failed on parameter 'XML_INPUT_FILE(XML_INPUT_FILE)' for Specified PRJ file does not exist 'mig_a.prj' . IP 'mig_ddr3'".  So you need to configure DDR3 by hands: 
    1. Double click 'mig_ddr3' in *IP Sources* window
    1. Click 'Next'
    1. Tick 'AXI Interface', Click 'Next'
    1. Click 'Next'
    1. Click 'Next'
    1. Select 'Data Width' as 64, Click 'Next'
    1. Click 'Next'
    1. Select 'Input Clock Period' as 5000ps(200MHz), Click 'Next'
    1. Select 'Reference Clock' as 'Use System Clock', Click 'Next'
    1. Click 'Next', 'Accept' and 'Generate' to the end. 
1. Click *Run Implementation* in the *Flow Navigator* window.
1. The whole process may take 5 ~ 10 hours depending on your machine. We also provide the generated post-implmementation report [here](https://drive.google.com/file/d/1CbHchAz-TM0XpVYArH6Jqb-hL9I-cfhj/view?usp=sharing).
 
### VCU Board

1. Go to the subfolder `VCU` by typing `cd ./VCU`.
1. Open vivado, type `source ./vivado_VCU_project.tcl` in the vivado tcl console to create the project. All the necessary files, IPs and testbenches will be imported and generated authomatically.
1. Wait until all the synthsis of IP blocks finish.
1. Click *Run Implementation* in the *Flow Navigator* window.
1. The whole process may take 8 ~ 20 hours depending on your machine. We also provide the generated post-implmementation report [here](https://drive.google.com/file/d/1CbHchAz-TM0XpVYArH6Jqb-hL9I-cfhj/view?usp=sharing).





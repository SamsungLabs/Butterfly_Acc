## Description

This folder contains the scripts to explore how does the use of ABfly and FBfly block affect the accuracy of Transformer. These scripts are used to generate Fig.16 in our paper. Note that the accuracy may slightly different due to the random seed and GPU you are using.

## Example

To evaluate on LAR-Image dataset, simply run:
```
bash acc_impact_image.sh
```
You can change the total number of attention layers by setting `Num_Att_layer` and `--num_layers` accordingly.

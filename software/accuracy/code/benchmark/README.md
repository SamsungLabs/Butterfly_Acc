## Description

This folder contains the scripts to obtain the accuracy of vanilla Tranformer and FNet in `LRA` dataset. These scripts are used to generate the results of the first two rows in Table 3. Note that the accuracy may slightly different due to the random seed and GPU you are using. 

## Accuracy on LRA Dataset

|             | ListOps       | Text          | Retrieval     | Image         | Pathfinder   | Avg      |
| ----------- | -----------   | -----------   | -----------   | -----------   |-----------   | ------   |
| Transformer | 0.373         | 0.637         | 0.783         | 0.379         | 0.709        | 0.576    |
| FNet        | 0.365         | 0.630         | 0.779         | 0.288         | 0.66         | 0.544    |

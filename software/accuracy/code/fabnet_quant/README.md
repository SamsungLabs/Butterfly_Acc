## Description

This folder contains the scripts to obtain the fp16 accuracy of our proposed FABNet with the optimized configuration generated in `grid_search` folder. These scripts are used to generate the results of the last row in Table 3. Note that the accuracy may slightly different due to the random seed and GPU you are using. 

## Accuracy on LRA Dataset

|             | ListOps       | Text          | Retrieval     | Image         | Pathfinder   | Avg      |
| ----------- | -----------   | -----------   | -----------   | -----------   |-----------   | ------   |
| FABNet      | 0.374         | 0.626         | 0.801         | 0.398         | 0.679        | 0.576    |
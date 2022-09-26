## Description

This folder contains the scripts to optimize the hyperparameters of FABNet for different task in `LRA` dataset (`image`, `listop`, `path`, `retrieval` and `text`).

## Optimized Configurations

| Task        | embedding_dim | hidden_dim    | num_layer     | dropout rate  |
| ----------- | -----------   | -----------   | -----------   | -----------   |
| image       | 32            | 128           | 1             | 0.3           |
| listop      | 128           | 256           | 2             | 0.1           |
| path        | 64            | 64            | 6             | 0.1           |
| retrieval   | 1024          | 256           | 2             | 0.1           |
| text        | 256           | 256           | 2             | 0.1           |
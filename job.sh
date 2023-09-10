#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --config ./config/sweeps/sweep_slowmo_h_m_lr_llr.py --name_suffix z

# test_models_wandb_example.py

# ./config/sweeps/sweep_fedavg.py
# ./config/sweeps/sweep_slowmo_h_m_lr_llr.py

# ./config/meta_test/small-image-mlp-fmst_fedavg.py
# ./config/meta_test/small-image-mlp-fmst_fedavg-slowmo.py

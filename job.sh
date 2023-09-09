#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --config ./config/meta_train/meta_train_fedlopt-adafac.py --num_local_steps 32  --name_suffix z --use_pmap --num_devices 2

# test_models_wandb_example.py

# ./config/sweeps/sweep_fedavg.py
# ./config/sweeps/sweep_slowmo_h_m_lr_llr.py

# ./config/meta_test/small-image-mlp-fmst_fedavg.py
# ./config/meta_test/small-image-mlp-fmst_fedavg-slowmo.py
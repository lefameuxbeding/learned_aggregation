#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type benchmark --optimizer fedavg --task small-conv --name fedavg_small-conv_K8_H4_0.5 --num_inner_steps 5000

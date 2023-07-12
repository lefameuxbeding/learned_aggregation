#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type meta-train --optimizer fedlopt --task small-image-mlp --name fedlopt32_small-image-mlp_K8_H16_0.5 --local_learning_rate 5e-1 --num_local_steps 16

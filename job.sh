#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type benchmark --optimizer fedavg --task image-mlp --num_inner_steps 5000 --num_outer_steps 20000

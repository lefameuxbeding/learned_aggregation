#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type benchmark --optimizer lagg --task image-mlp --batch_size 128 --num_grads 8 --num_inner_steps 2000

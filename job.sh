#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type benchmark --optimizer fedlagg --task image-mlp --batch_size 2048 --num_grads 8 --num_local_steps 4 --num_inner_steps 2000 --num_outer_steps 20000

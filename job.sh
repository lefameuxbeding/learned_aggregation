#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type benchmark --optimizer adam --task small-conv --name adam_0.001 --learning_rate 1e-3 --num_inner_steps 5000

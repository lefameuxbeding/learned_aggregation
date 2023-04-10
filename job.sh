#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type meta-train --optimizer lagg --task conv --num_grads 128

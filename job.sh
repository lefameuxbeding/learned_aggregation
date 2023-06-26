#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type benchmark --from_checkpoint --optimizer fedavg --task small-image-mlp --num_inner_steps 5000

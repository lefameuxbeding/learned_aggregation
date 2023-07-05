#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type benchmark --optimizer fedlopt-adafac --task small-image-mlp --name fedlopt-adafac32_small-image-mlp_K8_H4_0.5

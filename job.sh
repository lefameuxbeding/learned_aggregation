#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type benchmark --optimizer fedlagg --task small-image-mlp-fmst --name fedlagg256_small-image-mlp-fmst_K8_H4_0.5

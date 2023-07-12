#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type meta-train --optimizer fedlopt --task small-image-mlp-fmst --name fedlopt32_small-image-mlp-fmst_K8_H4_0.5

#!/bin/bash

module load anaconda/3

conda activate learned_aggregation

python ./src/main.py --run_type benchmark --optimizer nadamw --task image_mlp

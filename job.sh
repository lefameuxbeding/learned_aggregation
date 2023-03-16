#!/bin/bash

module load anaconda/3

conda activate env

python -m src.benchmark.lagg

#!/bin/bash

module load anaconda/3

conda activate env

python ./src/meta_train_lopt.py

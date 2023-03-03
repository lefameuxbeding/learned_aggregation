#!/bin/bash

module load python/3
module load cuda/11.0

source env/bin/activate

python main.py

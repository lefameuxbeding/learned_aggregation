#!/bin/bash

module load python/3.9
module load cuda/11.0

python -m venv env
source env/bin/activate

pip install --upgrade pip
pip install wandb black
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install git+https://github.com/google/learned_optimization.git

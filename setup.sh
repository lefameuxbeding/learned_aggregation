#!/bin/bash

module load anaconda/3

conda create -y -f -n env
conda activate env

conda install -y jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
pip install wandb black
pip install git+https://github.com/lefameuxbeding/learned_optimization

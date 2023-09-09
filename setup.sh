#!/bin/bash

module load anaconda/3

conda create -y -f -n learned_aggregation
conda activate learned_aggregation

conda install -y jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
pip install --no-cache-dir wandb black mmengine tqdm
pip install --no-cache-dir git+https://github.com/lefameuxbeding/learned_optimization
#!/bin/bash

module --quiet load anaconda/3

conda create -y -f -n venv
conda activate venv

conda install -y jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
conda install -y pip
pip install -r requirements.txt --no-cache-dir
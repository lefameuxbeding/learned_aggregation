#!/bin/bash

module --quiet load anaconda/3
conda activate venv

conda install -y -c conda-forge cudnn
conda install -y pip
pip install -r requirements.txt --no-cache-dir

python main.py
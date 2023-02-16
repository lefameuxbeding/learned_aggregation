#!/bin/bash

module load python/3.8

python -m venv ./.venv
source ./.venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
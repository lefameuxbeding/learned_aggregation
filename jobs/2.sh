#!/bin/bash
#SBATCH --partition=short-unkillable
#SBATCH -J 8192_muada
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH -t 3:00:00
#SBATCH --mail-user btherien@uwaterloo.ca
#SBATCH --mail-type=END
#SBATCH -o /home/mila/b/benjamin.therien/log/out.%j
#SBATCH -e /home/mila/b/benjamin.therien/log/err.%j

module purge

source /home/mila/b/benjamin.therien/setup.sh

cd /home/mila/b/benjamin.therien/github/learned_aggregation
# source script_new_install.sh
export WANDB_API_KEY=9fa3792d90a05640029b7725e310b9904ac00119


CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py \
    --name_suffix _m_muadam_final \
    --local_batch_size 4096 \
    --test_project mup-meta-testing \
    --task "mumlp-w8192-d3_imagenet-32x32x3" \
    --optimizer muadam \
    --num_runs 5 \
    --learning_rate 0.1 \
    --num_inner_steps 5000 \
    --gradient_accumulation_steps 1 \
    --needs_state \
    --mup_input_mult 0.25 \
    --mup_output_mult 0.25 \
    --mup_hidden_lr_mult 4 \
    --test_interval 50 
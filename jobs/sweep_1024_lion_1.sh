#!/bin/bash
#SBATCH --partition=long
#SBATCH -J lion_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 168:00:00
#SBATCH --mail-user btherien@uwaterloo.ca
#SBATCH --mail-type=END
#SBATCH -o /home/mila/b/benjamin.therien/log/out.%j
#SBATCH -e /home/mila/b/benjamin.therien/log/err.%j

module purge

source /home/mila/b/benjamin.therien/setup.sh

cd /home/mila/b/benjamin.therien/github/learned_aggregation
# source script_new_install.sh
export WANDB_API_KEY=9fa3792d90a05640029b7725e310b9904ac00119



# CUDA_VISIBLE_DEVICES=0 python src/main.py \
#     --config config/meta_test/small-image-mlp-fmst_fedavg-slowmo.py \
#     --name_suffix _lion_sweep \
#     --local_batch_size 4096 \
#     --test_project mup-meta-testing \
#     --task "mlp-w1024-d3_imagenet-32x32x3" \
#     --optimizer adam \
#     --num_runs 1 \
#     --learning_rate 3e-4 \
#     --num_inner_steps 1000 \
#     --gradient_accumulation_steps 1 \
#     --needs_state \
#     --mup_input_mult 1 \
#     --mup_output_mult 1 \
#     --mup_hidden_lr_mult 1 \
#     --test_interval 50


CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --config config/sweeps/mulo_sweep_lion.py \
    --name_suffix _lion_sweep \
    --local_batch_size 4096 \
    --test_project mup-meta-testing \
    --task "mlp-w1024-d3_imagenet-32x32x3" \
    --optimizer lion \
    --num_runs 1 \
    --learning_rate 3e-4 \
    --num_inner_steps 1000 \
    --gradient_accumulation_steps 1 \
    --needs_state \
    --mup_input_mult 1 \
    --mup_output_mult 1 \
    --mup_hidden_lr_mult 1 \
    --test_interval 50 \
    --sweep_id p10h596a
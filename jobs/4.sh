#!/bin/bash
#SBATCH --partition=short-unkillable
#SBATCH -J resnet_slowmo
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
    --config config/meta_test/meta_test_base.py \
    --name_suffix _multi-task-lopt_mup_no-clip \
    --local_batch_size 128 \
    --test_project mup-meta-testing \
    --task "mutransformer-w2048-d3_lm1b-s64-v32k" \
    --optimizer mup_small_fc_mlp \
    --wandb_checkpoint_id eb-lab/mup-meta-training/4swslf7l \
    --num_runs 10 \
    --num_inner_steps 2000 \
    --gradient_accumulation_steps 1 \
    --needs_state \
    --test_interval 50
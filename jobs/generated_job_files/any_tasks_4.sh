#!/bin/bash
#SBATCH --partition=long
#SBATCH -J any_tasks_4
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
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


CUDA_VISIBLE_DEVICES=0 python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _m_sp_final         --local_batch_size 4096         --test_project mup-meta-testing         --task mlp-w512-d3_imagenet-64x64x3         --optimizer sgd         --wandb_checkpoint_id eb-lab/mup-meta-training/2tfh8kuw         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.01         --gradient_accumulation_steps 1         --test_interval 100 --benchmark_momentum 0.777 --learning_rate 0.546         --use_bf16 --seed 1

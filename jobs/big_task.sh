#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH -J 2048_transformer
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 48:00:00
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
# --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py \
# --name_suffix _m_mup_final \
# --local_batch_size 4096 \
# --test_project mup-meta-testing \
# --task muvit-w2048-d3_imagenet-32x32x3 \
# --optimizer mup_small_fc_mlp \
# --wandb_checkpoint_id eb-lab/mup-meta-training/woz3g9l0 \
# --num_runs 5 \
# --num_inner_steps 5000 \
# --needs_state \
# --adafac_step_mult 0.01 \
# --gradient_accumulation_steps 1 \
# --test_interval 100 \
# --use_bf16



# CUDA_VISIBLE_DEVICES=0 python src/main.py \
# --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py \
# --name_suffix _m_mup_final \
# --local_batch_size 128 \
# --test_project mup-meta-testing \
# --task mutransformer-w2048-d16_lm1b-s64-v32k \
# --optimizer velo \
# --wandb_checkpoint_id eb-lab/mup-meta-training/woz3g9l0 \
# --num_runs 5 \
# --num_inner_steps 5000 \
# --needs_state \
# --adafac_step_mult 0.01 \
# --gradient_accumulation_steps 1 \
# --test_interval 100 \
# --use_bf16




# CUDA_VISIBLE_DEVICES=0 python src/main.py \
# --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py \
# --name_suffix _m_mup_final \
# --local_batch_size 128 \
# --test_project mup-meta-testing \
# --task mutransformer-w2048-d3_lm1b-s64-v32k \
# --optimizer mup_small_fc_mlp \
# --wandb_checkpoint_id eb-lab/mup-meta-training/woz3g9l0 \
# --num_runs 5 \
# --num_inner_steps 5000 \
# --needs_state \
# --adafac_step_mult 0.01 \
# --gradient_accumulation_steps 1 \
# --test_interval 100 \
# --use_bf16



python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _velo_final         --local_batch_size 128         --test_project mup-meta-testing         --task transformer-w2048-d3_lm1b-s64-v32k         --optimizer velo         --wandb_checkpoint_id eb-lab/mup-meta-training/2tfh8kuw         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.001         --gradient_accumulation_steps 1         --test_interval 100         --use_bf16 --seed 1


python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _s_sp_final         --local_batch_size 128         --test_project mup-meta-testing         --task transformer-w2048-d3_lm1b-s64-v32k         --optimizer small_fc_mlp         --wandb_checkpoint_id eb-lab/mup-meta-training/dcjjqqyt         --num_runs 5         --num_inner_steps 5000         --needs_state         --adafac_step_mult 0.001         --gradient_accumulation_steps 1         --test_interval 100         --use_bf16 --seed 1
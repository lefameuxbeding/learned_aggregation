
# CUDA_VISIBLE_DEVICES=0 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=4_inet_resnet50 \
# 	--num_local_steps 4 \
# 	--local_learning_rate 0.1 \
# 	--task "resnet50_imagenet_64" \
# 	--wandb_checkpoint_id eb-lab/learned_aggregation_meta_train/p92wfvor \
# 	--test_project leared-aggregation-final-testing \
# 	--num_inner_steps 1000 \
# 	--needs_state

# CUDA_VISIBLE_DEVICES=0 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=4_inet_deit-small \
# 	--num_local_steps 4 \
# 	--local_learning_rate 0.1 \
# 	--task "deit_small_imagenet_64" \
# 	--wandb_checkpoint_id eb-lab/learned_aggregation_meta_train/p92wfvor \
# 	--test_project leared-aggregation-final-testing \
# 	--num_inner_steps 1000

# CUDA_VISIBLE_DEVICES=0 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=4_inet_deit-tiny \
# 	--num_local_steps 4 \
# 	--local_learning_rate 0.1 \
# 	--task "deit_tiny_imagenet_64" \
# 	--wandb_checkpoint_id eb-lab/learned_aggregation_meta_train/p92wfvor \
# 	--test_project leared-aggregation-final-testing \
# 	--num_inner_steps 1000







# CUDA_VISIBLE_DEVICES=0 python src/main.py \
# 	--config config/meta_train/meta_train_fedlagg-adafac32_image-mlp-fmst_schedule_3e-3_5000_d3:1.py \
# 	--name_suffix _H=1_inet_mlp_widths \
# 	--num_local_steps 1 \
# 	--num_grads 1 \
# 	--local_learning_rate 0.1 \
# 	--task "imagenet_32_mlp_width-128-512-2048" \
# 	--num_inner_steps 1000 \
# 	--local_batch_size 4096 \
# 	--auto_resume

# CUDA_VISIBLE_DEVICES=0 python src/main.py \
# 	--config config/meta_train/meta_train_fedlagg-adafac32_image-mlp-fmst_schedule_3e-3_5000_d3:1.py \
# 	--name_suffix _H=1_inet_mlp_widths \
# 	--num_local_steps 1 \
# 	--num_grads 1 \
# 	--local_learning_rate 0.1 \
# 	--task "imagenet_32_mlp_width-256-1024-4096" \
# 	--num_inner_steps 1000 \
# 	--local_batch_size 4096 \
# 	--auto_resume







# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py \
# 	--config config/meta_train/meta_train_fedlagg-adafac32_image-mlp-fmst_schedule_3e-3_5000_d3:1.py \
# 	--name_suffix _H=1_inet_mlp_widths \
# 	--num_local_steps 1 \
# 	--num_grads 1 \
# 	--local_learning_rate 0.1 \
# 	--task "imagenet_32_mlp_width" \
# 	--num_inner_steps 1000 \
# 	--local_batch_size 4096 \
# 	--use_pmap \
# 	--num_devices 4




python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _25k-setps_m_mup_final         --local_batch_size 128         --test_project mup-meta-testing         --task mutransformer-w1024-d3_lm1b-s64-v32k         --optimizer mup_small_fc_mlp         --wandb_checkpoint_id eb-lab/mup-meta-training/woz3g9l0         --num_runs 5         --num_inner_steps 25000         --needs_state         --adafac_step_mult 0.01         --gradient_accumulation_steps 1         --test_interval 100


python src/main.py         --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py         --name_suffix _25k-setps_m_sp_final         --local_batch_size 128         --test_project mup-meta-testing         --task transformer-w1024-d3_lm1b-s64-v32k         --optimizer small_fc_mlp         --wandb_checkpoint_id eb-lab/mup-meta-training/byuo0ixg         --num_runs 5         --num_inner_steps 25000         --needs_state         --adafac_step_mult 0.001         --gradient_accumulation_steps 1         --test_interval 100
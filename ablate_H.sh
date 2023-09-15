# SLOWMO on FMNIST 28x28
CUDA_VISIBLE_DEVICES=5 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=16 \
--num_local_steps 16 \
--optimizer fedavg-slowmo \
--test_project leared-aggregation-final-testing \
--local_learning_rate 0.1 --slowmo_learning_rate 0.1 --beta 0.95

CUDA_VISIBLE_DEVICES=5 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=8 \
--num_local_steps 8 \
--optimizer fedavg-slowmo \
--test_project leared-aggregation-final-testing \
--local_learning_rate 0.1 --slowmo_learning_rate 0.1 --beta 0.95

CUDA_VISIBLE_DEVICES=5 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=4 \
--num_local_steps 4 \
--optimizer fedavg-slowmo \
--test_project leared-aggregation-final-testing \
--local_learning_rate 0.1 --slowmo_learning_rate 0.1 --beta 0.95

####################################
# FEDavg on FMNIST 28x28
####################################
CUDA_VISIBLE_DEVICES=5 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=16 \
--num_local_steps 16 \
--optimizer fedavg \
--test_project leared-aggregation-final-testing \
--local_learning_rate 0.3

CUDA_VISIBLE_DEVICES=5 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=8 \
--num_local_steps 8 \
--optimizer fedavg \
--test_project leared-aggregation-final-testing \
--local_learning_rate 0.5

CUDA_VISIBLE_DEVICES=5 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=4 \
--num_local_steps 4 \
--optimizer fedavg \
--test_project leared-aggregation-final-testing \
--local_learning_rate 0.5







####################################
# FEDLOPT-ADAFAC on FMNIST 28x28
####################################
CUDA_VISIBLE_DEVICES=4 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=16_llr=0.5_5k_iters \
--local_learning_rate 0.5 --num_local_steps 16 \
--optimizer fedlopt-adafac \
--wandb_checkpoint_id eb-lab/learned_aggregation_meta_train/9yi0nq2b \
--test_project leared-aggregation-final-testing

CUDA_VISIBLE_DEVICES=4 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=8_llr=0.5_5k_iters \
--local_learning_rate 0.5 --num_local_steps 8 \
--optimizer fedlopt-adafac \
--wandb_checkpoint_id eb-lab/learned_aggregation_meta_train/mn2a8lde \
--test_project leared-aggregation-final-testing

CUDA_VISIBLE_DEVICES=4 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=4_llr=0.5_5k_iters \
--local_learning_rate 0.5 --num_local_steps 4 \
--optimizer fedlopt-adafac \
--wandb_checkpoint_id eb-lab/learned_aggregation_meta_train/lmpor1c6 \
--test_project leared-aggregation-final-testing





####################################
# FEDLAGG-ADAFAC on FMNIST 
####################################
CUDA_VISIBLE_DEVICES=4 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=16_llr=0.5_5k_iters \
--local_learning_rate 0.5 --num_local_steps 16 \
--optimizer fedlagg-adafac \
--wandb_checkpoint_id eb-lab/learned_aggregation_meta_train/9kbqr0j9 \
--test_project leared-aggregation-final-testing

CUDA_VISIBLE_DEVICES=4 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=8_llr=0.5_5k_iters \
--local_learning_rate 0.5 --num_local_steps 8 \
--optimizer fedlagg-adafac \
--wandb_checkpoint_id eb-lab/learned_aggregation_meta_train/tjqrs7gu \
--test_project leared-aggregation-final-testing

CUDA_VISIBLE_DEVICES=4 python src/main.py --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py --name_suffix _H=4_llr=0.5_5k_iters \
--local_learning_rate 0.5 --num_local_steps 4 \
--optimizer fedlagg-adafac \
--wandb_checkpoint_id eb-lab/learned_aggregation_meta_train/n840quas \
--test_project leared-aggregation-final-testing

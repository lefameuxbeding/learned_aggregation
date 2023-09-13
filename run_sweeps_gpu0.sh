CUDA_VISIBLE_DEVICES=0 python src/main.py --config config/meta_test/small-image-mlp-fmst_fedavg.py --name_suffix **_H=4_llr=1 --local_learning_rate 1 --num_local_steps 4
CUDA_VISIBLE_DEVICES=0 python src/main.py --config config/meta_test/small-image-mlp-fmst_fedavg.py --name_suffix **_H=8_llr=0.3 --local_learning_rate 0.3 --num_local_steps 8
CUDA_VISIBLE_DEVICES=0 python src/main.py --config config/meta_test/small-image-mlp-fmst_fedavg.py --name_suffix **_H=16_llr=0.5 --local_learning_rate 0.5 --num_local_steps 16
CUDA_VISIBLE_DEVICES=0 python src/main.py --config config/meta_test/small-image-mlp-fmst_fedavg.py --name_suffix **_H=32_llr=0.3 --local_learning_rate 0.3 --num_local_steps 32








# #image mlp fmst slowmo
# CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
# python ./src/main.py \
# --config config/sweeps/sweep_slowmo_m_lr.py \
# --task image-mlp-fmst


# #image mlp fmst slowmo
# CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
# python ./src/main.py \
# --config config/sweeps/sweep_fedavg.py \
# --task image-mlp-fmst


# #small conv c10 slowmo
# CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
# python ./src/main.py \
# --config config/sweeps/sweep_slowmo_m_lr.py \
# --task small-conv-c10


# #small conv c10 slowmo
# CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
# python ./src/main.py \
# --config config/sweeps/sweep_fedavg.py \
# --task small-conv-c10


# #cifar 10 slowmo
# CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
# python ./src/main.py \
# --config config/sweeps/sweep_slowmo_m_lr.py \
# --task conv-c10

# #cifar 10 fedavg
# CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
# python ./src/main.py \
# --config config/sweeps/sweep_fedavg.py \
# --task conv-c10

#imagenet 64 slowmo
XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false CUDA_VISIBLE_DEVICES=1 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
python ./src/main.py \
--config config/sweeps/sweep_slowmo_m_lr.py \
--task conv-imagenet64

#imagenet 64 fedavg
XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false CUDA_VISIBLE_DEVICES=1 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
python ./src/main.py \
--config config/sweeps/sweep_fedavg.py \
--task conv-imagenet64
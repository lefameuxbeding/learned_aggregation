#image mlp fmst slowmo
CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
python ./src/main.py \
--config config/sweeps/sweep_slowmo_m_lr.py \
--task image-mlp-fmst


#image mlp fmst slowmo
CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
python ./src/main.py \
--config config/sweeps/sweep_fedavg.py \
--task image-mlp-fmst


#small conv c10 slowmo
CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
python ./src/main.py \
--config config/sweeps/sweep_slowmo_m_lr.py \
--task small-conv-c10


#small conv c10 slowmo
CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
python ./src/main.py \
--config config/sweeps/sweep_fedavg.py \
--task small-conv-c10


#cifar 10 slowmo
CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
python ./src/main.py \
--config config/sweeps/sweep_slowmo_m_lr.py \
--task conv-c10

#cifar 10 fedavg
CUDA_VISIBLE_DEVICES=0 TFDS_DATA_DIR=/home/new_datasets/tensorflow_datasets \
python ./src/main.py \
--config config/sweeps/sweep_fedavg.py \
--task conv-c10
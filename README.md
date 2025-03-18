# Welcome to Learned Aggregation!
[**TMLR Version**](https://arxiv.org/abs/2305.10210) | [**ArXiv Version**](https://arxiv.org/abs/2312.02204)

This repository contains the research code for [Meta-learning Optimizers for Communication-Efficient Learning](https://openreview.net/forum?id=uRbf9ANAns&noteId=laeorzVP1b).

# Citation 
If you found this code useful for your research, please consider citing our paper:
```bibtex
@article{
joseph2025metalearning,
title={Meta-learning Optimizers for Communication-Efficient Learning},
author={Charles-{\'E}tienne Joseph and Benjamin Th{\'e}rien and Abhinav Moudgil and Boris Knyazev and Eugene Belilovsky},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=uRbf9ANAns},
}
```

# Installation

Run the following code:
```
python -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install nvidia-pyindex
python -m pip install -r requirements.txt
```

# Quickstart

As a quickstart tutorial, we will replicate the experiments at different H values from the paper for the LAgg-A model.

### H=4, K=8
```
python src/main.py \
--config config/meta_train/meta_train_fedlagg-adafac32_image-mlp-fmst_schedule_3e-3_10000_d3:1.py \
--num_local_steps 4 --num_grads 8 \
--optimizer fedlagg-adafac \
--local_learning_rate 0.5
```

### H=8, K=8
```
python src/main.py \
--config config/meta_train/meta_train_fedlagg-adafac32_image-mlp-fmst_schedule_3e-3_10000_d3:1.py \
--num_local_steps 8 --num_grads 8 \
--optimizer fedlagg-adafac \
--local_learning_rate 0.5
```

### H=16, K=8
```
python src/main.py \
--config config/meta_train/meta_train_fedlagg-adafac32_image-mlp-fmst_schedule_3e-3_10000_d3:1.py \
--num_local_steps 4 --num_grads 8 \
--optimizer fedlagg-adafac \
--local_learning_rate 0.5
```

# Config file structure

Using MMengine's config file parser, we can write config files directly in Python and use an inheritance config structure to avoid redundant configurations. This can be achieved by specifying config files to inherit from using the 
```_base_=['my_config.py']``` 
special variable at the top of config files. More information is available at [mmengine config docs](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html).

In learned_aggragation, configuration files are logically separated into different directories based on the task to be executed: ```config/meta_test```,```config/meta_train```, and ```config/sweeps```. 

# Setting up a sweep
To sweep over the hyperparameters of a model during meta-testing, one can simply specify a sweep configuration using the ```sweep_config``` variable.


# Checkpointing during meta training
The ```checkpoints_to_keep``` and ```save_iter``` config variables control the number of checkpoints that should be kept and the checkpointing multiple, respectively. Default values of ```checkpoints_to_keep=10``` and ```save_iter=1000``` ensure that at most 10 previous checkpoints will be kept and that a checkpoint will be saved every 1000 iterations.

# Loading from a checkpoint during meta training
When a checkpoint is logged, it is saved under ```checkpoints/<meta-train-dir>``` where ```<meta-train-dir>``` is the dynamically assigned meta-train-name. Whenever a new checkpoint is logged, a file called ```latest``` is updated with the name of the most recent checkpoint. When resuming from a checkpoint the user simply has to set the ```--from_checkpoint``` flag and meta training will automatically resume to the checkpoint specified in the ```latest``` file.


# Citation 
If you found this code useful for your research, please consider citing our paper:
```bibtex
@article{
joseph2025metalearning,
title={Meta-learning Optimizers for Communication-Efficient Learning},
author={Charles-{\'E}tienne Joseph and Benjamin Th{\'e}rien and Abhinav Moudgil and Boris Knyazev and Eugene Belilovsky},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=uRbf9ANAns},
}
```

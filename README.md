# learned_aggregation

# How Config files are setup.

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

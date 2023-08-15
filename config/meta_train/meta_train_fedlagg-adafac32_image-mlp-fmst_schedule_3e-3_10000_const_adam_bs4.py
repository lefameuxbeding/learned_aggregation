_base_ = ["./meta_train_base.py"]

schedule = dict()
learning_rate=3e-3

num_outer_steps = 10000
task = "image-mlp-fmst"
optimizer = "fedlagg-adafac"
num_tasks = 4
name_suffix = "_3e-3_10000_const_adam_bs4"
num_local_steps = 16
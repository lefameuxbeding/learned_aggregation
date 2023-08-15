_base_ = ["./meta_train_base.py"]

schedule = dict()
learning_rate=3e-3

num_outer_steps = 5000
task = "image-mlp-fmst"
optimizer = "fedlagg-adafac"
name_suffix = "_3e-3_5000_const_adam"
num_local_steps = 16
_base_ = ["./meta_train_base.py"]

optimizer = "fedlagg"
task = "image-mlp-fmst"

num_local_steps = 16

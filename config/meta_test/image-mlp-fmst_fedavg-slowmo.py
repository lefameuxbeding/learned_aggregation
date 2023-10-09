_base_ = ["./meta_test_base.py"]

optimizer = "fedavg-slowmo"
task = "image-mlp-fmst"
num_inner_steps = 1000

num_grads = 32
num_local_steps = 4

# values determined by sweep
local_learning_rate = 0.3
slowmo_learning_rate = 0.5
beta = 0.9

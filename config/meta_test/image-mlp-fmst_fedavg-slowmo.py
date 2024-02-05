_base_ = ["./meta_test_base.py"]

optimizer = "fedavg-slowmo"
task = "mlp128x128x128_imagenet_32"
num_inner_steps = 1000

num_grads = 8
num_local_steps = 4

# values determined by sweep
local_learning_rate = 0.1
slowmo_learning_rate = 0.1
beta = 0.85

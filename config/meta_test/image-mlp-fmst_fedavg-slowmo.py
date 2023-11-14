_base_ = ["./meta_test_base.py"]

optimizer = "fedavg-slowmo"
task = "mlp128x128_fmnist_32"
num_inner_steps = 1000

num_grads = 8
num_local_steps = 16

# values determined by sweep
local_learning_rate = 0.3
slowmo_learning_rate = 0.01
beta = 0.9

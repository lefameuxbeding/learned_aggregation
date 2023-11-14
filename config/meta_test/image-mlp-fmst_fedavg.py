_base_ = ["./meta_test_base.py"]


optimizer = "fedavg"
task = "mlp128x128_fmnist_32"
num_inner_steps = 1000

num_grads = 8
num_local_steps = 32

# value determined by sweep
local_learning_rate = 0.1

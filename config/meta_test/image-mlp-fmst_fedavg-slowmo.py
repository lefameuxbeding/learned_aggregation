_base_ = ["./meta_test_base.py"]

optimizer = "fedavg-slowmo"
task = "resnet18_imagenet_32"
num_inner_steps = 1000

num_grads = 8
num_local_steps = 4

# values determined by sweep
local_learning_rate = 0.5
slowmo_learning_rate = 0.9
beta = 0.8

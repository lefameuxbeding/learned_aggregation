_base_ = ["./meta_test_base.py"]

optimizer = "sgd"
task = "conv-c10"
num_inner_steps = 1000

num_grads = 32
num_local_steps = 4

# value determined by sweep
learning_rate = 0.5

_base_ = ["./meta_test_base.py"]

optimizer = "adam"
task = "mlp128x128x128_imagenet_32"
num_inner_steps = 1000

num_grads = 8
num_local_steps = 4

# value determined by sweep
learning_rate = 0.001

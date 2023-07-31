_base_ = ["./meta_test_base.py"]


optimizer = "fedavg"
task = "image-mlp-fmst"
num_inner_steps = 5000

local_learning_rate = 0.1
num_local_steps = 8

_base_ = ["./meta_test_base.py"]

optimizer = "fedavg-slowmo"
task = "image-mlp-fmst"

# values determined by sweep
local_learning_rate = 0.1
slowmo_learning_rate = 0.1
beta = 0.95

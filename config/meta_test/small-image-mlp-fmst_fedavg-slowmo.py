_base_ = ["./meta_test_base.py"]

# values determined by sweep
beta = 0.85
local_learning_rate = 0.5
task = "small-image-mlp-fmst"
optimizer = "fedavg-slowmo"

_base_ = ["./meta_test_base.py"]

# values determined by sweep
beta = 0.9
local_learning_rate = 0.05
optimizer = "fedavg-slowmo"
task = "conv-c10"

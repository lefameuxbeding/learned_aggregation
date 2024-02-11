_base_ = ["./meta_test_base.py"]

optimizer = "fedavg"
task = "conv-c10"

# values determined by sweep
local_learning_rate = 1

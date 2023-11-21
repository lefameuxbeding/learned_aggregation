_base_ = ["./meta_test_base.py"]

optimizer = "fedavg-slowmo"
task = "conv-c10"

# values determined by sweep
slowmo_learning_rate = 0.1
beta = 0.95
local_learning_rate = 0.3

# must be overriden
run_type = None
from_checkpoint = None

# common
optimizer = "fedavg"
task = "small-image-mlp-fmst"
hidden_size = 32
local_learning_rate = 0.5
local_batch_size = 128
num_grads = 8
num_local_steps = 4
num_inner_steps = 2000
learning_rate = 0.0001

# meta training only
num_outer_steps = 50000

# meta testing only
num_runs = 10

# for slowmo
beta = 0.99

# sweeps only
sweep_config = dict()
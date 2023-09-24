# must be overriden
run_type = None

# common
optimizer = "fedavg"
task = "small-image-mlp-fmst"
hidden_size = 32
local_learning_rate = 0.5
local_batch_size = 128
num_grads = 8
num_local_steps = 4
num_inner_steps = 1000
learning_rate = 0.0001
name_suffix = ""

# meta training only
num_outer_steps = 5000
from_checkpoint = False
num_devices = 1
use_pmap = False
auto_resume = False
meta_loss_split = None

# meta testing only
num_runs = 10
wandb_checkpoint_id = None
test_project = "learned_aggregation_meta_test"
# for slowmo
beta = 0.99

# sweeps only
sweep_config = dict()

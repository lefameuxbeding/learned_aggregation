_base_ = ["../config_base.py"]

run_type = "meta-train"
save_iter = 50
checkpoints_to_keep = 10
schedule = dict()


# default 5k outer iters, 1k inner iters, 8 tasks
num_tasks = 8
num_outer_steps = 5000
num_inner_steps = 1000

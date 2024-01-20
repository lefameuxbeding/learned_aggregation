_base_ = ["./meta_test_base.py"]

optimizer = "diloco"
task = "mlp128x128_fmnist_32"
num_inner_steps = 1000

num_grads = 8
num_local_steps = 32

# value determined by sweep
learning_rate = 0.001
adamw_kwargs = dict(learning_rate=4e-4, b1=0.9, b2=0.999, eps=1e-08, eps_root=0.0, mu_dtype=None, weight_decay=0.0001, mask=None)
slowmo_learning_rate = 0.1
local_learning_rate = 0.1
nesterov_learning_rate = 0.7
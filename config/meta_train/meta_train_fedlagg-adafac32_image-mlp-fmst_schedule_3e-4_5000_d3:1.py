_base_ = ["./meta_train_base.py"]

schedule = dict(
    init_value=3e-10,
    peak_value=3e-4,
    end_value=1e-4,
    warmup_steps=100,
    decay_steps=4900,
    exponent=1.0,
)
learning_rate = 3e-4
num_outer_steps = 5000
task = "image-mlp-fmst"
optimizer = "fedlagg-adafac"
name_suffix = "_3e-4_5000_d3:1"

num_local_steps = 16

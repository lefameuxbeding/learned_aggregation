_base_ = ["./meta_train_base.py"]

schedule = dict(
    init_value=3e-10,
    peak_value=1e-3,
    end_value=0.333e-3,
    warmup_steps=100,
    decay_steps=4900,
    exponent=1.0,
)
learning_rate = 1e-3
num_outer_steps = 5000
task = "image-mlp-fmst"
optimizer = "fedlagg-adafac"
name_suffix = "_1e-3_5000_d1:03"

num_local_steps = 16

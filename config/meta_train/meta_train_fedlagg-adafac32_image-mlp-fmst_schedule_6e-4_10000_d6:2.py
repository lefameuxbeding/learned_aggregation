_base_ = ["./meta_train_base.py"]

schedule = dict(
    init_value=3e-10,
    peak_value=6e-4,
    end_value=2e-4,
    warmup_steps=100,
    decay_steps=9900,
    exponent=1.0,
)
learning_rate = 6e-4
num_outer_steps = 10000
task = "image-mlp-fmst"
optimizer = "fedlagg-adafac"
name_suffix = "_6e-4_10000_d6:2"

num_local_steps = 16

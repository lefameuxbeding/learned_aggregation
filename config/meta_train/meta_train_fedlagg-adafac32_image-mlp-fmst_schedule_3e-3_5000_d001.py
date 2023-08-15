_base_ = ["./meta_train_base.py"]

schedule = dict(
    init_value=3e-10,
    peak_value=3e-3,
    end_value=3e-5,
    warmup_steps=100,
    decay_steps=4900,
    exponent=1.0,
)
learning_rate=3e-3
num_outer_steps = 5000
task = "image-mlp-fmst"
optimizer = "fedlagg-adafac"
name_suffix = "_3e-3_5000_d001"
num_local_steps = 16

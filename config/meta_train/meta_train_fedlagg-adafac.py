_base_ = ["./meta_train_base.py"]

optimizer = "fedlagg-adafac"
task = "image-mlp-fmst"

num_outer_steps = 10000

schedule = dict(
    init_value=3e-10,
    peak_value=3e-3,
    end_value=1e-3,
    warmup_steps=100,
    decay_steps=9900,
    exponent=1.0,
)

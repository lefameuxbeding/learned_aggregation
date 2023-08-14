_base_ = ["./meta_train_base.py"]

schedule = dict(
    init_value=3e-10,
    peak_value=1e-3,
    end_value=1e-5,
    warmup_steps=100,
    decay_steps=1900,
    exponent=1.0,
)
num_outer_steps = 2000
task = "image-mlp-fmst"
optimizer = "fedlagg-adafac"

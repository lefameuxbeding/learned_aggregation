_base_ = ["./meta_train_base.py"]

optimizer = "fedlagg-adafac"
task = "image-mlp-fmst"

schedule = dict(
    init_value=3e-10,
    peak_value=1e-3,
    end_value=0.333e-3,
    warmup_steps=100,
    decay_steps=4900,
    exponent=1.0,
)
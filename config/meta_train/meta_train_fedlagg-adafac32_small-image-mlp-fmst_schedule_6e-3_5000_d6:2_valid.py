_base_ = ["./meta_train_base.py"]

num_outer_steps = 5000
schedule = dict(
    init_value=3e-10,
    peak_value=6e-3,
    end_value=2e-3,
    warmup_steps=100,
    decay_steps=4900,
    exponent=1.0,
)
hidden_size = 32

optimizer = "fedlagg-adafac"
task = "small-image-mlp-fmst"

outer_data_split = "outer_valid"

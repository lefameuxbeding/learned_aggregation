_base_ = ["./meta_train_base.py"]

optimizer = "fedlagg-adafac"
task = "small-image-mlp-fmst"
schedule = dict(init_value=3e-10, peak_value=3e-4, warmup_steps=300, decay_steps=9700, end_value=3e-5, exponent=1.0)
num_outer_steps = 10000
_base_ = ["./meta_train_fedlagg-adafac32_small-image-mlp-fmst_schedule_3e-4.py"]

schedule = dict(peak_value=6e-3, end_value=6e-5, warmup_steps=100, decay_steps=3900,)
num_outer_steps = 4000
optimizer = "fedlagg"
task = "small-image-mlp-fmst"
schedule = dict(init_value=3e-10, peak_value=6e-3, end_value=6e-5, warmup_steps=100, decay_steps=3900, exponent=1.0)
hidden_size = 32
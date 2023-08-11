_base_ = ["./meta_train_fedlagg-adafac32_small-image-mlp-fmst_schedule_3e-4.py"]

schedule = dict(peak_value=6e-3, end_value=6e-4, warmup_steps=100, decay_steps=900,)
num_outer_steps = 1000
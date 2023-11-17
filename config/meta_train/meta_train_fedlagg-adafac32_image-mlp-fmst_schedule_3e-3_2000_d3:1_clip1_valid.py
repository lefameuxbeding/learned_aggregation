_base_ = ["./meta_train_base.py"]

schedule = dict(
    init_value=3e-10,
    peak_value=3e-3,
    end_value=1e-3,
    warmup_steps=100,
    decay_steps=1900,
    exponent=1.0,
    clip=True,
)
learning_rate = 3e-3
num_outer_steps = 2000
task = "image-mlp-fmst"
optimizer = "fedlagg-adafac"
name_suffix = "_H4_3e-3_2000_d3:1_clip1_valid"
meta_loss_split = "outer_valid"
num_local_steps = 4

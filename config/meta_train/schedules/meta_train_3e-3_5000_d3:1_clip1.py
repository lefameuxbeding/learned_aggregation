_base_ = ["../meta_train_base.py"]

schedule = dict(
    init_value=3e-10,
    peak_value=3e-3,
    end_value=1e-3,
    warmup_steps=100,
    decay_steps=4900,
    exponent=1.0,
    clip=True,
)
num_outer_steps = 5000
learning_rate = 3e-3
name_suffix = "_3e-3_5000_d3:1_clip1"

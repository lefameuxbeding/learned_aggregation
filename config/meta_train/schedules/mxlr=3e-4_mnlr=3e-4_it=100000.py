_base_ = ["../meta_train_base.py"]
schedule = dict(
    init_value=3e-10,
    peak_value=3e-4,
    end_value=3e-4,
    warmup_steps=50,
    decay_steps=99950,
    exponent=1.0,
    clip=True,
)



num_outer_steps = 100000
num_inner_steps = 1000
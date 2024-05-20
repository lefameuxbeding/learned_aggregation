_base_ = ["../meta_train_base.py"]
schedule = dict(
    init_value=3e-10,
    peak_value=1e-3,
    end_value=( 1e-3 ) * 1/3,
    warmup_steps=50,
    decay_steps=950,
    exponent=1.0,
    clip=True,
)



num_outer_steps = 1000
num_inner_steps = 1000
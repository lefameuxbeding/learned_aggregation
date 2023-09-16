_base_ = ["./sweeps_base.py"]

optimizer = "adam"
task = "image-mlp-fmst"
num_inner_steps = 1000

num_local_steps = 4

sweep_config = dict(
    method="grid",
    metric=dict(name="test loss", goal="minimize"),
    parameters=dict(
        num_grads=dict(
            values=[
                16,
                32,
            ]
        ),
        learning_rate=dict(
            values=[
                0.5,
                0.1,
                0.05,
                0.01,
                0.005,
                0.001,
                0.0005,
                0.0001,
                0.00005,
                0.00001,
            ]
        ),
    ),
)

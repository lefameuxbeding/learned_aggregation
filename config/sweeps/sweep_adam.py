_base_ = ["./sweeps_base.py"]

optimizer = "adam"
task = "conv-c10"
num_inner_steps = 1000

num_grads = 8

sweep_config = dict(
    method="grid",
    metric=dict(name="test loss", goal="minimize"),
    parameters=dict(
        num_local_steps=dict(
            values=[4, 8, 16, 32,]
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

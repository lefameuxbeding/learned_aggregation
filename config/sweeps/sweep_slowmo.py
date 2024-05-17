_base_ = ["./sweeps_base.py"]

optimizer = "fedavg-slowmo"
task = "mlp-w128-d3_imagenet-32x32x3"

num_inner_steps = 1000
num_grads = 8
# num_local_steps = 4

sweep_config = dict(
    method="grid",
    metric=dict(name="test loss", goal="minimize"),
    parameters=dict(
        num_local_steps=dict(
            values=[
                32,
                64,
                128,
            ]
        ),
        slowmo_learning_rate=dict(
            values=[
                0.5,
                0.1,
                0.05,
                0.01,
                0.005,
                0.001,
            ]
        ),
        local_learning_rate=dict(
            values=[
                1.0,
                0.5,
                0.3,
                0.1,
            ]
        ),
        beta=dict(values=[0.99, 0.95, 0.9, 0.85, 0.8]),
    ),
)

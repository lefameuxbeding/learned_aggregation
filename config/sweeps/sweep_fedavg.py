_base_ = ["./sweeps_base.py"]

optimizer = "fedavg"
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
        local_learning_rate=dict(
            values=[
                # 1.0,
                # 0.5,
                0.3,
                0.1,
            ]
        ),
    ),
)

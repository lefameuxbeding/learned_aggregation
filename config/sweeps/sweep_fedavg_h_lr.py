_base_ = ["./sweeps_base.py"]

optimizer = "fedavg"
task = "small-image-mlp-fmst"
num_inner_steps = 1000

sweep_config = dict(
    method="grid",
    metric=dict(name="test loss", goal="minimize"),
    parameters=dict(
        local_learning_rate=dict(
            values=[
                1.0,
                0.5,
                0.3,
                0.1,
                # 0.05,
                # 0.03,
                # 0.01,
            ]
        ),
        num_local_steps=dict(values=[4, 8, 16, 32]),
    ),
)

_base_ = ["./sweeps_base.py"]

optimizer = "fedavg-slowmo"
task = "conv-c10"
num_inner_steps = 5000

sweep_config = dict(
    method="grid",
    metric=dict(name="test loss", goal="minimize"),
    parameters=dict(
        local_learning_rate=dict(
            values=[
                0.5,
                0.3,
                0.1,
                0.05,
                0.03,
                0.01,
            ]
        ),
        num_local_steps=dict(values=[4, 8, 16]),
        beta=dict(values=[0.9, 0.85, 0.8, 0.75]),
    ),
)

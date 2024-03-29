_base_ = ["./sweeps_base.py"]

optimizer = "fedavg-slowmo"
task = "conv-c10"
num_inner_steps = 1000

sweep_config = dict(
    method="grid",
    metric=dict(name="outer valid loss", goal="minimize"),
    parameters=dict(
        num_local_steps=dict(values=[8, 16, 32]),
        slowmo_learning_rate=dict(
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
        local_learning_rate=dict(
            values=[
                1.0,
                0.5,
                0.3,
                0.1,
            ]
        ),
        beta=dict(values=[0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]),
    ),
)



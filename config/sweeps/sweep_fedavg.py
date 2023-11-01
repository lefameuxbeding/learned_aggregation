_base_ = ["./sweeps_base.py"]

optimizer = "fedavg"
task = "conv-c10"
num_inner_steps = 1000

sweep_config = dict(
    method="grid",
    metric=dict(name="test loss", goal="minimize"),
    parameters=dict(
        local_learning_rate=dict(
            values=[
                1,
                # 0.5,
                # 0.1,
                # 0.05,
                # 0.01,
                # 0.005,
                # 0.001,
                # 0.0005,
                # 0.0001,
                # 0.00005,
                # 0.00001,
            ]
        ),
    ),
)

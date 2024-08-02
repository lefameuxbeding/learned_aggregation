_base_ = ["./sweeps_base.py"]


num_inner_steps = 1000

sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        learning_rate=dict(
            values=[0.1,
                    0.0492,
                    0.0242,
                    0.0119,
                    0.00588,
                    0.00289,
                    0.00143,
                    0.000702,
                    0.000346,
                    0.00017,
                    8.38e-05,
                    4.12e-05,
                    2.03e-05,
                    1e-05]
        ),
        benchmark_b1=dict(
            values=[
                0.9,0.95,0.99,
            ]
        ),

        benchmark_b2=dict(
            values=[
                0.95,0.99,0.999,
            ]
        ),

        benchmark_weight_decay=dict(
            values=[
                1.0,
                0.1,
                0.01,
                0.001,
            ]
        ),
    ),
)

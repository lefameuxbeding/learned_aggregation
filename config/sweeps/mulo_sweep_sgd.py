_base_ = ["./sweeps_base.py"]


num_inner_steps = 1000

sweep_config = dict(
    method="grid",
    metric=dict(name="train loss", goal="minimize"),
    parameters=dict(
        learning_rate=dict(
            values=[1.0,
                    0.546,
                    0.298,
                    0.162,
                    0.0886,
                    0.0483,
                    0.0264,
                    0.0144,
                    0.00785,
                    0.00428,
                    0.00234,
                    0.00127,
                    0.000695,
                    0.000379,
                    0.000207,
                    0.000113,
                    6.16e-05,
                    3.36e-05,
                    1.83e-05,
                    1e-05]
        ),
        benchmark_momentum=dict(
            values=[0.0,
                    0.046,
                    0.091,
                    0.137,
                    0.183,
                    0.229,
                    0.274,
                    0.32,
                    0.366,
                    0.411,
                    0.457,
                    0.503,
                    0.549,
                    0.594,
                    0.64,
                    0.686,
                    0.731,
                    0.777,
                    0.823,
                    0.869,
                    0.914,
                    0.96,
                    0.99,
                    0.999,
                    0.9999]
        ),
    ),
)






import os
import pickle
import sys

import jax
from learned_optimization.learned_optimizers import adafac_mlp_lopt
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import (
    gradient_learner,
    lopt_truncated_step,
    truncated_pes,
    truncation_schedule,
)
from learned_optimization.tasks import base as tasks_base
from learned_optimization.tasks.fixed import image_mlp

import wandb


if __name__ == "__main__":
    """Environment"""

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")

    """Setup"""

    key = jax.random.PRNGKey(0)

    num_runs = 10
    num_inner_steps = 500
    num_outer_steps = 10000

    lopt = adafac_mlp_lopt.AdafacMLPLOpt()
    lopt_str = "lopt"

    meta_opt = opt_base.Adam(1e-4)

    """Meta-tasks"""

    def grad_est_fn(task_family):
        trunc_sched = truncation_schedule.LogUniformLengthSchedule(
            min_length=100, max_length=num_inner_steps
        )
        truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
            task_family,
            lopt,
            trunc_sched,
            num_tasks=16,
            random_initial_iteration_offset=num_inner_steps,
        )
        return truncated_pes.TruncatedPES(
            truncated_step=truncated_step, trunc_length=50
        )

    mlp_task_family = tasks_base.single_task_to_family(
        image_mlp.ImageMLP_FashionMnist_Relu128x128()
    )
    gradient_estimators = [
        grad_est_fn(mlp_task_family),
    ]

    outer_trainer = gradient_learner.SingleMachineGradientLearner(
        lopt, gradient_estimators, meta_opt
    )

    """Meta-train"""

    run = wandb.init(project="learned_aggregation", group=lopt_str)

    key, key1 = jax.random.split(key)
    outer_trainer_state = outer_trainer.init(key1)

    for _ in range(num_outer_steps):
        outer_trainer_state, meta_loss, _ = outer_trainer.update(
            outer_trainer_state, key, with_metrics=False
        )
        run.log({"meta loss": meta_loss})

    run.finish()

    with open(lopt_str + ".pickle", "wb") as f:
        pickle.dump(
            outer_trainer_state.gradient_learner_state.theta_opt_state.params, f
        )

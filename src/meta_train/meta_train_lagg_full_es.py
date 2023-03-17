import sys
import os
import jax
import pickle
import wandb

from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks.fixed import conv
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.outer_trainers import full_es
from learned_optimization.tasks import base as tasks_base
from learned_optimization.outer_trainers import gradient_learner

from ..learned_aggregators import mlp_lagg
from ..outer_trainers import lagg_truncated_step


if __name__ == "__main__":
    """Environment"""

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")

    """Setup"""

    key = jax.random.PRNGKey(0)

    num_runs = 10
    num_inner_steps = 500
    num_outer_steps = 10000

    meta_train_str = "FullES"

    lagg = mlp_lagg.MLPLAgg()
    lagg_str = "PerParamMLPAgg_" + str(lagg.num_grads) + "_" + meta_train_str

    meta_opt = opt_base.Adam(1e-4)

    """Meta-tasks"""

    def grad_est_fn(task_family):
        trunc_sched = truncation_schedule.NeverEndingTruncationSchedule()
        truncated_step = lagg_truncated_step.VectorizedLAggTruncatedStep(
            task_family,
            lagg,
            trunc_sched,
            num_tasks=16,
            random_initial_iteration_offset=num_inner_steps,
        )
        trunc_sched = truncation_schedule.ConstantTruncationSchedule(num_inner_steps)
        return full_es.FullES(
            truncated_step,
            truncation_schedule=trunc_sched
        )

    mlp_task_family = tasks_base.single_task_to_family(
        image_mlp.ImageMLP_FashionMnist_Relu128x128()
    )
    conv_task_family = tasks_base.single_task_to_family(
        conv.Conv_Cifar100_32x64x64()
    )
    gradient_estimators = [
        grad_est_fn(mlp_task_family),
        grad_est_fn(conv_task_family),
    ]

    outer_trainer = gradient_learner.SingleMachineGradientLearner(
        lagg, gradient_estimators, meta_opt
    )

    """Meta-train"""

    run = wandb.init(project="learned_aggregation", group=lagg_str)

    key, key1 = jax.random.split(key)
    outer_trainer_state = outer_trainer.init(key1)

    for _ in range(num_outer_steps):
        outer_trainer_state, meta_loss, _ = outer_trainer.update(
            outer_trainer_state, key, with_metrics=False
        )
        run.log({meta_train_str + " meta loss 2": meta_loss})

    run.finish()

    with open(lagg_str + ".pickle", "wb") as f:
        pickle.dump(
            outer_trainer_state.gradient_learner_state.theta_opt_state.params, f
        )

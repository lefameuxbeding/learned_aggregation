import wandb

from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks import quadratics
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.tasks import base as tasks_base
from learned_optimization.outer_trainers import gradient_learner


def meta_train_lopt(lopt, lopt_str, key, inner_steps, outer_steps):
    meta_opt = opt_base.Adam(1e-4)

    trunc_sched = truncation_schedule.LogUniformLengthSchedule(
        min_length=100, max_length=inner_steps
    )

    def grad_est_fn(task_family):
        truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
            task_family,
            lopt,
            trunc_sched,
            num_tasks=16,
            random_initial_iteration_offset=inner_steps,
        )
        return truncated_pes.TruncatedPES(
            truncated_step=truncated_step, trunc_length=50
        )

    mlp_task_family = tasks_base.single_task_to_family(
        image_mlp.ImageMLP_FashionMnist8_Relu32()
    )
    gradient_estimators = [
        grad_est_fn(quadratics.FixedDimQuadraticFamily(10)),
        grad_est_fn(mlp_task_family),
    ]

    outer_trainer = gradient_learner.SingleMachineGradientLearner(
        lopt, gradient_estimators, meta_opt
    )
    outer_trainer_state = outer_trainer.init(key)

    run = wandb.init(project="learned_aggregation", group=lopt_str)

    for _ in range(outer_steps):
        outer_trainer_state, meta_loss, _ = outer_trainer.update(
            outer_trainer_state, key, with_metrics=False
        )
        run.log({"meta loss": meta_loss})

    run.finish()

    return outer_trainer_state.gradient_learner_state.theta_opt_state.params


def meta_train_lagg(lopt, lopt_str, key, inner_steps, outer_steps):
    return lopt.init(key)

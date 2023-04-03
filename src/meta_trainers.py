from learned_optimization.learned_optimizers import adafac_mlp_lopt
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import (
    gradient_learner,
    lopt_truncated_step,
    truncated_pes,
    truncation_schedule,
)
from learned_optimization.tasks import base as tasks_base

from adafac_mlp_lagg import AdafacMLPLAgg
from lagg_truncated_step import VectorizedLAggTruncatedStep
from tasks import get_task


def _lagg_meta_trainer(args):
    lagg = AdafacMLPLAgg()

    meta_opt = opt_base.Adam(1e-4)

    def grad_est_fn(task_family):
        trunc_sched = truncation_schedule.LogUniformLengthSchedule(
            min_length=100, max_length=args.num_inner_steps
        )
        truncated_step = VectorizedLAggTruncatedStep(
            task_family,
            lagg,
            trunc_sched,
            num_tasks=16,
            random_initial_iteration_offset=args.num_inner_steps,
        )
        return truncated_pes.TruncatedPES(
            truncated_step=truncated_step, trunc_length=50
        )

    task_family = tasks_base.single_task_to_family(get_task(args))
    gradient_estimators = [
        grad_est_fn(task_family),
    ]

    meta_trainer = gradient_learner.SingleMachineGradientLearner(
        lagg, gradient_estimators, meta_opt
    )

    return meta_trainer, "lagg_" + str(lagg.num_grads)


def _lopt_meta_trainer(args):
    lopt = adafac_mlp_lopt.AdafacMLPLOpt()

    meta_opt = opt_base.Adam(1e-4)

    def grad_est_fn(task_family):
        trunc_sched = truncation_schedule.LogUniformLengthSchedule(
            min_length=100, max_length=args.num_inner_steps
        )
        truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
            task_family,
            lopt,
            trunc_sched,
            num_tasks=16,
            random_initial_iteration_offset=args.num_inner_steps,
        )
        return truncated_pes.TruncatedPES(
            truncated_step=truncated_step, trunc_length=50
        )

    task_family = tasks_base.single_task_to_family(get_task(args))
    gradient_estimators = [
        grad_est_fn(task_family),
    ]

    meta_trainer = gradient_learner.SingleMachineGradientLearner(
        lopt, gradient_estimators, meta_opt
    )

    return meta_trainer, "lopt"


def get_meta_trainer(args):
    meta_trainers = {
        "lopt": _lopt_meta_trainer,
        "lagg": _lagg_meta_trainer,
    }

    return meta_trainers[args.optimizer](args)

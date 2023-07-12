from learned_optimization.learned_optimizers.adafac_mlp_lopt import AdafacMLPLOpt
from learned_optimization.learned_optimizers.mlp_lopt import MLPLOpt
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import (
    gradient_learner,
    truncated_pes,
    truncation_schedule,
)
from learned_optimization.tasks import base as tasks_base

from adafac_mlp_lagg import AdafacMLPLAgg
from fedlagg_truncated_step import VectorizedFedLAggTruncatedStep
from fedlopt_truncated_step import VectorizedFedLOptTruncatedStep
from mlp_lagg import MLPLAgg
from tasks import get_task


def _fedlopt_meta_trainer(args):
    lopt_class = AdafacMLPLOpt if args.optimizer == "fedlopt-adafac" else MLPLOpt
    lopt = lopt_class(hidden_size=args.hidden_size)

    meta_opt = opt_base.Adam(args.learning_rate)

    def grad_est_fn(task_family):
        trunc_sched = truncation_schedule.LogUniformLengthSchedule(
            min_length=100, max_length=args.num_inner_steps
        )
        truncated_step = VectorizedFedLOptTruncatedStep(
            task_family,
            lopt,
            trunc_sched,
            num_tasks=8,
            random_initial_iteration_offset=args.num_inner_steps,
            local_learning_rate= args.local_learning_rate,
            num_local_steps=args.num_local_steps,
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

    return meta_trainer


def _fedlagg_meta_trainer(args):
    lagg_class = AdafacMLPLAgg if args.optimizer == "fedlagg-adafac" else MLPLAgg
    with_avg = True if args.optimizer == "fedlagg-wavg" else False
    lagg = lagg_class(
        num_grads=args.num_grads, hidden_size=args.hidden_size, with_avg=with_avg
    )

    meta_opt = opt_base.Adam(args.learning_rate)

    def grad_est_fn(task_family):
        trunc_sched = truncation_schedule.LogUniformLengthSchedule(
            min_length=100, max_length=args.num_inner_steps
        )
        truncated_step = VectorizedFedLAggTruncatedStep(
            task_family,
            lagg,
            trunc_sched,
            num_tasks=8,
            random_initial_iteration_offset=args.num_inner_steps,
            local_learning_rate= args.local_learning_rate,
            num_local_steps=args.num_local_steps,
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

    return meta_trainer


def get_meta_trainer(args):
    meta_trainers = {
        "fedlopt": _fedlopt_meta_trainer,
        "fedlopt-adafac": _fedlopt_meta_trainer,
        "fedlagg": _fedlagg_meta_trainer,
        "fedlagg-wavg": _fedlagg_meta_trainer,
        "fedlagg-adafac": _fedlagg_meta_trainer,
    }

    return meta_trainers[args.optimizer](args)

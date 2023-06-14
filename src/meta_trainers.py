from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import (
    gradient_learner,
    truncated_pes,
    truncation_schedule,
)
from learned_optimization.tasks import base as tasks_base

from adafac_mlp_lagg import AdafacMLPLAgg
from fedlagg_truncated_step import VectorizedFedLAggTruncatedStep
from mlp_lagg import MLPLAgg
from tasks import get_task


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

    return (
        meta_trainer,
        args.optimizer
        + str(args.hidden_size)
        + "_"
        + args.task
        + "_K"
        + str(args.num_grads)
        + "_H"
        + str(args.num_local_steps)
        + "_"
        + str(args.local_learning_rate),
    )


def get_meta_trainer(
    args,
):  # TODO Find a better way to organize this since we're only using a single function each time
    meta_trainers = {
        "fedlagg": _fedlagg_meta_trainer,
        "fedlagg-wavg": _fedlagg_meta_trainer,
        "fedlagg-adafac": _fedlagg_meta_trainer,
    }

    return meta_trainers[args.optimizer](args)

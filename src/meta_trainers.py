from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import (
    gradient_learner,
    truncated_pes,
    truncation_schedule,
)
from learned_optimization.tasks import base as tasks_base

from fed_adafac_mlp_lopt import FedAdafacMLPLOpt
from fed_truncated_step import VectorizedFedLOptTruncatedStep
from fed_mlp_lopt import FedMLPLOpt
from tasks import get_task

import optax
from optimizers import AdamWLinearCosine, AdamW


def _fedlagg_meta_trainer(args):
    lagg_class = (
        FedAdafacMLPLOpt
        if args.optimizer in ["fedlopt-adafac", "fedlagg-adafac"]
        else FedMLPLOpt
    )
    with_all_grads = (
        True
        if args.optimizer in ["fedlagg", "fedlagg-wavg", "fedlagg-adafac"]
        else False
    )
    with_avg = (
        True
        if args.optimizer in ["fedlopt", "fedlopt-adafac", "fedlagg-wavg"]
        else False
    )
    lagg = lagg_class(
        num_grads=args.num_grads,
        hidden_size=args.hidden_size,
        with_all_grads=with_all_grads,
        with_avg=with_avg,
    )

    if args.schedule != {}:
        print("Using learning rate scheduler")
        if args.schedule.get("use_adamw", False):
            del args.schedule["use_adamw"]
            meta_opt = AdamW(**args.schedule)
        else:
            meta_opt = AdamWLinearCosine(**args.schedule)
    else:
        meta_opt = opt_base.Adam(args.learning_rate)

    def grad_est_fn(task_family):
        trunc_sched = truncation_schedule.LogUniformLengthSchedule(
            min_length=100, max_length=args.num_inner_steps
        )
        truncated_step = VectorizedFedLOptTruncatedStep(
            task_family,
            lagg,
            trunc_sched,
            num_tasks=args.num_tasks,
            random_initial_iteration_offset=args.num_inner_steps,
            local_learning_rate=args.local_learning_rate,
            num_local_steps=args.num_local_steps,
            meta_loss_split=args.meta_loss_split,
        )

        if args.use_pmap:
            return truncated_pes.TruncatedPESPMAP(
                truncated_step=truncated_step,
                trunc_length=50,
                num_devices=args.num_devices,
            )
        else:
            return truncated_pes.TruncatedPES(
                truncated_step=truncated_step, trunc_length=50
            )

    tasks = get_task(args)

    if type(tasks) is list:
        gradient_estimators = [
            grad_est_fn(tasks_base.single_task_to_family(task)) for task in tasks
        ]
    else:
        task_family = tasks_base.single_task_to_family(tasks)
        gradient_estimators = [
            grad_est_fn(task_family),
        ]

    meta_trainer = gradient_learner.SingleMachineGradientLearner(
        lagg, gradient_estimators, meta_opt
    )

    return meta_trainer, meta_opt


def get_meta_trainer(args):
    meta_trainers = {
        "fedlopt": _fedlagg_meta_trainer,
        "fedlopt-adafac": _fedlagg_meta_trainer,
        "fedlagg": _fedlagg_meta_trainer,
        "fedlagg-wavg": _fedlagg_meta_trainer,
        "fedlagg-adafac": _fedlagg_meta_trainer,
    }

    return meta_trainers[args.optimizer](args)  # TODO Find better way to do this

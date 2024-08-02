from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import (
    gradient_learner,
    truncated_pes,
    truncation_schedule,
)
from learned_optimization.tasks import base as tasks_base
from learned_optimization.outer_trainers.lopt_truncated_step import VectorizedLOptTruncatedStep
from learned_optimization.learned_optimizers.adafac_mlp_lopt import AdafacMLPLOpt

from fed_adafac_mlp_lopt import FedAdafacMLPLOpt
from fed_truncated_step import VectorizedFedLOptTruncatedStep
from fed_mlp_lopt import FedMLPLOpt
from tasks import get_task
import jax
import optax
from optimizers import AdamWLinearCosine, AdamW
from mup_adafac_mlp_lopt import MuAdafacMLPLOpt
import pickle
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import abc
import functools
import time
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import gin
import jax
import jax.numpy as jnp
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.optimizers import base as opt_base
import numpy as onp
from mup_hyper import MuHyperV2

from learned_optimization.outer_trainers.gradient_learner import WorkerWeights, GradientEstimator, GradientEstimatorState, \
    WorkerComputeOut, AggregatedGradient, SingleMachineState, _tree_zeros_on_device, _nan_to_num

PRNGKey = jnp.ndarray

@gin.configurable
@profile.wrap()
def gradient_worker_compute_distributed(
    worker_weights: WorkerWeights,
    gradient_estimators: Sequence[GradientEstimator],
    unroll_states: Sequence[GradientEstimatorState],
    key: PRNGKey,
    with_metrics: bool,
    clip_nan_loss_to_value: Optional[float] = 20.0,
    extra_metrics: bool = True,
    device: Optional[jax.lib.xla_client.Device] = None) -> WorkerComputeOut:
  """Compute a gradient signal to meta-train with.

  This function performs unrolls for each of the unroll_states with the
  corresponding gradient_estimator. The results from each of the gradient
  estimators get's merged into a single gradient. This aggregation is done
  to save bandwidth when collecting gradients from workers.

  Args:
    worker_weights: Weights created by the GradientLearner and represent the
      current parameters and model state of the learned optimizer.
    gradient_estimators: The gradient estimators used to update the unroll state
    unroll_states: state of the gradient estimator (e.g. inner problem weights)
    key: jax rng
    with_metrics: compute with summary metrics or not
    clip_nan_loss_to_value: float, value to set nan losses to
    extra_metrics: log out additional metrics.
    device: The jax device to run the computation on

  Returns:
    worker_compute_out: The results of the computation.
      This contains a gradient estimate, the next unroll states, metrics.
      A subset of which get passed to the GradientLearner.
  """
  print("in distributed gradient worker compute")
  if device is None:
    device = jax.local_devices(0)[0]

  theta = worker_weights.theta
  theta_model_state = worker_weights.theta_model_state

  theta_shape = jax.tree_util.tree_map(
      lambda x: jax.core.ShapedArray(x.shape, x.dtype), theta)
  grads_accum = _tree_zeros_on_device(theta_shape, device)

  metrics_list = []
  unroll_states_out = []
  losses = []
  event_info = []
  
  assert len(gradient_estimators) == len(unroll_states)

  for si, (estimator,
           unroll_state) in enumerate(zip(gradient_estimators, unroll_states)):
    with profile.Profile(f"estimator{si}"):
      stime = time.time()
      key, rng = jax.random.split(key)

      cfg_name = estimator.cfg_name()

      logging.info(
          "compute_gradient_estimate for estimator name %s and cfg name %s",
          estimator.task_name(), estimator.cfg_name())
      with profile.Profile(f"unroll__metrics{with_metrics}"):
        # print("\n\n before estimator.compute_gradient_estimate()\n\n")
        estimator_out, metrics = estimator.compute_gradient_estimate(
            worker_weights, rng, unroll_state, with_summary=with_metrics)
        # print("\n\n after estimator.compute_gradient_estimate()\n\n")

      unroll_states_out.append(estimator_out.unroll_state)
      losses.append(estimator_out.mean_loss)
      with profile.Profile("tree_add"):
        grads_accum = tree_utils.tree_add(grads_accum, estimator_out.grad)

      # grab a random iteration from the trajectory
      if estimator_out.unroll_info:
        idx = onp.random.randint(0, len(estimator_out.unroll_info.loss))

        def extract_one(idx, x):
          return x[idx] if x is not None else None

        fn = functools.partial(extract_one, idx)
        onp_task_params = jax.tree_util.tree_map(
            onp.asarray, estimator_out.unroll_info.task_param)
        iteration = estimator_out.unroll_info.iteration[
            idx] if estimator_out.unroll_info.iteration is not None else None
        event_info.append({
            "loss": estimator_out.unroll_info.loss[idx, :],
            "task_param": jax.tree_util.tree_map(fn, onp_task_params),
            "iteration": iteration,
            "outer_iteration": worker_weights.outer_state.outer_iteration,
        })
      else:
        logging.warn("No out specified by learner. "
                     "Not logging any events data.")

      metrics = {k: v for k, v in metrics.items()}
      if extra_metrics:
        family_name = estimator.task_name()
        cfg_name = estimator.cfg_name()
        if with_metrics:
          # Metrics don't take into account which task they are comming from.
          # Let's add additional metrics with the task name pulled out.
          with profile.Profile("metric_computation"):
            keys = list(metrics.keys())
            for k in keys:
              v = metrics[k]
              assert "||" in k, f"bad metric format? Got: {k}"
              agg, name = k.split("||")
              metrics[f"{agg}||{family_name}/{name}"] = v
              metrics[f"{agg}||{cfg_name}/{name}"] = v

            mean_abs = tree_utils.tree_mean_abs(estimator_out.grad)
            metrics[f"mean||{family_name}/grad_mean_abs"] = mean_abs
            metrics[f"mean||{cfg_name}/grad_mean_abs"] = mean_abs

            norm = tree_utils.tree_norm(estimator_out.grad)
            metrics[f"mean||{family_name}/grad_norm"] = norm
            metrics[f"mean||{cfg_name}/grad_norm"] = norm
        metrics[f"mean||{family_name}/mean_loss"] = estimator_out.mean_loss
        metrics[f"mean||{cfg_name}/mean_loss"] = estimator_out.mean_loss
        metrics[f"sample||{family_name}/time"] = time.time() - stime
        metrics[f"sample||{cfg_name}/time"] = time.time() - stime

      metrics_list.append(metrics)


  # where communication should happen

  with profile.Profile("mean_grads"):
    grads_accum = tree_utils.tree_div(grads_accum, len(gradient_estimators) * jax.process_count() )
    print("before AR", jax.tree_map(lambda x: jax.numpy.mean(x), grads_accum))
    grads_accum = jax.lax.psum(grads_accum)
    print("after AR", jax.tree_map(lambda x: jax.numpy.mean(x), grads_accum))

    mean_loss = jnp.mean(jnp.asarray(losses))

  exit(0)

  # block here to better account for costs with profile profiling.
  with profile.Profile("blocking"):
    stime = time.time()
    mean_loss.block_until_ready()
    block_time = time.time() - stime

  with profile.Profile("summary_aggregation"):
    metrics = summary.aggregate_metric_list(metrics_list)
  metrics["mean||block_time"] = block_time

  with profile.Profile("strip_nan"):
    # this should ideally never be NAN
    # TODO(lmetz) check if we need these checks.
    grads_accum = _nan_to_num(grads_accum, 0.0, use_jnp=True)
    if clip_nan_loss_to_value:
      mean_loss = _nan_to_num(mean_loss, clip_nan_loss_to_value, use_jnp=True)

  with profile.Profile("grads_to_onp"):
    to_put = AggregatedGradient(
        theta_grads=grads_accum,
        theta_model_state=theta_model_state,
        mean_loss=mean_loss)

    return WorkerComputeOut(
        to_put=jax.tree_util.tree_map(onp.asarray, to_put),
        unroll_states=unroll_states_out,
        metrics=metrics,
        event_info=event_info)



class EnhancedSingleMachineGradientLearner(gradient_learner.SingleMachineGradientLearner):
  """Train with gradient estimators on a single machine.

  This is a convience wrapper calling the multi-worker interface -- namley
  both `GradientLearner` and `gradient_worker_compute`.
  """

  def __init__(self,
               meta_init: gradient_learner.MetaInitializer,
               gradient_estimators: Sequence[gradient_learner.GradientEstimator],
               theta_opt: opt_base.Optimizer,
               init_theta_from_path: Optional[str] = None,
               num_steps: Optional[int] = None,
               device: Optional[jax.lib.xla_client.Device] = None):
    """Initializer.

    Args:
      meta_init: Class containing an init function to construct outer params.
      gradient_estimators: Sequence of gradient estimators used to calculate
        gradients.
      theta_opt: The optimizer used to train the weights of the learned opt.
      num_steps: Number of meta-training steps used by optimizer for schedules.
    """
    self.gradient_learner = gradient_learner.GradientLearner(
        meta_init, theta_opt, num_steps=num_steps, init_theta_from_path=init_theta_from_path)
    self.gradient_estimators = gradient_estimators

#   def update(
#       self,
#       state,
#       key: PRNGKey,
#       with_metrics: Optional[bool] = False
#   ) -> Tuple[SingleMachineState, jnp.ndarray, Mapping[str, jnp.ndarray]]:
#     """Perform one outer-update to train the learned optimizer.

#     Args:
#       state: State of this class
#       key: jax rng
#       with_metrics: To compute metrics or not

#     Returns:
#       state: The next state from this class
#       loss: loss from the current iteration
#       metrics: dictionary of metrics computed
#     """
#     # print("in update()\n\n")
#     key1, key2 = jax.random.split(key)
#     worker_weights = self.gradient_learner.get_state_for_worker(
#         state.gradient_learner_state)
    
#     # print("\n\nbefore grad worker compute\n\n")

#     #this  is where we perform the full unroll
#     worker_compute_out = gradient_worker_compute_distributed(
#         worker_weights,
#         self.gradient_estimators,
#         state.gradient_estimator_states,
#         key=key1,
#         with_metrics=with_metrics)
#     # print("\n\nafter grad worker compute\n\n")

#     next_gradient_estimator_states = worker_compute_out.unroll_states

#     next_theta_state, metrics = self.gradient_learner.update(
#         state.gradient_learner_state, [worker_compute_out.to_put],
#         key=key2,
#         with_metrics=with_metrics)

#     metrics = summary.aggregate_metric_list(
#         [worker_compute_out.metrics, metrics])

#     return (SingleMachineState(
#         gradient_learner_state=next_theta_state,
#         gradient_estimator_states=next_gradient_estimator_states),
#             worker_compute_out.to_put.mean_loss, metrics)
  



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
            min_length=args.truncation_schedule_min_length, 
            max_length=args.num_inner_steps
        )
        truncated_step = VectorizedFedLOptTruncatedStep(
            task_family=task_family,
            learned_opt=lagg,
            trunc_sched=trunc_sched,
            num_tasks=args.num_tasks,
            meta_loss_split=args.meta_loss_split,
            random_initial_iteration_offset=50,#args.num_inner_steps,
            outer_data_split="train",
            meta_loss_with_aux_key=None,
            local_learning_rate=args.local_learning_rate,
            task_name=task_family.datasets.extra_info['name'],
            num_local_steps=args.num_local_steps,
            keep_batch_in_gpu_memory=args.keep_batch_in_gpu_memory,
        )

        return truncated_pes.TruncatedPES(
            # num_devices=2,
            truncated_step=truncated_step, 
            trunc_length=50,
            std=0.01,
            steps_per_jit=args.steps_per_jit,
            stack_antithetic_samples= False, #default
            sign_delta_loss_scalar= None, #default
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
        meta_init=lagg, 
        gradient_estimators=gradient_estimators, 
        theta_opt=meta_opt, 
        device=jax.local_devices(0)[0]
    )

    return meta_trainer, meta_opt



def _default_meta_trainer(args):
    if args.optimizer.lower() == "MuHyperV2".lower():
        print("\n\n loading MuHyperV2 \n\n")
        lopt = MuHyperV2(
            lstm_hidden_size=128,
            ff_hidden_size=4,
            ff_hidden_layers=2,
            initial_momentum_decays=(0.9, 0.99, 0.999),
            initial_rms_decays=(0.999,),
            initial_adafactor_decays=(0.9, 0.99, 0.999),
            param_inits=64,
            mix_layers=True,
            exp_mult=0.001,
            step_mult=args.adafac_step_mult,
            validation_mode=False,
            with_validation_feature_dim=False,)
    
    elif 'mup' in args.optimizer:

        lopt = MuAdafacMLPLOpt(exp_mult=0.001,
                            step_mult=args.adafac_step_mult,
                            hidden_size=args.hidden_size,
                            hidden_layers=2,
                            initial_momentum_decays=(0.9, 0.99, 0.999),
                            initial_rms_decays=(0.999,),
                            initial_adafactor_decays=(0.9, 0.99, 0.999),
                            concat_weights=True,
                            make_separate_weights=False,
                            split_weights=False,
                            clip_grad=args.lo_clip_grad,)
                            # mup_lrs=args.runtime_mup_lrs)

    else:
        
        lopt = AdafacMLPLOpt(exp_mult=0.001,
                            step_mult=0.001,
                            hidden_size=args.hidden_size,
                            hidden_layers=2,
                            initial_momentum_decays=(0.9, 0.99, 0.999),
                            initial_rms_decays=(0.999,),
                            initial_adafactor_decays=(0.9, 0.99, 0.999),
                            concat_weights=True,
                            make_separate_weights=False,
                            split_weights=False,
                            clip_grad=args.lo_clip_grad,)
        
    # if args.start_from_test_ckpt:
    #     with open(args.test_checkpoint, "rb") as f:
    #         meta_params = pickle.load(f)
        
    #     import pdb; pdb.set_trace()
    #     lopt = lopt.opt_fn(meta_params)

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
            min_length=args.truncation_schedule_min_length, 
            max_length=args.num_inner_steps
        )
        truncated_step = VectorizedLOptTruncatedStep(
            task_family=task_family,
            learned_opt=lopt,
            trunc_sched=trunc_sched,
            num_tasks=args.num_tasks,
            meta_loss_split=args.meta_loss_split,
            random_initial_iteration_offset=50,#args.num_inner_steps,
            outer_data_split="train",
            meta_loss_with_aux_key=None,
            task_name=task_family.datasets.extra_info['name'],
        )

        return truncated_pes.TruncatedPES(
            # num_devices=2,
            truncated_step=truncated_step, 
            trunc_length=args.truncation_length,
            std=0.01,
            steps_per_jit=args.steps_per_jit,
            stack_antithetic_samples= False, #default
            sign_delta_loss_scalar= None, #default
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

    meta_trainer = EnhancedSingleMachineGradientLearner(
        meta_init=lopt, 
        gradient_estimators=gradient_estimators,
        init_theta_from_path=args.test_checkpoint,
        theta_opt=meta_opt, 
        device=jax.local_devices(0)[0]
    )

    return meta_trainer, meta_opt


def get_meta_trainer(args):
    meta_trainers = {
        "fedlopt": _fedlagg_meta_trainer,
        "fedlopt-adafac": _fedlagg_meta_trainer,
        "fedlagg": _fedlagg_meta_trainer,
        "fedlagg-wavg": _fedlagg_meta_trainer,
        "fedlagg-adafac": _fedlagg_meta_trainer,
        'small_fc_mlp': _default_meta_trainer,
        'mup_small_fc_mlp': _default_meta_trainer,
        "MuHyperV2": _default_meta_trainer,
    }

    return meta_trainers[args.optimizer](args)  # TODO Find better way to do this

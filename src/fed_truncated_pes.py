# Adapted from https://github.com/google/learned_optimization/blob/main/learned_optimization/outer_trainers/truncated_pes.py and https://github.com/google/learned_optimization/blob/main/learned_optimization/outer_trainers/common.py

import gin
import jax
import jax.numpy as jnp
import haiku as hk
import chex

from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers.truncated_pes import (
    truncated_step_mod,
    PRNGKey,
    PESWorkerState,
    compute_pes_grad
)
from learned_optimization import profile
from learned_optimization.outer_trainers import common
from learned_optimization import summary
from learned_optimization import jax_utils
from learned_optimization import tree_utils

from typing import Optional, Sequence, Any, Mapping, Tuple
import functools

import globals


@functools.partial(
    jax.jit,
    static_argnames=("truncated_step", "with_summary", "unroll_length",
                     "theta_is_vector", "wrap_step_fn"),
)
@functools.partial(summary.add_with_summary, static_argnums=(0, 1, 2, 3, 9))
def truncated_unroll(
    truncated_step: truncated_step_mod.VectorizedTruncatedStep,
    unroll_length: int,
    theta_is_vector: bool,
    theta: common.MetaParams,
    key: chex.PRNGKey,
    state: common.UnrollState,
    datas: Any,
    outer_state: Any,
    override_num_steps: Optional[int] = None,
    p = None,
    with_summary: bool = False,  # used by add_with_summary. pylint: disable=unused-argument
    wrap_step_fn: Optional[Any] = None,
) -> Tuple[Tuple[common.UnrollState, truncated_step_mod.TruncatedUnrollOut], Mapping[
    str, jnp.ndarray]]:
  """Unroll train a single state some number of steps."""

  if jax.tree_util.tree_leaves(datas):
    assert tree_utils.first_dim(datas) == unroll_length, (
        f"got a mismatch in data size. Expected to have data of size: {unroll_length} "
        f"but got data of size {tree_utils.first_dim(datas)}")

  def step_fn(state, xs):
    key, data = xs
    if override_num_steps is not None:
      extra_kwargs = {"override_num_steps": override_num_steps}
    else:
      extra_kwargs = {}

    s, c_s = state
    # print("inside scan before unroll_step()")
    (state, clients_state), outs = truncated_step.unroll_step(
        theta,
        s,
        key,
        data,
        outer_state=outer_state,
        theta_is_vector=theta_is_vector,
        clients_state=c_s,
        **extra_kwargs)
    # print("inside scan after unroll_step()")
    return (state, clients_state), outs
  
  key_and_data = jax.random.split(key, unroll_length), datas
  if wrap_step_fn is not None:
    step_fn = wrap_step_fn(step_fn)

  initial_clients_state = jax.tree_util.tree_map(lambda x : jnp.array([jnp.array([jnp.zeros_like(x) for _ in range(globals.num_grads)]) for _ in range(globals.num_tasks)]), p)

  # print("before jax.lax.scan(step_fn)")
  (state, _), ys = jax.lax.scan(step_fn, (state, initial_clients_state), key_and_data)
  # print("after jax.lax.scan(step_fn)")
  return state, ys


def maybe_stacked_es_unroll(
    truncated_step: truncated_step_mod.VectorizedTruncatedStep,
    unroll_length: int,
    stack_antithetic_samples: bool,
    vec_p_theta: Any,
    vec_n_theta: Any,
    p_state: common.UnrollState,
    n_state: common.UnrollState,
    key: chex.PRNGKey,
    datas: Any,
    outer_state: Any,
    with_summary: bool = False,
    sample_rng_key: Optional[chex.PRNGKey] = None,
    override_num_steps: Optional[int] = None,
    p = None,
) -> Tuple[common.UnrollState, common.UnrollState, truncated_step_mod.TruncatedUnrollOut,
           truncated_step_mod.TruncatedUnrollOut, Mapping[str, jnp.ndarray]]:
  """Run's truncated_unroll one time with stacked antithetic samples or 2x."""
  theta_is_vector = True
  static_args = [
      truncated_step,
      unroll_length,
      theta_is_vector,
  ]
  # we provide 2 ways to compute the antithetic unrolls:
  # First, we stack the positive and negative states and compute things
  # in parallel
  # Second, we do this serially in python.
  # (lmetz) this assumes that the truncated functions operate on batches
  # of tasks. Somehow assert this.
  if stack_antithetic_samples:

    (pn_state, pn_ys), m = truncated_unroll(  # pylint: disable=unbalanced-tuple-unpacking,unexpected-keyword-arg,redundant-keyword-arg
        *(static_args + [
            common._stack(vec_p_theta, vec_n_theta),
            key,
            common._stack(p_state, n_state),
            common._stack(datas, datas, axis=1),
            outer_state,
            override_num_steps,
            p,
        ]),
        with_summary=with_summary,
        sample_rng_key=sample_rng_key)
    p_state, n_state = common._split_tree(pn_state)
    p_ys, n_ys = common._split_tree(pn_ys, axis=1)
  else:
    (p_state, p_ys), m = truncated_unroll(  # pylint: disable=unbalanced-tuple-unpacking,unexpected-keyword-arg,redundant-keyword-arg
        *(static_args +
          [vec_p_theta, key, p_state, datas, outer_state, override_num_steps, p]),
        with_summary=with_summary,
        sample_rng_key=sample_rng_key)
    (n_state, n_ys), _ = truncated_unroll(  # pylint: disable=unbalanced-tuple-unpacking,unexpected-keyword-arg,redundant-keyword-arg
        *(static_args +
          [vec_n_theta, key, n_state, datas, outer_state, override_num_steps, p]),
        with_summary=False)

  return p_state, n_state, p_ys, n_ys, m



@gin.configurable
class FedTruncatedPES(gradient_learner.GradientEstimator):
  """GradientEstimator for computing PES gradient estimates.

  Persistent Evolution Strategies (PES) is a gradient estimation technique
  for computing unbiased gradients in a unrolled computation graph. It does this
  by building of of Evolutionary Strategies but additionally keeping a running
  buffer of all the previously used perturbations. See the paper for more
  details (http://proceedings.mlr.press/v139/vicol21a.html).

  In practice, PES is higher variance than pure truncated ES but lower bias.
  """

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      trunc_length=10,
      std=0.01,
      steps_per_jit=10,
      stack_antithetic_samples: bool = False,
      sign_delta_loss_scalar: Optional[float] = None,
      p = None,
  ):
    self.truncated_step = truncated_step
    self.std = std

    self.trunc_length = trunc_length
    self.steps_per_jit = steps_per_jit
    self.stack_antithetic_samples = stack_antithetic_samples
    self.sign_delta_loss_scalar = sign_delta_loss_scalar
    
    self.p = p

    if self.trunc_length % self.steps_per_jit != 0:
      raise ValueError("Pass a trunc_length and steps_per_jit that are"
                       " multiples of each other.")

  def task_name(self) -> str:
    return self.truncated_step.task_name()

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> PESWorkerState:
    theta = worker_weights.theta

    pos_unroll_state = self.truncated_step.init_step_state(
        theta, worker_weights.outer_state, key, theta_is_vector=False)
    neg_unroll_state = pos_unroll_state

    accumulator = jax.tree_util.tree_map(
        lambda x: jnp.zeros([self.truncated_step.num_tasks] + list(x.shape)),
        theta)

    return PESWorkerState(
        pos_state=pos_unroll_state,
        neg_state=neg_unroll_state,
        accumulator=accumulator)

  @profile.wrap()
  def get_datas(self):
    return [
        self.truncated_step.get_batch(self.steps_per_jit)
        for _ in range(self.trunc_length // self.steps_per_jit)
    ]

  @profile.wrap()
  def compute_gradient_estimate(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state: PESWorkerState,
      with_summary: bool = False,
      datas_list: Optional[Sequence[Any]] = None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    p_state = state.pos_state
    n_state = state.neg_state
    accumulator = state.accumulator
    rng = hk.PRNGSequence(key)

    theta = worker_weights.theta

    vec_pos, vec_p_theta, vec_n_theta = common.vector_sample_perturbations(
        theta, next(rng), self.std, self.truncated_step.num_tasks)

    p_yses = []
    n_yses = []
    metrics = []

    # print("\nBefore maybe_stacked_es_unroll() loop\n")

    # (lmetz) consider switching this to be a jax.lax.scan when inside jit.
    for i in range(self.trunc_length // self.steps_per_jit):
      if datas_list is None:
        if jax_utils.in_jit():
          raise ValueError("Must pass data in when using a jit gradient est.")
        datas = self.truncated_step.get_batch(self.steps_per_jit)
      else:
        datas = datas_list[i]

      # force all to be non weak type. This is for cache hit reasons.
      # (lmetz) consider instead just setting the weak type flag?
      p_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       p_state)
      n_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       n_state)

      key = next(rng)

      p_state, n_state, p_ys, n_ys, m = maybe_stacked_es_unroll(
          self.truncated_step,
          self.steps_per_jit,
          self.stack_antithetic_samples,
          vec_p_theta,
          vec_n_theta,
          p_state,
          n_state,
          key,
          datas,
          worker_weights.outer_state,
          with_summary=with_summary,
          sample_rng_key=next(rng),
          p=self.p)

      metrics.append(m)
      p_yses.append(p_ys)
      n_yses.append(n_ys)

    # print("\nBefore maybe_stacked_es_unroll() loop\n")

    # print("\nBefore compute_pes_grad()\n")

    loss, es_grad, new_accumulator, p_ys, delta_loss = compute_pes_grad(
        p_yses,
        n_yses,
        accumulator,
        vec_pos,
        self.std,
        sign_delta_loss_scalar=self.sign_delta_loss_scalar)
    
    # print("\nafter compute_pes_grad()\n")

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        unroll_state=PESWorkerState(p_state, n_state, new_accumulator),
        unroll_info=unroll_info)

    metrics = summary.aggregate_metric_list(
        metrics, use_jnp=jax_utils.in_jit(), key=next(rng))
    if with_summary:
      metrics["sample||delta_loss_sample"] = summary.sample_value(
          key, jnp.abs(delta_loss))
      metrics["mean||delta_loss_mean"] = jnp.abs(delta_loss)
      if hasattr(p_state, "inner_step"):
        metrics["sample||inner_step"] = p_state.inner_step[0]
        metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics

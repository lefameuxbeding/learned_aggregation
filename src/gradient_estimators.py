import functools
from typing import Any, Mapping, Optional, Sequence, Tuple

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import jax_utils
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import gradient_learner

from learned_optimization.outer_trainers.truncated_pes import TruncatedPES, PESWorkerState, compute_pes_grad
import gin


PRNGKey = jnp.ndarray
MetaParams = Any
TruncatedUnrollState = Any

def flatten_dict(d, parent_key='', sep='/'):
    """
    Flattens a nested dictionary. Each key in the output dictionary is a concatenation
    of the keys from all levels in the input dictionary, joined by `sep`.

    Args:
    - d (dict): The dictionary to flatten.
    - parent_key (str, optional): The concatenated key constructed from the upper levels.
    - sep (str, optional): The separator used to join keys.

    Returns:
    - dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Recursive call for nested dictionaries
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)




@gin.configurable
class TruncatedPESLog(TruncatedPES):

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
    max_list = []
    min_list = []

    # print("\nBefore maybe_stacked_es_unroll() loop\n")

    # TODO(lmetz) consider switching this to be a jax.lax.scan when inside jit.
    for i in range(self.trunc_length // self.steps_per_jit):
      if datas_list is None:
        if jax_utils.in_jit():
          raise ValueError("Must pass data in when using a jit gradient est.")
        datas = self.truncated_step.get_batch(self.steps_per_jit)
      else:
        datas = datas_list[i]

      # force all to be non weak type. This is for cache hit reasons.
      # TODO(lmetz) consider instead just setting the weak type flag?
      p_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       p_state)
      n_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       n_state)

      key = next(rng)

      p_state, n_state, p_ys, n_ys, m = common.maybe_stacked_es_unroll(
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
          sample_rng_key=next(rng))
      
      max_ = jax.tree_map(lambda x,y : max([jnp.max(x),jnp.max(y)]),
                   n_state.inner_opt_state.log,
                   p_state.inner_opt_state.log)

      min_ = jax.tree_map(lambda x,y : min([jnp.min(x),jnp.min(y)]),
                   n_state.inner_opt_state.log,
                   p_state.inner_opt_state.log)
      
      max_list.append(max_)
      min_list.append(min_)

      metrics.append(m)
      p_yses.append(p_ys)
      n_yses.append(n_ys)

    loss, es_grad, new_accumulator, p_ys, delta_loss = compute_pes_grad(
        p_yses,
        n_yses,
        accumulator,
        vec_pos,
        self.std,
        sign_delta_loss_scalar=self.sign_delta_loss_scalar)

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
      max_ = functools.reduce(lambda x, y: jax.tree_map(max, x, y), max_list)
      min_ = functools.reduce(lambda x, y: jax.tree_map(max, x, y), min_list)
      metrics.update({'sample||'+k+'_max':v for k,v in flatten_dict(max_).items()})
      metrics.update({'sample||'+k+'_min':v for k,v in flatten_dict(min_).items()})

      if hasattr(p_state, "inner_step"):
        metrics["sample||inner_step"] = p_state.inner_step[0]
        metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics
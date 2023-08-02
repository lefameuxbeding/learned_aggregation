import numpy as np
import jax.numpy as jnp
import jax
import os

from matplotlib import pylab as plt

from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.outer_trainers import lopt_truncated_step

from learned_optimization.tasks import quadratics
from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks import base as tasks_base

from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import mlp_lopt
from learned_optimization.optimizers import base as opt_base

from learned_optimization import optimizers
from learned_optimization import training
from learned_optimization import eval_training

import haiku as hk
import tqdm

lopt = mlp_lopt.MLPLOpt()


theta_opt = opt_base.Adam(1e-3)
central_learner = gradient_learner.GradientLearner(lopt, theta_opt)

key = jax.random.PRNGKey(0)
central_state = central_learner.init(key)
jax.tree_util.tree_map(lambda x: jnp.asarray(x).shape, central_state)

# We can see here that this just contains the weights of the learned optimizer, plus the extra accumulators used by adam.
# Next, we can compute gradient estimators, but first we must get the required state from the learner.

worker_weights = central_learner.get_state_for_worker(central_state)


# Next, we can compute gradients on a given worker. As before we need to get a list of gradient estimators. We can use the same set we used before.

max_length = 300
trunc_sched = truncation_schedule.LogUniformLengthSchedule(
    min_length=100, max_length=max_length)


def grad_est_fn(task_family):
  trunc_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
      task_family,
      lopt,
      trunc_sched,
      num_tasks=16,
      random_initial_iteration_offset=max_length)
  return truncated_pes.TruncatedPES(trunc_step, trunc_length=50)


mlp_task_family = tasks_base.single_task_to_family(
    image_mlp.ImageMLP_FashionMnist8_Relu32())

gradient_estimators = [
    grad_est_fn(quadratics.FixedDimQuadraticFamily(10)),
    grad_est_fn(mlp_task_family)
]

#Next, we need to kick things off by first computing the initial states for each of the gradient estimators.
unroll_states = [
    grad.init_worker_state(worker_weights, key=jax.random.fold_in(key, i))
    for (i, grad) in enumerate(gradient_estimators)
]

#Next we can use these states to estimate a meta-gradient!
print("len(gradient_estimators)",len(gradient_estimators))
print("len(unroll_states)",len(unroll_states))


out = jax.pmap(gradient_learner.gradient_worker_compute)(
    jnp.array([worker_weights,worker_weights]),
    jnp.array(gradient_estimators),
    jnp.array(unroll_states),
    key=jnp.array([key,key]),
    with_metrics=jnp.array([False,False]))

#This produces a couple of different outputs bundled together in a dataclass.

print([x for x in dir(out) if not x.startswith("__") and x != "replace"])

exit(0)

# Most importantly we have to_put which contains information that should be sent to the central learner, and unroll_states which contains the next unroll states.

# Now with more than one worker, we would pass back a list of these gradients. In this demo, we will just use a single one, and pass 
# this directly into the central learner to get the next meta-iteration. With more workers, this would contain a different gradient estimator from each worker.


key1, key = jax.random.split(key)
grads_list = [out.to_put]
central_state, metrics = central_learner.update(central_state, grads_list, key=key1)

losses = []

outer_train_steps = int(os.environ.get("LOPT_META_TRAIN_LENGTH", 500))

for i in tqdm.trange(outer_train_steps):
  worker_weights = central_learner.get_state_for_worker(central_state)

  key1, key = jax.random.split(key)
  out = gradient_learner.gradient_worker_compute(
      worker_weights,
      gradient_estimators,
      unroll_states,
      key=key1,
      with_metrics=False)
  # extract the next unroll state output for the next iteration.
  unroll_states = out.unroll_states

  key1, key = jax.random.split(key)
  central_state, metrics = central_learner.update(
      central_state, [out.to_put], key=key1)
  losses.append(out.to_put.mean_loss)







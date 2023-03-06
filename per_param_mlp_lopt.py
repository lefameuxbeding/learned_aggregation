# Adapted from https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/Part6_custom_learned_optimizers.ipynb

import flax
from typing import Any
import jax
import jax.numpy as jnp
import haiku as hk

from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base


MetaParams = Any  # typing definition to label some types below


@flax.struct.dataclass
class LOptState:
    params: Any
    model_state: Any
    iteration: jnp.ndarray


@flax.struct.dataclass
class PerParamState:
    params: Any
    model_state: Any
    iteration: jnp.ndarray
    momentums: Any


class PerParamMLPLOpt(lopt_base.LearnedOptimizer):
    def __init__(self, decay=0.9, hidden_size=64):
        self.decay = decay
        self.hidden_size = hidden_size

        def forward(grads, momentum, params):
            features = jnp.asarray([params, momentum, grads])
            # transpose to have features dim last. The MLP will operate on this,
            # and treat the leading dimensions as a batch dimension.
            features = jnp.transpose(
                features, list(range(1, 1 + len(grads.shape))) + [0]
            )

            outs = hk.nets.MLP([self.hidden_size, 2])(features)

            scale = outs[..., 0]
            mag = outs[..., 1]
            # Compute a step as follows.
            return scale * 0.01 * jnp.exp(mag * 0.01)

        self.net = hk.without_apply_rng(hk.transform(forward))

    def init(self, key) -> MetaParams:
        """Initialize the weights of the learned optimizer."""
        # to initialize our neural network, we must pass in a batch that looks like
        # data we might train on.
        # Because we are operating per parameter, the shape of this batch doesn't
        # matter.
        fake_grads = fake_params = fake_mom = jnp.zeros([10, 10])
        return {"nn": self.net.init(key, fake_grads, fake_mom, fake_params)}

    def opt_fn(self, theta: MetaParams) -> opt_base.Optimizer:
        # define an anonymous class which implements the optimizer.
        # this captures over the meta-parameters, theta.

        parent = self

        class _Opt(opt_base.Optimizer):
            def init(self, params, model_state=None, **kwargs) -> LOptState:
                # In addition to params, model state, and iteration, we also need the
                # initial momentum values.

                momentums = jax.tree_util.tree_map(jnp.zeros_like, params)

                return PerParamState(
                    params=params,
                    model_state=model_state,
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                    momentums=momentums,
                )

            def update(
                self, opt_state: LOptState, grads, model_state=None, **kwargs
            ) -> LOptState:
                """Perform the actual update."""

                # update all the momentums
                def _update_one_momentum(m, g):
                    return m * parent.decay + (g * (1 - parent.decay))

                next_moms = jax.tree_util.tree_map(
                    _update_one_momentum, opt_state.momentums, grads
                )

                # Update all the params
                def _update_one(g, m, p):
                    step = parent.net.apply(theta["nn"], g, m, p)
                    return p - step

                next_params = jax.tree_util.tree_map(
                    _update_one, opt_state.params, grads, next_moms
                )

                # Pack the new parameters back up
                return PerParamState(
                    params=next_params,
                    model_state=model_state,
                    iteration=opt_state.iteration + 1,
                    momentums=next_moms,
                )

        return _Opt()

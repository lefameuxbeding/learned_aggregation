# Adapted from https://github.com/google/learned_optimization/blob/main/learned_optimization/learned_optimizers/mlp_lopt.py

from typing import Any, Optional
import gin
import haiku as hk
import jax
import jax.numpy as jnp

from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base

from learned_optimization.learned_optimizers.mlp_lopt import (
    PRNGKey,
    MLPLOptState,
    _tanh_embedding,
    _second_moment_normalizer,
)


@gin.configurable
class MLPLAgg(lopt_base.LearnedOptimizer):
    """Learned optimizer leveraging a per parameter MLP.
    This is also known as LOLv2.
    """

    def __init__(
        self,
        exp_mult=0.001,
        step_mult=0.001,
        hidden_size=32,
        hidden_layers=2,
        compute_summary=True,
        num_grads=4,  # TODO
    ):
        super().__init__()
        self._step_mult = step_mult
        self._exp_mult = exp_mult
        self._compute_summary = compute_summary
        self.num_grads = num_grads  # TODO

        def ff_mod(inp):
            return hk.nets.MLP([hidden_size] * hidden_layers + [2])(inp)

        self._mod = hk.without_apply_rng(hk.transform(ff_mod))

    def init(self, key: PRNGKey) -> lopt_base.MetaParams:
        # There are 19 features used as input. For now, hard code this.
        return self._mod.init(key, jnp.zeros([0, 19 - 1 + self.num_grads]))  # TODO

    def opt_fn(
        self, theta: lopt_base.MetaParams, is_training: bool = False
    ) -> opt_base.Optimizer:
        decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])

        mod = self._mod
        exp_mult = self._exp_mult
        step_mult = self._step_mult
        compute_summary = self._compute_summary

        class _Opt(opt_base.Optimizer):
            """Optimizer instance which has captured the meta-params (theta)."""

            def __init__(self, num_grads=4): # TODO
                self.num_grads = num_grads

            def init(
                self,
                params: lopt_base.Params,
                model_state: Any = None,
                num_steps: Optional[int] = None,
                key: Optional[PRNGKey] = None,
            ) -> MLPLOptState:
                """Initialize inner opt state."""

                return MLPLOptState(
                    params=params,
                    state=model_state,
                    rolling_features=common.vec_rolling_mom(decays).init(params),
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                )

            def update(
                self,
                opt_state: MLPLOptState,
                grads: Any,  # TODO
                loss: float,
                model_state: Any = None,
                is_valid: bool = False,
                key: Optional[PRNGKey] = None,
            ) -> MLPLOptState:
                
                # Makes sure we are feeding the correct number of sample gradients
                assert len(grads) == self.num_grads

                overall_grad = jax.tree_util.tree_map(lambda g, *gs : jnp.mean(jnp.array(gs + (g,))), grads[0], *grads[1:])
                next_rolling_features = common.vec_rolling_mom(decays).update(
                    opt_state.rolling_features, overall_grad  # TODO
                )

                training_step_feature = _tanh_embedding(opt_state.iteration)

                def _update_tensor(p, m, *gs):  # TODO
                    # this doesn't work with scalar parameters, so let's reshape.
                    if not p.shape:
                        p = jnp.expand_dims(p, 0)
                        g_list = []
                        for g in gs:
                            g_list.append(jnp.expand_dims(g, 0)) # TODO
                        m = jnp.expand_dims(m, 0)
                        did_reshape = True
                    else:
                        g_list = list(gs) # TODO
                        did_reshape = False

                    inps = []

                    # feature consisting of raw gradient values
                    for g in g_list:
                        batch_g = jnp.expand_dims(g, axis=-1)  # TODO
                        inps.append(batch_g)

                    # feature consisting of raw parameter values
                    batch_p = jnp.expand_dims(p, axis=-1)
                    inps.append(batch_p)

                    # feature consisting of all momentum values
                    inps.append(m)

                    inp_stack = jnp.concatenate(inps, axis=-1)
                    axis = list(range(len(p.shape)))

                    inp_stack = _second_moment_normalizer(inp_stack, axis=axis)

                    # once normalized, add features that are constant across tensor.
                    # namly the training step embedding.
                    stacked = jnp.reshape(
                        training_step_feature,
                        [1] * len(axis) + list(training_step_feature.shape[-1:]),
                    )
                    stacked = jnp.tile(stacked, list(p.shape) + [1])

                    inp = jnp.concatenate([inp_stack, stacked], axis=-1)

                    # apply the per parameter MLP.
                    output = mod.apply(theta, inp)

                    # split the 2 outputs up into a direction and a magnitude
                    direction = output[..., 0]
                    magnitude = output[..., 1]

                    # compute the step
                    step = direction * jnp.exp(magnitude * exp_mult) * step_mult
                    step = step.reshape(p.shape)
                    new_p = p - step
                    if did_reshape:
                        new_p = jnp.squeeze(new_p, 0)

                    if compute_summary:
                        for fi, f in enumerate(inp):
                            summary.summary(
                                f"mlp_lopt/inp{fi}/mean_abs", jnp.mean(jnp.abs(f))
                            )

                        avg_step_size = jnp.mean(jnp.abs(step))
                        summary.summary("mlp_lopt/avg_step_size", avg_step_size)

                        summary.summary(
                            "mlp_lopt/avg_step_size_hist",
                            avg_step_size,
                            aggregation="collect",
                        )

                        summary.summary(
                            "mlp_lopt/direction/mean_abs", jnp.mean(jnp.abs(direction))
                        )
                        summary.summary(
                            "mlp_lopt/magnitude/mean_abs", jnp.mean(jnp.abs(magnitude))
                        )
                        summary.summary("mlp_lopt/magnitude/mean", jnp.mean(magnitude))

                        # summary.summary(
                        #     "mlp_lopt/grad/mean_abs", jnp.mean(jnp.abs(g))
                        # )

                    return new_p

                next_params = jax.tree_util.tree_map(_update_tensor, opt_state.params, next_rolling_features.m, *grads) # TODO
                next_opt_state = MLPLOptState(
                    params=tree_utils.match_type(next_params, opt_state.params),
                    rolling_features=tree_utils.match_type(
                        next_rolling_features, opt_state.rolling_features
                    ),
                    iteration=opt_state.iteration + 1,
                    state=model_state,
                )
                return next_opt_state

        return _Opt(self.num_grads) # TODO

import copy
import pdb
import functools
from typing import Any, Callable, Optional, Sequence, Union
import jax
import jax.numpy as jnp
import optax
import chex
import gin


from learned_optimization.optimizers.optax_opts import OptaxOptimizer, OptaxState


ModelState = Any
Params = Any
Gradient = Params
OptState = Any


@gin.configurable
class SGDSlowMo(OptaxOptimizer):
    """Stochastic gradient descent with momentum."""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum_state = None
        opt = optax.sgd(learning_rate)
        super().__init__(opt)

    @property
    def name(self):
        return f"SGDSlowMo_lr{self.learning_rate}_m{self.momentum}"

    def init(
        self,
        params: Params,
        momentum: Optional[Params] = None,
        model_state: Optional[ModelState] = None,
        num_steps: Optional[int] = None,
        key: Optional[chex.PRNGKey] = None,
    ):
        return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            params=params,
            optax_opt_state=[
                self.opt.init(params),
                {"momentum": copy.deepcopy(params)}
                if momentum is None
                else {"momentum": momentum},
            ],
            state=model_state,
            iteration=0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        opt_state: OptaxState,
        grad: Gradient,
        loss: Optional[jnp.ndarray] = None,
        model_state: Optional[ModelState] = None,
        key: Optional[chex.PRNGKey] = None,
        **kwargs,
    ):
        del loss
        update, new_opt_state = self.opt.update(
            grad, opt_state.optax_opt_state[0], opt_state.params
        )
        return OptaxState(
            state=model_state,
            params=optax.apply_updates(opt_state.params, update),
            optax_opt_state=[
                new_opt_state,
                opt_state.optax_opt_state[1],
            ],
            iteration=opt_state.iteration + 1,
        )

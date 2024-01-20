import copy
import pdb
import functools
from typing import Any, Callable, Optional, Sequence, Union
import jax
import jax.numpy as jnp
import optax
import chex
import gin
import haiku as hk

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





@gin.configurable
class AdamW_kw(OptaxOptimizer):
    """Adam with a piecewise linear learning rate schedule."""

    def __init__(
        self,
        **kwargs,
    ):
        self.num_workers = None
        self.momentum_state = None
        opt = optax.adamw(**kwargs)
        super().__init__(opt)

    def init(self,
           params: Params,
           model_state: Optional[ModelState] = None,
           num_steps: Optional[int] = None,
           key: Optional[chex.PRNGKey] = None):
        return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            params=params,
            optax_opt_state=self.opt.init(params),
            state=model_state,
            iteration=0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self,
                opt_state: OptaxState,
                grad: Gradient,
                loss: Optional[jnp.ndarray] = None,
                model_state: Optional[ModelState] = None,
                key: Optional[chex.PRNGKey] = None,
                **kwargs):
        del loss
        update, new_opt_state = self.opt.update(grad, opt_state.optax_opt_state,
                                                opt_state.params)
        return OptaxState(
            state=model_state,
            params=optax.apply_updates(opt_state.params, update),
            optax_opt_state=new_opt_state,
            iteration=opt_state.iteration + 1,
        )
    

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_params(self,
                params: Params,
                opt_state: OptaxState,
                grad: Gradient,
                loss: Optional[jnp.ndarray] = None,
                model_state: Optional[ModelState] = None,
                key: Optional[chex.PRNGKey] = None,
                **kwargs):
        return OptaxState(
            state=model_state,
            params=params,
            optax_opt_state=opt_state,
            iteration=opt_state.iteration,
        )

    # def init(
    #     self,
    #     params: Params,
    #     momentum: Optional[Params] = None,
    #     model_state: Optional[ModelState] = None,
    #     num_steps: Optional[int] = None,
    #     key: Optional[chex.PRNGKey] = None,
    # ):
    #     return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
    #         params=params,
    #         optax_opt_state=self.opt.init(params),
    #         state=model_state,
    #         iteration=0,
    #     )

    # @functools.partial(jax.jit, static_argnums=(0,))
    # def update(
    #     self,
    #     opt_state: OptaxState,
    #     grad: Gradient,
    #     loss: Optional[jnp.ndarray] = None,
    #     model_state: Optional[ModelState] = None,
    #     key: Optional[chex.PRNGKey] = None,
    #     **kwargs,
    # ):
    #     del loss
    #     update, new_opt_state = self.opt.update(
    #         grad, opt_state.optax_opt_state, opt_state.params
    #     )
    #     return OptaxState(
    #         state=model_state,
    #         params=optax.apply_updates(opt_state.params, update),
    #         optax_opt_state=new_opt_state,
    #         iteration=opt_state.iteration + 1,
    #     )




@gin.configurable
class AdamWDiLoCo(OptaxOptimizer):
    """Stochastic gradient descent with momentum."""

    def __init__(self, adamw_kwargs, num_workers):
        self.num_workers = num_workers
        self.momentum_state = None
        self.opt = [AdamW_kw(**adamw_kwargs) for _ in range(num_workers)]
        self.local_states = [None for _ in range(num_workers)]

    @property
    def name(self):
        return f"AdamWDiLoCo_lr{self.learning_rate}_m{self.momentum}"

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_k(self,k,*args, **kwargs):
        return self.opt[k].update(*args, **kwargs)

    def init_k(self,k,*args, **kwargs):
        return self.opt[k].init(*args, **kwargs)

    def get_state_k(self,k,*args, **kwargs):
        return self.opt[k].get_state(*args, **kwargs)

    def get_params_k(self,k,*args, **kwargs):
        return self.opt[k].get_params(*args, **kwargs)

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
                [o.opt.init(params) for o in self.opt],
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
        params: Params,
        opt_state: OptaxState,
        grad: Gradient,
        loss: Optional[jnp.ndarray] = None,
        model_state: Optional[ModelState] = None,
        key: Optional[chex.PRNGKey] = None,
        **kwargs,
    ):
        del loss
        # update, new_opt_state = [o.update(
        #     grad, opt_state.optax_opt_state[i], opt_state[i].params
        # ) for i,o in enumerate(self.opt)]

        return OptaxState(
            state=model_state,
            params=params,#optax.apply_updates(opt_state.params, update),
            optax_opt_state=[
                opt_state[0],
                opt_state.optax_opt_state[1],
            ],
            iteration=opt_state.iteration + 1,
        )

    def get_state_k(self, state, k):
        return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            params=state.params,
            optax_opt_state = state.optax_opt_state[0][k],
            state=state.state,
            iteration=state.iteration,
        )

    def get_params_state_k(self, params, state, k):
        return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            params=params,
            optax_opt_state=state.optax_opt_state[0][k],
            state=state.state,
            iteration=state.iteration,
        )


    # def init(self, *args, **kwargs):
    #     return self.init_k(0,*args, **kwargs)

    # @functools.partial(jax.jit, static_argnums=(0,))
    # def update(self, *args, **kwargs):
    #     return self.update_k(0,*args, **kwargs)

    # def cond_set_local_states(self,states):

    #     def true_fn(states):
    #         return [states for x in range(self.num_workers)]
    #         # return states

    #     def false_fn(states):
    #         return self.local_states

    #     return jax.lax.cond(self.local_states[0] == None, true_fn, false_fn, states)

    # def update_local_states(self,updated_params,momentum,avg_state):
    #     self.local_states = [o.init(updated_params, momentum=momentum, model_state=avg_state) for o in self.opt]



    # def init_k(
    #     self,
    #     k: int,
    #     params: Params,
    #     momentum: Optional[Params] = None,
    #     model_state: Optional[ModelState] = None,
    #     num_steps: Optional[int] = None,
    #     key: Optional[chex.PRNGKey] = None,
    # ):
    #     return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
    #         params=params,
    #         optax_opt_state=[
    #             self.opt[k].init(params),
    #             {"momentum": copy.deepcopy(params)}
    #             if momentum is None
    #             else {"momentum": momentum},
    #         ],
    #         state=model_state,
    #         iteration=0,
    #     )

    # @functools.partial(jax.jit, static_argnums=(0,))
    # def update_k(
    #     self,
    #     k: int,
    #     opt_state: OptaxState,
    #     grad: Gradient,
    #     loss: Optional[jnp.ndarray] = None,
    #     model_state: Optional[ModelState] = None,
    #     key: Optional[chex.PRNGKey] = None,
    #     **kwargs,
    # ):
    #     del loss
    #     update, new_opt_state = self.opt[k].update(
    #         grad, opt_state.optax_opt_state[0], opt_state.params
    #     )
    #     return OptaxState(
    #         state=model_state,
    #         params=optax.apply_updates(opt_state.params, update),
    #         optax_opt_state=[
    #             new_opt_state,
    #             opt_state.optax_opt_state[1],
    #         ],
    #         iteration=opt_state.iteration + 1,
    #     )



    # def init(
    #     self,
    #     params: Params,
    #     momentum: Optional[Params] = None,
    #     model_state: Optional[ModelState] = None,
    #     num_steps: Optional[int] = None,
    #     key: Optional[chex.PRNGKey] = None,
    # ):
    #     return OptaxState(  # pytype: disable=wrong-arg-types  # jax-ndarray
    #         params=params,
    #         optax_opt_state=[
    #             self.opt[0].init(params),
    #             {"momentum": copy.deepcopy(params)}
    #             if momentum is None
    #             else {"momentum": momentum},
    #         ],
    #         state=model_state,
    #         iteration=0,
    #     )

    # @functools.partial(jax.jit, static_argnums=(0,))
    # def update(
    #     self,
    #     opt_state: OptaxState,
    #     grad: Gradient,
    #     loss: Optional[jnp.ndarray] = None,
    #     model_state: Optional[ModelState] = None,
    #     key: Optional[chex.PRNGKey] = None,
    #     **kwargs,
    # ):
    #     del loss
    #     update, new_opt_state = self.opt[0].update(
    #         grad, opt_state.optax_opt_state[0], opt_state.params
    #     )
    #     return OptaxState(
    #         state=model_state,
    #         params=optax.apply_updates(opt_state.params, update),
    #         optax_opt_state=[
    #             new_opt_state,
    #             opt_state.optax_opt_state[1],
    #         ],
    #         iteration=opt_state.iteration + 1,
    #     )

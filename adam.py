import jax
import jax.numpy as jnp
import functools


class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def init(self, params):
        return (
            tuple(params),
            jnp.asarray(0),
            tuple([jnp.zeros_like(p) for p in params]),
            tuple([jnp.zeros_like(p) for p in params]),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, state, grads):
        params, iteration, momentum, rms = state
        iteration += 1
        momentum = tuple(
            [m * self.beta1 + (1 - self.beta1) * g for m, g in zip(momentum, grads)]
        )
        rms = tuple(
            [v * self.beta2 + (1 - self.beta2) * (g**2) for v, g in zip(rms, grads)]
        )
        mhat = [m / (1 - self.beta1**iteration) for m in momentum]
        vhat = [v / (1 - self.beta2**iteration) for v in rms]
        params = tuple(
            [
                p - self.lr * m / (jnp.sqrt(v) + self.epsilon)
                for p, m, v in zip(params, mhat, vhat)
            ]
        )
        return (params, iteration, momentum, rms)

import jax
import jax.numpy as jnp
import functools


class LAggOpt:
    def __init__(self, num_samples=4, decay=0.9):
        self.num_samples = num_samples
        self.decay = decay
        self.hidden_size = 64

    def init_meta_params(self, key):
        """Initialize the learned optimizer weights -- in this case the weights of

        the per parameter mlp.
        """
        key1, key2 = jax.random.split(key)
        input_feats = (
            2 + self.num_samples
        )  # parameter value, momentum value, and gradient samples

        # the optimizer is a 2 hidden layer MLP.
        w0 = jax.random.normal(key1, [input_feats, self.hidden_size])
        b0 = jnp.zeros([self.hidden_size])

        w1 = jax.random.normal(key2, [self.hidden_size, 2])
        b1 = jnp.zeros([2])
        return (w0, b0, w1, b1)

    def initial_inner_opt_state(self, meta_params, params):
        # The inner opt state contains the parameter values, and the momentum values.
        momentum = [jnp.zeros_like(p) for p in params]
        return tuple(params), tuple(momentum)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_inner_opt_state(self, meta_params, inner_opt_state, inner_grads):
        "Perform 1 step of learning using the learned optimizer." ""
        params, momentum = inner_opt_state

        # compute momentum
        momentum = [
            m * self.decay + (jnp.mean(g, axis=0) * (1 - self.decay))
            for m, g in zip(momentum, inner_grads)
        ]

        def predict_step(features):
            """Predict the update for a single ndarray."""
            w0, b0, w1, b1 = meta_params
            outs = jax.nn.relu(jax.lax.stop_gradient(features) @ w0 + b0) @ w1 + b1
            # slice out the last 2 elements
            scale = outs[..., 0]
            mag = outs[..., 1]
            # Compute a step as follows.
            return scale * 0.01 * jnp.exp(mag * 0.01)

        out_params = []
        for p, m, g in zip(params, momentum, inner_grads):

            @jax.vmap
            def get_grads_mean(grads):
                return jnp.mean(grads, axis=0)

            sample_grads = get_grads_mean(jnp.array(jnp.split(g, self.num_samples)))
            features = jnp.concatenate((jnp.asarray([p, m]), sample_grads))
            # transpose to have features dim last. The MLP will operate on this,
            # and treat the leading dimensions as a batch dimension.
            features = jnp.transpose(features, list(range(1, 1 + len(p.shape))) + [0])

            step = predict_step(features)
            out_params.append(p - step)

        return tuple(out_params), tuple(momentum)

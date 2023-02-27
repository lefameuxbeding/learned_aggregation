import jax
import jax.numpy as jnp


class MLP:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def init(self, key):
        key1, key2 = jax.random.split(key)
        w0 = jax.random.normal(key1, [self.input_size, 128]) * 0.02
        w1 = jax.random.normal(key2, [128, self.output_size]) * 0.02
        b0 = jnp.zeros([128])
        b1 = jnp.ones([self.output_size])
        return (w0, b0, w1, b1)

    def predict(self, params, input):
        w0, b0, w1, b1 = params
        return jax.nn.log_softmax(jax.nn.relu(input @ w0 + b0) @ w1 + b1)

    def loss(self, params, input, labels):
        logits = self.predict(params, input)
        return jnp.mean(-jnp.sum(labels * logits, axis=-1))

import jax
import jax.numpy as jnp
import numpy as np
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
import sys
import os
from adam import Adam
from mlp import MLP


class LOpt:
    def __init__(self, decay=0.9):
        self.decay = decay
        self.hidden_size = 64

    def init_meta_params(self, key):
        """Initialize the learned optimizer weights -- in this case the weights of

        the per parameter mlp.
        """
        key1, key2 = jax.random.split(key)
        input_feats = 3  # parameter value, momentum value, and gradient value

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
            m * self.decay + (g * (1 - self.decay))
            for m, g in zip(momentum, inner_grads)
        ]

        def predict_step(features):
            """Predict the update for a single ndarray."""
            w0, b0, w1, b1 = meta_params
            outs = jax.nn.relu(features @ w0 + b0) @ w1 + b1
            # slice out the last 2 elements
            scale = outs[..., 0]
            mag = outs[..., 1]
            # Compute a step as follows.
            return scale * 0.01 * jnp.exp(mag * 0.01)

        out_params = []
        for p, m, g in zip(params, momentum, inner_grads):
            features = jnp.asarray([p, m, g])
            # transpose to have features dim last. The MLP will operate on this,
            # and treat the leading dimensions as a batch dimension.
            features = jnp.transpose(features, list(range(1, 1 + len(p.shape))) + [0])

            step = predict_step(features)
            out_params.append(p - step)

        return tuple(out_params), tuple(momentum)


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
            outs = (
                jax.nn.relu(jax.lax.stop_gradient(features) @ w0 + b0) @ w1 + b1
            )  # Make sure the inputs to the MLP are dropped from the compute graph
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


def resize_and_scale(batch):
    batch["image"] = tf.image.resize(batch["image"], (8, 8)) / 255.0
    return batch


def get_batch_seq(seq_len):
    batches = [next(data_iterator) for _ in range(seq_len)]
    # stack the data to add a leading dim.
    return {
        "image": jnp.asarray([b["image"] for b in batches]),
        "label": jnp.asarray([b["label"] for b in batches]),
    }


def meta_loss_fn(meta_params, key, sequence_of_batches):
    def step(opt_state, batch):
        input = batch["image"]
        loss = loss_fn(
            opt_state[0],
            jnp.reshape(input, [input.shape[0], -1]),
            jax.nn.one_hot(batch["label"], 10),
        )
        grads = grad_fn(
            opt_state[0],
            jnp.reshape(input, [input.shape[0], -1]),
            jax.nn.one_hot(batch["label"], 10),
        )
        opt_state = optimizer.update_inner_opt_state(meta_params, opt_state, grads)
        return opt_state, loss

    params = task.init(key)
    opt_state = optimizer.initial_inner_opt_state(meta_params, params)
    # Iterate N times where N is the number of batches in sequence_of_batches
    opt_state, losses = jax.lax.scan(step, opt_state, sequence_of_batches)

    return jnp.mean(losses)


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    ds = tfds.load("fashion_mnist", split="train", data_dir=os.getenv("SLURM_TMPDIR"))
    ds = (
        ds.map(resize_and_scale)
        .cache()
        .repeat(-1)
        .shuffle(64 * 10)
        .batch(128)
        .prefetch(5)
    )
    data_iterator = ds.as_numpy_iterator()
    batch = next(data_iterator)

    ds_test = tfds.load("fashion_mnist", split="test", data_dir=os.getenv("SLURM_TMPDIR"))
    ds_test = ds_test.map(resize_and_scale).repeat(-1).batch(len(ds_test))
    test_iterator = ds_test.as_numpy_iterator()

    input_size = np.prod(batch["image"].shape[1:])
    output_size = 10
    task = MLP(input_size, output_size)

    num_runs = 10
    num_inner_steps = 10

    """ Adam """

    optimizer = Adam(0.01)
    loss_fn = jax.jit(task.loss)
    grad_fn = jax.jit(jax.grad(task.loss))

    for j in range(num_runs):
        run = wandb.init(project="learning-aggregation", group="Adam")

        key = jax.random.PRNGKey(j)
        params = task.init(key)
        opt_state = optimizer.init(params)

        for i in range(num_inner_steps):
            batch = next(data_iterator)
            input = batch["image"]
            loss = loss_fn(
                opt_state[0],
                jnp.reshape(input, [input.shape[0], -1]),
                jax.nn.one_hot(batch["label"], output_size),
            )
            grads = grad_fn(
                opt_state[0],
                jnp.reshape(input, [input.shape[0], -1]),
                jax.nn.one_hot(batch["label"], output_size),
            )
            opt_state = optimizer.update(opt_state, grads)

            run.log({"train_loss": loss})

        test = next(test_iterator)
        input = jnp.reshape(test["image"], [test["image"].shape[0], -1])
        labels = test["label"]
        logits = task.predict(opt_state[0], input)
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.sum((predictions == labels) * 1) / labels.size

        run.log({"test_accuracy": accuracy})

        run.finish()

    """ Learned optimizers """

    num_episodes = 300

    optimizers = [
        (
            "LOpt",
            LOpt(),
            jax.jit(task.loss),
            jax.jit(jax.grad(task.loss)),
        ),
        (
            "LAggOpt-8",
            LAggOpt(8),
            jax.jit(task.loss),
            jax.jit(jax.vmap(jax.grad(task.loss), in_axes=(None, 0, 0))),
        ),
    ]

    for optimizer_name, optimizer, loss_fn, grad_fn in optimizers:
        """Training"""

        key = jax.random.PRNGKey(0)
        params = task.init(key)
        meta_params = optimizer.init_meta_params(key)

        key = jax.random.PRNGKey(0)
        meta_value_grad_fn = jax.jit(jax.value_and_grad(meta_loss_fn))
        meta_opt = Adam(0.001)

        key = jax.random.PRNGKey(0)
        meta_params = optimizer.init_meta_params(key)
        meta_opt_state = meta_opt.init(meta_params)

        for i in range(num_episodes):
            data = get_batch_seq(num_inner_steps)
            key1, key = jax.random.split(key)
            _, meta_grad = meta_value_grad_fn(meta_opt_state[0], key1, data)
            meta_opt_state = meta_opt.update(meta_opt_state, meta_grad)

        meta_params = meta_opt_state[0]

        """ Benchmarking """

        all_losses = []

        for j in range(num_runs):
            run = wandb.init(project="learning-aggregation", group=optimizer_name)

            key = jax.random.PRNGKey(j + 1)  # Different problems than trained on
            params = task.init(key)
            opt_state = optimizer.initial_inner_opt_state(meta_params, params)

            for i in range(num_inner_steps):
                batch = next(data_iterator)
                input = batch["image"]
                loss = loss_fn(
                    opt_state[0],
                    jnp.reshape(input, [input.shape[0], -1]),
                    jax.nn.one_hot(batch["label"], output_size),
                )
                grads = grad_fn(
                    opt_state[0],
                    jnp.reshape(input, [input.shape[0], -1]),
                    jax.nn.one_hot(batch["label"], output_size),
                )
                opt_state = optimizer.update_inner_opt_state(
                    meta_params, opt_state, grads
                )

                run.log({"train_loss": loss})

            test = next(test_iterator)
            input = jnp.reshape(test["image"], [test["image"].shape[0], -1])
            labels = test["label"]
            logits = task.predict(opt_state[0], input)
            predictions = jnp.argmax(logits, axis=-1)
            accuracy = jnp.sum((predictions == labels) * 1) / labels.size

            run.log({"test_accuracy": accuracy})

            run.finish()

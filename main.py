import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
import sys
import os
from adam import Adam
from mlp import MLP
from lopt import LOpt
from laggopt import LAggOpt


def resize_and_scale(batch):
    batch["image"] = tf.image.resize(batch["image"], (8, 8)) / 255.0
    return batch


def get_batch_seq(data_iterator, seq_len):
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

    ds_test = tfds.load(
        "fashion_mnist", split="test", data_dir=os.getenv("SLURM_TMPDIR")
    )
    test_size = len(ds_test)
    ds_test = ds_test.map(resize_and_scale).repeat(-1).batch(test_size)
    test_iterator = ds_test.as_numpy_iterator()

    input_size = np.prod(batch["image"].shape[1:])
    output_size = 10
    task = MLP(input_size, output_size)

    num_runs = 10
    num_episodes = 300
    num_inner_training_steps = 30
    num_inner_steps = 50

    """ Random """

    for j in range(num_runs):
        run = wandb.init(project="learning-aggregation", group="Random")

        test = next(test_iterator)
        input = jnp.reshape(test["image"], [test["image"].shape[0], -1])
        labels = test["label"]
        predictions = np.random.randint(0, output_size - 1, test_size)
        accuracy = jnp.sum((predictions == labels) * 1) / labels.size

        run.log({"test accuracy": accuracy})

        run.finish()

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

            run.log({"train loss": loss})

        test = next(test_iterator)
        input = jnp.reshape(test["image"], [test["image"].shape[0], -1])
        labels = test["label"]
        logits = task.predict(opt_state[0], input)
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.sum((predictions == labels) * 1) / labels.size

        run.log({"test accuracy": accuracy})

        run.finish()

    """ Learned optimizers """

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
        meta_loss_value_grad_fn = jax.jit(jax.value_and_grad(meta_loss_fn))
        meta_opt = Adam(0.001)

        key = jax.random.PRNGKey(0)
        meta_params = optimizer.init_meta_params(key)
        meta_opt_state = meta_opt.init(meta_params)

        run = wandb.init(project="learning-aggregation", group=optimizer_name)

        for i in range(num_episodes):
            data = get_batch_seq(data_iterator, num_inner_training_steps)
            key1, key = jax.random.split(key)
            meta_loss, meta_grad = meta_loss_value_grad_fn(
                meta_opt_state[0], key1, data
            )
            meta_opt_state = meta_opt.update(meta_opt_state, meta_grad)

            run.log({"meta loss": meta_loss})

        run.finish()

        meta_params = meta_opt_state[0]

        """ Benchmarking """

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

                run.log({"train loss": loss})

            test = next(test_iterator)
            input = jnp.reshape(test["image"], [test["image"].shape[0], -1])
            labels = test["label"]
            logits = task.predict(opt_state[0], input)
            predictions = jnp.argmax(logits, axis=-1)
            accuracy = jnp.sum((predictions == labels) * 1) / labels.size

            run.log({"test accuracy": accuracy})

            run.finish()

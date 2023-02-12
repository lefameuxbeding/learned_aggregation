import jax
import jax.numpy as jnp
import numpy as np
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle


class MLPTask:
    def __init__(self, input_size):
        self.input_size = input_size

    def init(self, key):
        key1, key2 = jax.random.split(key)
        w0 = jax.random.normal(key1, [self.input_size, 128]) * 0.02
        w1 = jax.random.normal(key2, [128, 10]) * 0.02
        b0 = jnp.zeros([128])
        b1 = jnp.ones([10])
        return (w0, b0, w1, b1)

    def loss(self, params, input, labels):
        # input = jnp.reshape(batch["image"], [data.shape[0], -1])
        # labels = jax.nn.one_hot(batch["label"], 10)
        w0, b0, w1, b1 = params
        logits = jax.nn.log_softmax(jax.nn.relu(input @ w0 + b0) @ w1 + b1)
        return jnp.mean(-jnp.sum(labels * logits, axis=-1))


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

    ds = tfds.load("fashion_mnist", split="train")
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

    task = MLPTask(np.prod(batch["image"].shape[1:]))

    """ Adam """

    all_losses = []
    optimizer = Adam(0.01)
    loss_fn = jax.jit(task.loss)
    grad_fn = jax.jit(jax.grad(task.loss))

    for j in range(10):
        losses = []
        key = jax.random.PRNGKey(j)
        params = task.init(key)
        opt_state = optimizer.init(params)

        for i in range(10):
            batch = next(data_iterator)
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
            opt_state = optimizer.update(opt_state, grads)
            losses.append(loss)

        all_losses.append(losses)

    losses_mean = np.mean(all_losses, 0)
    losses_std = np.std(all_losses, 0)

    with open("Adam.pickle", "wb") as f:
        pickle.dump({"losses_mean": losses_mean, "losses_std": losses_std}, f)

    """ Learned optimizers """

    optimizers = [
        ("LOpt", LOpt(), jax.jit(task.loss), jax.jit(jax.grad(task.loss))),
        (
            "LAggOpt-4",
            LAggOpt(4),
            jax.jit(task.loss),
            jax.jit(jax.vmap(jax.grad(task.loss), in_axes=(None, 0, 0))),
        ),
        (
            "LAggOpt-8",
            LAggOpt(8),
            jax.jit(task.loss),
            jax.jit(jax.vmap(jax.grad(task.loss), in_axes=(None, 0, 0))),
        ),
        (
            "LAggOpt-16",
            LAggOpt(16),
            jax.jit(task.loss),
            jax.jit(jax.vmap(jax.grad(task.loss), in_axes=(None, 0, 0))),
        ),
        (
            "LAggOpt-32",
            LAggOpt(32),
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
        meta_loss, meta_grad = meta_value_grad_fn(meta_params, key, get_batch_seq(10))
        meta_opt = Adam(0.001)

        key = jax.random.PRNGKey(0)
        meta_params = optimizer.init_meta_params(key)
        meta_opt_state = meta_opt.init(meta_params)

        meta_losses = []

        for i in range(300):
            data = get_batch_seq(10)
            key1, key = jax.random.split(
                key
            )  # Are the starting weights random at each meta-training iteration?
            loss, meta_grad = meta_value_grad_fn(meta_opt_state[0], key1, data)
            meta_losses.append(meta_loss)
            meta_opt_state = meta_opt.update(meta_opt_state, meta_grad)

        meta_params = meta_opt_state[0]

        """ Benchmarking """

        all_losses = []

        for j in range(10):
            losses = []
            key = jax.random.PRNGKey(
                j + 1
            )  # Make sure the problems are different than the one it was trained on
            params = task.init(key)
            opt_state = optimizer.initial_inner_opt_state(meta_params, params)

            for i in range(10):
                batch = next(data_iterator)
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
                opt_state = optimizer.update_inner_opt_state(
                    meta_params, opt_state, grads
                )
                losses.append(loss)

            all_losses.append(losses)

        losses_mean = np.mean(all_losses, 0)
        losses_std = np.std(all_losses, 0)

        with open(optimizer_name + ".pickle", "wb") as f:
            pickle.dump({"losses_mean": losses_mean, "losses_std": losses_std}, f)

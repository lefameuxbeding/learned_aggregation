import pickle

import jax
import jax.numpy as jnp
from haiku._src.data_structures import FlatMap
from learned_optimization.learned_optimizers import adafac_mlp_lopt
from learned_optimization.optimizers import optax_opts, nadamw

from adafac_mlp_lagg import AdafacMLPLAgg
from tasks import get_task


def _lagg(args):
    lagg = AdafacMLPLAgg(num_grads=args.num_grads)
    agg_str = args.optimizer + "_" + args.task + "_" + str(args.num_grads)
    with open(agg_str + ".pickle", "rb") as f:
        meta_params = pickle.load(f)
    agg = lagg.opt_fn(meta_params)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = agg.get_params(opt_state)
        loss = task.loss(params, key, batch)

        def sample_grad_fn(image, label):
            sub_batch_dict = {}
            sub_batch_dict["image"] = image
            sub_batch_dict["label"] = label
            sub_batch = FlatMap(sub_batch_dict)

            return jax.grad(task.loss)(params, key, sub_batch)

        split_image = jnp.split(batch["image"], lagg._num_grads)
        split_label = jnp.split(batch["label"], lagg._num_grads)
        grads = [
            sample_grad_fn(split_image[i], split_label[i])
            for i in range(lagg._num_grads)
        ]

        overall_grad = jax.grad(task.loss)(params, key, batch)  # TODO

        opt_state = agg.update(opt_state, overall_grad, grads, loss=loss)

        return opt_state, loss

    return agg, agg_str, update


def _lopt(args):
    lopt = adafac_mlp_lopt.AdafacMLPLOpt()
    opt_str = args.optimizer + "_" + args.task
    with open(opt_str + ".pickle", "rb") as f:
        meta_params = pickle.load(f)
    opt = lopt.opt_fn(meta_params)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)
        loss, grad = jax.value_and_grad(task.loss)(params, key, batch)
        opt_state = opt.update(opt_state, grad, loss=loss)

        return opt_state, loss

    return opt, opt_str, update


def _nadamw(args):
    opt = nadamw.NAdamW(learning_rate=args.learning_rate)
    opt_str = args.optimizer + str(args.learning_rate)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)
        loss, grad = jax.value_and_grad(task.loss)(params, key, batch)
        opt_state = opt.update(opt_state, grad, loss=loss)

        return opt_state, loss

    return opt, opt_str, update


def _adam(args):
    opt = optax_opts.Adam(learning_rate=args.learning_rate)
    opt_str = args.optimizer + str(args.learning_rate)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)
        loss, grad = jax.value_and_grad(task.loss)(params, key, batch)
        opt_state = opt.update(opt_state, grad, loss=loss)

        return opt_state, loss

    return opt, opt_str, update


def get_optimizer(args):
    optimizers = {
        "nadamw": _nadamw,
        "adam": _adam,
        "lopt": _lopt,
        "lagg": _lagg,
    }

    return optimizers[args.optimizer](args)

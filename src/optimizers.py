import pickle

import jax
import jax.numpy as jnp
from haiku._src.data_structures import FlatMap
from learned_optimization.learned_optimizers import adafac_mlp_lopt
from learned_optimization.optimizers import nadamw, optax_opts

from adafac_mlp_lagg import AdafacMLPLAgg
from tasks import get_task


def _lagg(args):
    lagg = AdafacMLPLAgg(num_grads=args.num_grads, hidden_size=args.hidden_size)
    agg_str = args.optimizer + "_" + args.task + "_" + str(args.num_grads)
    with open(agg_str + ".pickle", "rb") as f:
        meta_params = pickle.load(f)
    agg = lagg.opt_fn(meta_params)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = agg.get_params(opt_state)

        # TODO Could try to use vmap
        split_image = jnp.split(batch["image"], args.num_grads)
        split_label = jnp.split(batch["label"], args.num_grads)
        split_batch = []
        for i in range(args.num_grads):
            sub_batch_dict = {}
            sub_batch_dict["image"] = split_image[i]
            sub_batch_dict["label"] = split_label[i]
            split_batch.append(FlatMap(sub_batch_dict))

        losses_grads = [
            jax.value_and_grad(task.loss)(params, key, b)
            for b in split_batch
        ]
        loss = jnp.mean(jnp.array([l[0] for l in losses_grads]))
        grads = [grad[1] for grad in losses_grads]
        overall_grad = jax.tree_util.tree_map(
            lambda g, *gs: jnp.mean(jnp.array(gs + (g,)), axis=0), grads[0], *grads[1:]
        )

        opt_state = agg.update(opt_state, overall_grad, grads, loss=loss)

        return opt_state, loss

    return agg, agg_str, update


def _lopt(args):
    lopt = adafac_mlp_lopt.AdafacMLPLOpt(hidden_size=args.hidden_size)
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
    opt_str = args.optimizer + "_" + str(args.learning_rate)

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
    opt_str = args.optimizer + "_" + str(args.learning_rate)

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

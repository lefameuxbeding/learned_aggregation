import copy
import pickle

import jax
import jax.numpy as jnp
from haiku._src.data_structures import FlatMap
from learned_optimization.optimizers import base as opt_base
from learned_optimization.optimizers import optax_opts

from adafac_mlp_lagg import AdafacMLPLAgg
from mlp_lagg import MLPLAgg
from tasks import get_task


def _adam(args):
    opt = opt_base.Adam(args.learning_rate)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)
        loss, grad = jax.value_and_grad(task.loss)(params, key, batch)

        return opt.update(opt_state, grad, loss=loss), loss

    return opt, update


def _fedlagg(args):
    lagg_class = AdafacMLPLAgg if args.optimizer in ["fedlopt-adafac", "fedlagg-adafac"] else MLPLAgg
    with_all_grads = True if args.optimizer in ["fedlagg", "fedlagg-wavg"] else False
    with_avg = True if args.optimizer in ["fedlopt, fedlagg-wavg"] else False
    lagg = lagg_class(
        num_grads=args.num_grads, hidden_size=args.hidden_size, with_all_grads=with_all_grads, with_avg=with_avg
    )

    with open("./" + args.name + ".pickle", "rb") as f:
        meta_params = pickle.load(f)
    agg = lagg.opt_fn(meta_params)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        local_opt = optax_opts.SGD(learning_rate=args.local_learning_rate)
        params = agg.get_params(opt_state)
        local_opt_state = local_opt.init(params)

        images = jnp.array(batch["image"])
        labels = jnp.array(batch["label"])

        def split(arr, split_factor):
            """Splits the first axis of `arr` evenly across the number of devices."""
            return arr.reshape(
                split_factor, arr.shape[0] // split_factor, *arr.shape[1:]
            )

        images = split(images, agg.num_grads)
        labels = split(labels, agg.num_grads)

        def local_updates(im, lab):
            l_opt_state = copy.deepcopy(local_opt_state)
            s_c_images = split(im, args.num_local_steps)
            s_c_labels = split(lab, args.num_local_steps)

            s_c_batch = []
            for i in range(args.num_local_steps):
                sub_batch_dict = {}
                sub_batch_dict["image"] = s_c_images[i]
                sub_batch_dict["label"] = s_c_labels[i]
                s_c_batch.append(FlatMap(sub_batch_dict))

            losses = []

            for sub_client_batch in s_c_batch:
                params = local_opt.get_params(l_opt_state)
                loss, grad = jax.value_and_grad(task.loss)(
                    params, key, sub_client_batch
                )
                losses.append(loss)
                l_opt_state = local_opt.update(l_opt_state, grad, loss=loss)

            old_params = local_opt.get_params(local_opt_state)
            new_params = local_opt.get_params(l_opt_state)
            delta = jax.tree_util.tree_map(
                lambda old_p, new_p: new_p - old_p, old_params, new_params
            )

            return jnp.mean(jnp.array(losses)), delta

        losses, deltas = jax.vmap(local_updates)(images, labels)
        loss = jnp.mean(jnp.array(losses))

        return agg.update(opt_state, deltas, loss=loss), loss

    return agg, update


def _fedavg(args):
    opt = optax_opts.SGD(learning_rate=args.local_learning_rate)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        images = jnp.array(batch["image"])
        labels = jnp.array(batch["label"])

        def split(arr, split_factor):
            """Splits the first axis of `arr` evenly across the number of devices."""
            return arr.reshape(
                split_factor, arr.shape[0] // split_factor, *arr.shape[1:]
            )

        images = split(images, args.num_grads)
        labels = split(labels, args.num_grads)

        def local_updates(im, lab):
            local_opt_state = copy.deepcopy(opt_state)
            s_c_images = split(im, args.num_local_steps)
            s_c_labels = split(lab, args.num_local_steps)

            s_c_batch = []
            for i in range(args.num_local_steps):
                sub_batch_dict = {}
                sub_batch_dict["image"] = s_c_images[i]
                sub_batch_dict["label"] = s_c_labels[i]
                s_c_batch.append(FlatMap(sub_batch_dict))

            losses = []

            for sub_client_batch in s_c_batch:
                params = opt.get_params(local_opt_state)
                l, grad = jax.value_and_grad(task.loss)(params, key, sub_client_batch)
                losses.append(l)
                local_opt_state = opt.update(local_opt_state, grad, loss=l)

            return jnp.mean(jnp.array(losses)), opt.get_params(local_opt_state)

        losses, new_params = jax.vmap(local_updates)(images, labels)
        avg_params = jax.tree_util.tree_map(
            lambda p, nps: jnp.mean(nps, axis=0), opt.get_params(opt_state), new_params
        )

        return opt.init(avg_params), jnp.mean(jnp.array(losses))

    return opt, update


def get_optimizer(args):
    optimizers = {
        "adam": _adam,
        "fedavg": _fedavg,
        "fedlopt": _fedlagg,
        "fedlopt-adafac": _fedlagg,
        "fedlagg": _fedlagg,
        "fedlagg-wavg": _fedlagg,
        "fedlagg-adafac": _fedlagg,
    }

    return optimizers[args.optimizer](args) # TODO Find better way to do this

import copy

import jax
import jax.numpy as jnp
from haiku._src.data_structures import FlatMap
from learned_optimization import checkpoints
from learned_optimization.optimizers import optax_opts

from adafac_mlp_lagg import AdafacMLPLAgg
from meta_trainers import get_meta_trainer
from mlp_lagg import MLPLAgg
from tasks import get_task


def _fedlagg(args):
    lagg_class = AdafacMLPLAgg if args.optimizer == "fedlagg-adafac" else MLPLAgg
    with_avg = True if args.optimizer == "fedlagg-wavg" else False
    lagg = lagg_class(
        num_grads=args.num_grads, hidden_size=args.hidden_size, with_avg=with_avg
    )
    meta_trainer, agg_str = get_meta_trainer(args)

    key = jax.random.PRNGKey(0)
    key, key1 = jax.random.split(key)
    outer_trainer_state = meta_trainer.init(key1)
    outer_trainer_state = checkpoints.load_state(
        "./" + agg_str + ".ckpt", outer_trainer_state
    )
    meta_params = outer_trainer_state.gradient_learner_state.theta_opt_state.params

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

        opt_state = agg.update(opt_state, deltas, loss=loss)

        return opt_state, loss

    return agg, agg_str, update


def _fedavg(args):
    opt = optax_opts.SGD(learning_rate=args.local_learning_rate)
    opt_str = (
        args.optimizer
        + "_"
        + args.task
        + "_K"
        + str(args.num_grads)
        + "_H"
        + str(args.num_local_steps)
        + "_"
        + str(args.local_learning_rate)
    )

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

        loss = jnp.mean(jnp.array(losses))

        avg_params = jax.tree_util.tree_map(
            lambda p, nps: jnp.mean(nps, axis=0), opt.get_params(opt_state), new_params
        )
        opt_state = opt.init(avg_params)

        return opt_state, loss

    return opt, opt_str, update


def get_optimizer(
    args,
):  # TODO Find a better way to organize this since we're only using a single function for some of them
    optimizers = {
        "fedavg": _fedavg,
        "fedlagg": _fedlagg,
        "fedlagg-wavg": _fedlagg,
        "fedlagg-adafac": _fedlagg,
    }

    return optimizers[args.optimizer](args)

import copy

import jax
import jax.numpy as jnp
from haiku._src.data_structures import FlatMap
from learned_optimization import checkpoints
from learned_optimization.optimizers import optax_opts

from meta_trainers import get_meta_trainer
from mlp_lagg import MLPLAgg
from tasks import get_task


def _fedlagg(args):
    lagg = MLPLAgg(num_grads=args.num_grads, hidden_size=args.hidden_size)
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

            loss = 0

            for sub_client_batch in s_c_batch:
                params = local_opt.get_params(l_opt_state)
                loss, grad = jax.value_and_grad(task.loss)(
                    params, key, sub_client_batch
                )
                l_opt_state = local_opt.update(l_opt_state, grad, loss=loss)

            old_params = local_opt.get_params(local_opt_state)
            new_params = local_opt.get_params(l_opt_state)
            delta = jax.tree_util.tree_map(
                lambda old_p, new_p: new_p - old_p, old_params, new_params
            )

            return loss, delta

        losses, deltas = jax.vmap(local_updates)(images, labels)
        print("DELTAS HERE", deltas)

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
        s_batch = split_batch(batch, args.num_grads)

        losses = []
        new_params = []
        for client_batch in s_batch:  # TODO Vectorize if possible
            local_opt_state = copy.deepcopy(opt_state)
            s_c_batch = split_batch(client_batch, args.num_local_steps)
            for sub_client_batch in s_c_batch:
                params = opt.get_params(local_opt_state)
                l, grad = jax.value_and_grad(task.loss)(params, key, sub_client_batch)
                losses.append(l)
                local_opt_state = opt.update(local_opt_state, grad, loss=l)
            new_params.append(opt.get_params(local_opt_state))

        loss = jnp.mean(jnp.array(losses))

        overall_params = jax.tree_util.tree_map(
            lambda p, *ps: jnp.mean(jnp.array(ps + (p,)), axis=0),
            new_params[0],
            *new_params[1:],
        )
        opt_state = opt.init(overall_params)

        return opt_state, loss

    return opt, opt_str, update


def get_optimizer(args):
    optimizers = {
        "fedavg": _fedavg,
        "fedlagg": _fedlagg,
    }

    return optimizers[args.optimizer](args)

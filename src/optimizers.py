import pickle
import copy

import jax
import jax.numpy as jnp
from learned_optimization.learned_optimizers import adafac_mlp_lopt
from learned_optimization.optimizers import nadamw, optax_opts

from adafac_mlp_lagg import AdafacMLPLAgg
from tasks import get_task
from utils import split_batch


def _fedavg(args):
    opt = optax_opts.SGD(learning_rate=args.learning_rate)
    opt_str = args.optimizer + "_K" + str(args.num_grads) + "_H" + str(args.num_local_steps)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        s_batch = split_batch(batch, args.num_grads)

        losses = []
        new_params = []
        for client_batch in s_batch: # TODO Vectorize if possible
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
            lambda p, *ps: jnp.mean(jnp.array(ps + (p,)), axis=0), new_params[0], *new_params[1:]
        )
        opt_state = opt.init(overall_params)

        return opt_state, loss

    return opt, opt_str, update


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

        s_batch = split_batch(batch, args.num_grads)

        losses_grads = [
            jax.value_and_grad(task.loss)(params, key, b)
            for b in s_batch
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
        "fedavg": _fedavg,
    }

    return optimizers[args.optimizer](args)

import functools
import pickle

import gin
import jax
import jax.numpy as jnp
import optax
from learned_optimization.optimizers import OptaxOptimizer
from learned_optimization.optimizers import base as opt_base
from learned_optimization.optimizers import optax_opts

from fed_adafac_mlp_lopt import FedAdafacMLPLOpt
from fed_mlp_lopt import FedMLPLOpt
from slowmo import SGDSlowMo
from tasks import get_task


@gin.configurable
class AdamWLinearCosine(OptaxOptimizer):
    """Adam with a piecewise linear learning rate schedule."""

    def __init__(
        self,
        init_value=3e-10,
        peak_value=3e-4,
        warmup_steps=300,
        decay_steps=9700,
        end_value=3e-5,
        exponent=1.0,
        clip=False,
    ):
        self.schedule_ = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=peak_value,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_value,
            exponent=exponent,
        )
        if clip:
            opt = optax.chain(
                optax.adamw(self.schedule_),
                optax.clip_by_global_norm(1.0),
            )
        else:
            opt = optax.adamw(self.schedule_)

        super().__init__(opt)


@gin.configurable
class AdamW(OptaxOptimizer):
    """Adam with a piecewise linear learning rate schedule."""

    def __init__(
        self,
        learning_rate,
    ):
        opt = optax.adamw(learning_rate)
        super().__init__(opt)


def _sgd(args):
    opt = optax_opts.SGD(learning_rate=args.learning_rate)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)

        if args.needs_state:
            state = opt.get_state(opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(
                params, state, key, batch
            )
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key, batch)
            s = None

        return opt.update(opt_state, grad, loss=l, model_state=s), l

    return opt, update


def _adam(args):
    opt = opt_base.Adam(args.learning_rate)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)

        if args.needs_state:
            state = opt.get_state(opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(
                params, state, key, batch
            )
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key, batch)
            s = None

        return opt.update(opt_state, grad, loss=l, model_state=s), l

    return opt, update


def _fedlagg(args):
    lagg_class = (
        FedAdafacMLPLOpt
        if args.optimizer in ["fedlopt-adafac", "fedlagg-adafac"]
        else FedMLPLOpt
    )
    with_all_grads = (
        True
        if args.optimizer in ["fedlagg", "fedlagg-wavg", "fedlagg-adafac"]
        else False
    )
    with_avg = (
        True
        if args.optimizer in ["fedlopt", "fedlopt-adafac", "fedlagg-wavg"]
        else False
    )
    lagg = lagg_class(
        num_grads=args.num_grads,
        hidden_size=args.hidden_size,
        with_all_grads=with_all_grads,
        with_avg=with_avg,
    )

    with open(args.test_checkpoint, "rb") as f:
        meta_params = pickle.load(f)
    agg = lagg.opt_fn(meta_params)
    local_opt = optax_opts.SGD(learning_rate=args.local_learning_rate)
    task = get_task(args)

    def local_step(local_opt_state_and_key, local_batch):
        local_opt_state, key = local_opt_state_and_key
        params = local_opt.get_params(local_opt_state)
        key, key1 = jax.random.split(key)

        if args.needs_state:
            state = local_opt.get_state(local_opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, local_batch)
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key1, local_batch)
            s = None

        return (local_opt.update(local_opt_state, grad, loss=l, model_state=s), key), l

    @functools.partial(jax.pmap, in_axes=(None, 0, 0), out_axes=(None, 0, None, None), axis_name="num_grads")
    def pmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)
        delta = jax.tree_util.tree_map(
            lambda new_p, old_p: new_p - old_p,
            local_opt.get_params(final_local_opt_state),
            local_opt.get_params(init_local_opt_state),
        )

        return (
            jax.lax.pmean(jnp.mean(local_losses), axis_name="num_grads"),
            delta,
            jax.lax.pmean(delta, axis_name="num_grads"),
            jax.lax.pmean(local_opt.get_state(final_local_opt_state), axis_name="num_grads") if args.needs_state else None
        )

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def vmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        return (
            jnp.mean(local_losses),
            jax.tree_util.tree_map(
                lambda new_p, old_p: new_p - old_p,
                local_opt.get_params(final_local_opt_state),
                local_opt.get_params(init_local_opt_state),
            ),
            local_opt.get_state(final_local_opt_state) if args.needs_state else None,
        )

    def update(opt_state, key, batch):
        splitted_batches = jax.tree_util.tree_map(lambda x: x.reshape((args.num_grads, args.num_local_steps, args.local_batch_size) + x.shape[1:]), batch)
        init_local_opt_state = local_opt.init(agg.get_params(opt_state), model_state=agg.get_state(opt_state))

        keys = jax.random.split(key, args.num_grads)
        if args.use_pmap:
            assert args.num_devices == args.num_grads, "The number of devices for pmap should be equal to the number of clients (gradients)"
            loss, deltas, avg_delta, avg_state = pmap_local_updates(init_local_opt_state, keys, splitted_batches)
        else:
            losses, deltas, new_state = vmap_local_updates(init_local_opt_state, keys, splitted_batches)
            loss = jnp.mean(losses)
            avg_delta = jax.tree_util.tree_map(
                    lambda ds: jnp.mean(ds, axis=0), deltas
            )
            if args.needs_state:
                avg_state = jax.tree_util.tree_map(
                    lambda s, ns: jnp.mean(ns, axis=0),
                    local_opt.get_state(init_local_opt_state),
                    new_state,
                )
            else:
                avg_state = None

        return agg.update(opt_state, deltas, avg_delta, loss=loss, model_state=avg_state), loss

    return agg, update


def _fedavg(args):
    local_opt = optax_opts.SGD(learning_rate=args.local_learning_rate)
    task = get_task(args)

    def local_step(local_opt_state_and_key, local_batch):
        local_opt_state, key = local_opt_state_and_key
        params = local_opt.get_params(local_opt_state)
        key, key1 = jax.random.split(key)

        if args.needs_state:
            state = local_opt.get_state(local_opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, local_batch)
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key1, local_batch)
            s = None

        return (local_opt.update(local_opt_state, grad, loss=l, model_state=s), key), l

    @functools.partial(jax.pmap, in_axes=(None, 0, 0), out_axes=(None, None, None), axis_name="num_grads")
    def pmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        return (
            jax.lax.pmean(jnp.mean(local_losses), axis_name="num_grads"),
            jax.lax.pmean(
                local_opt.get_params(final_local_opt_state), axis_name="num_grads"
            ),
            jax.lax.pmean(
                local_opt.get_state(final_local_opt_state), axis_name="num_grads"
            ) if args.needs_state else None
        )

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def vmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        return (
            jnp.mean(local_losses),
            local_opt.get_params(final_local_opt_state),
            local_opt.get_state(final_local_opt_state) if args.needs_state else None,
        )

    def update(opt_state, key, batch):
        splitted_batches = jax.tree_util.tree_map(lambda x: x.reshape((args.num_grads, args.num_local_steps, args.local_batch_size) + x.shape[1:]), batch)

        keys = jax.random.split(key, args.num_grads)
        if args.use_pmap:
            assert args.num_devices == args.num_grads, "The number of devices for pmap should be equal to the number of clients (gradients)"
            loss, avg_params, avg_state = pmap_local_updates(opt_state, keys, splitted_batches)
        else:
            losses, new_params, new_state = vmap_local_updates(opt_state, keys, splitted_batches)
            loss = jnp.mean(losses)
            avg_params = jax.tree_util.tree_map(
                lambda p, nps: jnp.mean(nps, axis=0),
                local_opt.get_params(opt_state),
                new_params,
            )
            if args.needs_state:
                avg_state = jax.tree_util.tree_map(
                    lambda s, ns: jnp.mean(ns, axis=0),
                    local_opt.get_state(opt_state),
                    new_state,
                )
            else:
                avg_state = None

        return local_opt.init(avg_params, model_state=avg_state), loss

    return local_opt, update


def _fedavg_slowmo(args):
    opt = SGDSlowMo(learning_rate=args.local_learning_rate)
    task = get_task(args)

    def local_step(local_opt_state_and_key, local_batch):
        local_opt_state, key = local_opt_state_and_key
        params = opt.get_params(local_opt_state)
        key, key1 = jax.random.split(key)

        if args.needs_state:
            state = opt.get_state(local_opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key1, local_batch)
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key1, local_batch)
            s = None

        return (opt.update(local_opt_state, grad, loss=l, model_state=s), key), l

    @functools.partial(jax.pmap, in_axes=(None, 0, 0), out_axes=(None, None, None), axis_name="num_grads")
    def pmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        return (
            jax.lax.pmean(jnp.mean(local_losses), axis_name="num_grads"),
            jax.lax.pmean(
                opt.get_params(final_local_opt_state), axis_name="num_grads"
            ),
            jax.lax.pmean(
                opt.get_state(final_local_opt_state), axis_name="num_grads"
            ) if args.needs_state else None
        )

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def vmap_local_updates(init_local_opt_state, key, client_batch):
        (final_local_opt_state, _), local_losses = jax.lax.scan(local_step, (init_local_opt_state, key), client_batch)

        return (
            jnp.mean(local_losses),
            opt.get_params(final_local_opt_state),
            opt.get_state(final_local_opt_state) if args.needs_state else None,
        )

    def update(opt_state, key, batch):
        splitted_batches = jax.tree_util.tree_map(lambda x: x.reshape((args.num_grads, args.num_local_steps, args.local_batch_size) + x.shape[1:]), batch)

        keys = jax.random.split(key, args.num_grads)
        if args.use_pmap:
            assert args.num_devices == args.num_grads, "The number of devices for pmap should be equal to the number of clients (gradients)"
            loss, avg_params, avg_state = pmap_local_updates(opt_state, keys, splitted_batches)
        else:
            losses, new_params, new_state = vmap_local_updates(opt_state, keys, splitted_batches)
            loss = jnp.mean(losses)
            avg_params = jax.tree_util.tree_map(
                lambda p, nps: jnp.mean(nps, axis=0), opt.get_params(opt_state), new_params
            )
            if args.needs_state:
                avg_state = jax.tree_util.tree_map(
                    lambda s, ns: jnp.mean(ns, axis=0), opt.get_state(opt_state), new_state
                )
            else:
                avg_state = None

        ##### SLOW MO UPDATE (TODO not optimal) #####

        def update_momentum(momentum, avg_params, current_params, beta, local_learning_rate):
            return beta * momentum + (1 / local_learning_rate) * (current_params - avg_params)

        def update_params(current_params, momentum, local_learning_rate):
            return current_params - local_learning_rate * momentum

        # Get the momentum and current parameters
        momentum = opt_state.optax_opt_state[1]["momentum"]
        current_params = opt.get_params(opt_state)

        # Update the momentum
        momentum = jax.tree_util.tree_map(
            update_momentum,
            momentum,
            avg_params,
            current_params,
            jax.tree_util.tree_map(lambda x: args.beta, momentum),
            jax.tree_util.tree_map(lambda x: args.local_learning_rate, momentum),
        )

        # Update the parameters
        updated_params = jax.tree_util.tree_map(
            update_params,
            current_params,
            momentum,
            jax.tree_util.tree_map(lambda x: args.slowmo_learning_rate, current_params),
        )

        return opt.init(updated_params, momentum=momentum, model_state=avg_state), loss

    return opt, update


def get_optimizer(args):
    optimizers = {
        "adam": _adam,
        "sgd": _sgd,
        "fedavg": _fedavg,
        "fedavg-slowmo": _fedavg_slowmo,
        "fedlopt": _fedlagg,
        "fedlopt-adafac": _fedlagg,
        "fedlagg": _fedlagg,
        "fedlagg-wavg": _fedlagg,
        "fedlagg-adafac": _fedlagg,
    }

    return optimizers[args.optimizer](args)  # TODO Find better way to do this

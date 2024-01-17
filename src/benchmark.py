import argparse

import jax
import jax.numpy as jnp
import wandb
from tqdm import tqdm

from optimizers import get_optimizer
from tasks import get_task


def rename_batch(batch, label_map):
    return {label_map[k]:v for k,v in batch.items()}


def count_parameters(params):
    return sum(jnp.size(param) for param in jax.tree_util.tree_leaves(params))


def benchmark(args):
    key = jax.random.PRNGKey(0)

    task = get_task(args)
    test_task = get_task(args, is_test=True)

    opt, update = get_optimizer(args)

    data_label_map = {'obs':'image',
                    'target':'label',
                    'image':'image',
                    'label':'label'}

    for _ in tqdm(range(args.num_runs), ascii=True, desc="Outer Loop"):
        run = wandb.init(project=args.test_project, group=args.name, config=vars(args))

        key, key1 = jax.random.split(key)
        if args.needs_state:
            params, state = task.init_with_state(key1)
        else:
            params, state = task.init(key1), None

        params_count = count_parameters(params)
        print("Model parameters (M): ", params_count/1000000)
        
        opt_state = opt.init(params, model_state=state, num_steps=args.num_inner_steps)
        clients_state = jax.tree_util.tree_map(lambda x : jnp.array([jnp.zeros_like(x) for _ in range(args.num_grads)]), params)

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            batch = rename_batch(next(task.datasets.train), data_label_map)
            key, key1 = jax.random.split(key)
            opt_state, loss, clients_state = update(opt_state, clients_state, key1, batch, int(params_count * args.top_k_value))

            key, key1 = jax.random.split(key)
            params = opt.get_params(opt_state)

            test_batch = rename_batch(next(test_task.datasets.test), data_label_map)
            #log loss and accuracy if implemented
            try:
                test_loss, test_acc = test_task.loss_and_accuracy(params, key1, test_batch)
                test_log = {
                    "test loss": test_loss,
                    "test accuracy": test_acc,
                }
            except AttributeError as e:
                Warning("test_task does not have loss_and_accuracy method, defaulting to loss")
                if args.needs_state:
                    state = opt.get_state(opt_state)
                    test_loss = test_task.loss(params, state, key1, test_batch)
                else:
                    test_loss = test_task.loss(params, key1, test_batch)

                test_log = {"test loss": test_loss}


            outer_valid_batch = rename_batch(next(test_task.datasets.outer_valid), data_label_map)
            if args.needs_state:
                state = opt.get_state(opt_state)
                outer_valid_loss = test_task.loss(params, state, key1, outer_valid_batch)
            else:
                outer_valid_loss = test_task.loss(params, key1, outer_valid_batch)
            
            to_log = {
                    "train loss": loss,
                    "outer valid loss": outer_valid_loss
                }
            to_log.update(test_log)

            run.log(to_log)

        run.finish()


def sweep(args):
    def sweep_fn(args=args):
        key = jax.random.PRNGKey(0)

        run = wandb.init(
            project="learned_aggregation_meta_test", group=args.name, config=vars(args)
        )
        args = argparse.Namespace(**run.config)

        task = get_task(args)
        test_task = get_task(args, is_test=True)

        opt, update = get_optimizer(args)

        data_label_map = {'obs':'image',
                        'target':'label',
                        'image':'image',
                        'label':'label'}

        key, key1 = jax.random.split(key)
        if args.needs_state:
            params, state = task.init_with_state(key1)
        else:
            params, state = task.init(key1), None
        
        params_count = count_parameters(params)
        print("Model parameters (M): ", params_count/1000000)
        
        opt_state = opt.init(params, model_state=state, num_steps=args.num_inner_steps)
        clients_state = jax.tree_util.tree_map(lambda x : jnp.array([jnp.zeros_like(x) for _ in range(args.num_grads)]), params)

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            batch = rename_batch(next(task.datasets.train), data_label_map)
            key, key1 = jax.random.split(key)
            opt_state, loss, clients_state = update(opt_state, clients_state, key1, batch, int(params_count * args.top_k_value))

            key, key1 = jax.random.split(key)
            params = opt.get_params(opt_state)

            test_batch = rename_batch(next(test_task.datasets.test), data_label_map)
            #log loss and accuracy if implemented
            try:
                test_loss, test_acc = test_task.loss_and_accuracy(params, key1, test_batch)
                test_log = {
                    "test loss": test_loss,
                    "test accuracy": test_acc,
                }
            except AttributeError as e:
                Warning("test_task does not have loss_and_accuracy method, defaulting to loss")
                if args.needs_state:
                    state = opt.get_state(opt_state)
                    test_loss = test_task.loss(params, state, key1, test_batch)
                else:
                    test_loss = test_task.loss(params, key1, test_batch)

                test_log = {"test loss": test_loss}


            outer_valid_batch = rename_batch(next(test_task.datasets.outer_valid), data_label_map)
            if args.needs_state:
                state = opt.get_state(opt_state)
                outer_valid_loss = test_task.loss(params, state, key1, outer_valid_batch)
            else:
                outer_valid_loss = test_task.loss(params, key1, outer_valid_batch)
            
            to_log = {
                    "train loss": loss,
                    "outer valid loss": outer_valid_loss
                }
            to_log.update(test_log)

            run.log(to_log)

        run.finish()

    if args.sweep_id is None:
        args.sweep_id = wandb.sweep(
            sweep=args.sweep_config, project="learned_aggregation_meta_test"
        )

    wandb.agent(args.sweep_id, sweep_fn, project="learned_aggregation_meta_test")

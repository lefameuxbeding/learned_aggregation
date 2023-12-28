import argparse

import jax
import wandb
from tqdm import tqdm

from optimizers import get_optimizer
from tasks import get_task

import jax.numpy as jnp


def flatten_dict(d, parent_key='', sep='/'):
    """
    Flattens a nested dictionary. Each key in the output dictionary is a concatenation
    of the keys from all levels in the input dictionary, joined by `sep`.

    Args:
    - d (dict): The dictionary to flatten.
    - parent_key (str, optional): The concatenated key constructed from the upper levels.
    - sep (str, optional): The separator used to join keys.

    Returns:
    - dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Recursive call for nested dictionaries
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def rename_batch(batch, label_map):
    return {label_map[k]:v for k,v in batch.items()}

def count_parameters(params):
    return sum(jnp.size(param) for param in jax.tree_leaves(params))

def benchmark(args):
    key = jax.random.PRNGKey(0)

    task = get_task(args)
    test_task = get_task(args, is_test=True)

    key, key1 = jax.random.split(key)
    if args.needs_state:
        print("Using stateful model")
        params, state = task.init_with_state(key1)
    else:
        params, state = task.init(key1), None

    try:
        args.muadamw_schedule_kwargs['mup_lrs'] = state['mu_mlp']['mup_lrs']
    except KeyError as e:
        print(e)
        print("Ignoring if not training mu optimizers muadamw_schedule_kwargs")
    except TypeError as e:
        print(e)
        print("Ignoring if not training mu optimizers muadamw_schedule_kwargs")

    if 'dllr' in args.optimizer:
        args.local_learning_rate = jax.tree_map(lambda x : args.local_learning_rate, params)

    opt, update = get_optimizer(args)


    data_label_map = {'obs':'image',
                    'target':'label',
                    'image':'image',
                    'label':'label'}

    test_batch = rename_batch(next(test_task.datasets.test), data_label_map)
    outer_valid_batch = rename_batch(next(test_task.datasets.outer_valid), data_label_map)

    for _ in tqdm(range(args.num_runs), ascii=True, desc="Outer Loop"):
        run = wandb.init(project=args.test_project, group=args.name, config=vars(args))
        
        if _ == 0:
            print("Model parameters (M): ", count_parameters(params)/1000000)
        else:
            key, key1 = jax.random.split(key)
            if args.needs_state:
                params, state = task.init_with_state(key1)
            else:
                params, state = task.init(key1), None
    
        opt_state = opt.init(params, model_state=state, num_steps=args.num_inner_steps)
        # print("Opt state llr: ", opt_state.llr)
        # print(params,state)

        for inner_step in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            batch = rename_batch(next(task.datasets.train), data_label_map)
            key, key1 = jax.random.split(key)

            
            opt_state, loss = update(opt_state, key1, batch)
            # print("Opt state llr: ", opt_state.llr)

            key, key1 = jax.random.split(key)
            params = opt.get_params(opt_state)
            
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
                    test_loss, _ = test_task.loss_with_state(params, state, key1, test_batch)
                else:
                    test_loss = test_task.loss(params, key1, test_batch)

                test_log = {"test loss": test_loss}


            if args.needs_state:
                state = opt.get_state(opt_state)
                outer_valid_loss, _ = test_task.loss_with_state(params, state, key1, outer_valid_batch)
            else:
                outer_valid_loss = test_task.loss(params, key1, outer_valid_batch)
            
            to_log = {
                    "train loss": loss,
                    "outer valid loss": outer_valid_loss
                }
            to_log.update(test_log)
            to_log.update(flatten_dict(opt_state.llr))

            # DONT delete the following comment
            # if inner_step < 10:
            #     state = opt.get_state(opt_state)
            #     to_log.update({k + f"_iter_{inner_step}" : v for k,v in state['mu_mlp'].items() if 'act' in k})

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

        key, key1 = jax.random.split(key)
        if args.needs_state:
            params, state = task.init_with_state(key1)
        else:
            params, state = task.init(key1), None
        
        opt_state = opt.init(params, model_state=state, num_steps=args.num_inner_steps)


        tmp = {'obs':'image',
                'target':'label',
                'image':'image',
                'label':'label'}

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            batch = next(task.datasets.train)
            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, batch)

            key, key1 = jax.random.split(key)
            params = opt.get_params(opt_state)

            test_batch = next(test_task.datasets.test)
            test_batch = {tmp[k]:v for k,v in test_batch.items()}
            if args.needs_state:
                state = opt.get_state(opt_state)
                test_loss = test_task.loss(params, state, key1, test_batch)
            else:
                test_loss = test_task.loss(params, key1, test_batch)

            outer_valid_batch = next(test_task.datasets.outer_valid)
            outer_valid_batch = {tmp[k]:v for k,v in outer_valid_batch.items()}
            if args.needs_state:
                state = opt.get_state(opt_state)
                outer_valid_loss = test_task.loss(params, state, key1, outer_valid_batch)
            else:
                outer_valid_loss = test_task.loss(params, key1, outer_valid_batch)

            run.log(
                {   
                    "train loss": loss, 
                    "test loss": test_loss,
                    "outer valid loss": outer_valid_loss
                }
            )

        run.finish()

    if args.sweep_id is None:
        args.sweep_id = wandb.sweep(
            sweep=args.sweep_config, project="learned_aggregation_meta_test"
        )

    wandb.agent(args.sweep_id, sweep_fn, project="learned_aggregation_meta_test")

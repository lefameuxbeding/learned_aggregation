import argparse

import jax
import wandb
from tqdm import tqdm

from optimizers import get_optimizer
from tasks import get_task



def rename_batch(batch, label_map):
    return {label_map[k]:v for k,v in batch.items()}


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
        
        opt_state = opt.init(params, model_state=state, num_steps=args.num_inner_steps)

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            batch = rename_batch(next(task.datasets.train), data_label_map)
            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, batch)

            # key, key1 = jax.random.split(key)
            # params = opt.get_params(opt_state)

            # test_batch = rename_batch(next(test_task.datasets.test), data_label_map)
            # #log loss and accuracy if implemented
            # try:
            #     test_loss, test_acc = test_task.loss_and_accuracy(params, key1, test_batch)
            #     test_log = {
            #         "test loss": test_loss,
            #         "test accuracy": test_acc,
            #     }
            # except AttributeError as e:
            #     Warning("test_task does not have loss_and_accuracy method, defaulting to loss")
            #     if args.needs_state:
            #         state = opt.get_state(opt_state)
            #         test_loss = test_task.loss(params, state, key1, test_batch)
            #     else:
            #         test_loss = test_task.loss(params, key1, test_batch)

            #     test_log = {"test loss": test_loss}


            # outer_valid_batch = rename_batch(next(test_task.datasets.outer_valid), data_label_map)
            # if args.needs_state:
            #     state = opt.get_state(opt_state)
            #     outer_valid_loss = test_task.loss(params, state, key1, outer_valid_batch)
            # else:
            #     outer_valid_loss = test_task.loss(params, key1, outer_valid_batch)
            
            to_log = {
                    "train loss": loss,
                    # "outer valid loss": outer_valid_loss
                }
            # to_log.update(test_log)

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

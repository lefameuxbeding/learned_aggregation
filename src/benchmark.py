import argparse

import jax
import wandb
from tqdm import tqdm

from optimizers import get_optimizer
from tasks import get_task


def benchmark(args):
    key = jax.random.PRNGKey(0)

    task = get_task(args)
    test_task = get_task(args, is_test=True)

    opt, update = get_optimizer(args)

    for _ in tqdm(range(args.num_runs), ascii=True, desc="Outer Loop"):
        run = wandb.init(project=args.test_project, group=args.name, config=vars(args))

        key, key1 = jax.random.split(key)
        params = task.init(key1)
        opt_state = opt.init(params, num_steps=args.num_inner_steps)

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            batch = next(task.datasets.train)
            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, batch)

            key, key1 = jax.random.split(key)
            params = opt.get_params(opt_state)

            test_batch = next(test_task.datasets.test)
            test_loss = test_task.loss(params, key1, test_batch)

            run.log({"train loss": loss, "test loss": test_loss})

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
        params = task.init(key1)
        opt_state = opt.init(params, num_steps=args.num_inner_steps)

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            batch = next(task.datasets.train)
            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, batch)

            key, key1 = jax.random.split(key)
            params = opt.get_params(opt_state)

            test_batch = next(test_task.datasets.test)
            test_loss = test_task.loss(params, key1, test_batch)

            run.log({"train loss": loss, "test loss": test_loss})

        run.finish()

    sweep_id = wandb.sweep(
        sweep=args.sweep_config, project="learned_aggregation_meta_test"
    )
    wandb.agent(sweep_id, sweep_fn)

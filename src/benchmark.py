import jax
import wandb

from optimizers import get_optimizer
from tasks import get_task


def benchmark(args):
    key = jax.random.PRNGKey(0)

    task = get_task(args)
    opt, update = get_optimizer(args)

    for _ in range(args.num_runs):
        run = wandb.init(project="learned_aggregation", group=args.name)

        key, key1 = jax.random.split(key)
        params = task.init(key1)
        opt_state = opt.init(params, num_steps=args.num_inner_steps)

        for _ in range(args.num_inner_steps):
            batch = next(task.datasets.train)
            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, batch)

            run.log({args.task + " train loss": loss})

        run.finish()

import jax
import wandb

from tqdm import tqdm
from optimizers import get_optimizer
from tasks import get_task
import jax.numpy as jnp

def benchmark(args):
    key = jax.random.PRNGKey(0)

    task = get_task(args)
    opt, update = get_optimizer(args)

    for _ in tqdm(range(args.num_runs), ascii=True, desc="Outer Loop"):
        run = wandb.init(project="learned_aggregation", group=args.name)

        key, key1 = jax.random.split(key)
        params = task.init(key1)
        opt_state = opt.init(params, num_steps=args.num_inner_steps)

        for _ in tqdm(range(args.num_inner_steps),ascii=True, desc="Inner Loop"):
            batch = next(task.datasets.train)
            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, batch)

            key, key1 = jax.random.split(key)
            params = opt.get_params(opt_state)

            test_batch = next(task.datasets.test)
            test_loss = task.loss(params, key1, test_batch)

            run.log(
                {args.task + " train loss": loss, args.task + " test loss": test_loss}
            )

        run.finish()

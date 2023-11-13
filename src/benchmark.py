import argparse

import jax
import jax.numpy as jnp
import wandb
from tqdm import tqdm
from typing import Any, Optional, Sequence, Callable
import haiku as hk

from optimizers import get_optimizer
from tasks import get_task


# Could probably be integrated within tasks
def conv_logits(
    hidden_units: Sequence[int] = [32, 64, 64],
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    initializers: Optional[hk.initializers.Initializer] = None,
    norm_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    pool: str = "avg",
    num_classes: int = 10):
  """Haiku function for a conv net with pooling and cross entropy loss."""
  if not initializers:
    initializers = {}

  def _fn(batch):
    net = batch["image"]
    strides = [2] + [1] * (len(hidden_units) - 1)
    for hs, ks, stride in zip(hidden_units, [3] * len(hidden_units), strides):
      net = hk.Conv2D(hs, ks, stride=stride)(net)
      net = activation_fn(net)
      net = norm_fn(net)

    if pool == "avg":
      net = jnp.mean(net, [1, 2])
    elif pool == "max":
      net = jnp.max(net, [1, 2])
    else:
      raise ValueError("pool type not supported")

    logits = hk.Linear(num_classes)(net)

    return logits

  return _fn

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
        params = task.init(key1)
        opt_state = opt.init(params, num_steps=args.num_inner_steps)

        for _ in tqdm(range(args.num_inner_steps), ascii=True, desc="Inner Loop"):
            batch = rename_batch(next(task.datasets.train), data_label_map)
            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, batch)

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
                test_loss = test_task.loss(params, key1, test_batch)
                test_log = {"test loss": test_loss}


            outer_valid_batch = rename_batch(next(test_task.datasets.outer_valid), data_label_map)
            outer_valid_loss = test_task.loss(params, key1, outer_valid_batch)
            
            to_log = {
                    "train loss": loss,
                    "outer valid loss": outer_valid_loss
                }
            to_log.update(test_log)

            run.log(to_log)

        test_data = next(test_task.datasets.test)
        mod = hk.transform(conv_logits())
        key, key1 = jax.random.split(key)
        test_logits = mod.apply(params, key1, test_data)
        test_predictions = jnp.argmax(test_logits, axis=-1)
        test_accuracy = jnp.sum((test_predictions == test_data["label"]) * 1) / test_data["label"].size

        run.log({"test accuracy" : test_accuracy})

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
            test_loss = test_task.loss(params, key1, test_batch)

            outer_valid_batch = next(test_task.datasets.outer_valid)
            outer_valid_batch = {tmp[k]:v for k,v in outer_valid_batch.items()}
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

import jax
import wandb

from learned_optimization import checkpoints

from meta_trainers import get_meta_trainer


def meta_train(args):
    meta_trainer, lopt_str = get_meta_trainer(args)

    key = jax.random.PRNGKey(0)
    key, key1 = jax.random.split(key)
    outer_trainer_state = meta_trainer.init(key1)

    if args.from_checkpoint:
        outer_trainer_state = checkpoints.load_state(
            "./" + lopt_str + ".ckpt", outer_trainer_state
        )

    run = wandb.init(project="learned_aggregation", group=lopt_str)

    for i in range(args.num_outer_steps):
        key, key1 = jax.random.split(key)
        outer_trainer_state, meta_loss, _ = meta_trainer.update(
            outer_trainer_state, key1, with_metrics=False
        )
        run.log({args.task + " meta loss": meta_loss})

        if (i + 1) % 1000 == 0:  # Checkpoint every 1000th iteration
            checkpoints.save_state("./" + lopt_str + ".ckpt", outer_trainer_state)

    run.finish()

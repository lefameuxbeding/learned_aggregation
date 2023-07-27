import pickle

from tqdm import tqdm
import jax
import wandb
from learned_optimization import checkpoints

from meta_trainers import get_meta_trainer


def meta_train(args):
    meta_trainer = get_meta_trainer(args)

    key = jax.random.PRNGKey(0)
    key, key1 = jax.random.split(key)
    outer_trainer_state = meta_trainer.init(key1)

    if args.from_checkpoint:
        outer_trainer_state = checkpoints.load_state(
            "./" + args.meta_train_name + ".ckpt", outer_trainer_state
        )

    run = wandb.init(
        project="learned_aggregation_meta_train", group=args.meta_train_name
    )

    for i in tqdm(range(args.num_outer_steps), ascii=True, desc="Outer Loop"):
        key, key1 = jax.random.split(key)
        outer_trainer_state, meta_loss, _ = meta_trainer.update(
            outer_trainer_state, key1, with_metrics=False
        )
        run.log({args.task + " meta loss": meta_loss})

        if (i + 1) % 1000 == 0:  # Checkpoint every 1000th iteration
            checkpoints.save_state(
                "./" + args.meta_train_name + ".ckpt", outer_trainer_state
            )
            with open("./" + args.meta_train_name + ".pickle", "wb") as f:
                pickle.dump(
                    outer_trainer_state.gradient_learner_state.theta_opt_state.params, f
                )

    run.finish()

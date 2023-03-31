import pickle
import jax
import wandb
from meta_trainers import get_meta_trainer


def meta_train(args):
    meta_trainer, lopt_str = get_meta_trainer(args.optimizer, args.task, args.num_inner_steps)

    key = jax.random.PRNGKey(0)
    key, key1 = jax.random.split(key)
    outer_trainer_state = meta_trainer.init(key1)

    run = wandb.init(project="learned_aggregation", group=lopt_str)

    for _ in range(args.num_outer_steps):
        key, key1 = jax.random.split(key)
        outer_trainer_state, meta_loss, _ = meta_trainer.update(outer_trainer_state, key1, with_metrics=False)
        run.log({"meta loss": meta_loss})

    run.finish()

    with open(lopt_str + ".pickle", "wb") as f:
        pickle.dump(outer_trainer_state.gradient_learner_state.theta_opt_state.params, f)

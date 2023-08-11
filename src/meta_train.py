import pickle
import os.path as osp
from tqdm import tqdm
import jax
import wandb
from learned_optimization import checkpoints

from meta_trainers import get_meta_trainer

from glob import glob
import os
import shutil
import re


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def delete_old_checkpoints(save_dir, n_to_keep):
    ckpt_dir_regex = r"global_step[\d]*"
    if save_dir.endswith("/"):
        save_dir = save_dir.strip("/")
    all_ckpts = natural_sort(
        [
            i
            for i in glob(f"{save_dir}/*")
            if i.endswith(".ckpt") and re.search(ckpt_dir_regex, i)
        ]
    )
    all_pkl = natural_sort(
        [
            i
            for i in glob(f"{save_dir}/*")
            if i.endswith(".pickle") and re.search(ckpt_dir_regex, i)
        ]
    )

    n_to_delete = len(all_ckpts) - n_to_keep
    if n_to_delete > 0:
        to_delete_ckpt = all_ckpts[:n_to_delete]
        to_delete_pkl = all_pkl[:n_to_delete]
        print(
            f"WARNING: Deleting old checkpoints: \n\t{', '.join(to_delete_ckpt + to_delete_pkl)}"
        )
        for ckpt in to_delete_ckpt + to_delete_pkl:
            try:
                os.remove(ckpt)
            except FileNotFoundError:
                pass



def save_checkpoint(prefix,i,args,outer_trainer_state):  # Checkpoint every 1000th iteration
    save_dir = osp.join("checkpoints", prefix + args.meta_train_name)
    checkpoints.save_state(
        osp.join(
            save_dir,
            "global_step{}.ckpt".format(i + 1),
        ),
        outer_trainer_state,
    )
    pickle_filename = osp.join(save_dir, "global_step{}.pickle".format(i + 1),)
    with open(pickle_filename, "wb",) as f:
        pickle.dump(
            outer_trainer_state.gradient_learner_state.theta_opt_state.params, f
        )

    with open(osp.join(save_dir, "latest"), "w") as f:
        f.write("global_step{}".format(i + 1))

    delete_old_checkpoints(
        save_dir=save_dir,
        n_to_keep=args.checkpoints_to_keep,
    )

    return pickle_filename


def meta_train(args):
    meta_trainer = get_meta_trainer(args)
    # print(meta_trainer)
    # print(meta_trainer.__dict__)
    # print(meta_trainer.gradient_learner._theta_opt.opt)
    # exit(0)

    key = jax.random.PRNGKey(0)
    key, key1 = jax.random.split(key)
    outer_trainer_state = meta_trainer.init(key1)

    if args.from_checkpoint:
        dirname = osp.join("checkpoints", args.meta_train_name)
        ckpt = open(osp.join(dirname, "latest"), "r").readline().strip()
        outer_trainer_state = checkpoints.load_state(
            osp.join(dirname, "{}.ckpt".format(ckpt)), outer_trainer_state
        )

    run = wandb.init(
        project="learned_aggregation_meta_train", group=args.meta_train_name, config=vars(args)
    )

    for i in tqdm(range(args.num_outer_steps), ascii=True, desc="Outer Loop"):
        key, key1 = jax.random.split(key)
        outer_trainer_state, meta_loss, _ = meta_trainer.update(
            outer_trainer_state, key1, with_metrics=False
        )
        # print(outer_trainer_state.gradient_learner_state.theta_opt_state)
        # exit(0)
        run.log({args.task + " meta loss": meta_loss})

        if (i + 1) % args.save_iter == 0:  # Checkpoint every 1000th iteration
            save_checkpoint(prefix=run.id,
                            i=i,
                            args=args,
                            outer_trainer_state=outer_trainer_state)

    savepath = save_checkpoint(
            prefix=run.id,
            i=i,
            args=args,
            outer_trainer_state=outer_trainer_state
        )

    wandb.save(savepath)

    run.finish()



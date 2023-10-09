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


def save_checkpoint(
    prefix, i, args, outer_trainer_state
):  # Checkpoint every 1000th iteration
    save_dir = osp.join("checkpoints", prefix + args.meta_train_name)
    checkpoints.save_state(
        osp.join(
            save_dir,
            "global_step{}.ckpt".format(i + 1),
        ),
        outer_trainer_state,
    )
    pickle_filename = osp.join(
        save_dir,
        "global_step{}.pickle".format(i + 1),
    )
    with open(
        pickle_filename,
        "wb",
    ) as f:
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


def get_ckpt_dirs(ckpt_dir, meta_train_name):
    a = os.listdir(ckpt_dir)
    keep = []
    for x in a:
        if osp.isdir(osp.join(ckpt_dir, x)) and x[8:] == meta_train_name:
            keep.append(x)
    return keep


def get_ckpt_to_load(ckpt_dir, dirs):
    def nat_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key[1])]
        return sorted(l, key=alphanum_key)

    sortable = []
    for x in dirs:
        if osp.isfile(osp.join(ckpt_dir, x, "latest")):
            ckpt = open(osp.join(ckpt_dir, x, "latest"), "r").readline().strip()
            sortable.append(
                (
                    osp.join(ckpt_dir, x, ckpt),
                    ckpt,
                )
            )
    sortable = nat_sort(sortable)

    keep = []
    for x in sortable:
        if x[1] == sortable[-1][1]:
            keep.append(x)
    if len(keep) > 1:
        print(
            "[Warning] multiple directories contain a checkpoint at the same latest iteration. Selecting one arbitrarily."
        )

    return keep[0]


def get_resume_ckpt(ckpt_dir, meta_train_name):
    dirs = get_ckpt_dirs(ckpt_dir, meta_train_name)
    if len(dirs) == 0:
        print("[Info] No existing checkpoint found. Starting from scratch.")
        return None
    ckpt_path, suffix = get_ckpt_to_load(ckpt_dir, dirs)
    print("[Info] Loading checkpoint from {}".format(ckpt_path))
    return ckpt_path


def meta_train(args):
    meta_trainer, meta_opt = get_meta_trainer(args)

    key = jax.random.PRNGKey(0)
    key, key1 = jax.random.split(key)
    outer_trainer_state = meta_trainer.init(key1)

    if args.from_checkpoint:
        dirname = osp.join("checkpoints", args.meta_train_name)
        ckpt = open(osp.join(dirname, "latest"), "r").readline().strip()
        outer_trainer_state = checkpoints.load_state(
            osp.join(dirname, "{}.ckpt".format(ckpt)), outer_trainer_state
        )
    elif args.auto_resume:
        ckpt = get_resume_ckpt("checkpoints", args.meta_train_name)
        if ckpt is not None:
            outer_trainer_state = checkpoints.load_state(
                "{}.ckpt".format(ckpt), outer_trainer_state
            )

    run = wandb.init(
        project="learned_aggregation_meta_train",
        group=args.meta_train_name,
        config=vars(args),
    )

    iteration = int(
        outer_trainer_state.gradient_learner_state.theta_opt_state.iteration
    )
    for i in tqdm(
        range(iteration, args.num_outer_steps),
        initial=iteration,
        total=args.num_outer_steps,
        ascii=True,
        desc="Outer Loop",
    ):
        key, key1 = jax.random.split(key)
        outer_trainer_state, meta_loss, _ = meta_trainer.update(
            outer_trainer_state, key1, with_metrics=False
        )
        run.log(
            {
                "iteration": i,
                args.task + " meta loss": meta_loss,
                "learning rate": meta_opt.__dict__.get(
                    "schedule_", lambda x: args.learning_rate
                )(
                    outer_trainer_state.gradient_learner_state.theta_opt_state.iteration
                    - 1
                ),
            }
        )

        if (i + 1) % args.save_iter == 0:  # Checkpoint every 1000th iteration
            save_checkpoint(
                prefix=run.id, i=i, args=args, outer_trainer_state=outer_trainer_state
            )

    savepath = save_checkpoint(
        prefix=run.id, i=i, args=args, outer_trainer_state=outer_trainer_state
    )

    wandb.save(savepath)

    run.finish()

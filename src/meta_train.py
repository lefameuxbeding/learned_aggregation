import pickle
import os.path as osp
from tqdm import tqdm
import jax
import wandb
from learned_optimization import checkpoints

from meta_trainers import get_meta_trainer
import globals

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


import csv

def save_timings_to_csv(timings, filename, column_name):
    """
    Saves the timings to a CSV file.

    :param timings: List of execution times.
    :param filename: Name of the file to save the data.
    :param column_name: Name of the column under which timings are saved.
    """
    # Calculate and print the average timing
    average_timing = sum(timings) / len(timings)
    print(f"Average timing: {average_timing} seconds")

    # Save the timings to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([column_name])  # Write the header
        for timing in timings:
            writer.writerow([timing])


def meta_train(args):
    meta_trainer, meta_opt = get_meta_trainer(args)

    key = jax.random.PRNGKey(0)
    key, key1 = jax.random.split(key)
    outer_trainer_state = meta_trainer.init(key1)

    # import pprint
    # pprint.pprint(jax.tree_map(lambda x:x.shape if type(x) != int else x,outer_trainer_state.__dict__))
    # exit(0)

    # Set up globals used in truncated step for meta-training
    globals.needs_state = args.needs_state
    globals.num_grads = args.num_grads
    globals.num_local_steps = args.num_local_steps
    globals.local_batch_size = args.local_batch_size
    globals.use_pmap = args.use_pmap
    globals.num_devices = args.num_devices

    if args.use_pmap:
        assert args.num_grads % args.num_devices == 0, "The number of devices for parallelism should be a divisor of the number of clients (gradients)"

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
        outer_trainer_state, meta_loss, metrics = meta_trainer.update(
            outer_trainer_state, key1, with_metrics=True
        )

        # ts = meta_trainer.gradient_estimators[0].truncated_step
        # if len(ts.timings) >= 200:
        #     save_timings_to_csv(ts.timings, 
        #                         ts._task_name + '_spj:10_nt:{}_bs:{}_ls:{}_k:{}.csv'.format(args.num_tasks,
        #                                                                          args.local_batch_size, 
        #                                                                          args.num_local_steps, 
        #                                                                          args.num_grads), 
        #                         ts._task_name)
        #     exit(0)
        # exit(0) 
        # print()

        more_to_log = {
                "iteration": i,
                args.task + " meta loss": meta_loss,
                "learning rate": meta_opt.__dict__.get(
                    "schedule_", lambda x: args.learning_rate
                )(
                    outer_trainer_state.gradient_learner_state.theta_opt_state.iteration
                    - 1
                ),
            }
        
        metrics.update(more_to_log)
        run.log(
            metrics
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

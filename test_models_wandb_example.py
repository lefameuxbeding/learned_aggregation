import os
import wandb

import os.path as osp
from datetime import datetime

# Initialize API
api = wandb.Api()

# Define the workspace and project
workspace = "eb-lab"
project = "learned_aggregation_meta_train"  # Replace with your project's name

compare_date_str = "2023-08-11T00:00:00.000Z"
compare_date = datetime.strptime(compare_date_str, "%Y-%m-%dT%H:%M:%S.%fZ")


def parse_date(date_str):
    """Parse a date string which might or might not have fractional seconds."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")


tested = {
    "fedlagg": {(h, ll): False for h, ll in [(4, 1.0), (8, 0.5), (16, 0.5), (32, 0.5)]},
    "fedlopt": {(h, ll): False for h, ll in [(4, 1.0), (8, 1.0), (16, 0.5), (32, 0.3)]},
    "fedlagg-adafac": {
        (h, ll): False for h, ll in [(4, 0.5), (8, 0.3), (16, 0.3), (32, 0.1)]
    },
}

config_map = {
    "fedlagg": "config/meta_test/small-image-mlp-fmst_fedlagg.py",
    "fedlopt": "config/meta_test/small-image-mlp-fmst_fedlopt.py",
    "fedlagg-adafac": "config/meta_test/small-image-mlp-fmst_fedlagg-adafac.py",
    "fedlagg-wavg": "config/meta_test/small-image-mlp-fmst_fedlagg-wavg.py",
    "fedlopt-adafac": "config/meta_test/small-image-mlp-fmst_fedlopt-adafac.py",
}


def get_checkpoint(files, download_dir="./"):
    ckpts = [x for x in files if "global_step" in x.name]
    if len(ckpts) > 1:
        print(ckpts)

    assert (
        len(ckpts) <= 1
    ), "multiple checkpoints exist can't determine which one to use"

    if len(ckpts) == 0:
        return None

    ckpts[0].download(download_dir, replace=True)
    return osp.join(download_dir, ckpts[0].name)


cmd_prefix = ""  # "CUDA_VISIBLE_DEVICES=1 "
runs = api.runs(f"{workspace}/{project}")
for i, run in enumerate(runs):
    try:
        run.config["name_suffix"]
    except KeyError:
        continue

    if (
        run.config["name_suffix"] == "z"
        and (
            run.config["optimizer"] == "fedlopt-adafac"
            or run.config["optimizer"] == "fedlagg-adafac"
        )
        and run.config["num_grads"] == 16
        and run.config["num_local_steps"] == 4
    ):
        # print(run.created_at)
        # run_date = parse_date(run.created_at)
        # if run_date > compare_date and run.config['optimizer'] in list(config_map.keys()):

        #     # print(run.created_at)
        k = run.config["num_grads"]
        h = run.config["num_local_steps"]
        llr = run.config["local_learning_rate"]
        opt = run.config["optimizer"]

        #     try:
        #         tested[opt][(h,llr)]
        #     except KeyError:
        #         continue

        #     if tested[opt][(h,llr)]:
        #         continue

        ckpt_name = get_checkpoint(run.files(), download_dir="./")

        #     if ckpt_name is None:
        #          continue

        task = "image-mlp-fmst"

        command = (
            cmd_prefix
            + "python src/main.py --config {} --name_suffix {} --local_learning_rate {} --num_grads {} --num_local_steps {} --test_checkpoint {} --task {}".format(
                config_map[opt],
                "_5K_iters_on_{}".format(task),
                llr,
                k,
                h,
                ckpt_name,
                task,
            )
        )

        print(command)
        print(run.id)
        # os.system(command)

    #     tested[opt][(h,llr)] = True

    else:
        continue

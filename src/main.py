import argparse
import os
import sys

from jax.lib import xla_bridge
import wandb
import os.path as osp

from benchmark import benchmark, sweep
from meta_train import meta_train
import tensorflow as tf

from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_type", type=str, choices=["benchmark", "meta-train","sweep"])
    parser.add_argument("--optimizer", type=str, choices=["sgd",
                                                          "adam", 
                                                          "fedavg", 
                                                          "fedavg-slowmo", 
                                                          "fedlopt", 
                                                          "fedlopt-adafac", 
                                                          "fedlagg", 
                                                          "fedlagg-wavg", 
                                                          "fedlagg-adafac"])
    parser.add_argument("--task", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--local_learning_rate", type=float)
    parser.add_argument("--local_batch_size", type=int)
    parser.add_argument("--num_grads", type=int)
    parser.add_argument("--num_local_steps", type=int)
    parser.add_argument("--num_runs", type=int)
    parser.add_argument("--num_inner_steps", type=int)
    parser.add_argument("--num_outer_steps", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--sweep_config", type=str)
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--test_checkpoint", type=str)
    parser.add_argument("--use_pmap", action="store_true")
    parser.add_argument("--num_devices", type=int)
    parser.add_argument("--num_tasks", type=int)
    parser.add_argument("--name_suffix", type=str)
    parser.add_argument("--slowmo_learning_rate", type=float)
    parser.add_argument("--wandb_checkpoint_id", type=str)
    parser.add_argument("--meta_loss_split", type=str)
    parser.add_argument("--test_project", type=str)
    parser.add_argument("--tfds_data_dir", type=str, default=os.getenv("SLURM_TMPDIR"))
    parser.add_argument("--wandb_dir", type=str, default=os.getenv("SCRATCH"))
    parser.add_argument("--auto_resume", action="store_true")
    # fmt: on

    return parser.parse_args()


def assert_args(args):
    # fmt: off
    if args.run_type == "benchmark" and args.optimizer in ["fedlopt", "fedlopt-adafac", "fedlagg", "fedlagg-wavg", "fedlagg-adafac"]:
        assert os.path.exists(args.test_checkpoint), "need to meta-train learned optimizer before benchmarking"
        assert args.test_checkpoint.endswith('.pickle'), "optimizer checkpoints must be saved as .pickle files"
    if args.run_type == "meta-train":
        assert args.optimizer not in ["adam", "fedavg", "fedavg-slowmo"], "can't meta-train a non learned optimizer"
    # fmt: on


def download_wandb_checkpoint(cfg):
    api = wandb.Api()
    run = api.run(cfg.wandb_checkpoint_id)

    ckpts = [x for x in run.files() if "global_step" in x.name]
    if len(ckpts) > 1:
        print(ckpts)

    assert (
        len(ckpts) <= 1
    ), "multiple checkpoints exist can't determine which one to use"

    if len(ckpts) == 0:
        return None

    ckpts[0].download("/tmp", replace=True)
    return osp.join("/tmp", ckpts[0].name)


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], "GPU")

    print(xla_bridge.get_backend().platform)

    args = parse_args()

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = args.tfds_data_dir
    os.environ["WANDB_DIR"] = args.wandb_dir

    cfg = Config.fromfile(args.config)

    # override args from the command line
    for k, v in vars(args).items():
        if v is not None:
            print("[INFO] Overriding config value: {}={}".format(k, v))
            cfg._cfg_dict[k] = v

    cfg.name = "{}{}_{}{}".format(
        cfg.optimizer, cfg.hidden_size, cfg.task, cfg.name_suffix
    )
    cfg.meta_train_name = "{}{}_{}_K{}_H{}_{}{}".format(
        cfg.optimizer,
        cfg.hidden_size,
        cfg.task,
        cfg.num_grads,
        cfg.num_local_steps,
        cfg.local_learning_rate,
        cfg.name_suffix,
    )

    if cfg.wandb_checkpoint_id is not None:
        cfg.test_checkpoint = download_wandb_checkpoint(cfg)

    args = argparse.Namespace(**cfg._cfg_dict)

    assert_args(args)

    run_types = {"benchmark": benchmark, "meta-train": meta_train, "sweep": sweep}

    run_types[args.run_type](args)

import argparse
import os
import sys

from jax.lib import xla_bridge

from benchmark import benchmark, sweep
from meta_train import meta_train
import tensorflow as tf

from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--run_type", type=str, choices=["benchmark", "meta-train","sweep"], required=False)
    parser.add_argument("--optimizer", type=str, choices=["adam", "fedavg", "fedavg-slowmo", "fedlopt", "fedlopt-adafac", "fedlagg", "fedlagg-wavg", "fedlagg-adafac"], required=False)
    parser.add_argument("--task", type=str, choices=["image-mlp-fmst", "small-image-mlp-fmst", "conv-c10", "small-conv-c10", 'conv-imagenet', 'conv-imagenet64'], required=False)
    parser.add_argument("--name", type=str, required=False)
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
    parser.add_argument("--sweep_config", type=str, required=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--test_checkpoint", type=str)
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


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], "GPU")

    print(xla_bridge.get_backend().platform)

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")
    os.environ["WANDB_DIR"] = os.getenv("SCRATCH")

    args = parse_args()
    cfg = Config.fromfile(args.config)

    # override args from the command line
    for k, v in vars(args).items():
        if v is not None:
            print("[INFO] Overriding config value: {}={}".format(k, v))
            cfg._cfg_dict[k] = v

    cfg.name = "{}_{}".format(cfg.optimizer, cfg.task)
    cfg.meta_train_name = "{}{}_{}_K{}_H{}_{}".format(
        cfg.optimizer,
        cfg.hidden_size,
        cfg.task,
        cfg.num_grads,
        cfg.num_local_steps,
        cfg.learning_rate,
    )
    args = argparse.Namespace(**cfg._cfg_dict)

    assert_args(args)

    run_types = {"benchmark": benchmark, "meta-train": meta_train, "sweep": sweep}

    run_types[args.run_type](args)

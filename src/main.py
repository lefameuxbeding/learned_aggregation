import argparse
import os
import sys

from jax.lib import xla_bridge

from benchmark import benchmark, sweep
from meta_train import meta_train
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--run_type", type=str, choices=["benchmark", "meta-train","sweep"], required=True)
    parser.add_argument("--optimizer", type=str, choices=["adam", "fedavg", "fedavg-slowmo", "fedlopt", "fedlopt-adafac", "fedlagg", "fedlagg-wavg", "fedlagg-adafac"], required=True)
    parser.add_argument("--task", type=str, choices=["image-mlp-fmst", "small-image-mlp-fmst", "conv-c10", "small-conv-c10", 'conv-imagenet', 'conv-imagenet64'], required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--local_learning_rate", type=float, default=5e-1)
    parser.add_argument("--local_batch_size", type=int, default=128)
    parser.add_argument("--num_grads", type=int, default=8)
    parser.add_argument("--num_local_steps", type=int, default=4)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--num_inner_steps", type=int, default=2000)
    parser.add_argument("--num_outer_steps", type=int, default=50000)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--sweep_config", type=str, required=False)
    parser.add_argument("--from_checkpoint", action="store_true")
    # fmt: on

    return parser.parse_args()


def assert_args(args):
    # fmt: off
    if args.run_type == "benchmark" and args.optimizer in ["fedlopt", "fedlopt-adafac", "fedlagg", "fedlagg-wavg", "fedlagg-adafac"]:
        assert os.path.exists( "./models/small-image-mlp/" + args.name + ".pickle"), "need to meta-train learned optimizer before benchmarking"
    if args.run_type == "meta-train":
        assert args.optimizer not in ["adam", "fedavg", "fedavg-slowmo"], "can't meta-train a non learned optimizer"
    # fmt: on


if __name__ == "__main__":

    tf.config.experimental.set_visible_devices([], "GPU")

    print(xla_bridge.get_backend().platform)

    sys.path.append(os.getcwd())
    # os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")
    # os.environ["WANDB_DIR"] = os.getenv("SCRATCH")

    args = parse_args()
    assert_args(args)

    # print(args.__dict__)
    # print(vars(args))
    # print(argparse.Namespace(**args.__dict__))
    # exit(0)

    run_types = {"benchmark": benchmark, 
                 "meta-train": meta_train,
                 "sweep": sweep}

    run_types[args.run_type](args)

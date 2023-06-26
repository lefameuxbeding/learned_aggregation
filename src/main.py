import argparse
import os
import sys

from jax.lib import xla_bridge

from benchmark import benchmark
from meta_train import meta_train


def parse_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--run_type", type=str, choices=["benchmark", "meta-train"], required=True)
    parser.add_argument("--optimizer", type=str, choices=["fedavg", "fedlagg", "fedlagg-wavg", "fedlagg-adafac"], required=True)
    parser.add_argument("--task", type=str, choices=["image-mlp", "small-image-mlp", "conv"], required=True)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--local_learning_rate", type=float, default=5e-1)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_grads", type=int, default=8)
    parser.add_argument("--num_local_steps", type=int, default=4)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--num_inner_steps", type=int, default=2000)
    parser.add_argument("--num_outer_steps", type=int, default=20000)
    parser.add_argument("--from_checkpoint", action="store_true")
    # fmt: on

    return parser.parse_args()


def assert_args(args):
    # fmt: off
    # if args.run_type == "benchmark" and args.optimizer in ["lopt", "lagg"]:
    #     assert os.path.exists(args.optimizer + ".pickle"), "need to meta-train learned optimizer before benchmarking" # TODO
    if args.run_type == "meta-train":
        assert args.optimizer not in ["nadamw", "adam"], "can't meta-train a non learned optimizer"
    # fmt: on


if __name__ == "__main__":
    print(xla_bridge.get_backend().platform)

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")
    os.environ["WANDB_DIR"] = os.getenv("SCRATCH")

    args = parse_args()
    assert_args(args)

    run_types = {"benchmark": benchmark, "meta-train": meta_train}
    run_types[args.run_type](args)

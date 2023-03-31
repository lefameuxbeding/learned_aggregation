import os
import sys
import argparse
from benchmark import benchmark
from meta_train import meta_train


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_type", type=str, choices=["benchmark", "meta-train"], required=True)
    parser.add_argument("--optimizer", type=str, choices=["nadamw", "lopt"], required=True)
    parser.add_argument("--task", type=str, choices=["image_mlp"], required=True)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--num_inner_steps", type=int, default=500)
    parser.add_argument("--num_outer_steps", type=int, default=10000)

    return parser.parse_args()


def assert_args(args):
    if args.run_type == "benchmark" and args.optimizer != "nadamw":
        assert os.path.exists(args.optimizer + ".pickle"), "need to meta-train learned optimizer before benchmarking"
    if args.run_type == "meta-train":
        assert args.optimizer != "nadamw", "can't meta-train a non learned optimizer"


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")
    os.environ["WANDB_DIR"] = os.getenv("SCRATCH")

    args = parse_args()
    assert_args(args)

    run_types = {
        "benchmark" : benchmark,
        "meta-train" : meta_train
    }
    run_types[args.run_type](args)
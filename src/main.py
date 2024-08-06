import argparse
import os
import sys

from jax.lib import xla_bridge
import jax
from jax import lax
import numpy as np
import jax.numpy as jnp
import wandb
import os.path as osp

from benchmark import benchmark, sweep
from meta_train import meta_train
import tensorflow as tf

from mmengine.config import Config

from helpers import print_rank_0

def comma_separated_strings(string):
    # This function will be used to parse the comma-separated string into a list
    return string.split(',')

def parse_args():
    parser = argparse.ArgumentParser()

    

    # fmt: off
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_type", type=str, choices=["benchmark", "meta-train","sweep"])
    parser.add_argument("--optimizer", type=str, choices=["sgd",
                                                          "adam", 
                                                          'muadam',
                                                          "adamw",
                                                          "fedavg", 
                                                          "fedavg-slowmo", 
                                                          "fedlopt", 
                                                          "fedlopt-adafac", 
                                                          "fedlagg", 
                                                          "fedlagg-wavg", 
                                                          "fedlagg-adafac",
                                                          'small_fc_mlp',
                                                          'mup_small_fc_mlp',
                                                          "velo",
                                                          "lion","RNNMLPLOpt",
                                                          "MuRNNMLPLOpt",
                                                          'MuHyperV2'])
    parser.add_argument("--task", type=comma_separated_strings)
    parser.add_argument("--needs_state", action="store_true")
    parser.add_argument("--name", type=str)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--local_learning_rate", type=float)
    parser.add_argument("--local_batch_size", metavar='N', type=int, nargs='+',
                        help='an integer for the list')
    parser.add_argument("--num_grads", type=int)
    parser.add_argument("--num_local_steps", type=int)
    parser.add_argument("--steps_per_jit", type=int)
    parser.add_argument("--num_runs", type=int)
    parser.add_argument("--num_inner_steps", type=int)
    parser.add_argument("--num_outer_steps", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--sweep_config", type=str)
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--test_checkpoint", type=str)
    parser.add_argument("--use_pmap", action="store_true")
    parser.add_argument("--num_tasks", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--num_devices", type=int)
    parser.add_argument("--name_suffix", type=str)
    parser.add_argument("--slowmo_learning_rate", type=float)
    parser.add_argument("--wandb_checkpoint_id", type=str)
    parser.add_argument("--meta_loss_split", type=str)
    parser.add_argument("--test_project", type=str)
    parser.add_argument("--train_project", type=str)
    parser.add_argument("--tfds_data_dir", type=str, default="/network/scratch/b/benjamin.therien/data/tensorflow_datasets") # os.getenv("SLURM_TMPDIR") "/network/scratch/b/benjamin.therien/data/tensorflow_datasets"
    parser.add_argument("--wandb_dir", type=str, default=os.getenv("SCRATCH"))
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--truncation_schedule_min_length", type=int)
    parser.add_argument("--sweep_id", type=str)
    parser.add_argument("--lo_clip_grad", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--test_interval", type=int)
    parser.add_argument("--prefetch_batches", type=int)
    parser.add_argument("--adafac_step_mult", type=float)
    parser.add_argument("--mup_input_mult", type=float)
    parser.add_argument("--mup_output_mult", type=float)
    parser.add_argument("--mup_hidden_lr_mult", type=float)
    parser.add_argument("--keep_batch_in_gpu_memory", action="store_true")
    parser.add_argument("--seed", type=int)
    
    parser.add_argument("--truncation_length", type=int)

    parser.add_argument("--start_from_test_ckpt", action="store_true")
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



def test_bf16_support_on_gpu():
    # Check if there is any GPU available
    gpus = jax.devices()#[device for device in jax.devices() if 'gpu' in device.device_kind.lower()]
    if not gpus:
        print("No GPU devices found.")
        return
    
    # Select the first GPU device
    gpu = gpus[0]
    jax.devices().append(gpu)
    print(f"Testing on GPU: {gpu}")

    try:
        # Create test data in BF16
        a = lax.convert_element_type(np.array([1.0, 2.0, 3.0]), jnp.bfloat16)
        b = lax.convert_element_type(np.array([1.0, 2.0, 3.0]), jnp.bfloat16)
        
        # Perform an addition operation on GPU
        result = lax.add(a, b)

        # Print the results to verify
        print("BF16 operation successful on GPU. Result:", result)
    except Exception as e:
        print(f"Failed to perform BF16 operations on GPU: {e}")





if __name__ == "__main__":

    # Replace 'coordinator-service' with your actual service name or IP address
    # coordinator_address = "coordinator-service:1234"

    # jax.distributed.initialize(coordinator_address=coordinator_address)


    # Initialize the distributed environment
    # jax.distributed.initialize(coordinator_address=) 

    tf.config.experimental.set_visible_devices([], "GPU")

    print(xla_bridge.get_backend().platform)
    print(jax.devices())


    args = parse_args()

    assert len(args.local_batch_size) == len(args.task), f"local batch size and task length mismatch: {len(args.local_batch_size)} != {len(args.task)}"

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = args.tfds_data_dir
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.999'
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
        cfg.task[0] if len(cfg.task) == 1 else "multi-task-with"+cfg.task[0],
        cfg.num_grads,
        cfg.num_local_steps,
        cfg.local_learning_rate,
        cfg.name_suffix,
    )
    cfg.num_devices = len(jax.devices())

    if cfg.wandb_checkpoint_id is not None:
        cfg.test_checkpoint = download_wandb_checkpoint(cfg)

    args = argparse.Namespace(**cfg._cfg_dict)
    assert_args(args)

    if args.use_bf16 and test_bf16_support_on_gpu():
        print('setting bf 16 as default supported')
        jax.config.update('jax_default_matmul_precision', 'bfloat16')# coordinator_address='10.0.0.1:1234', num_processes=2)

   
    # Check the rank of the current process# Check the rank of the current process
    args.rank = jax.process_index()
    # print(f"Process rank: {rank}")

    # Get the world size
    args.world_size = jax.process_count()
    # print(f"World size: {world_size}")

    if args.world_size > 1:
        #setup distributed
        assert len(args.task) % args.world_size == 0, "world size must divide the number of tasks"

        args.task = np.array(args.task).reshape(args.world_size, -1)[args.rank].tolist()

        args.local_batch_size = np.array(args.local_batch_size).reshape(args.world_size, -1)[args.rank].tolist()

        print(args.rank, args.task)


    # exit(0)

    
    run_types = {"benchmark": benchmark, 
                 "meta-train": meta_train, 
                 "sweep": sweep}
    run_types[args.run_type](args)




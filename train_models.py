 parser = argparse.ArgumentParser()

    # fmt: off
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--run_type", type=str, choices=["benchmark", "meta-train","sweep"])
parser.add_argument("--optimizer", type=str, choices=["adam", "fedavg", "fedavg-slowmo", "fedlopt", "fedlopt-adafac", "fedlagg", "fedlagg-wavg", "fedlagg-adafac"])
parser.add_argument("--task", type=str, choices=["image-mlp-fmst", "small-image-mlp-fmst", "conv-c10", "small-conv-c10", 'conv-imagenet', 'conv-imagenet64'])
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
parser.add_argument("--from_checkpoint", type=bool)
parser.add_argument("--test_checkpoint", type=str)
parser.add_argument("--use_pmap", action="store_true")
parser.add_argument("--num_devices", type=int)
parser.add_argument("--name_suffix", type=str)
parser.add_argument("--gpu", type=str)
# fmt: on

args = parser.parse_args()


prefix = "CUDA_VISIBLE_DEVICES={} ".format(args.gpu)
for h in [4,8,16,32]:
    for local_learning_rate in [0.1,0.3,0.5,1.]:
        command = prefix + "python ./src/main.py --config {} --task {} --local_learning_rate {} --num_local_steps {}".format(
            args.config, args.task, local_learning_rate)


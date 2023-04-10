from learned_optimization.tasks.fixed.conv import Conv_Cifar10_32x64x64
from learned_optimization.tasks.fixed.image_mlp import ImageMLP_FashionMnist_Relu128x128


def get_task(args):
    tasks = {
        "image-mlp": ImageMLP_FashionMnist_Relu128x128(),
        "conv": Conv_Cifar10_32x64x64(),
    }

    return tasks[args.task]

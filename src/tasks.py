import gin
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask


@gin.configurable
def My_ImageMLP_FashionMnist_Relu128x128(args):
    """A 2 hidden layer, 128 hidden unit MLP designed for fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=args.batch_size)
    return _MLPImageTask(datasets, [128, 128])


def get_task(args):
    tasks = {
        "image-mlp": My_ImageMLP_FashionMnist_Relu128x128(args),
    }

    return tasks[args.task]

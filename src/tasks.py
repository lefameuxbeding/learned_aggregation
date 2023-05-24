import gin
import jax
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.fixed.conv import _ConvTask, _cross_entropy_pool_loss
from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask


@gin.configurable
def My_Conv_Cifar10_32x64x64(args):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=10)
    datasets = image.cifar10_datasets(batch_size=args.batch_size)
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def My_ImageMLP_FashionMnist_Relu128x128(args):
    """A 2 hidden layer, 128 hidden unit MLP designed for fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=args.batch_size)
    return _MLPImageTask(datasets, [128, 128])


def get_task(args):
    tasks = {
        "image-mlp": My_ImageMLP_FashionMnist_Relu128x128,
        "conv": My_Conv_Cifar10_32x64x64,
    }

    return tasks[args.task](args)

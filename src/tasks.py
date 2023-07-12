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
def My_Conv_Cifar10_8_16x32(args):
    """A 2 hidden layer convnet designed for 8x8 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=10)
    datasets = image.cifar10_datasets(batch_size=args.batch_size, image_size=(8, 8))
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def My_ImageMLP_FashionMnist_Relu128x128(args):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=args.batch_size)
    return _MLPImageTask(datasets, [128, 128])


@gin.configurable
def My_ImageMLP_FashionMnist8_Relu32(args):
    """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
    datasets = image.fashion_mnist_datasets(
        batch_size=args.batch_size, image_size=(8, 8)
    )
    return _MLPImageTask(datasets, [32])


def get_task(args):
    tasks = {
        "image-mlp-fmst": My_ImageMLP_FashionMnist_Relu128x128,
        "small-image-mlp-fmst": My_ImageMLP_FashionMnist8_Relu32,
        "conv-c10": My_Conv_Cifar10_32x64x64,
        "small-conv-c10": My_Conv_Cifar10_8_16x32,
    }

    return tasks[args.task](args)

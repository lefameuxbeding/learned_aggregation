import gin
import jax
from typing import Tuple
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.datasets import base
from learned_optimization.tasks.fixed.conv import _ConvTask, _cross_entropy_pool_loss
from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask

from learned_optimization.tasks import base as tasks_base
from learned_optimization import training

@gin.configurable
def My_Conv_Cifar10_32x64x64(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=10)
    datasets = image.cifar10_datasets(batch_size=batch_size,
                                      prefetch_batches=50,)
    return _ConvTask(base_model_fn, datasets)


ct = My_Conv_Cifar10_32x64x64(128)

task_family = tasks_base.single_task_to_family(ct)


def get_batch(steps, task_family, num_tasks=4, meta_loss_split=None):
    if steps is not None:
        data_shape = (steps, num_tasks)
    else:
        data_shape = (num_tasks,)
    tr_batch = training.get_batches(
        task_family, data_shape, numpy=True, split="train"
    )

    if meta_loss_split == "same_data" or meta_loss_split is None:
        return tr_batch
    else:
        outer_batch = training.get_batches(
            task_family, data_shape, numpy=True, split=meta_loss_split
        )
        return (tr_batch, outer_batch)
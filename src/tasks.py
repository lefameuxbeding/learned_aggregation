
"""
The following file defines tasks.

Tasks follow the following naming conventions:
MODEL_DATASET

for language modelling:
    DATASET = {dataset name}-s{sequence length}-v{vocab size}

for image datasets:
    DATASET = {dataset name}-{HxWxC}

for MLPs:
    MODEL = {model name}-w{width}-d{depth}

"""

import gin
import jax
import functools
import ml_collections
from typing import Tuple

from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.datasets import base
from learned_optimization.tasks.fixed.conv import _ConvTask, _cross_entropy_pool_loss
from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask
from learned_optimization.tasks.fixed.transformer_lm import _TransformerTask
from learned_optimization.tasks.fixed.vit import (VisionTransformerTask, wide16_config, 
            tall16_config, vit_p16_h128_m512_nh4_nl10_config, deit_tiny_config, deit_small_config)
from learned_optimization.tasks.fixed.vit_test import VITTest
from learned_optimization.tasks.parametric.image_resnet import ParametricImageResNet
from learned_optimization.tasks.resnet import ResNet
from learned_optimization.tasks.fixed.resnet import _ResnetTaskDataset

from fast_imagenet import fast_imagenet_datasets
from custom_tasks import  _MuTransformerTask, _MuMLPImageTask
from learned_optimization.tasks.datasets.language import _make_datasets, get_32k_sentence_piece_vocab

@gin.configurable
def mlp128x128_fastinet_32(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    h5_path = "/mnt/raid0/imagenet_hdf5/ilsvrc2012.hdf5"
    datasets = fast_imagenet_datasets(h5_path, 
        batch_size, 
        workers=48, 
        distributed=False,
        image_size=(32,32,),
        output_channel=(3,)
    )
    return _MLPImageTask(datasets, [128, 128])


@base.dataset_lru_cache
@gin.configurable
def imagenet_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (224, 224),
    **kwargs,
) -> base.Datasets:
    splits = ("train", "validation", "validation", "test")
    return base.tfds_image_classification_datasets(
        datasetname="imagenet2012",
        splits=splits,
        batch_size=batch_size,
        image_size=image_size,
        stack_channels=1,
        prefetch_batches=50,
        shuffle_buffer_size=10000,
        normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
        convert_to_black_and_white=False,
        **kwargs,
    )


@base.dataset_lru_cache
@gin.configurable
def imagenet_64_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (64, 64),
    # prefetch_batches=[20,1,1,1],
    data_fraction=1.0,
    **kwargs,
) -> base.Datasets:
    perc = max(1, int(80 * data_fraction))
    splits = (f"train[0:{perc}%]", "train[80%:90%]", "train[90%:]", "validation")
    return base.preload_tfds_image_classification_datasets(
        datasetname="imagenet_resized",
        splits=splits,
        batch_size=batch_size,
        image_size=image_size,
        stack_channels=1,
        # prefetch_batches=prefetch_batches,
        # shuffle_buffer_size=10000,
        normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
        convert_to_black_and_white=False,
        # cache=True,
        **kwargs,
    )

def func_create_func(task_fun, ds_args, model_args):
    ds_fun = ds_args['fun']
    # model_fun = model_args['fun']
    return task_fun(ds_fun(*ds_args['args'],**ds_args['kwargs']),**model_args)


def add_MLP_tasks(tasks, image_datasets, widths, depths):
    for k,ds in image_datasets.items():
        for mlp_width in widths:
            for mlp_depth in depths:
                tasks['mlp-w{}-d{}_{}'.format(mlp_width,mlp_depth,k)] = functools.partial(func_create_func, _MLPImageTask, ds,
                                                                                            dict(hidden_sizes=[mlp_width] * mlp_depth))
                
                tasks['mumlp-w{}-d{}_{}'.format(mlp_width,mlp_depth,k)] = functools.partial(func_create_func, 
                                                                                            _MuMLPImageTask, 
                                                                                            ds,
                                                                                            dict(hidden_sizes=[mlp_width] * mlp_depth))


def add_transformer_lm_tasks(tasks, lm_datasets, widths, depths):
    for k,ds in lm_datasets.items():
        for w,heads in widths:
            for d in depths:
                cfg = {
                    "num_heads": heads,
                    "d_model": w,
                    "num_layers": d,
                    "dropout_rate": 0.1,
                }
                name = 'transformer-w{}-d{}_{}'.format(w,d,k)
                tasks[name] = functools.partial(func_create_func, 
                                                _TransformerTask, 
                                                ds,
                                                dict(cfg=cfg,name=name))
                
                name = 'mutransformer-w{}-d{}_{}'.format(w,d,k)
                tasks[name] = functools.partial(func_create_func, 
                                                _MuTransformerTask, 
                                                ds,
                                                dict(cfg=cfg,name=name))



def deit_tiny_config():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 192
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 768
  config.transformer.num_heads = 3
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config


def deit_small_config():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 384
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 384 * 4
  config.transformer.num_heads = 6
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config



def add_vision_transformer_tasks(tasks, image_datasets, widths, depths):
    for k,ds in image_datasets.items():
        # for w,heads in widths:
        #     for d in depths:
        w=384
        d=12
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        _TransformerTask, 
                                        ds,
                                        dict(cfg=deit_small_config()))
        

        w=192
        d=12
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        _TransformerTask, 
                                        ds,
                                        dict(cfg=deit_tiny_config()))
        


def add_resnet_tasks(tasks, image_datasets, widths, depths):
    for k,ds in image_datasets.items():
        # for w,heads in widths:
        #     for d in depths:
        cfg = dict(initial_conv_kernel_size=7,
                initial_conv_stride=2,
                resnet_v2=False, 
                max_pool=True,
                **ResNet.CONFIGS[50])
        w=64
        d=50
        name = 'resnet50-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        _TransformerTask, 
                                        ds,
                                        dict(cfg=cfg))
        

        cfg = dict(initial_conv_kernel_size=7,
                initial_conv_stride=2,
                resnet_v2=False, 
                max_pool=True,
                **ResNet.CONFIGS[18])
        w=256
        d=18
        name = 'resnet50-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        _TransformerTask, 
                                        ds,
                                        dict(cfg=cfg))


def get_test_batch_size(task):
    if 'cifar' in task:
        return 10000
    elif 'food101' in task:
        return 10000
    elif 'fashionmnist' in task:
        return 10000
    elif 'imagenet' in task:
        return 50000
    elif 'lm1b' in task:
        return 128
    else:
        raise ValueError(f"Unknown task: {task}")
                
def get_task(args, is_test=False):

    created_tasks = []

    for chosen_task in args.task:

        if args.run_type == 'benchmark':
            batch_size = (args.meta_testing_batch_size,1,1,get_test_batch_size(chosen_task),)
            prefetch_batches = (2,1,1,2)
        else:
            if args.meta_loss_split is not None:
                batch_size = (args.meta_training_batch_size,1,4096,1,)
                prefetch_batches = (20,1,20,1)
            else:
                batch_size = (args.meta_training_batch_size,1,1,1,)
                prefetch_batches = (50,1,1,1)


        ds_kwargs = dict(prefetch_batches=prefetch_batches,
                            batch_shape=args.batch_shape,
                            label_sharding=args.label_sharding,
                            image_sharding=args.image_sharding,)


        IMAGE_DATASET_REGISTY = {
            'imagenet-32x32x3': dict(fun=imagenet_64_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(32, 32), **ds_kwargs)),
            'imagenet-64x64x3':  dict(fun=imagenet_64_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(64, 64), **ds_kwargs)),
            'imagenet-128x128x3':  dict(fun=imagenet_64_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(128, 128), **ds_kwargs)),
            'imagenet-224x224x3':  dict(fun=imagenet_64_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(224, 224), **ds_kwargs)),

            'cifar10-32x32x3': dict(fun=image.cifar10_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(32, 32), **ds_kwargs)),
            'food101-32x32x3': dict(fun=image.food101_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(32, 32), **ds_kwargs)),
            'fashionmnist-28x28x1': dict(fun=image.fashion_mnist_datasets,args=[],kwargs=dict(batch_size=batch_size, **ds_kwargs)),
        }

        LANGUAGE_DATASET_REGISTY = {
            'lm1b-s2048-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=2048)),
            'lm1b-s1024-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=1024)),
            'lm1b-s512-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=512)),
            'lm1b-s256-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=256)),
            'lm1b-s128-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=128)),
            'lm1b-s64-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=64)),
            'lm1b-s32-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=32)),
        }
        
        tasks = {}

        add_MLP_tasks(tasks, 
                    image_datasets=IMAGE_DATASET_REGISTY, 
                    widths=[2**i for i in range(16)], 
                    depths=[3,6,12])

        add_transformer_lm_tasks(tasks, 
                                lm_datasets=LANGUAGE_DATASET_REGISTY, 
                                widths=[(128,2),(192,3),(384,6),(768,12)], 
                                depths=[3,6,12])
        
        add_vision_transformer_tasks(tasks,
                                    image_datasets=IMAGE_DATASET_REGISTY, widths=[], depths=[])


        add_resnet_tasks(tasks, 
                        image_datasets=IMAGE_DATASET_REGISTY, widths=[], depths=[])
        
        created_tasks.append(tasks[chosen_task]())
    
    if len(created_tasks) == 1:
        return created_tasks[0]
    else:
        return created_tasks
    
    # return  tasks[args.task]()



    # print(tasks.keys())
    # exit(0)

    # if is_test:
    #     batch_size = test_batch_size[args.task]

    task = tasks[args.task]

    if type(task) is list:
        return [task(batch_size,
                     prefetch_batches=prefetch_batches,
                     batch_shape=args.batch_shape,
                     label_sharding=args.label_sharding,
                     image_sharding=args.image_sharding,) \
                for task in task]
    else:
        return tasks[args.task](batch_size,
                                prefetch_batches=prefetch_batches,
                                batch_shape=args.batch_shape,
                                label_sharding=args.label_sharding,
                                image_sharding=args.image_sharding,)

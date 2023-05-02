import jax.numpy as jnp
from haiku._src.data_structures import FlatMap


# TODO Could try to use vmap
def split_batch(batch, split):
    split_image = jnp.split(batch["image"], split)
    split_label = jnp.split(batch["label"], split)

    split_batch = []
    for i in range(split):
        sub_batch_dict = {}
        sub_batch_dict["image"] = split_image[i]
        sub_batch_dict["label"] = split_label[i]
        split_batch.append(FlatMap(sub_batch_dict))

    return split_batch
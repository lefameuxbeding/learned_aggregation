import sys
import os
import jax
import jax.numpy as jnp
import pickle
import wandb

from haiku._src.data_structures import FlatMap

from learned_optimization.tasks.fixed import image_mlp

from ..learned_aggregators import mlp_lagg


if __name__ == "__main__":
    """Environment"""

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")

    """Setup"""

    key = jax.random.PRNGKey(0)

    num_runs = 10
    num_inner_steps = 500

    task = image_mlp.ImageMLP_FashionMnist_Relu128x128()

    lagg = mlp_lagg.MLPLAgg(num_grads=8)
    agg_str = "PerParamMLPAgg_" + str(lagg.num_grads) + "_PES"
    with open(agg_str + ".pickle", "rb") as f:
        meta_params = pickle.load(f)
    agg = lagg.opt_fn(meta_params)

    @jax.jit
    def update(opt_state, key, batch):
        params = agg.get_params(opt_state)
        loss = task.loss(params, key, batch)

        def sample_grad_fn(image, label):
            sub_batch_dict = {}
            sub_batch_dict["image"] = image
            sub_batch_dict["label"] = label
            sub_batch = FlatMap(sub_batch_dict)

            return jax.grad(task.loss)(params, key, sub_batch)

        split_image = jnp.split(batch["image"], lagg.num_grads)
        split_label = jnp.split(batch["label"], lagg.num_grads)
        grads = [
            sample_grad_fn(split_image[i], split_label[i])
            for i in range(lagg.num_grads)
        ]

        opt_state = agg.update(opt_state, grads, loss=loss)

        return opt_state, loss

    """Benchmarking"""

    for j in range(num_runs):
        run = wandb.init(project="learned_aggregation", group=agg_str)

        key, key1 = jax.random.split(key)
        params = task.init(key1)
        opt_state = agg.init(params)

        for i in range(num_inner_steps):
            batch = next(task.datasets.train)
            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, batch)

            run.log({task.name + " train loss 4": loss})

        run.finish()

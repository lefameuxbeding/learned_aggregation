import sys
import os
import jax
import jax.numpy as jnp
import wandb

from haiku._src.data_structures import FlatMap

from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.optimizers import base as opt_base
from learned_optimization.learned_optimizers import mlp_lopt

from meta_train import meta_train_lopt, meta_train_lagg
from mlp_lagg import MLPLAgg


if __name__ == "__main__":
    """Environment"""

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")

    """Setup"""

    num_runs = 10
    num_inner_steps = 500
    num_outer_steps = 10000

    key = jax.random.PRNGKey(0)
    task = image_mlp.ImageMLP_FashionMnist8_Relu32()

    """Adam"""

    adam = opt_base.Adam(1e-2)

    @jax.jit
    def adam_update(opt_state, key, batch):
        key, key1 = jax.random.split(key)
        params = adam.get_params(opt_state)
        loss, grad = jax.value_and_grad(task.loss)(params, key1, batch)
        opt_state = adam.update(opt_state, grad, loss=loss)

        return opt_state, key, loss

    """Per Parameter MLP Optimizer"""

    per_param_mlp_lopt = mlp_lopt.MLPLOpt()
    per_param_mlp_opt_str = "PerParamMLPOpt"

    key, key1 = jax.random.split(key)
    per_param_mlp_lopt_meta_params = meta_train_lopt(
        per_param_mlp_lopt,
        per_param_mlp_opt_str,
        key1,
        num_inner_steps,
        num_outer_steps,
    )

    per_param_mlp_opt = per_param_mlp_lopt.opt_fn(per_param_mlp_lopt_meta_params)

    @jax.jit
    def per_param_mlp_opt_update(opt_state, key, batch):
        key, key1 = jax.random.split(key)
        params = per_param_mlp_opt.get_params(opt_state)
        loss, grad = jax.value_and_grad(task.loss)(params, key1, batch)
        opt_state = per_param_mlp_opt.update(opt_state, grad, loss=loss)

        return opt_state, key, loss

    """Per Parameter MLP Aggregator"""

    per_param_mlp_lagg = MLPLAgg()
    per_param_mlp_agg_str = "PerParamMLPAgg"

    key, key1 = jax.random.split(key)
    per_param_mlp_lagg_meta_params = meta_train_lagg(
        per_param_mlp_lagg,
        per_param_mlp_agg_str,
        key1,
        num_inner_steps,
        num_outer_steps,
    )

    per_param_mlp_agg = per_param_mlp_lagg.opt_fn(per_param_mlp_lagg_meta_params)

    @jax.jit
    def per_param_mlp_agg_update(opt_state, key, batch):
        key, key1 = jax.random.split(key)
        params = per_param_mlp_agg.get_params(opt_state)
        loss = task.loss(params, key1, batch)

        def sample_grad_fn(image, label):
            sub_batch_dict = {}
            sub_batch_dict["image"] = image
            sub_batch_dict["label"] = label
            sub_batch = FlatMap(sub_batch_dict)

            return jax.grad(task.loss)(params, key1, sub_batch)

        split_image = jnp.split(batch["image"], per_param_mlp_lagg.num_grads)
        split_label = jnp.split(batch["label"], per_param_mlp_lagg.num_grads)
        grads = [
            sample_grad_fn(split_image[i], split_label[i])
            for i in range(per_param_mlp_lagg.num_grads)
        ]

        opt_state = per_param_mlp_agg.update(opt_state, grads, loss=loss)

        return opt_state, key, loss

    """Benchmarking"""

    optimizers = [
        ("Adam", adam, adam_update),
        (per_param_mlp_opt_str, per_param_mlp_opt, per_param_mlp_opt_update),
        (per_param_mlp_agg_str, per_param_mlp_agg, per_param_mlp_agg_update),
    ]

    for opt_str, opt, update in optimizers:
        for j in range(num_runs):
            run = wandb.init(project="learned_aggregation", group=opt_str)

            params = task.init(key)
            opt_state = opt.init(params)

            for i in range(num_inner_steps):
                batch = next(task.datasets.train)
                opt_state, key, train_loss = update(opt_state, key, batch)

                run.log({"train loss": train_loss})

            run.finish()

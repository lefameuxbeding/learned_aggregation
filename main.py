import sys
import os
import jax
import wandb

from learned_optimization.tasks.fixed import image_mlp


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")

    key = jax.random.PRNGKey(0)
    task = image_mlp.ImageMLP_FashionMnist8_Relu32()
    grad_fn = jax.jit(jax.value_and_grad(task.loss))
    lr = 0.1

    for j in range(10):
        run = wandb.init(project="learned_aggregation", group="SGD")

        params = task.init(key)

        for i in range(100):
            key, key1 = jax.random.split(key)
            batch = next(task.datasets.train)
            l, grads = grad_fn(params, key1, batch)
            params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
            test_l = task.loss(params, key, next(task.datasets.test))
            run.log({"train loss": l, "test loss": test_l})

        run.finish()

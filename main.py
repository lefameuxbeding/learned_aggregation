import sys
import os
import jax
import wandb

from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.optimizers import base as opt_base


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")

    key = jax.random.PRNGKey(0)
    task = image_mlp.ImageMLP_FashionMnist8_Relu32()
    opt = opt_base.Adam(1e-2)

    @jax.jit
    def update(opt_state, key, batch):
        key, key1 = jax.random.split(key)
        params, _ = opt.get_params_state(opt_state)
        loss, grads = jax.value_and_grad(task.loss)(params, key1, batch)
        opt_state = opt.update(opt_state, grads, loss=loss)

        return opt_state, key, loss

    for j in range(10):
        run = wandb.init(project="learned_aggregation", group="Adam")

        params = task.init(key)
        opt_state = opt.init(params)

        for i in range(10):
            batch = next(task.datasets.train)
            opt_state, key, train_loss = update(opt_state, key, batch)
            test_loss = task.loss(
                opt.get_params_state(opt_state)[0], key, next(task.datasets.test)
            )
            run.log({"train loss": train_loss, "test loss": test_loss})

        run.finish()

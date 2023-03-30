import sys
import os
import jax
import pickle
import wandb

from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.learned_optimizers import adafac_mlp_lopt


if __name__ == "__main__":
    """Environment"""

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")

    """Setup"""

    key = jax.random.PRNGKey(0)

    num_runs = 10
    num_inner_steps = 500

    task = image_mlp.ImageMLP_FashionMnist_Relu128x128()

    lopt = adafac_mlp_lopt.AdafacMLPLOpt()
    opt_str = "lopt"
    with open(opt_str + ".pickle", "rb") as f:
        meta_params = pickle.load(f)
    opt = lopt.opt_fn(meta_params)

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)
        loss, grad = jax.value_and_grad(task.loss)(params, key, batch)
        opt_state = opt.update(opt_state, grad, loss=loss)

        return opt_state, loss

    """Benchmarking"""

    for j in range(num_runs):
        run = wandb.init(project="learned_aggregation", group=opt_str)

        key, key1 = jax.random.split(key)
        params = task.init(key1)
        opt_state = opt.init(params)

        for i in range(num_inner_steps):
            batch = next(task.datasets.train)
            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, batch)

            run.log({task.name + " train loss": loss})

        run.finish()

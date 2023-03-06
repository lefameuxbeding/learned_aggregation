import sys
import os
import jax
import wandb

from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.optimizers import base as opt_base
from learned_optimization.learned_optimizers import mlp_lopt
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.tasks import base as tasks_base
from learned_optimization.outer_trainers import gradient_learner


if __name__ == "__main__":
    """Environment"""

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")

    """Setup"""

    num_runs = 10
    num_inner_steps = 100
    num_outer_steps = 300

    key = jax.random.PRNGKey(0)
    task = image_mlp.ImageMLP_FashionMnist8_Relu32()

    """Adam"""

    adam = opt_base.Adam(1e-2)

    @jax.jit
    def adam_update(opt_state, key, batch):
        key, key1 = jax.random.split(key)
        params = adam.get_params(opt_state)
        loss, grads = jax.value_and_grad(task.loss)(params, key1, batch)
        opt_state = adam.update(opt_state, grads, loss=loss)

        return opt_state, key, loss

    """Per Parameter MLP Optimizer"""

    per_param_mlp_lopt = mlp_lopt.MLPLOpt()

    # TODO Meta-training
    meta_opt = opt_base.Adam(1e-3)

    trunc_sched = truncation_schedule.LogUniformLengthSchedule(
        min_length=100, max_length=300
    )

    def grad_est_fn(task_family):
        truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
            task_family,
            per_param_mlp_lopt,
            trunc_sched,
            num_tasks=1,
            random_initial_iteration_offset=300,
        )
        return truncated_pes.TruncatedPES(
            truncated_step=truncated_step, trunc_length=50
        )

    gradient_estimators = [grad_est_fn(tasks_base.single_task_to_family(task))]

    outer_trainer = gradient_learner.SingleMachineGradientLearner(
        per_param_mlp_lopt, gradient_estimators, meta_opt
    )
    outer_trainer_state = outer_trainer.init(key)

    run = wandb.init(project="learned_aggregation", group="PerParamMLPOpt")

    for _ in range(num_outer_steps):
        outer_trainer_state, meta_loss, _ = outer_trainer.update(
            outer_trainer_state, key, with_metrics=False
        )

        run.log({"meta loss": meta_loss})

    run.finish()
    # ---

    per_param_mlp_opt = per_param_mlp_lopt.opt_fn(outer_trainer_state.gradient_learner_state.theta_opt_state.params)

    @jax.jit
    def per_param_mlp_opt_update(opt_state, key, batch):
        key, key1 = jax.random.split(key)
        params = per_param_mlp_opt.get_params(opt_state)
        loss, grads = jax.value_and_grad(task.loss)(params, key1, batch)
        opt_state = per_param_mlp_opt.update(opt_state, grads, loss=loss)

        return opt_state, key, loss

    """Benchmarking"""

    optimizers = [
        ("Adam", adam, adam_update),
        ("PerParamMLPOpt", per_param_mlp_opt, per_param_mlp_opt_update),
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

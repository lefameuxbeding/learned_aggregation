import pickle
import jax
from learned_optimization.optimizers import nadamw
from learned_optimization.learned_optimizers import adafac_mlp_lopt


def _lopt(task):
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
    
    return (opt, opt_str, update)


def _nadamw(task):
    opt = nadamw.NAdamW()
    opt_str = "nadamw_" + str(opt.config["learning_rate"])

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)
        loss, grad = jax.value_and_grad(task.loss)(params, key, batch)
        opt_state = opt.update(opt_state, grad, loss=loss)

        return opt_state, loss
    
    return (opt, opt_str, update)


def get_optimizer(optimizer, task):
    optimizers = {
        "nadamw" : _nadamw,
        "lopt" : _lopt,
    }

    return optimizers[optimizer](task)

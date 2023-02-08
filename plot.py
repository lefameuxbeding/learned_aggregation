import matplotlib.pyplot as plt
import pickle
import numpy as onp


if __name__ == "__main__":

    optimizers = [("LOpt", "blue"), ("LAggOpt", "red")]

    for optimizer, color in optimizers:

        with open(optimizer + ".pickle", "rb") as f:
            results = pickle.load(f)

        plt.plot(results["losses_mean"], label=optimizer, color=color)
        plt.fill_between(
            onp.arange(10), results["losses_mean"] - results["losses_std"], results["losses_mean"] + results["losses_std"], alpha=0.1, color=color
        )

    plt.ylim(1.0, 2.3)

    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("loss")

    plt.show()
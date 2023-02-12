import matplotlib.pyplot as plt
import pickle
import numpy as np


if __name__ == "__main__":

    optimizers = [
        ("Adam", "red"),
        ("LOpt", "orange"),
        ("LAggOpt-4", "gold"),
        ("LAggOpt-8", "green"),
        ("LAggOpt-16", "blue"),
        ("LAggOpt-32", "purple"),
    ]

    for optimizer_name, color in optimizers:

        with open(optimizer_name + ".pickle", "rb") as f:
            results = pickle.load(f)

        plt.plot(results["losses_mean"], label=optimizer_name, color=color)
        plt.fill_between(
            np.arange(10),
            results["losses_mean"] - results["losses_std"],
            results["losses_mean"] + results["losses_std"],
            alpha=0.1,
            color=color,
        )

    plt.ylim(1.0, 2.4)

    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("loss")

    plt.show()

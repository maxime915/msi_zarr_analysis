import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

assert len(sys.argv) > 1, "expected at least one argument"

def save(path: str):
    assert path[-4:] == ".npz", "expected a numpy z archive"
    prefix = "evaluation_prog_binning_"
    assert path[: len(prefix)] == prefix, f"filename should start with {prefix=}"

    suffix = path[len(prefix) : -4]

    cls_description = " / ".join(
        map(
            lambda s: s.replace("positive AREA", "+").replace("negative AREA", "-"),
            suffix.split("_"),
        )
    )

    archive = np.load(path)

    scores = archive["scores"]

    mean = 100 * np.mean(scores, axis=0)
    std = 100 * np.std(scores, axis=0, ddof=1)

    xs = np.arange(scores.shape[1])

    # learning curve

    fig, ax = plt.subplots()
    ax.plot(xs, mean, label=cls_description)
    ax.fill_between(xs, mean - std, mean + std, alpha=0.3)
    ax.set_title("Accuracy evolution during the progressive binning")
    ax.set_xlabel("Binning step")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.set_ylim((0, 110))

    plt.tight_layout()
    fig.savefig(f"evaluation_prog_bin_{suffix}.png")
    plt.show()

# list() to consume the iterator
list(map(save, sys.argv[1:]))
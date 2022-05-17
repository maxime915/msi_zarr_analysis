import sys
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

assert len(sys.argv) == 2, "expected exactly one argument"
path = sys.argv[1]
assert path[-4:] == ".npz", "expected a numpy z archive"
prefix = "fine_saved_prog_binning_"
assert path[: len(prefix)] == prefix, f"filename should start with {prefix=}"

suffix = path[len(prefix) : -4]

cls_description = " / ".join(
    map(
        lambda s: s.replace("positive AREA", "+").replace("negative AREA", "-"),
        suffix.split("_"),
    )
)

archive = np.load(path)

collected_bin_hi = archive["collected_bin_hi"]
collected_bin_lo = archive["collected_bin_lo"]
collected_indices = archive["collected_indices"]
collected_scores = archive["collected_scores"]

df = pd.read_csv("mz value + lipid name.csv", sep=None, engine="python")

n_iteration = len(collected_scores)
blueish = cycle([0.5, 0.7])


fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(f"Progressive Binning ({cls_description})")

ax.get_yaxis().set_visible(False)
ax.invert_yaxis()

# show known lipids
for mz in df["m/z"]:
    ax.axvline(mz, color=(0.9, 0.1, 0.1, 0.3))

for i in range(collected_bin_hi.shape[0]):
    bin_lo = collected_bin_lo[i]
    bin_hi = collected_bin_hi[i]

    for lo, hi in zip(bin_lo, bin_hi):
        ax.fill_between((lo, hi), i, i+1, color=(0.11, next(blueish), 0.93))

ax.set_xticks(range(100, 1200, 50))

plt.tight_layout()
fig.savefig(f"fine_progressive_binning{suffix}.svg")
# plt.show()

# learning curve

fig, ax = plt.subplots()
ax.plot(1 + np.arange(len(collected_scores)), 100 * collected_scores, label=cls_description)
ax.set_title("Accuracy evolution during the progressive binning")
ax.set_xlabel("Binning step")
ax.set_ylabel("Accuracy (%)")
ax.legend()
ax.set_ylim((0, 100))

plt.tight_layout()
fig.savefig(f"fine_accuracy_evolution_{suffix}.png")
# plt.show()

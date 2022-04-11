
import matplotlib.pyplot as plt
import pandas as pd

ds = pd.read_csv("res_floor.csv", index_col=None)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# problem: tol was 0.001 but mean error is close to 0.015

ax1.boxplot(ds.true_scale - ds.pred_scale)
ax1.set_title("Scale Error\n(true - prediction)")
ax1.get_xaxis().set_visible(False)

ax2.boxplot(ds.true_y - ds.pred_y)
ax2.set_title("Coordinate (y) Error\n(true - prediction)")
ax2.get_xaxis().set_visible(False)

ax3.boxplot(ds.true_x - ds.pred_x)
ax3.set_title("Coordinate (x) Error\n(true - prediction)")
ax3.get_xaxis().set_visible(False)

fig.tight_layout()
fig.savefig("res_floor.png")

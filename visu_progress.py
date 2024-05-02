import numpy as np
import matplotlib.pyplot as plt

from partial import PartialOT1d


def plot_distrib(points, y=0, color="k"):
    plt.axhline(y=y, linestyle="dashed", color="k", zorder=-1)
    plt.scatter(points, [y] * len(points), s=30, color=color)


def plot_matches(x, y):
    x_s = np.sort(x)
    y_s = np.sort(y)
    for x_i, y_j in zip(x_s, y_s):
        plt.plot([x_i, y_j], [0.5, 0.], color="k", zorder=-1)


np.random.seed(0)
n = 12
x = np.random.rand(n)
x.sort()
y = np.random.rand(n)
y.sort()
indices_x, indices_y, costs = PartialOT1d(max_iter=n).fit(x, y)
    
plt.figure(figsize=(12, 4))
for i in range(n):
    plt.subplot(3, 4, i + 1)
    plot_distrib(x, y=.5, color="orange")
    plot_distrib(y, y=0, color="navy")
    plot_matches(x[indices_x[:i + 1]], y[indices_y[:i + 1]])
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Iteration {i + 1}")
plt.tight_layout()
plt.show()

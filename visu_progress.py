import numpy as np
import matplotlib.pyplot as plt

from partial import partial_ot_1d


def plot_distrib(points, y=0, color="k"):
    plt.axhline(y=y, linestyle="dashed", color="k", zorder=-1)
    plt.scatter(points, [y] * len(points), s=30, color=color)


def plot_matches(groups, x, y):
    for g in groups:
        # print(g)
        for i, j in zip(range(g.i_x, g.i_x + g.length), range(g.i_y, g.i_y + g.length)):
            # print(i, j)
            plt.plot([x[i], y[j]], [0.5, 0.], color="k", zorder=-1)


np.random.seed(0)
n = 12
x = np.random.rand(n)
x.sort()
y = np.random.rand(n)
y.sort()
solutions = partial_ot_1d(x, y)
    
plt.figure(figsize=(12, 4))
for i in range(n):
    plt.subplot(3, 4, i + 1)
    plot_distrib(x, y=.5, color="orange")
    plot_distrib(y, y=0, color="navy")
    print(f"Iteration {i + 1} (i={i})")
    plot_matches(solutions[i], x, y)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Iteration {i + 1}")
plt.tight_layout()
plt.show()

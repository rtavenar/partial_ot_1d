import numpy as np
import matplotlib.pyplot as plt

from partial import partial_ot_1d

np.random.seed(0)
n_repeat = 10

n = 1000

steps = list(range(1, n + 1))
n_groups = np.zeros((n_repeat, n), dtype=int)
for i in range(n_repeat):
    x = np.random.rand(n)
    y = np.random.rand(n)

    solutions = partial_ot_1d(x, y)
    n_groups[i] = [len(groups) for groups in solutions]
    
plt.figure()
plt.plot(steps, np.mean(n_groups, axis=0), linestyle="solid", color="tab:blue")
plt.plot(steps, np.mean(n_groups, axis=0) + np.std(n_groups, axis=0), linestyle="dashed", color="tab:blue")
plt.plot(steps, np.mean(n_groups, axis=0) - np.std(n_groups, axis=0), linestyle="dashed", color="tab:blue")
plt.xlabel("Step")
plt.ylabel("Average number of groups (+/- standard deviation)")
plt.tight_layout()
plt.grid()
plt.show()
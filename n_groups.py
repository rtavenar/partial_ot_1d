import numpy as np
import matplotlib.pyplot as plt

from partial import partial_ot_1d

np.random.seed(0)
n_repeat = 10

n = 1000

steps = list(range(1, n + 1))
n_groups = np.zeros((n, ), dtype=int)
for _ in range(n_repeat):
    x = np.random.rand(n)
    y = np.random.rand(n)

    solutions = partial_ot_1d(x, y)
    n_groups += [len(groups) for groups in solutions]
n_groups /= n_repeat
    
plt.figure()
plt.plot(steps, n_groups)
plt.xlabel("Step")
plt.ylabel("Average number of groups")
plt.tight_layout()
plt.grid()
plt.show()
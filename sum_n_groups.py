import numpy as np
import matplotlib.pyplot as plt

from partial import partial_ot_1d

np.random.seed(0)
n_repeat = 10

values_for_n = [10, 30, 100, 300, 1000]

sum_n_groups = np.zeros((n_repeat, len(values_for_n)), dtype=int)
for idx_n, n in enumerate(values_for_n):
    for i in range(n_repeat):
        x = np.random.rand(n)
        y = np.random.rand(n)

        solutions = partial_ot_1d(x, y)
        sum_n_groups[i, idx_n] = np.sum([len(groups) for groups in solutions])
    
plt.figure()
plt.loglog(values_for_n, np.mean(sum_n_groups, axis=0), linestyle="solid", color="tab:blue")
plt.loglog(values_for_n, np.mean(sum_n_groups, axis=0) + np.std(sum_n_groups, axis=0), linestyle="dashed", color="tab:blue")
plt.loglog(values_for_n, np.mean(sum_n_groups, axis=0) - np.std(sum_n_groups, axis=0), linestyle="dashed", color="tab:blue")
plt.xlabel("Number of samples")
plt.ylabel("Sum of the number of groups over all steps in the algorithm")
plt.tight_layout()
plt.grid()
plt.show()
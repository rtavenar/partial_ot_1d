import matplotlib.pyplot as plt
import numpy as np

from partial import PartialOT1d

n = 30
pb = PartialOT1d(max_iter=n)
np.random.seed(0)
ind_outliers = np.random.choice(2, p=[.9, .1], size=n)
x = np.random.rand(n, ) + ind_outliers * 3
np.random.shuffle(ind_outliers)
y = np.random.rand(n, ) - ind_outliers * 3
indices_x, indices_y, marginal_costs = pb.fit(x, y)
cum_costs = np.cumsum(marginal_costs)

n_outliers = np.sum(ind_outliers)
plt.bar(x=np.arange(1, n - n_outliers + 1), height=cum_costs[0:n - n_outliers])
plt.bar(x=np.arange(n - n_outliers + 1, n + 1), height=cum_costs[n - n_outliers:], color="red")
plt.title(f"Total costs after each iteration of the partial OT algorithm in 1D\n({n_outliers} outliers in x and y here)")
plt.show()

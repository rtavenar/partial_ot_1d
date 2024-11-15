import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from mpl_sizes import get_format
import matplotlib

from partial import partial_ot_1d_elbow

n = 30
np.random.seed(0)
ind_outliers = np.random.choice(2, p=[.9, .1], size=n)
x = np.random.rand(n, ) + ind_outliers * 1.
np.random.shuffle(ind_outliers)
y = np.random.rand(n, ) - ind_outliers * 1.
indices_x, indices_y, marginal_costs, elbow = partial_ot_1d_elbow(x, y, return_all_solutions=True)
cum_costs = np.cumsum(marginal_costs)
n_inliers = elbow
n_outliers = np.sum(ind_outliers)

plt.style.use(['science'])
matplotlib.rcParams.update({'font.size': 8})

plt.bar(x=np.arange(1, n_inliers + 1), height=cum_costs[0:n_inliers])
plt.bar(x=np.arange(n_inliers + 1, n + 1), height=cum_costs[n_inliers:])
plt.ylabel("PAWL$_k$ cost")
plt.xlabel("$k$")
# plt.title(f"Total costs after each iteration of the partial OT algorithm in 1D\n({n_outliers} outliers in x and y here)")
plt.savefig("elbow.pdf")

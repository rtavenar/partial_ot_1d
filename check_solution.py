import ot
import numpy as np 
from partial import partial_ot_1d

np.random.seed(0)

n = 200
x = np.random.rand(n)
y = np.random.rand(n)

for p in [1, 2]:
    print(f"Testing for p={p} (if nothing prints, it's OK)")
    M = ot.dist(x[:, None], y[:, None], metric="minkowski", p=p) ** p
    indices_x, indices_y, marginal_costs = partial_ot_1d(x, y, max_iter=n, p=p)
    costs = np.cumsum(marginal_costs)

    for i in range(1, n):
        diff = ot.partial.partial_wasserstein2([], [], M, m=i / n) - costs[i-1] / n
        if np.abs(diff) > 1e-8:
            print("--------i", i, "diff", diff, "val", costs[i-1] / n)
        
        t = ot.partial.partial_wasserstein([], [], M, m=i / n)
        ind_x, ind_y = np.where(t > 1e-6)

        np.testing.assert_array_equal(np.sort(indices_x[:i]), np.sort(ind_x))
        np.testing.assert_array_equal(np.sort(indices_y[:i]), np.sort(ind_y))

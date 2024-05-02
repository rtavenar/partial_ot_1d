import ot
import numpy as np 
from partial import PartialOT1d

np.random.seed(0)

n = 200
x = np.random.rand(n)
y = np.random.rand(n)

M = ot.dist(x[:, None], y[:, None], metric="minkowski", p=1)
pb = PartialOT1d(max_iter=n)
marginal_costs = pb.fit(x, y)[-1]
costs = np.cumsum(marginal_costs)

for i in range(1, n):
    print("--------i", i, "diff", ot.partial.partial_wasserstein2([], [], M, m=i / n) - costs[i-1] / n)

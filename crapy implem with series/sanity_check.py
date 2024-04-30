import ot
import numpy as np 
from partialOT_crapy import partial_ot

np.random.seed(0)

n = 6
x = np.random.rand(n)
y = np.random.rand(n)

x_s = np.sort(x)
y_s = np.sort(y)
M = ot.dist(x_s.reshape(-1, 1), y_s.reshape(-1, 1), metric="minkowski", p=1)
current_series_idx, current_series_cost, active_x, active_y, active_cost = partial_ot(x_s, y_s)
for i in np.arange(1,len(x)+1):
    print("--------i", i)
    mass = i/ len(x)  
    print("diff", ot.partial.partial_wasserstein2([], [], M, mass) - active_cost[i-1])
    if (np.abs(ot.partial.partial_wasserstein2([], [], M, mass) - active_cost[i-1]) > 1e-8):
        print(ot.partial.partial_wasserstein2([], [], M, mass),active_cost[i-1] )
        print(active_x)
        print(active_y)
        print("current series and costs")
        print(current_series_idx)
        print(current_series_cost)
        t = ot.partial.partial_wasserstein([], [], M, mass)
        print(np.where(t > 1e-6))

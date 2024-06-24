import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt

n = 200
total_costs = np.arange(n)
total_costs[-50:] = (total_costs[-50:] - total_costs[-50]) * 3 + total_costs[-50]
total_costs[-100:] = (total_costs[-100:] - total_costs[-100]) * 3 + total_costs[-100]


kneedle = KneeLocator(x=np.arange(len(total_costs)), 
                        y=total_costs, 
                        S=1e-5, 
                        curve="convex", 
                        direction="increasing")
if kneedle.elbow is None:
    # No elbow has been detected
    idx_elbow = n - 1
else:
    idx_elbow = int(kneedle.elbow)

print(idx_elbow, kneedle.all_elbows)

plt.bar(x=np.arange(1, idx_elbow + 1), height=total_costs[0:idx_elbow])
plt.bar(x=np.arange(idx_elbow + 1, n + 1), height=total_costs[idx_elbow:], color="red")
plt.title(f"Total costs after each iteration of the partial OT algorithm in 1D\n(outliers in x and y here)")
plt.show()
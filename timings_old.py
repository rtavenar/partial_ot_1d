import time
import numpy as np
import matplotlib.pyplot as plt

from partial_old import partial_ot_1d

np.random.seed(0)
n_repeat = 10

values_for_n = [10, 30, 100, 300, 1000]
timings = []

for n in values_for_n:
    x = np.random.rand(n)
    y = np.random.rand(n)

    t0 = time.time()
    for _ in range(n_repeat):
        partial_ot_1d(x, y)
    avg_time = (time.time() - t0) / n_repeat
    timings.append(avg_time)
    print(f"n={n:04d}\ttime={avg_time}")
    
plt.figure()
plt.loglog(values_for_n, timings)
plt.xlabel("Number of points in x (resp. y)")
plt.ylabel("Average time (in seconds) for the resolution of all partial problems")
plt.tight_layout()
plt.grid()
plt.show()

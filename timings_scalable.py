import time
import numpy as np
import matplotlib.pyplot as plt

from scalable_partial import PartialOT1d

np.random.seed(0)
n_repeat = 10

values_for_n =  [10, 30, 100, 300, 1000, 3000, 10000]
timings = []

for n in values_for_n:
    x = np.random.rand(n)
    y = np.random.rand(n)

    t0 = time.time()
    for _ in range(n_repeat):
        PartialOT1d(max_iter=n).fit(x, y)
    avg_time = (time.time() - t0) / n_repeat
    timings.append(avg_time)
    print(f"n={n:04d}\ttime={avg_time}\tvalue of constant C in complexity (C * n log n)={avg_time / (n * np.log(n))}")
    
plt.figure()
plt.loglog(values_for_n, timings)
plt.xlabel("Number of points in x (resp. y)")
plt.ylabel("Average time (in seconds) for the resolution of a partial problem")
plt.tight_layout()
plt.grid()
plt.show()

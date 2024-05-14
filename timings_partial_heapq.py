import time
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from mpl_sizes import get_format

from partial import PartialOT1d
from partial_heapq import PartialOT1d as PartialOT1dHQ
from partial_nb import partial_ot_1d as partial_ot_1d_nb

np.random.seed(0)
n_repeat = 3

values_for_n =  [1000, 3000, 10000, 30000, 100000]  #, 300000, 1000000]
timings_ours = []
timings_ours_hq = []
timings_ours_nb = []

# Cache
partial_ot_1d_nb(np.random.rand(10), np.random.rand(10), max_iter=100)

for n in values_for_n:
    sum_time_ours = 0.
    sum_time_ours_hq = 0.
    sum_time_ours_nb = 0.
    for _ in range(n_repeat):
        x = np.random.rand(n)
        y = np.random.rand(n)

        # Ours
        t0 = time.time()
        PartialOT1d(max_iter=n).fit(x, y)
        sum_time_ours += (time.time() - t0)

        # Ours using heap queues
        t0 = time.time()
        PartialOT1dHQ(max_iter=n).fit(x, y)
        sum_time_ours_hq += (time.time() - t0)

        # Ours using heap queues
        t0 = time.time()
        partial_ot_1d_nb(x, y, max_iter=n)
        sum_time_ours_nb += (time.time() - t0)
    timings_ours.append(sum_time_ours / n_repeat)
    timings_ours_hq.append(sum_time_ours_hq / n_repeat)
    timings_ours_nb.append(sum_time_ours_nb / n_repeat)

plt.style.use(['science'])
formatter = get_format("NeurIPS") # options: ICLR, ICML, NeurIPS, InfThesis
plt.figure(figsize=formatter.line_width_plot(aspect_ratio="normal"))
plt.loglog(values_for_n, timings_ours, label="Our method")
plt.loglog(values_for_n, timings_ours_hq, label="Our method using heap queues")
plt.loglog(values_for_n, timings_ours_nb, label="Our method using heap queues and numba")
plt.xlabel("Number of points in x (resp. y)")
plt.ylabel("Average time (in seconds)")
plt.tight_layout()
plt.grid()
legend = plt.legend(frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
plt.savefig("timings_partial.pdf")

import time
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from mpl_sizes import get_format

from partial_nb import partial_ot_1d
from baselines.cvpr23_bai import solve_opt as CVPR23Bai_partial_solver
from baselines.aistats22_sejourne import solve_uot as AISTATS22Sejourne_unbalanced_solver

np.random.seed(0)
n_repeat = 3

values_for_n =  [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
timings_ours = []
timings_aistats22 = []
lambdas_cvpr23 = [1e-3]  # lambda_hat = lambda / median_dist
rho_aistats22 = 1.
timings_cvpr23 = {k: [] for k in lambdas_cvpr23}

# Cache stuff
CVPR23Bai_partial_solver(np.sort(np.random.rand(10)), np.sort(np.random.rand(10)), 10)
AISTATS22Sejourne_unbalanced_solver(np.ones(10), np.ones(10),
                                    np.sort(np.random.rand(10)), np.sort(np.random.rand(10)), 
                                    p=1, rho1=rho_aistats22, rho2=rho_aistats22)
partial_ot_1d(np.random.rand(10), np.random.rand(10), max_iter=10)

for n in values_for_n:
    sum_time_ours = 0.
    sum_time_aistats22 = 0.
    sum_time_cvpr23 = {k: 0. for k in lambdas_cvpr23}
    transported_mass_cvpr23 = {k: [] for k in lambdas_cvpr23}
    for _ in range(n_repeat):
        x = np.random.rand(n)
        y = np.random.rand(n)
        med_dist = np.median((np.sort(x) - np.sort(y)) ** 2)

        # Ours
        t0 = time.time()
        partial_ot_1d(x, y, max_iter=n)
        sum_time_ours += (time.time() - t0)

        # CVPR23
        for l in lambdas_cvpr23:
            l_times_meddist = l * med_dist
            t0 = time.time()
            objective,phi,psi,piRow,piCol = CVPR23Bai_partial_solver(np.sort(x), np.sort(y), l_times_meddist)
            sum_time_cvpr23[l] += (time.time() - t0)
            transported_mass_cvpr23[l].append(np.sum(piRow > 1e-6) / n)

        # AISTATS22
        t0 = time.time()
        AISTATS22Sejourne_unbalanced_solver(
            np.ones(n), np.ones(n),
            np.sort(x), np.sort(y), 
            p=1, rho1=rho_aistats22, rho2=rho_aistats22
        )
        sum_time_aistats22 += (time.time() - t0)
    timings_ours.append(sum_time_ours / n_repeat)
    for l in lambdas_cvpr23:
        timings_cvpr23[l].append(sum_time_cvpr23[l] / n_repeat)
        print(f"n={n:07d}, transported masses for lambda={l}: {transported_mass_cvpr23[l]}")
    timings_aistats22.append(sum_time_aistats22)

plt.style.use(['science'])
formatter = get_format("ICLR-large") # options: ICLR, ICML, NeurIPS, InfThesis
plt.figure(figsize=formatter.line_width_plot(aspect_ratio="normal"))
plt.loglog(values_for_n, timings_ours, label="PAWL")
for l in lambdas_cvpr23:
    plt.loglog(values_for_n, timings_cvpr23[l], label=f"OPT, $\lambda={l} \cdot  \\text{{median}}(dists)$")  # (Bai et al, CVPR 2023)")
plt.loglog(values_for_n, timings_aistats22, label=f"Fast-UOT, $\\rho_1 = \\rho_2 = {rho_aistats22}$")  # (Séjourné et al, AISTATS 2022)")
plt.xlabel("Number of points in x (resp. y)")
plt.ylabel("Average time (in seconds)")
plt.tight_layout()
plt.grid()
legend = plt.legend(frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
plt.savefig("timings_partial.pdf")

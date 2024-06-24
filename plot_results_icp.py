import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import scienceplots
from mpl_sizes import get_format

baselines = {"icp_du": "ICP (Du)",
             "spot_bonneel": "SPOT", 
             "ours_apriori": "Ours (uses $n_0$)", 
            #  "swgg_apriori": "SWGG (uses $n_0$)", 
             "ours_elbow": "Ours (elbow method)",
             "sopt": "SOPT (uses $n_0$)"
             }
baselines_uses_apriori = {"icp_du": False, "spot_bonneel": False, 
                          "sopt": True,  "ours_apriori": True, "ours_elbow": False}
datasets = ["dragon", 'stanford_bunny', 'mumble_sitting', 'witchcastle']

plt.style.use(['science'])
formatter = get_format("NeurIPS") # options: ICLR, ICML, NeurIPS, InfThesis
figx, figy = formatter.line_width_plot(aspect_ratio="normal")

for seed in [0, 42, 10, 100, 1000]:
    plt.figure(figsize=(1.5 * figx, 1.5 * figy))
    for idx_p, problem in enumerate(datasets):
        for idx_pc, percent in enumerate([5, 7]):
            for idx_n, n_source in enumerate([9 * 1000, 10 * 1000]):
                plt.subplot(4, 4, idx_pc * 8 + idx_n * 4 + idx_p + 1)
                if idx_pc == 0 and idx_n == 0:
                    plt.title(problem)
                if idx_p == 0:
                    plt.ylabel(f"$n_0={n_source}$\n$p={percent}\\%$")
                for baseline in baselines:
                    print(problem, percent, n_source, baseline)
                    if not os.path.exists(f"results_icp/{problem}_p{percent}_n_source{n_source}_baseline-{baseline}_seed{seed}.npz"):
                        continue
                    data = torch.load(f"shape_data_bai/{problem}.pt")
                    rotation_matrix = data["param"]["rotation_op"]
                    scaling_factor = data["param"]["scalar_op"]
                    shift = data["param"]["beta_op"]

                    source = data[f"S-{n_source // 1000}k-{percent}p"]
                    std = np.sqrt(np.trace(np.cov(source.T)))

                    mat_gt = np.concatenate((shift[:, None] / std, scaling_factor * rotation_matrix), axis=1)

                    data_results = np.load(f"results_icp/{problem}_p{percent}_n_source{n_source}_baseline-{baseline}_seed{seed}.npz")
                    rotation_list = data_results["rotation_list"]
                    scalar_list = data_results["scalar_list"]
                    beta_list = data_results["beta_list"]
                    norm_error_list = [
                        np.linalg.norm(
                            mat_gt - (np.concatenate((b[:, None] / std, s * rot), axis=1))
                        )
                        for rot, s, b in zip(rotation_list, scalar_list, beta_list)
                    ]
                    timings_list = data_results["timings_list"]
                    plt.plot(timings_list, norm_error_list,
                             label=baselines[baseline],
                             linestyle="dashed" if baselines_uses_apriori[baseline] else "solid")
                plt.xlim([0, 700])
    plt.legend(bbox_to_anchor=(1.04, 0.), loc="lower left")
    plt.tight_layout()
    plt.savefig(f"shape_registration_error_vs_timings_seed{seed}.pdf")

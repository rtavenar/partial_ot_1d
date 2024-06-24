import numpy as np
import torch
import matplotlib.pyplot as plt
import os

aggregation_fun = np.median

baselines = {"icp_du": "ICP (Du)",
             "spot_bonneel": "SPOT", 
             "ours_apriori": "Ours",  
             "ours_elbow": "Ours",
             "sopt": "SOPT"
             }
datasets = ["dragon", #'stanford_bunny', 
            'mumble_sitting', 'witchcastle']

def is_best(errors, baseline, list_baselines, label, t):
    baseline_score = aggregation_fun(errors[baseline, label, t])
    for other_baseline in list_baselines:
        if other_baseline == baseline:
            continue
        other_baseline_score = aggregation_fun(errors[other_baseline, label, t])
        if other_baseline_score < baseline_score:
            return False
    return True

errors = {}

for seed in [0, 42, 10, 100, 1000]:
    for problem in datasets:
        for baseline in baselines:
            for percent in [5, 7]:
                for n_source in [9 * 1000, 10 * 1000]:
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
                        for t in [30, 60]:
                            i = 0
                            while timings_list[i] < t:
                                i += 1
                                if i >= len(timings_list):
                                    i -= 1
                                    print(f"Could not find anything for t={t}")
                                    break
                            key = (baseline, f"p{percent}_n_source{n_source}", t)
                            errors[key] = errors.get(key, []) + [norm_error_list[i]]
                        key = (baseline, f"p{percent}_n_source{n_source}", "end")
                        errors[key] = errors.get(key, []) + [norm_error_list[-1]]

# print(errors)

s = r"""\begin{table}
    \tiny
    \caption{Performance for the shape registration experiment (error norms, the lower the better)}
    \label{tab:perf_shape}
    \centering
    \begin{tabular}{lllllllllllll}
        % \toprule
         & \multicolumn{3}{c}{10k,5\%} & \multicolumn{3}{c}{10k,7\%} & \multicolumn{3}{c}{9k,5\%} & \multicolumn{3}{c}{9k,7\%} \\
        \cmidrule(r){2-4} \cmidrule(r){5-7} \cmidrule(r){8-10} \cmidrule(r){11-13} 
         & 30s. & 60s. & End & 30s. & 60s. & End & 30s. & 60s. & End & 30s. & 60s. & End \\
        \midrule
"""
for baseline in ["icp_du", "spot_bonneel", "ours_elbow"]:
    s += f"        {baselines[baseline]} "
    for label in ['p5_n_source10000', 'p7_n_source10000', 'p5_n_source9000', 'p7_n_source9000']:
        for t in [30, 60, 'end']:
            if is_best(errors, baseline, ["icp_du", "spot_bonneel", "ours_elbow"], label, t):
                s += f"& \\textbf{{{aggregation_fun(errors[baseline, label, t]):.3f}}} "
            else:
                s += f"& {aggregation_fun(errors[baseline, label, t]):.3f} "
    s += "\\\\\n"
s += "        \midrule\n"
for baseline in ["sopt", "ours_apriori"]:
    s += f"        {baselines[baseline]} "
    for label in ['p5_n_source10000', 'p7_n_source10000', 'p5_n_source9000', 'p7_n_source9000']:
        for t in [30, 60, 'end']:
            if is_best(errors, baseline, ["sopt", "ours_apriori"], label, t):
                s += f"& \\textbf{{{aggregation_fun(errors[baseline, label, t]):.3f}}} "
            else:
                s += f"& {aggregation_fun(errors[baseline, label, t]):.3f} "
    s += "\\\\\n"
s += r"""        \bottomrule
    \end{tabular}
\end{table}
"""


fname_out = "results_icp/tex/table.tex"
fp = open(fname_out, "wt")
fp.write(s)
fp.close()


# for seed in [0, 42, 1000]:
#     for idx_p, problem in enumerate(datasets):
#         for idx_pc, percent in enumerate([5, 7]):
#             for idx_n, n_source in enumerate([9 * 1000, 10 * 1000]):
#                 plt.subplot(4, 4, idx_pc * 8 + idx_n * 4 + idx_p + 1)
#                 if idx_pc == 0 and idx_n == 0:
#                     plt.title(problem)
#                 if idx_p == 0:
#                     plt.ylabel(f"$n_0={n_source}$\n$p={percent}\\%$")
#                 for baseline in baselines:
#                     print(problem, percent, n_source, baseline)
#                     if not os.path.exists(f"results_icp/{problem}_p{percent}_n_source{n_source}_baseline-{baseline}_seed{seed}.npz"):
#                         continue
#                     data = torch.load(f"shape_data_bai/{problem}.pt")
#                     rotation_matrix = data["param"]["rotation_op"]
#                     scaling_factor = data["param"]["scalar_op"]
#                     shift = data["param"]["beta_op"]

#                     source = data[f"S-{n_source // 1000}k-{percent}p"]
#                     std = np.sqrt(np.trace(np.cov(source.T)))

#                     mat_gt = np.concatenate((shift[:, None] / std, scaling_factor * rotation_matrix), axis=1)

#                     data_results = np.load(f"results_icp/{problem}_p{percent}_n_source{n_source}_baseline-{baseline}_seed{seed}.npz")
#                     rotation_list = data_results["rotation_list"]
#                     scalar_list = data_results["scalar_list"]
#                     beta_list = data_results["beta_list"]
#                     norm_error_list = [
#                         np.linalg.norm(
#                             mat_gt - (np.concatenate((b[:, None] / std, s * rot), axis=1))
#                         )
#                         for rot, s, b in zip(rotation_list, scalar_list, beta_list)
#                     ]
#                     timings_list = data_results["timings_list"]
#                     plt.plot(timings_list, norm_error_list,
#                              label=baselines[baseline],
#                              linestyle="dashed" if baselines_uses_apriori[baseline] else "solid")
#                 plt.xlim([0, 600])
#     plt.legend(bbox_to_anchor=(1.04, 0.), loc="lower left")
#     plt.tight_layout()
#     plt.savefig(f"shape_registration_error_vs_timings_seed{seed}.pdf")

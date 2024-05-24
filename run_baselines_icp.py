import numpy as np
import torch

from baselines.cvpr23_bai_icp_xp import shape_image, vis_param_list, icp_du, spot_bonneel, sopt_main, ours_main

viz = False

for problem in ["dragon", 'stanford_bunny', 'mumble_sitting', 'witchcastle']:
    for percent in [5, 7]:
        for n_source in [9 * 1000, 10 * 1000]:
            baseline_fun = {
                "icp_du": lambda s, t: icp_du(s, t, n_iterations=100),
                "spot_bonneel": lambda s, t: spot_bonneel(s, t, n_projections=20, n_iterations=100),
                "sopt": lambda s, t: sopt_main(s, t, n_iterations=2000, N0=n_source),  # N0=# of clean source data points
                "ours_apriori": lambda s, t: ours_main(s, t, n_iterations=10000, N0=n_source),  # N0=# of clean source data points
                "ours_elbow": lambda s, t: ours_main(s, t, n_iterations=10000, N0="elbow")
            }
            for baseline in baseline_fun.keys():
                print(problem, percent, n_source, baseline)
                # data = np.load(f"processed_data/{problem}_p{p}_n_source{n_source}.npz")
                data = torch.load(f"shape_data_bai/{problem}.pt")
                source_data = data[f"S-{n_source // 1000}k-{percent}p"].astype(np.float64)
                target_data = data[f"T-{percent}p"].astype(np.float64)

                try:
                    rotation_list, scalar_list, beta_list, timings_list = baseline_fun[baseline](source_data, target_data)

                    if viz:
                        n_iter = len(rotation_list)
                        step_viz = n_iter // 10
                        for rotation, scalar, beta in zip(rotation_list[::step_viz], scalar_list[::step_viz], beta_list[::step_viz]):
                            transformed_source_data = source_data @ rotation * scalar + beta
                            shape_image(target_data, transformed_source_data, param=vis_param_list[problem])
                    print(baseline, timings_list[-1])
                    np.savez(f"results_icp/{problem}_p{percent}_n_source{n_source}_baseline-{baseline}.npz",
                            rotation_list=rotation_list,
                            scalar_list=scalar_list,
                            beta_list=beta_list,
                            timings_list=timings_list)
                except Exception as inst:
                    print(f"Baseline {baseline} did not run properly")
                    print(inst)
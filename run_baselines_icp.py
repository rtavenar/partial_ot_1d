import numpy as np
import time

from baselines.cvpr23_bai_icp_xp import shape_image, vis_param_list, icp_du, spot_bonneel, sopt_main

viz = False

for problem in ["dragon", 'stanford_bunny', 'mumble_sitting', 'witchcastle']:
    for p in [.05, .07]:
        for n_target in [9 * 1000, 10 * 1000]:
            for baseline in ["icp_du", "spot_bonneel", "sopt"]:
                print(problem, p, n_target, baseline)
                baseline_fun = {
                    "icp_du": lambda s, t: icp_du(s, t, n_iterations=100),
                    "spot_bonneel": lambda s, t: spot_bonneel(s, t, n_projections=20, n_iterations=100),
                    "sopt": lambda s, t: sopt_main(s, t, n_iterations=2000, N0=10 * 1000)  # N0=# of clean source data points
                }
                data = np.load(f"processed_data/{problem}_p{p}_n_target{n_target}.npz")
                source_data = data["source_data"]
                target_data = data["target_data"]

                t0 = time.time()
                rotation_list, scalar_list, beta_list = baseline_fun[baseline](source_data, target_data)
                total_time = time.time() - t0

                if viz:
                    n_iter = len(rotation_list)
                    step_viz = n_iter // 10
                    for rotation, scalar, beta in zip(rotation_list[::step_viz], scalar_list[::step_viz], beta_list[::step_viz]):
                        transformed_source_data = source_data @ rotation * scalar + beta
                        shape_image(target_data, transformed_source_data, param=vis_param_list[problem])
                np.savez(f"results_icp/{problem}_p{p}_n_target{n_target}_baseline-{baseline}.npz",
                         rotation_list=rotation_list,
                         scalar_list=scalar_list,
                         beta_list=beta_list,
                         total_time=total_time)

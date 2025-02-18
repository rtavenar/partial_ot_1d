import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import scienceplots
from mpl_sizes import get_format
import matplotlib

from baselines.cvpr23_bai_icp_xp import vis_param_list

def shape_image(T_data,S_data, ax, keep_centered=False, param=None):
    if param!=None:
        xlim,ylim,zlim,view_init,(dx,dy,dz)=param
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(view_init[0],view_init[1],vertical_axis=view_init[2])
        
    ax.scatter(T_data[::5,0]+dx,T_data[::5,1]+dy,T_data[::5,2]+dz,alpha=1.,s=.5,marker='.')#,c='C2')
    ax.scatter(S_data[::5,0]+dx,S_data[::5,1]+dy,S_data[::5,2]+dz,alpha=1.,s=.5,marker='.')#,c='C1')
    
    # ax.set_facecolor('black') 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.grid(True)
    # ax.axis('off')

baselines = {
             "spot_bonneel": "SPOT",
             "sopt": "SOPT", 
             "pawl_apriori": "PAWL", 
             "pawl_elbow": "PAWL (elbow)"
             }
datasets = ["stanford_bunny"]
timings = 30

plt.style.use(['science'])
matplotlib.rcParams.update({'font.size': 22})

for seed in [10]:
    for idx_p, problem in enumerate(datasets):
        for idx_pc, percent in enumerate([5, 7]):
            for idx_n, n_source in enumerate([9 * 1000, 10 * 1000]):
                print(problem, percent, n_source)
                data = torch.load(f"shape_data_bai/{problem}.pt")
                source = data[f"S-{n_source // 1000}k-{percent}p"]
                target = data[f"T-{percent}p"]

                init_rot = np.eye(3)
                init_scalar = 1.
                init_beta = target.mean(axis=0) - source.mean(axis=0)

                for idx_m, method in enumerate(baselines.keys()):
                    if not os.path.exists(f"results_icp/{problem}_p{percent}_n_source{n_source}_baseline-{method}_seed{seed}.npz"):
                        continue

                    data_results = np.load(f"results_icp/{problem}_p{percent}_n_source{n_source}_baseline-{method}_seed{seed}.npz")
                    rotation_list = [init_rot] + list(data_results["rotation_list"])
                    scalar_list = [init_scalar] + list(data_results["scalar_list"])
                    beta_list = [init_beta] + list(data_results["beta_list"])
                    timings_list = [0.0] + list(data_results["timings_list"])
                    ax = plt.figure(figsize=(5, 5)).add_subplot(projection='3d')
                    i = 0
                    while timings_list[i] < timings:
                        i += 1
                        if i >= len(timings_list):
                            i -= 1
                            print(f"Could not find anything for t={timings}")
                            break
                    rotation = rotation_list[i]
                    scalar = scalar_list[i]
                    beta = beta_list[i]
                    transformed_source_data = source @ rotation * scalar + beta
                    shape_image(target, transformed_source_data, param=vis_param_list[problem], ax=plt.gca())
                        # if idx_t == 0:
                        #     plt.title(baselines[method])
                    plt.tight_layout()
                    # plt.subplots_adjust(wspace=-0.2, hspace=-0.55)
                    plt.savefig(f"shape_figs/viz_{problem}_{timings}sec_{method}_p{percent}_n_source{n_source}_seed{seed}.pdf")

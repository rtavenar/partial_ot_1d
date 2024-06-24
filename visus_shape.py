import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import scienceplots
from mpl_sizes import get_format
import matplotlib

from baselines.cvpr23_bai_icp_xp import vis_param_list

def shape_image(T_data,S_data, ax, param=None):
    if param!=None:
        xlim,ylim,zlim,view_init,(dx,dy,dz)=param
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(view_init[0],view_init[1],vertical_axis=view_init[2])
        
    ax.scatter(T_data[:,0]+dx,T_data[:,1]+dy,T_data[:,2]+dz,alpha=.5,s=.05,marker='.')#,c='C2')
    ax.scatter(S_data[:,0]+dx,S_data[:,1]+dy,S_data[:,2]+dz,alpha=.5,s=.05,marker='.')#,c='C1')
    
    # ax.set_facecolor('black') 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.grid(True)
    # ax.axis('off')

baselines = {"icp_du": "ICP (Du)",
             "spot_bonneel": "SPOT",
             "sopt": "SOPT (uses $n_0$)", 
             "ours_apriori": "Ours (uses $n_0$)", 
            #  "swgg_apriori": "SWGG (uses $n_0$)", 
             "ours_elbow": "Ours (elbow method)"
             }
datasets = ["dragon", 'stanford_bunny', 'mumble_sitting', 'witchcastle']

plt.style.use(['science'])
matplotlib.rcParams.update({'font.size': 22})

for seed in [0, 42, 1000]:
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

                # formatter = get_format("NeurIPS") # options: ICLR, ICML, NeurIPS, InfThesis
                # figx, figy = formatter.line_width_plot(aspect_ratio="normal")
                # fig = plt.figure(figsize=(1.5 * figx, 1.5 * figy))
                fig = plt.figure(figsize=(20, 20))

                for idx_m, method in enumerate(baselines.keys()):
                    if not os.path.exists(f"results_icp/{problem}_p{percent}_n_source{n_source}_baseline-{method}_seed{seed}.npz"):
                        continue

                    data_results = np.load(f"results_icp/{problem}_p{percent}_n_source{n_source}_baseline-{method}_seed{seed}.npz")
                    rotation_list = [init_rot] + list(data_results["rotation_list"])
                    scalar_list = [init_scalar] + list(data_results["scalar_list"])
                    beta_list = [init_beta] + list(data_results["beta_list"])
                    timings_list = [0.0] + list(data_results["timings_list"])
                    for idx_t, t in enumerate([-1, 20, 40, 60]):
                        ax = fig.add_subplot(len(baselines), 4, idx_m * 4 + idx_t + 1, projection='3d')
                        i = 0
                        while timings_list[i] < t:
                            i += 1
                            if i >= len(timings_list):
                                i -= 1
                                print(f"Could not find anything for t={t}")
                                break
                        rotation = rotation_list[i]
                        scalar = scalar_list[i]
                        beta = beta_list[i]
                        transformed_source_data = source @ rotation * scalar + beta
                        shape_image(target, transformed_source_data, param=vis_param_list[problem], ax=ax)
                        # print(ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
                        if idx_m == 0:
                            if idx_t == 0:
                                plt.title(f"At initialization")
                            else:
                                plt.title(f"After {t} seconds")
                        if idx_t == 0:
                            plt.ylabel(baselines[method])
                plt.tight_layout()
                plt.savefig(f"shape_figs/viz_{problem}_p{percent}_n_source{n_source}_seed{seed}.pdf")

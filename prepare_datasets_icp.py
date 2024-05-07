import pyminiply
import os
import torch
import numpy as np
from baselines.cvpr23_bai_icp_xp import rotation_matrix_3d, shape_image, vis_param_list

def random_subsample(vertices, target_n):
    ind = np.random.choice(vertices.shape[0], size=target_n, replace=False)
    vertices_out = vertices[ind]
    return vertices_out

vertices_per_problem = {
    # Coming from http://graphics.stanford.edu/data/3Dscanrep/
    "dragon": pyminiply.read(os.path.join("data", "dragon_recon", "dragon_vrip.ply"))[0],
    # Coming from http://graphics.stanford.edu/data/3Dscanrep/
    'stanford_bunny': pyminiply.read(os.path.join("data", "bunny", "reconstruction", "bun_zipper.ply"))[0],
    # Coming from https://github.com/nbonneel/spot/tree/master/Datasets/Pointsets/3D
    'mumble_sitting': np.loadtxt(os.path.join("data", "mumble_sitting_80000.pts")),
    # Coming from https://github.com/nbonneel/spot/tree/master/Datasets/Pointsets/3D
    'witchcastle': np.loadtxt(os.path.join("data", "WitchCastle_150000.pts")),
}


np.random.seed(0)
torch.manual_seed(0)

for problem in vertices_per_problem.keys():
    for p in [.05, .07]:
        for n_target in [9 * 1000, 10 * 1000]:
            print(problem, p, n_target)
            vertices = vertices_per_problem[problem]

            n_source = 10 * 1000
            source = random_subsample(vertices, target_n=n_source)
            target = source[:n_target]
            rot_matrix = rotation_matrix_3d(2 * np.pi / 3 * torch.rand(3, ) - np.pi / 3).detach().numpy()
            rotated_target = target @ rot_matrix
            scaling_factor = 2 * np.random.rand(3, )
            scaled_target = scaling_factor * rotated_target
            std = np.std(scaled_target)
            shift = 4 * std * np.random.rand(3, ) - 2 * std
            shifted_target = scaled_target + shift
            M = np.max(np.linalg.norm(shifted_target, axis=1))
            shifted_target_with_noise = np.concatenate((shifted_target, 
                                                        4 * M * np.random.rand(int(p * n_source), 3) - 2 * M))

            M = np.max(np.linalg.norm(source, axis=1))
            source_with_noise = np.concatenate((source, 
                                                4 * M * np.random.rand(int(p * n_source), 3) - 2 * M))

            shape_image(source_with_noise, shifted_target_with_noise, 
                        name=f"processed_data/{problem}_p{p}_n_target{n_target}",
                        param=vis_param_list[problem])
            np.savez(f"processed_data/{problem}_p{p}_n_target{n_target}.npz",
                     source_data=source_with_noise,
                     target_data=shifted_target_with_noise,
                     rotation_matrix=rot_matrix,
                     scaling_factor=scaling_factor,
                     shift=shift)

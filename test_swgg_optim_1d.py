import numpy as np
import scipy
import torch

import matplotlib.pyplot as plt

from perturbed_sliced import PerturbedPartialSWGG

n = 100
d = 2
n_outliers = 10
np.random.seed(0)
torch.manual_seed(0)

x = np.random.randn(n, d) * 3
y = np.random.randn(n, d)
x[:, 1] /= 10
y[:, 1] /= 10

sliced = PerturbedPartialSWGG(max_iter_gradient=20, 
                                        max_iter_partial=n-n_outliers, 
                                        opt_lambda_fun=lambda param: torch.optim.SGD(param, lr=1e-2))
_, bool_ind_x, bool_ind_y, list_w, list_w_grad, w = sliced.fit(x, y)

n_angles = 1000
all_costs = []
all_gradients = []
for theta in np.linspace(-np.pi, np.pi, n_angles):
    # print(0., theta, 2 * np.pi)
    w = torch.Tensor([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])

    # Cost
    proj_x, proj_y = sliced.project_in_1d(x, y, torch.Tensor(w))
    ind_x, ind_y, marginal_costs = sliced.partial_problem.fit(proj_x, proj_y)

    sorted_ind_x = np.sort(ind_x)
    sorted_ind_y = np.sort(ind_y)
    subset_x = x[sorted_ind_x]
    subset_y = y[sorted_ind_y]
    cost = np.sum(np.abs(subset_x - subset_y))
    all_costs.append(cost)

colors = plt.cm.jet(np.linspace(0, 1, n_angles))
rank = np.array(all_costs).argsort().argsort()

plt.plot(np.linspace(-np.pi, np.pi, n_angles), all_costs)
plt.plot(np.linspace(-np.pi, np.pi, n_angles), scipy.ndimage.gaussian_filter(all_costs, sigma=10.))
for (w_x, w_y), (dx, dy) in zip(list_w, list_w_grad):
    theta = np.arctan2(w_y, w_x)
    print(w_x, w_y, theta)
    w = torch.Tensor([np.cos(theta), np.sin(theta)])

    # Cost
    proj_x, proj_y = sliced.project_in_1d(x, y, torch.Tensor(w))
    ind_x, ind_y, marginal_costs = sliced.partial_problem.fit(proj_x, proj_y)

    sorted_ind_x = np.sort(ind_x)
    sorted_ind_y = np.sort(ind_y)
    subset_x = x[sorted_ind_x]
    subset_y = y[sorted_ind_y]
    cost = np.sum(np.abs(subset_x - subset_y))
    plt.plot([theta], [cost], color="r", marker="o")

    epsilon = .3
    normal = np.array([-np.sin(theta), np.cos(theta)])
    grad = np.array([dx, dy]) @ normal
    plt.plot([theta - epsilon, theta + epsilon], 
             [cost - epsilon * grad, cost + epsilon * grad], 
             color="r")
    
    # Estimate gradient the other way (through local slope estimation)
    # proj_x, proj_y = sliced.project_in_1d(x, y, torch.Tensor(w + 1e-3 * normal))
    # ind_x, ind_y, marginal_costs = sliced.partial_problem.fit(proj_x, proj_y)

    # sorted_ind_x = np.sort(ind_x)
    # sorted_ind_y = np.sort(ind_y)
    # subset_x = x[sorted_ind_x]
    # subset_y = y[sorted_ind_y]
    # cost_plus = np.sum(np.abs(subset_x - subset_y))

    # proj_x, proj_y = sliced.project_in_1d(x, y, torch.Tensor(w - 1e-3 * normal))
    # ind_x, ind_y, marginal_costs = sliced.partial_problem.fit(proj_x, proj_y)

    # sorted_ind_x = np.sort(ind_x)
    # sorted_ind_y = np.sort(ind_y)
    # subset_x = x[sorted_ind_x]
    # subset_y = y[sorted_ind_y]
    # cost_minus = np.sum(np.abs(subset_x - subset_y))

    # slope = (cost_plus - cost_minus) / (2 * 1e-3)
    # plt.plot([theta - epsilon, theta + epsilon], 
    #          [cost - epsilon * slope, cost + epsilon * slope], 
    #          color="orange")

    
plt.ylabel("OT cost")
plt.xlabel("$\\theta$")
plt.show()

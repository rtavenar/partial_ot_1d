import numpy as np
import scipy
import torch

import matplotlib.pyplot as plt

from perturbed_sliced import PerturbedPartialSWGG, swgg_step, compute_cost

n = 100
d = 2
n_outliers = 0
np.random.seed(10)
torch.manual_seed(10)

x = np.random.randn(n, d) * 3
y = np.random.randn(n, d)
x[:, 1] /= 10
y[:, 1] /= 10

n_samples = 10
n_steps_gradient = 50
sigma=.1
plot_grads = False
plot_gd = True

n_angles = 100
list_theta = torch.linspace(-np.pi, np.pi, n_angles)

all_costs = []
all_gradients = []
all_F_epsilon = []
all_grad_F_epsilon = []
for i, theta in enumerate(list_theta):
    print(min(list_theta), theta, max(list_theta))
    theta.requires_grad_(True)
    w = torch.stack([torch.cos(theta), torch.sin(theta)])
    w.requires_grad_(True)

    # Cost
    cost = compute_cost(x, y, w.detach().numpy(), n - n_outliers)
    all_costs.append(cost)

    F_epsilon = swgg_step(w, x, y, n-n_outliers, n_samples, "normal", sigma, "cpu")
    all_F_epsilon.append(F_epsilon.detach().numpy())

    if i % 10 == 0 and plot_grads:
        F_epsilon.backward()
        all_grad_F_epsilon.append(np.copy(theta.grad.detach().numpy()))

plt.plot(list_theta.detach().numpy(), all_costs)
plt.plot(list_theta.detach().numpy(), all_F_epsilon)
if plot_grads:
    for i, theta in enumerate(list_theta):
        if i % 10 != 0:
            continue
        epsilon=.3
        grad = all_grad_F_epsilon[i // 10]
        plt.plot([theta - epsilon, theta + epsilon], 
                [all_costs[i] - epsilon * grad, all_costs[i] + epsilon * grad], 
                color="r")


if plot_gd:
    sliced = PerturbedPartialSWGG(max_iter_gradient=n_steps_gradient, 
                                max_iter_partial=n-n_outliers, 
                                perturbation_n_samples=n_samples,
                                perturbation_sigma=sigma,
                                opt_lambda_fun=lambda param: torch.optim.SGD(param, lr=1e-1, momentum=.9))
    _, bool_ind_x, bool_ind_y, list_costs, w = sliced.fit(x, y)
    list_w = sliced.list_w_
    list_w_grad = sliced.list_w_grad_
    for i, ((w_x, w_y), (dx, dy)) in enumerate(zip(list_w, list_w_grad)):
        theta = np.arctan2(w_y, w_x)
        w = torch.Tensor([w_x, w_y])

        # Cost
        cost = compute_cost(x, y, w.detach().numpy(), n - n_outliers)
        plt.plot([theta], [cost], color="r", marker="o")
        plt.annotate(str(i + 1), [theta, cost])

        if plot_grads:
            epsilon = .3
            normal = np.array([-np.sin(theta), np.cos(theta)])
            grad = np.array([dx, dy]) @ normal
            print(w_x, w_y, theta, dx, dy, grad)
            plt.plot([theta - epsilon, theta + epsilon], 
                    [cost - epsilon * grad, cost + epsilon * grad], 
                    color="r")
    
plt.ylabel("OT cost")
plt.xlabel("$\\theta$")
plt.show()

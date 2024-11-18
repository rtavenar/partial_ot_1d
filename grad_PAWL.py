import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scienceplots

from partial import partial_ot_1d

def theta2w(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def G(x, y, theta):
    w = theta2w(theta)
    _, _, marginal_costs = partial_ot_1d(x.dot(w), y.dot(w), max_iter=x.shape[0] // 2, p=2)
    return np.sum(marginal_costs)

def G_eps(x, y, theta, epsilon, n_samples):
    return np.mean([G(x, y, theta + epsilon * z) for z in np.random.randn(n_samples)])

def grad_G_eps_Stein(x, y, theta, epsilon, n_samples):
    G0 = G(x, y, theta)
    return np.mean([(G(x, y, theta + epsilon * z) - G0) * z / epsilon for z in np.random.randn(n_samples)])


n = 20
n_samples = 1000
epsilon = .1
d = 2

np.random.seed(0)

x = np.random.randn(n, d)
y = np.random.randn(n, d)

thetas = np.linspace(-np.pi, np.pi, num=200)
Gs = [G(x, y, theta) for theta in thetas]
Gs_eps = [G_eps(x, y, theta, epsilon, n_samples) for theta in thetas]
grad_Gs_eps_Stein = [grad_G_eps_Stein(x, y, theta, epsilon, n_samples) for theta in thetas]

plt.style.use(['science'])
matplotlib.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(9, 6))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(thetas, Gs, label="$\\text{PAWL}(\\theta)$")
ax1.plot(thetas, Gs_eps, label="$\\text{PAWL}_\\varepsilon(\\theta)$")
ax1.set_xlabel("$\\theta$")
ax1.legend(loc="upper right")
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(thetas, np.zeros_like(thetas), color='k', linestyle="dashed")
ax2.plot(thetas, grad_Gs_eps_Stein, label="$\\nabla_{\\theta} \\text{PAWL}_\\varepsilon$", color='#00B945')
ax2.set_xlabel("$\\theta$")
ax2.legend(loc="lower right")
plt.tight_layout()
plt.savefig("grad_PAWL.pdf")
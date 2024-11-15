import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scienceplots

from partial import partial_ot_1d

def theta2w(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def dw_dtheta(theta):
  return np.array([-np.sin(theta), np.cos(theta)])

def G(x, y, theta):
    w = theta2w(theta)
    _, _, marginal_costs = partial_ot_1d(x.dot(w), y.dot(w), max_iter=x.shape[0] // 2, p=2)
    return np.sum(marginal_costs)

def G_eps(x, y, theta, epsilon, n_samples):
    return np.mean([G(x, y, theta + epsilon * z) for z in np.random.randn(n_samples)])

def grad_G_eps_Stein(x, y, theta, epsilon, n_samples):
    return np.mean([G(x, y, theta + epsilon * z) * z / epsilon for z in np.random.randn(n_samples)])

def dCw_dw(x, y, w):
  x_y = x.T[:, :, None] - y.T[:, None, :]        # (d, n, n)
  xw_yw = x.dot(w)[:, None] - y.dot(w)[None, :]  # (n, n)
  return 2 * x_y * xw_yw[None, :, :]             # (d, n, n)

def pi_star(x, y, theta):
  w = theta2w(theta)
  x_1d = x.dot(w)
  y_1d = y.dot(w)

  pi = np.zeros((x.shape[0], y.shape[0]))
  pi[np.argsort(x_1d), np.argsort(y_1d)] = 1.

  return pi

def pi_star_epsilon(x, y, theta, epsilon, n_samples):
  return np.mean(
      [pi_star(x, y, theta + epsilon * z) for z in np.random.randn(n_samples)],
      axis=0
  )

def grad_G_theta_Berthet(x, y, theta, epsilon, n_samples):
  grad_G_C = pi_star_epsilon(x, y, theta, epsilon, n_samples)  # (n, n)
  dC_dw = dCw_dw(x, y, theta2w(theta))                         # (d, n, n)
  dG_dw = - np.sum(dC_dw * grad_G_C[None, :, :], axis=(1, 2))  # (d, )
  return dG_dw.dot(dw_dtheta(theta))

def grad_G(x, y, theta):
  grad_G_C = pi_star(x, y, theta)                              # (n, n)
  dC_dw = dCw_dw(x, y, theta2w(theta))                         # (d, n, n)
  dG_dw = - np.sum(dC_dw * grad_G_C[None, :, :], axis=(1, 2))  # (d, )
  return dG_dw.dot(dw_dtheta(theta))


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
grad_Gs_eps_Berthet = [grad_G_theta_Berthet(x, y, theta, epsilon, n_samples) for theta in thetas]
grad_Gs = [grad_G(x, y, theta) for theta in thetas]

plt.style.use(['science'])
fig = plt.figure(figsize=(9, 6))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(thetas, Gs, label="$\\text{PAWL}(\\theta)$")
ax1.plot(thetas, Gs_eps, label="$\\text{PAWL}_\\varepsilon(\\theta)$")
ax1.set_xlabel("$\\theta$")
ax1.legend(loc="upper right")
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(thetas, grad_Gs, label="$\\nabla_{\\theta} \\text{PAWL}$")
ax2.plot(thetas, grad_Gs_eps_Berthet, label="$\\nabla_{\\theta} \\text{PAWL}_\\varepsilon$")
ax2.set_xlabel("$\\theta$")
ax2.legend(loc="upper right")
plt.savefig("grad_PAWL.pdf")
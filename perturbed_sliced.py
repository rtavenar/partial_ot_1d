import numpy as np
import torch
import perturbations as perturbations

from sliced import SlicedPartialOT, PartialOT1d

class PerturbedMaheySlicedPartialOT(SlicedPartialOT):
    def __init__(self, max_iter_gradient, max_iter_partial=None, opt_lambda_fun=None, device=None) -> None:
        self.max_iter_gradient = max_iter_gradient
        self.max_iter_partial = max_iter_partial
        self.partial_problem = PartialOT1d(self.max_iter_partial)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.opt_lambda_fun = opt_lambda_fun

    def project_in_1d(self, x, y, w):
        # w /= torch.sqrt(torch.sum(w ** 2, axis=-1, keepdims=True))
        proj_x = (w @ x.T).T
        proj_y = (w @ y.T).T
        return proj_x, proj_y

    def fit(self, x, y):
        assert x.shape[:2] == y.shape[:2]
        n, d = x.shape[:2]
        eta = .001

        def action(w):
            proj_x, proj_y = self.project_in_1d(x, y, w)
            perturbed_costs = []
            # Iterate over the perturbation dimension
            for proj_x_i, proj_y_i in zip(proj_x.T, proj_y.T):
                ind_x, ind_y, _ = self.partial_problem.fit(proj_x_i, proj_y_i)

                sorted_ind_x = np.sort(ind_x)
                sorted_ind_y = np.sort(ind_y)
                subset_x = x[sorted_ind_x]
                subset_y = y[sorted_ind_y]
                cost = np.sum(np.abs(subset_x - subset_y))
                perturbed_costs.append(cost)

            return torch.tensor(perturbed_costs)
        
        pert_action = perturbations.perturbed(action,
                                              num_samples=1000,
                                              sigma=0.5,
                                              noise='gumbel',
                                              batched=False, 
                                              device=self.device)

        min_cost = np.inf
        bool_indices_x, bool_indices_y = None, None
        w = torch.tensor(self.draw_direction(d), requires_grad=True)
        if self.opt_lambda_fun is not None:
            opt = self.opt_lambda_fun([w])
        else:
            opt = torch.optim.SGD([w], lr=0.01, momentum=0.9)
        for _ in range(self.max_iter_gradient):
            opt.zero_grad()
            output = pert_action(w)
            output.backward(torch.ones_like(output))
            opt.step()

            cost = float(output)
            print(cost)

            if cost < min_cost:
                best_w = w.detach().numpy()
                min_cost = cost
        
        proj_x, proj_y = self.project_in_1d(x, y, best_w)
        ind_x, ind_y, _ = self.partial_problem.fit(proj_x, proj_y)
        
        bool_indices_x = np.array([(i in ind_x) for i in range(n)], dtype=bool)
        bool_indices_y = np.array([(i in ind_y) for i in range(n)], dtype=bool)
        
        return min_cost, bool_indices_x, bool_indices_y



if __name__ == "__main__":
    n = 100
    d = 50
    n_outliers = 10
    np.random.seed(0)

    outliers_x = np.random.choice(n, size=n_outliers, replace=False)
    ind_outliers_x = np.zeros((n, d))
    ind_outliers_x[outliers_x] = 1.5
    x = np.random.rand(n, d) + ind_outliers_x

    outliers_y = np.random.choice(n, size=n_outliers, replace=False)
    ind_outliers_y = np.zeros((n, d))
    ind_outliers_y[outliers_y] = 1.5
    y = np.random.rand(n, d) - ind_outliers_y
    
    sliced = PerturbedMaheySlicedPartialOT(max_iter_gradient=5, 
                                           max_iter_partial=n-n_outliers)  #, opt_lambda_fun=lambda param: torch.optim.Adam(param, lr=1e-3))
    _, bool_ind_x, bool_ind_y = sliced.fit(x, y)
    print("x")
    print(np.sum(bool_ind_x), np.sum(bool_ind_x[outliers_x]))
    print("y")
    print(np.sum(bool_ind_y), np.sum(bool_ind_y[outliers_y]))
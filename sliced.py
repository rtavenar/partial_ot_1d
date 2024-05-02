import numpy as np

from partial import PartialOT1d


class SlicedPartialOT:
    def __init__(self, n_proj, max_iter_partial=None) -> None:
        self.n_proj = n_proj
        self.max_iter_partial = max_iter_partial
        self.partial_problem = PartialOT1d(self.max_iter_partial)

    def project_in_1d(self, x, y):
        assert x.shape[1] == y.shape[1]
        d = x.shape[1]
        w = np.random.randn(d)
        proj_x = np.dot(x, w)
        proj_y = np.dot(y, w)
        return proj_x, proj_y, w


class MonteCarloSlicedPartialOT(SlicedPartialOT):
    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]
        n = x.shape[0]

        selection_freq_x = np.zeros((n, ))
        selection_freq_y = np.zeros((n, ))
        avg_cost = 0.
        for _ in range(self.n_proj):
            proj_x, proj_y = self.project_in_1d(x, y)[:2]
            ind_x, ind_y, marginal_costs = self.partial_problem.fit(proj_x, proj_y)
            selection_freq_x[ind_x] += 1.
            selection_freq_y[ind_y] += 1.
            avg_cost += np.sum(marginal_costs) # Sum of marginal costs is the total cost
        
        return avg_cost / self.n_proj, selection_freq_x / self.n_proj, selection_freq_y / self.n_proj


class MaheySlicedPartialOT(SlicedPartialOT):
    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]

        min_cost = np.inf
        bool_indices_x, bool_indices_y = None, None
        for _ in range(self.n_proj):
            proj_x, proj_y = self.project_in_1d(x, y)[:2]
            ind_x, ind_y, _ = self.partial_problem.fit(proj_x, proj_y)

            sorted_ind_x = np.sort(ind_x)
            sorted_ind_y = np.sort(ind_y)
            subset_x = x[sorted_ind_x]
            subset_y = y[sorted_ind_y]
            cost = np.sum(np.abs(subset_x - subset_y))

            if cost < min_cost:
                bool_indices_x = np.array([(i in ind_x) for i in range(n)], dtype=bool)
                bool_indices_y = np.array([(i in ind_y) for i in range(n)], dtype=bool)
        
        return min_cost, bool_indices_x, bool_indices_y



if __name__ == "__main__":
    n = 30
    d = 20
    n_outliers = 10
    np.random.seed(0)

    outliers_x = np.random.choice(n, size=n_outliers, replace=False)
    ind_outliers_x = np.zeros((n, d))
    ind_outliers_x[outliers_x] = 3.
    x = np.random.rand(n, d) + ind_outliers_x

    outliers_y = np.random.choice(n, size=n_outliers, replace=False)
    ind_outliers_y = np.zeros((n, d))
    ind_outliers_y[outliers_y] = 3.
    y = np.random.rand(n, d) - ind_outliers_y
    
    sliced = MonteCarloSlicedPartialOT(n_proj=100, max_iter_partial=n-n_outliers)
    _, freq_x, freq_y = sliced.fit(x, y)
    print("x")
    print(freq_x)
    print(freq_x[outliers_x])
    print("y")
    print(freq_y)
    print(freq_y[outliers_y])
    
    sliced = MaheySlicedPartialOT(n_proj=100, max_iter_partial=n-n_outliers)
    _, freq_x, freq_y = sliced.fit(x, y)
    print("x")
    print(freq_x)
    print(freq_x[outliers_x])
    print("y")
    print(freq_y)
    print(freq_y[outliers_y])

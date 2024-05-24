import numpy as np

from partial_nb import partial_ot_1d


class SlicedPartialOT:
    def __init__(self, n_proj, max_iter_partial=None) -> None:
        self.n_proj = n_proj
        self.max_iter_partial = max_iter_partial

    def draw_direction(self, d):
        w = np.random.randn(d)
        return w

    def project_in_1d(self, x, y, w):
        w /= np.linalg.norm(w)
        proj_x = np.dot(x, w)
        proj_y = np.dot(y, w)
        return proj_x, proj_y


class MonteCarloSlicedPartialOT(SlicedPartialOT):
    def fit(self, x, y):
        assert x.shape[1] == y.shape[1]
        n, d = x.shape[:2]
        m, d = y.shape[:2]

        selection_freq_x = np.zeros((n, ))
        selection_freq_y = np.zeros((m, ))
        avg_cost = 0.
        for _ in range(self.n_proj):
            w = self.draw_direction(d)
            proj_x, proj_y = self.project_in_1d(x, y, w)
            ind_x, ind_y, marginal_costs = partial_ot_1d(proj_x, proj_y, self.max_iter_partial)
            selection_freq_x[ind_x] += 1.
            selection_freq_y[ind_y] += 1.
            avg_cost += np.sum(marginal_costs) # Sum of marginal costs is the total cost
        
        return selection_freq_x / self.n_proj, selection_freq_y / self.n_proj, avg_cost / self.n_proj


class MaheySlicedPartialOT(SlicedPartialOT):
    def fit(self, x, y, projections=None):
        assert x.shape[1] == y.shape[1]
        n, d = x.shape[:2]
        m, d = y.shape[:2]

        min_cost = np.inf
        indices_x, indices_y = None, None
        best_w = None
        for i in range(self.n_proj):
            if projections is None:
                w = self.draw_direction(d)
            else:
                w = projections[i]
            proj_x, proj_y = self.project_in_1d(x, y, w)
            ind_x, ind_y, _ = partial_ot_1d(proj_x, proj_y, self.max_iter_partial)

            sorted_ind_x = np.sort(ind_x)
            sorted_ind_y = np.sort(ind_y)
            subset_x = x[sorted_ind_x]
            subset_y = y[sorted_ind_y]
            cost = np.sum(np.abs(subset_x - subset_y))

            if cost < min_cost:
                indices_x = ind_x
                indices_y = ind_y
                min_cost = cost
                best_w = w
        
        return indices_x, indices_y, best_w, min_cost



if __name__ == "__main__":
    n = 100
    d = 50
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
    freq_x, freq_y, _ = sliced.fit(x, y)
    print("x")
    print(freq_x)
    print(freq_x[outliers_x])
    print("y")
    print(freq_y)
    print(freq_y[outliers_y])
    
    sliced = MaheySlicedPartialOT(n_proj=100, max_iter_partial=n-n_outliers)
    ind_x, ind_y, theta, min_cost = sliced.fit(x, y)
    print(min_cost)

import numpy as np
import tensorflow as tf
import perturbations

from sliced import SlicedPartialOT, PartialOT1d

class PerturbedMaheySlicedPartialOT(SlicedPartialOT):
    def __init__(self, max_iter_gradient, max_iter_partial=None) -> None:
        self.max_iter_gradient = max_iter_gradient
        self.max_iter_partial = max_iter_partial
        self.partial_problem = PartialOT1d(self.max_iter_partial)

    def project_in_1d(self, x, y, w):
        w /= np.sqrt(np.sum(w ** 2, axis=-1, keepdims=True))
        proj_x = np.dot(w, x.T).T
        proj_y = np.dot(w, y.T).T
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

            return tf.cast(tf.stack(perturbed_costs), dtype=tf.float32)
        
        pert_action = perturbations.perturbed(action,
                                              num_samples=1000,
                                              sigma=0.5,
                                              noise='gumbel',
                                              batched=False)

        min_cost = np.inf
        bool_indices_x, bool_indices_y = None, None
        w = tf.Variable(self.draw_direction(d), dtype=tf.float32)
        for _ in range(self.max_iter_gradient):
            with tf.GradientTape() as tape:
                theta = tf.Variable(w)
                pert_cost = pert_action(theta)
                grad_pert, = tape.gradient(pert_cost, [theta])
            
            w = tf.Variable(w - eta * grad_pert, dtype=tf.float32)

            cost = float(pert_cost)
            print(cost)

            if cost < min_cost:
                best_w = np.array(w)
        
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
    ind_outliers_x[outliers_x] = 10.
    x = np.random.rand(n, d) + ind_outliers_x

    outliers_y = np.random.choice(n, size=n_outliers, replace=False)
    ind_outliers_y = np.zeros((n, d))
    ind_outliers_y[outliers_y] = 10.
    y = np.random.rand(n, d) - ind_outliers_y
    
    sliced = PerturbedMaheySlicedPartialOT(max_iter_gradient=20, max_iter_partial=n-n_outliers)
    _, bool_ind_x, bool_ind_y = sliced.fit(x, y)
    print("x")
    print(np.sum(bool_ind_x), np.sum(bool_ind_x[outliers_x]))
    print("y")
    print(np.sum(bool_ind_y), np.sum(bool_ind_y[outliers_y]))
import numpy as np
import warnings

class PartialOT1d:
    def __init__(self, max_iter) -> None:
        self.max_iter = max_iter

    @property
    def x_sorted(self):
       return self.x[self.indices_sort_x]

    @property
    def y_sorted(self):
       return self.y[self.indices_sort_y]

    @property
    def n_x(self):
       return len(self.x)

    @property
    def n_y(self):
       return len(self.y)

    def preprocess(self, x, y):
        self.x = x
        self.y = y

        self.indices_sort_x = np.argsort(x)
        self.indices_sort_y = np.argsort(y)

        xy = np.concatenate((self.x_sorted, self.y_sorted))
        self.indices_sort_xy = np.argsort(xy)

        idx = np.concatenate((np.zeros(self.n_x, dtype=int), np.ones(self.n_y, dtype=int)))
        self.sorted_distrib_indicator = idx[self.indices_sort_xy]
    
    def _binary_search(self, arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return arr[left] if left < len(arr) else None

    # TODO: change this for indices to be a class index
    def _insert_constant_values(self, arr, indices):
        """
        Assumes indices is a boolean array.
        
        Takes arr as input. For each position i in arr, if indices[i] is True,
        the value from arr is copied, 
        otherwise, the previous value that was copied from arr is repeated.
        """
        arr_insert = []
        for i in range(len(arr)):
            if indices[i]:
                arr_insert.append(arr[i])
            elif i == 0:
                arr_insert.append(0)
            else:
                arr_insert.append(arr_insert[i-1])
        return np.array(arr_insert)

    def compute_cumulative_sum_differences(self):
        cum_sum_xs = np.cumsum(self.x_sorted)
        cum_sum_ys = np.cumsum(self.y_sorted)
        cum_sum = np.concatenate((cum_sum_xs, cum_sum_ys))
        cum_sum = cum_sum[self.indices_sort_xy]

        cum_sum_x = self._insert_constant_values(cum_sum, self.sorted_distrib_indicator==0)
        cum_sum_y = self._insert_constant_values(cum_sum, self.sorted_distrib_indicator==1)

        return cum_sum_x - cum_sum_y

    def compute_rank_differences(self):
        ranks_x, ranks_y = np.arange(self.n_x), np.arange(self.n_y)
        ranks_xy = np.concatenate((ranks_x, ranks_y))
        ranks_xy = ranks_xy[self.indices_sort_xy]

        ranks_xy_x = ranks_xy.copy()
        ranks_xy_x[self.sorted_distrib_indicator==1] = 0
        ranks_xy_x[self.sorted_distrib_indicator==0] += 1
        ranks_xy_x_cum = self._insert_constant_values(ranks_xy_x, self.sorted_distrib_indicator==0)

        ranks_xy_y = ranks_xy.copy()
        ranks_xy_y[self.sorted_distrib_indicator==0] = 0
        ranks_xy_y[self.sorted_distrib_indicator==1] += 1
        ranks_xy_y_cum = self._insert_constant_values(ranks_xy_y, self.sorted_distrib_indicator==1)

        diff_ranks = ranks_xy_x_cum - ranks_xy_y_cum

        d_cumranks_indices = {}
        for i in range(self.n_x + self.n_y):
            d_cumranks_indices[diff_ranks[i]] = d_cumranks_indices.get(diff_ranks[i], []) + [i]

        return ranks_xy, diff_ranks, d_cumranks_indices
    
    def compute_costs(self, diff_cum_sum, diff_ranks, d_cumranks_indices):
        l_costs = []
        for i in range(len(diff_ranks)):
            # For each item in either distrib, find the scope of the smallest
            # "group" that would start at that point and extend on the right, 
            # if one exists, and store the cost of this "group" by relying 
            # on differences of cumulative sums
            cur_rank = diff_ranks[i]
            if self.sorted_distrib_indicator[i] == 0:
                target_rank = cur_rank - 1
            else:
                target_rank = cur_rank + 1
            list_positions = d_cumranks_indices.get(target_rank, [])
            next_pos = self._binary_search(list_positions, i)
            if next_pos is not None:
                if i == 0:
                    cost = diff_cum_sum[next_pos] # - 0
                else:
                    cost = diff_cum_sum[next_pos]   - diff_cum_sum[i - 1]
                # i: start of the "group", "next_pos": end of the "group", abs(cost): cost of the group
                l_costs.append((i, next_pos, abs(cost)))
        return sorted(l_costs, key=lambda x: x[2])[:self.max_iter]

    def generate_solution(self, costs, ranks_xy):
        iter_enters_solution = {}
        for iter, (i, j, _) in enumerate(costs):
            if i not in iter_enters_solution:
                iter_enters_solution[i] = iter
            if j not in iter_enters_solution:
                iter_enters_solution[j] = iter
        superset_of_active_set = sorted(iter_enters_solution.keys())

        # Gather packs of adjacent points in `superset_of_active_set`
        idx_active = np.zeros((self.n_x + self.n_y + 1), dtype=int)
        idx_active[superset_of_active_set] = 1
        diff_idx = idx_active[1:] - idx_active[:-1]
        idx_start = np.where(diff_idx == 1)[0] + 1
        idx_end = np.where(diff_idx == -1)[0] + 1  # idx_end is excluded (first index outside the pack)

        # For each pack, remove extra points that prevent it from being an actual set of pairs
        # for i, j in zip(idx_start, idx_end):
        #     n_x_in_group = np.sum(self.sorted_distrib_indicator[i:j] == 0)
        #     n_y_in_group = np.sum(self.sorted_distrib_indicator[i:j] == 1)
        #     print("---", self.sorted_distrib_indicator[i:j], [iter_enters_solution.get(k, -1) for k in range(i, j)])
        #     while n_x_in_group != n_y_in_group:
        #         print(i, j, n_x_in_group, n_y_in_group)
        #         if iter_enters_solution[i] < iter_enters_solution[j - 1]:
        #             del iter_enters_solution[j - 1]
        #             if self.sorted_distrib_indicator[j - 1] == 0:
        #                 n_x_in_group -= 1
        #             else: 
        #                 n_y_in_group -= 1
        #             j -= 1
        #         elif iter_enters_solution[i] > iter_enters_solution[j - 1]:
        #             del iter_enters_solution[i]
        #             if self.sorted_distrib_indicator[i] == 0:
        #                 n_x_in_group -= 1
        #             else: 
        #                 n_y_in_group -= 1
        #             i -= 1
        for i, j in zip(idx_start, idx_end):  # WE KNOW THIS DOES NOT ALWAYS WORK, WE WILL HAVE TO CHANGE THAT
            if (j - i) % 2 != 0:
                if iter_enters_solution[i] < iter_enters_solution[j - 1]:
                    del iter_enters_solution[j - 1]
                elif iter_enters_solution[i] > iter_enters_solution[j - 1]:
                    del iter_enters_solution[i]

        # Write down solution with remaining points
        active_set = sorted(iter_enters_solution.keys())
        ranks_active_set = ranks_xy[active_set]
        which_distrib = self.sorted_distrib_indicator[active_set]

        return ranks_active_set[which_distrib == 0], ranks_active_set[which_distrib == 1]
    
    def check_solution_valid(self, indices_x, indices_y):
        if len(indices_x) != len(indices_y):
            warnings.warn("A valid solution should have as many x's as y's", RuntimeWarning)
        # We could implement other checks here

    def fit(self, x, y):
        # Sort distribs and keep track of their original indices (stored in instance attributes)
        self.preprocess(x, y)

        # Precompute useful quantities
        diff_cum_sum = self.compute_cumulative_sum_differences()
        ranks_xy, diff_ranks, d_cumranks_indices = self.compute_rank_differences()

        # Compute costs for "groups"
        costs = self.compute_costs(diff_cum_sum, diff_ranks, d_cumranks_indices)

        # Generate solution from sorted costs
        sol_indices_x_sorted, sol_indices_y_sorted = self.generate_solution(costs, ranks_xy)

        self.check_solution_valid(sol_indices_x_sorted, sol_indices_y_sorted)
        # TODO: write better docs for each method

        return self.indices_sort_x[sol_indices_x_sorted], self.indices_sort_y[sol_indices_y_sorted]



if __name__ == "__main__":
    pb = PartialOT1d(max_iter=10)
    np.random.seed(0)
    x = np.random.rand(10, )
    y = np.random.rand(10, )
    print(pb.fit(x, y))
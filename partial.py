import numpy as np
import warnings
from sortedcontainers import SortedList

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
        """Given two 1d distributions `x` and `y`:
        
        1. `self.indices_sort_x` sorts `x` (ie. `x[self.indices_sort_x]` is sorted) and
           `self.indices_sort_y` sorts `y` (ie. `y[self.indices_sort_y]` is sorted)
        
        2. stack them into a single distrib such that:
        
        * the new distrib is sorted with sort indices (wrt a stack of sorted x and sorted y) `self.indices_sort_xy`
        * `self.sorted_distrib_indicator` is a vector of zeros and ones where 0 means 
          "this point comes from x" and 1 means "this point comes from y"
        """
        self.x = x if type(x) is np.ndarray else np.array(x)
        self.y = y if type(y) is np.ndarray else np.array(y)

        self.indices_sort_x = np.argsort(x)
        self.indices_sort_y = np.argsort(y)

        xy = np.concatenate((self.x_sorted, self.y_sorted))
        self.indices_sort_xy = np.argsort(xy)

        idx = np.concatenate((np.zeros(self.n_x, dtype=int), np.ones(self.n_y, dtype=int)))
        self.sorted_distrib_indicator = idx[self.indices_sort_xy]
    
    def _binary_search(self, arr, target):
        """Return the first element in `arr` that is strictly greater than `target` 
        (returns `None` if all elements in `arr` are lower or equal than `target`).
        
        Note that `arr` is assumed to be sorted.

        Examples
        --------
        >>> p = PartialOT1d(-1)
        >>> p._binary_search([0, 2, 4, 5, 7], 4)
        5
        >>> p._binary_search([0, 2, 4], 4)  # None
        >>> p._binary_search([], 4)         # None
        """
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return arr[left] if left < len(arr) else None

    def _insert_constant_values(self, arr, distrib_index):
        """Takes `arr` as input. For each position `i` in `arr`, 
        if `self.sorted_distrib_indicator[i]==distrib_index`,
        the value from `arr` is copied, 
        otherwise, the previous value that was copied from `arr` is repeated.

        Examples
        --------
        >>> p = PartialOT1d(-1)
        >>> x = [0, 3, 4]
        >>> y = [1, 2, 5]
        >>> p.preprocess(x, y)
        >>> p._insert_constant_values([1, -1, -1, 2, 3, -1], 0)
        array([1, 1, 1, 2, 3, 3])
        """
        arr_insert = []
        for i in range(len(arr)):
            if self.sorted_distrib_indicator[i]==distrib_index:
                arr_insert.append(arr[i])
            elif i == 0:
                arr_insert.append(0)
            else:
                arr_insert.append(arr_insert[i-1])
        return np.array(arr_insert)

    def compute_cumulative_sum_differences(self):
        """Computes difference between cumulative sums for both distribs.

        The cumulative sum vector for a sorted x is:

            cumsum_x = [x_0, x_0 + x_1, ..., x_0 + ... + x_n]

        This vector is then extend to reach a length of 2*n by repeating 
        values at places that correspond to an y item.
        In other words, if the order of x and y elements on the real 
        line is something like x-y-y-x..., then the extended vector is 
        (note the repetitions):

            cumsum_x = [x_0, x_0, x_0, x_0 + x_1, ..., x_0 + ... + x_n]

        Overall, this function returns `cumsum_x - cumsum_y` where `cumsum_x`
        and `cumsum_y` are the extended versions.

        Examples
        --------
        >>> p = PartialOT1d(-1)
        >>> x = [-1, 3, 4]
        >>> y = [1, 2, 5]
        >>> p.preprocess(x, y)
        >>> p.sorted_distrib_indicator
        array([0, 1, 1, 0, 0, 1])
        >>> p.compute_cumulative_sum_differences()  # [-1, -1, -1, 2, 6, 6] - [0, 1, 3, 3, 3, 8]
        array([-1, -2, -4, -1,  3, -2])
        """
        cum_sum_xs = np.cumsum(self.x_sorted)
        cum_sum_ys = np.cumsum(self.y_sorted)
        cum_sum = np.concatenate((cum_sum_xs, cum_sum_ys))
        cum_sum = cum_sum[self.indices_sort_xy]

        cum_sum_x = self._insert_constant_values(cum_sum, 0)
        cum_sum_y = self._insert_constant_values(cum_sum, 1)

        return cum_sum_x - cum_sum_y

    def compute_rank_differences(self):
        """Precompute important rank-related quantities for better group generation.
        
        Three quantities are returned:

        * `ranks_xy` is an array that gathers ranks of the elements in 
          their original distrib, eg. if the distrib indicator is
          [0, 1, 1, 0, 0, 1], then `rank_xy` will be:
          [0, 0, 1, 1, 2, 2]
        * `diff_ranks` is computed from `ranks_xy_x_cum` and `ranks_xy_y_cum`.
          For the example above, we would have:
          
            ranks_xy_x_cum = [1, 1, 1, 2, 3, 3]
            ranks_xy_y_cum = [0, 1, 2, 2, 2, 3]

          And `diff_ranks` is just `ranks_xy_x_cum - ranks_xy_y_cum`.

        * `d_cumranks_indices` indicates, for each `diff_ranks` value, 
          the sorted lists of all indexes of its occurrences in `diff_ranks`

        Examples
        --------
        >>> p = PartialOT1d(max_iter=2)
        >>> x = [-2, 2, 3]
        >>> y = [-1, 1, 5]
        >>> p.preprocess(x, y)
        >>> ranks_xy, diff_ranks, d_cumranks_indices = p.compute_rank_differences()
        >>> ranks_xy
        array([0, 0, 1, 1, 2, 2])
        >>> diff_ranks
        array([ 1,  0, -1,  0,  1,  0])
        >>> d_cumranks_indices
        {1: [0, 4], 0: [1, 3, 5], -1: [2]}
        """
        ranks_x, ranks_y = np.arange(self.n_x), np.arange(self.n_y)
        ranks_xy = np.concatenate((ranks_x, ranks_y))
        ranks_xy = ranks_xy[self.indices_sort_xy]

        ranks_xy_x = ranks_xy.copy()
        ranks_xy_x[self.sorted_distrib_indicator==1] = 0
        ranks_xy_x[self.sorted_distrib_indicator==0] += 1
        ranks_xy_x_cum = self._insert_constant_values(ranks_xy_x, 0)

        ranks_xy_y = ranks_xy.copy()
        ranks_xy_y[self.sorted_distrib_indicator==0] = 0
        ranks_xy_y[self.sorted_distrib_indicator==1] += 1
        ranks_xy_y_cum = self._insert_constant_values(ranks_xy_y, 1)

        diff_ranks = ranks_xy_x_cum - ranks_xy_y_cum

        d_cumranks_indices = {}
        for i in range(self.n_x + self.n_y):
            d_cumranks_indices[diff_ranks[i]] = d_cumranks_indices.get(diff_ranks[i], []) + [i]

        return ranks_xy, diff_ranks, d_cumranks_indices
    
    def compute_costs(self, diff_cum_sum, diff_ranks, d_cumranks_indices):
        """For each element in sorted `x`, compute its group (cf note below).
        Then compute the cost for each group and sort all groups in increasing 
        cost order.

        Note: the "group" of a point x_i in x is the minimal set of adjacent 
        points (starting at x_i and extending to the right) that one should 
        take to get a balanced set (ie. a set in which we have as many 
        elements from x as elements from y)
        """
        l_costs = []
        self._group_starting_at = {}
        self._group_ending_at = {}
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
                self._group_starting_at[i] = {"ends_at": next_pos, "cost": abs(cost)}
                assert next_pos not in self._group_ending_at
                self._group_ending_at[next_pos] = {"starts_at": i, "cost": abs(cost)}
        return SortedList(l_costs, key=lambda x: x[2]), self.precompute_pack_costs_cumsum()
    
    def precompute_pack_costs_cumsum(self):
        # TODO: docstring
        pack_costs_cumsum = {}
        for i in range(self.n_x + self.n_y):
            # If there is a group ending there
            # (if not, this cannot be the end of a pack)
            if i in self._group_ending_at:
                start = self._group_ending_at[i]["starts_at"]
                additional_cost = self._group_ending_at[i]["cost"]
                pack_costs_cumsum[i] = pack_costs_cumsum.get(start - 1, 0) + additional_cost
        return pack_costs_cumsum

     
    @classmethod
    def _insert_new_pack(cls, packs: SortedList, candidate_pack):
        """Insert the `candidate_pack` into the sorted list of `packs`.
        `packs` is modified in-place and the pack in which the candidate 
        is inserted is returned.

        Examples
        --------
        >>> packs = SortedList([[2, 4], [8, 9]], key=lambda t: t[0])
        >>> PartialOT1d._insert_new_pack(packs, [5, 7])
        [2, 9]
        >>> packs # doctest: +ELLIPSIS
        SortedKeyList([[2, 9]], key=<function <lambda> at ...>)
        >>> 
        >>> packs = SortedList([[2, 4], [8, 9]], key=lambda t: t[0])
        >>> PartialOT1d._insert_new_pack(packs, [11, 12])
        [11, 12]
        >>> packs # doctest: +ELLIPSIS
        SortedKeyList([[2, 4], [8, 9], [11, 12]], key=<function <lambda> at ...>)
        >>> 
        >>> packs = SortedList([[2, 4], [8, 9]], key=lambda t: t[0])
        >>> PartialOT1d._insert_new_pack(packs, [5, 6])
        [2, 6]
        >>> packs # doctest: +ELLIPSIS
        SortedKeyList([[2, 6], [8, 9]], key=<function <lambda> at ...>)
        >>> 
        >>> packs = SortedList([[2, 4], [8, 9]], key=lambda t: t[0])
        >>> PartialOT1d._insert_new_pack(packs, [6, 7])
        [6, 9]
        >>> packs # doctest: +ELLIPSIS
        SortedKeyList([[2, 4], [6, 9]], key=<function <lambda> at ...>)
        """
        i, j = candidate_pack
        idx = packs.bisect_left(candidate_pack)
        # Is `i` adjacent to `packs[idx - 1]` 
        # or `j` adjacent to `packs[idx]`?
        if idx > 0 and packs[idx - 1][1] == i - 1:
            # Extend the pack up to `j`
            p = packs.pop(idx - 1)
            candidate_pack = [p[0], candidate_pack[1]]
        idx = packs.bisect_left(candidate_pack)
        if idx < len(packs) and packs[idx][0] == j + 1:
            # Extend the pack from `i` on
            p = packs.pop(idx)
            candidate_pack = [candidate_pack[0], p[1]]
        packs.add(candidate_pack)
        return candidate_pack
    
    def _compute_cost_for_pack(self, idx_start, idx_end, pack_costs_cumsum):
        """Compute the associated cost for a pack (set of contiguous points
        included in the solution) ranging from `idx_start` to `idx_end` (both included).
        
        # TODO: unit tests
        """
        return pack_costs_cumsum[idx_end] - pack_costs_cumsum.get(idx_start - 1, 0)
    
    def get_group_starting_at(self, i):
        return self._group_starting_at.get(i, {"ends_at": None, "cost": None})

    def generate_solution_using_marginal_costs(self, costs: SortedList, ranks_xy, pack_costs_cumsum):
        """Generate a solution from a sorted list of group costs.
        See the note in `compute_costs` docs for a definition of groups.

        The solution is a pair of lists. The first list contains the indices from `sorted_x`
        that are in the active set, and the second one contains the indices from `sorted_y`
        that are in the active set.

        Examples
        --------
        >>> p = PartialOT1d(max_iter=3)
        >>> x = [1, 2, 5, 6]
        >>> y = [3, 4, 11, 12]
        >>> p.preprocess(x, y)
        >>> diff_cum_sum = p.compute_cumulative_sum_differences()
        >>> ranks_xy, diff_ranks, d_cumranks_indices = p.compute_rank_differences()
        >>> costs, pack_costs_cumsum = p.compute_costs(diff_cum_sum, diff_ranks, d_cumranks_indices)
        >>> ranks_xy = p.compute_rank_differences()[0]
        >>> p.generate_solution_using_marginal_costs(costs, ranks_xy, pack_costs_cumsum)
        (array([1, 2, 3]), array([0, 1, 2]))
        """
        active_set = set()
        packs = SortedList(key=lambda t: t[0])
        list_marginal_costs = []
        while len(costs) > 0 and self.max_iter > len(active_set) // 2:
            i, j, c = costs.pop(index=0)
            if i in active_set or j in active_set:
                continue
            new_pack = None
            # Case 1: j == i + 1 => "Simple" insert
            if j == i + 1:
                active_set.add(i)
                active_set.add(j)
                new_pack = PartialOT1d._insert_new_pack(packs, [i, j])
            # Case 2: insert a group that contains a pack
            elif [i + 1, j - 1] in packs:
                active_set.add(i)
                active_set.add(j)
                packs.remove([i + 1, j - 1])
                new_pack = PartialOT1d._insert_new_pack(packs, [i, j])
            # There should be no "Case 3"
            else:
                self._print_current_status(active_set, i, j)
                raise ValueError
            list_marginal_costs.append(c)
            
            # We now need to update the groups wrt the pack we have just created
            p_s, p_e = new_pack
            if p_s == 0 or p_e == self.n_x + self.n_y - 1:
                continue
            # 1. If (p_s - 1, p_e + 1) is a group: remove it (it will be re-inserted later)
            if self.get_group_starting_at(p_s - 1)["ends_at"] == p_e + 1:
                previous_cost = self.get_group_starting_at(p_s - 1)["cost"]
                costs.remove((p_s - 1, p_e + 1, previous_cost))

            # 2. If (p_s - 1, p_s) and (p_e, p_e + 1) are matching groups, 
            #    remove them and insert the overall group (p_s - 1, p_e + 1) 
            #    with the adequate marginal cost
            if self.sorted_distrib_indicator[p_s - 1] != self.sorted_distrib_indicator[p_e + 1]:
                if (self.get_group_starting_at(p_s - 1)["ends_at"] == p_s 
                        and self.get_group_starting_at(p_e)["ends_at"] == p_e + 1):
                    # Below we use discard instead of remove (discard does not throw an error
                    # if the item is not in the list) since it could be that the group 
                    # has already been removed beforehand because of an overlap with
                    # another pack
                    costs.discard((p_s - 1, p_s, self.get_group_starting_at(p_s - 1)["cost"]))
                    costs.discard((p_e, p_e + 1, self.get_group_starting_at(p_e)["cost"]))

                # Insert (p_s - 1, p_e + 1) as a new pseudo-group with marginal cost
                marginal_cost = (self._compute_cost_for_pack(p_s - 1, p_e + 1, pack_costs_cumsum)
                                - self._compute_cost_for_pack(p_s, p_e, pack_costs_cumsum))
                costs.add((p_s - 1, p_e + 1, marginal_cost))

        # Generate active_set as sorted set of indexes
        active_set = sorted(active_set)

        # Write down solution with remaining points
        ranks_active_set = ranks_xy[active_set]
        which_distrib = self.sorted_distrib_indicator[active_set]

        return ranks_active_set[which_distrib == 0], ranks_active_set[which_distrib == 1], list_marginal_costs
    
    def check_solution_valid(self, indices_x, indices_y):
        """Check that a solution (given by two lists of indices in 
        sorted x and sorted y respectively) is valid.

        Examples
        --------
        >>> p = PartialOT1d(-1)
        >>> x = [-2, 3, 5]
        >>> y = [-1, 1, 3]
        >>> p.preprocess(x, y)
        >>> p.check_solution_valid([0, 1], [0, 2])
        >>> p.check_solution_valid([0, 1], [0, 1, 2])  # This one raises a warning
        >>> p.check_solution_valid([0, 1], [0, 3])     # This one too
        """
        if len(indices_x) != len(indices_y):
            warnings.warn("A valid solution should have as many x's as y's", RuntimeWarning)
        if not np.alltrue(np.array(indices_x) < self.n_x) or not np.alltrue(np.array(indices_x) >= 0):
            warnings.warn(f"All x indices should be between 0 and {self.n_x - 1}", RuntimeWarning)
        if not np.alltrue(np.array(indices_y) < self.n_y) or not np.alltrue(np.array(indices_y) >= 0):
            warnings.warn(f"All y indices should be between 0 and {self.n_y - 1}", RuntimeWarning)
        # We could implement other checks here

    def fit(self, x, y):
        """Main method for the class.
        
        Does:
        
        1. Preprocessing of the distribs (sorted & co)
        2. Precomputations (ranks, cumulative sums)
        3. Extraction of groups
        4. Generate and return solution

        Examples
        --------
        >>> p = PartialOT1d(max_iter=2)
        >>> x = [5, -2, 3.1]
        >>> y = [-1, 1, 3]
        >>> p.fit(x, y)
        (array([1, 2]), array([0, 2]))
        """
        # Sort distribs and keep track of their original indices (stored in instance attributes)
        self.preprocess(x, y)

        # Precompute useful quantities
        diff_cum_sum = self.compute_cumulative_sum_differences()
        ranks_xy, diff_ranks, d_cumranks_indices = self.compute_rank_differences()

        # Compute costs for "groups"
        costs, pack_costs_cumsum = self.compute_costs(diff_cum_sum, diff_ranks, d_cumranks_indices)

        # Generate solution from sorted costs
        sol_indices_x_sorted, sol_indices_y_sorted, sol_costs = self.generate_solution_using_marginal_costs(costs, ranks_xy, pack_costs_cumsum)

        self.check_solution_valid(sol_indices_x_sorted, sol_indices_y_sorted)

        # Convert back into indices in original `x` and `y` distribs
        return self.indices_sort_x[sol_indices_x_sorted], self.indices_sort_y[sol_indices_y_sorted], sol_costs

    def _print_current_status(self, active_set, i, j):
        print("=" * (15 + self.n_x + self.n_y) + "\nCurrent status")
        s = "Distribs:      "
        for v in self.sorted_distrib_indicator:
            if v == 0:
                s += "o"
            else:
                s += "-"
        print(s)
        s = "Active set:    "
        for pos in range(self.n_x + self.n_y):
            if pos in active_set:
                s += "x"
            else:
                s += " "
        print(s)
        s = "Current group: "
        for pos in range(self.n_x + self.n_y):
            if pos in [i, j]:
                s += "^"
            else:
                s += " "
        print(s)
        print("=" * (15 + self.n_x + self.n_y))

if __name__ == "__main__":
    pb = PartialOT1d(max_iter=40)
    np.random.seed(0)
    x = np.random.rand(30, )
    y = np.random.rand(30, )
    indices_x, indices_y, marginal_costs = pb.fit(x, y)
    print(len(indices_x), len(indices_y))
    print(indices_x)
    print(indices_y)
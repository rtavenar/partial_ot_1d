import numpy as np
import heapq
from kneed import KneeLocator

from numba import njit
from numba.typed import Dict
from numba.core import types

int64 = types.int64

@njit(cache=True, fastmath=True)
def bisect_left(arr, x):
    """Similar to bisect.bisect_left(), from the built-in library."""
    M = len(arr)
    for i in range(M):
        if arr[i] >= x:
            return i
    return M

@njit(cache=True, fastmath=True)
def insert_new_pack(packs_starting_at, packs_ending_at, candidate_pack):
    """Insert the `candidate_pack` into the already known packs stored in 
    `packs_starting_at` and `packs_ending_at`.
    `packs_starting_at` and `packs_ending_at` are modified in-place and 
    the pack in which the candidate is inserted is returned.

    Examples
    --------
    >>> packs_starting_at = {2: 4, 8: 9}
    >>> packs_ending_at = {4: 2, 9: 8}
    >>> insert_new_pack(packs_starting_at, packs_ending_at, [5, 7])
    (2, 9)
    >>> packs_starting_at
    {2: 9}
    >>> 
    >>> packs_starting_at = {2: 4, 8: 9}
    >>> packs_ending_at = {4: 2, 9: 8}
    >>> insert_new_pack(packs_starting_at, packs_ending_at, [11, 12])
    (11, 12)
    >>> packs_starting_at
    {2: 4, 8: 9, 11: 12}
    >>> 
    >>> packs_starting_at = {2: 4, 8: 9}
    >>> packs_ending_at = {4: 2, 9: 8}
    >>> insert_new_pack(packs_starting_at, packs_ending_at, [5, 6])
    (2, 6)
    >>> packs_starting_at
    {2: 6, 8: 9}
    >>> 
    >>> packs_starting_at = {2: 4, 8: 9}
    >>> packs_ending_at = {4: 2, 9: 8}
    >>> insert_new_pack(packs_starting_at, packs_ending_at, [6, 7])
    (6, 9)
    >>> packs_starting_at
    {2: 4, 6: 9}
    """
    i, j = candidate_pack
    if i - 1  in packs_ending_at:
        i = packs_ending_at[i - 1]
    if j + 1 in packs_starting_at:
        j = packs_starting_at[j + 1]
    if i in packs_starting_at:
        del packs_ending_at[packs_starting_at[i]]
    packs_starting_at[i] = j
    if j in packs_ending_at:
        del packs_starting_at[packs_ending_at[j]]
    packs_ending_at[j] = i
    return i, j

@njit(cache=True, fastmath=True)
def compute_cost_for_pack(idx_start, idx_end, pack_costs_cumsum):
    """Compute the associated cost for a pack (set of contiguous points
    included in the solution) ranging from `idx_start` to `idx_end` (both included).
    """
    return pack_costs_cumsum[idx_end] - pack_costs_cumsum.get(idx_start - 1, 0)

@njit(cache=True, fastmath=True)
def precompute_pack_costs_cumsum(group_ending_at, n):
    """For each position `i` at which a pack could end,
    Compute (using dynamic programming and the costs of groups that have been precomputed)
    the cost of the largest pack ending at `i`.

    This is useful because this can be used later, to compute the cost of any pack in O(1)
    (cf. `compute_cost_for_pack`).
    """
    pack_costs_cumsum = Dict.empty(key_type=types.int64, value_type=types.float64)
    for i in range(n):
        # If there is a group ending there
        # (if not, this cannot be the end of a pack)
        if i in group_ending_at:
            start, additional_cost = group_ending_at[i]
            pack_costs_cumsum[i] = pack_costs_cumsum.get(start - 1, 0) + additional_cost
    return pack_costs_cumsum

@njit(cache=True, fastmath=True)
def compute_costs(diff_cum_sum, diff_ranks, sorted_distrib_indicator):
    """For each element in sorted `x`, compute its group (cf note below).
    Then compute the cost for each group and sort all groups in increasing 
    cost order.

    Note: the "group" of a point x_i in x is the minimal set of adjacent 
    points (starting at x_i and extending to the right) that one should 
    take to get a balanced set (ie. a set in which we have as many 
    elements from x as elements from y)

    # Examples
    # --------
    # >>> p = PartialOT1d(max_iter=3)
    # >>> x = [1, 2, 5, 6]
    # >>> y = [3, 4, 11, 12]
    # >>> p.preprocess(x, y)
    # >>> diff_cum_sum = p.compute_cumulative_sum_differences()
    # >>> ranks_xy, diff_ranks = p.compute_rank_differences()
    # >>> costs, pack_costs_cumsum = p.compute_costs(diff_cum_sum, diff_ranks)
    # >>> list(costs)
    # [(1, 2, 1), (3, 4, 1), (5, 6, 5)]
    """
    l_costs = [(0.0, 0, 0) for _ in range(0)]
    _group_ending_at = {}
    last_pos_for_rank_x = Dict.empty(key_type=int64, value_type=int64)
    last_pos_for_rank_y = Dict.empty(key_type=int64, value_type=int64)
    n = len(diff_ranks)
    for i in range(n):
        # For each item in either distrib, find the scope of the smallest
        # "group" that would start at that point and extend on the right, 
        # if one exists, and store the cost of this "group" by relying 
        # on differences of cumulative sums
        idx_end = i
        cur_rank = diff_ranks[i]
        idx_start = -1
        if sorted_distrib_indicator[i] == 0:
            target_rank = cur_rank - 1
            if target_rank in last_pos_for_rank_y:
                idx_start = last_pos_for_rank_y[target_rank]
            last_pos_for_rank_x[cur_rank] = i
        else:
            target_rank = cur_rank + 1
            if target_rank in last_pos_for_rank_x:
                idx_start = last_pos_for_rank_x[target_rank]
            last_pos_for_rank_y[cur_rank] = i
        if idx_start != -1:
            if idx_start == 0:
                cost = diff_cum_sum[idx_end] # - 0
            else:
                cost = diff_cum_sum[idx_end]   - diff_cum_sum[idx_start - 1]
            if idx_end == idx_start + 1:
                heapq.heappush(l_costs, (abs(cost), idx_start, idx_end))
            assert idx_end not in _group_ending_at
            _group_ending_at[idx_end] = (idx_start, abs(cost))
    return l_costs, precompute_pack_costs_cumsum(_group_ending_at, n)

@njit(cache=True, fastmath=True)
def preprocess(x, y):
    """Given two 1d distributions `x` and `y`:
    
    1. `indices_sort_x` sorts `x` (ie. `x[indices_sort_x]` is sorted) and
       `indices_sort_y` sorts `y` (ie. `y[indices_sort_y]` is sorted)
    
    2. stack them into a single distrib such that:
    
    * the new distrib is sorted with sort indices (wrt a stack of sorted x and sorted y) `indices_sort_xy`
    * `sorted_distrib_indicator` is a vector of zeros and ones where 0 means 
        "this point comes from x" and 1 means "this point comes from y"
    """
    indices_sort_x = np.argsort(x)
    indices_sort_y = np.argsort(y)

    xy = np.concatenate((x[indices_sort_x], y[indices_sort_y]))
    indices_sort_xy = np.argsort(xy)

    idx = np.concatenate((np.zeros(x.shape[0], dtype=np.int64), np.ones(y.shape[0], dtype=np.int64)))
    sorted_distrib_indicator = idx[indices_sort_xy]

    return indices_sort_x, indices_sort_y, indices_sort_xy, sorted_distrib_indicator

@njit(cache=True, fastmath=True)
def generate_solution_using_marginal_costs(costs, ranks_xy, pack_costs_cumsum, max_iter, sorted_distrib_indicator):
    """Generate a solution from a sorted list of group costs.
    See the note in `compute_costs` docs for a definition of groups.

    The solution is a pair of lists. The first list contains the indices from `sorted_x`
    that are in the active set, and the second one contains the indices from `sorted_y`
    that are in the active set.

    # Examples
    # --------
    # >>> p = PartialOT1d(max_iter=3)
    # >>> x = [1, 2, 5, 6]
    # >>> y = [3, 4, 11, 12]
    # >>> p.preprocess(x, y)
    # >>> diff_cum_sum = p.compute_cumulative_sum_differences()
    # >>> ranks_xy, diff_ranks = p.compute_rank_differences()
    # >>> costs, pack_costs_cumsum = p.compute_costs(diff_cum_sum, diff_ranks)
    # >>> ranks_xy = p.compute_rank_differences()[0]
    # >>> p.generate_solution_using_marginal_costs(costs, ranks_xy, pack_costs_cumsum)
    # (array([1, 2, 3]), array([0, 1, 2]), [1, 1, 5])
    """
    # max_iter = self.max_iter if self.max_iter != "elbow" else min(self.n_x, self.n_y)
    active_set = set()
    packs_starting_at = Dict.empty(key_type=int64, value_type=int64)
    packs_ending_at = Dict.empty(key_type=int64, value_type=int64)
    list_marginal_costs = []
    list_active_set_inserts = []
    n = len(sorted_distrib_indicator)
    while len(costs) > 0 and max_iter > len(active_set) // 2:
        c, i, j = heapq.heappop(costs)
        if i in active_set or j in active_set:
            continue
        new_pack = None
        # Case 1: j == i + 1 => "Simple" insert
        if j == i + 1:
            new_pack = insert_new_pack(packs_starting_at, packs_ending_at, [i, j])
        # Case 2: insert a group that contains a pack
        elif j - 1 in packs_ending_at:
            del packs_starting_at[i + 1]
            del packs_ending_at[j - 1]
            new_pack = insert_new_pack(packs_starting_at, packs_ending_at, [i, j])
        # There should be no "Case 3"
        else:
            # self._print_current_status(active_set, i, j)
            raise ValueError
        active_set.update({i, j})
        list_active_set_inserts.append((i, j))
        list_marginal_costs.append(c)
        
        # We now need to update the groups wrt the pack we have just created
        p_s, p_e = new_pack
        if p_s == 0 or p_e == n - 1:
            continue
        if sorted_distrib_indicator[p_s - 1] != sorted_distrib_indicator[p_e + 1]:
            # Insert (p_s - 1, p_e + 1) as a new pseudo-group with marginal cost
            marginal_cost = (compute_cost_for_pack(p_s - 1, p_e + 1, pack_costs_cumsum)
                                - compute_cost_for_pack(p_s, p_e, pack_costs_cumsum))
            heapq.heappush(costs, (marginal_cost, p_s - 1, p_e + 1))

    # Generate index arrays in the order of insertion in the active set
    indices_sorted_x = np.array([ranks_xy[i] 
                                    if sorted_distrib_indicator[i] == 0 else ranks_xy[j] 
                                    for i, j in list_active_set_inserts])
    indices_sorted_y = np.array([ranks_xy[i] 
                                    if sorted_distrib_indicator[i] == 1 else ranks_xy[j] 
                                    for i, j in list_active_set_inserts])

    return (indices_sorted_x, indices_sorted_y, list_marginal_costs)

@njit(cache=True, fastmath=True)
def compute_cumulative_sum_differences(x_sorted, y_sorted, indices_sort_xy, sorted_distrib_indicator):
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

    # Examples
    # --------
    # >>> p = PartialOT1d(-1)
    # >>> x = [-1, 3, 4]
    # >>> y = [1, 2, 5]
    # >>> p.preprocess(x, y)
    # >>> p.sorted_distrib_indicator
    # array([0, 1, 1, 0, 0, 1])
    # >>> p.compute_cumulative_sum_differences()  # [-1, -1, -1, 2, 6, 6] - [0, 1, 3, 3, 3, 8]
    # array([-1, -2, -4, -1,  3, -2])
    """
    cum_sum_xs = np.cumsum(x_sorted)
    cum_sum_ys = np.cumsum(y_sorted)
    cum_sum = np.concatenate((cum_sum_xs, cum_sum_ys))
    cum_sum = cum_sum[indices_sort_xy]

    cum_sum_x = _insert_constant_values_float(cum_sum, 0, sorted_distrib_indicator)
    cum_sum_y = _insert_constant_values_float(cum_sum, 1, sorted_distrib_indicator)

    return cum_sum_x - cum_sum_y

@njit(cache=True, fastmath=True)
def _insert_constant_values_int(arr, distrib_index, sorted_distrib_indicator):
    """Takes `arr` as input. For each position `i` in `arr`, 
    if `self.sorted_distrib_indicator[i]==distrib_index`,
    the value from `arr` is copied, 
    otherwise, the previous value that was copied from `arr` is repeated.

    # Examples
    # --------
    # >>> p = PartialOT1d(-1)
    # >>> x = [0, 3, 4]
    # >>> y = [1, 2, 5]
    # >>> p.preprocess(x, y)
    # >>> p._insert_constant_values([1, -1, -1, 2, 3, -1], 0)
    # array([1, 1, 1, 2, 3, 3])
    """
    arr_insert = np.copy(arr)
    for i in range(len(arr)):
        if sorted_distrib_indicator[i]!=distrib_index:
            if i == 0:
                arr_insert[i] = 0
            else:
                arr_insert[i] = arr_insert[i-1]
    return arr_insert

@njit(cache=True, fastmath=True)
def _insert_constant_values_float(arr, distrib_index, sorted_distrib_indicator):
    """Takes `arr` as input. For each position `i` in `arr`, 
    if `self.sorted_distrib_indicator[i]==distrib_index`,
    the value from `arr` is copied, 
    otherwise, the previous value that was copied from `arr` is repeated.

    # Examples
    # --------
    # >>> p = PartialOT1d(-1)
    # >>> x = [0, 3, 4]
    # >>> y = [1, 2, 5]
    # >>> p.preprocess(x, y)
    # >>> p._insert_constant_values([1, -1, -1, 2, 3, -1], 0)
    # array([1, 1, 1, 2, 3, 3])
    """
    arr_insert = np.copy(arr)
    for i in range(len(arr)):
        if sorted_distrib_indicator[i]!=distrib_index:
            if i == 0:
                arr_insert[i] = 0.
            else:
                arr_insert[i] = arr_insert[i-1]
    return arr_insert


@njit(cache=True, fastmath=True)
def compute_rank_differences(indices_sort_xy, sorted_distrib_indicator):
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

    # Examples
    # --------
    # >>> p = PartialOT1d(max_iter=2)
    # >>> x = [-2, 2, 3]
    # >>> y = [-1, 1, 5]
    # >>> p.preprocess(x, y)
    # >>> ranks_xy, diff_ranks = p.compute_rank_differences()
    # >>> ranks_xy
    # array([0, 0, 1, 1, 2, 2])
    # >>> diff_ranks
    # array([ 1,  0, -1,  0,  1,  0])
    """
    n_x = np.sum(sorted_distrib_indicator == 0)
    n_y = np.sum(sorted_distrib_indicator == 1)
    ranks_x, ranks_y = np.arange(n_x), np.arange(n_y)
    ranks_xy = np.concatenate((ranks_x, ranks_y))
    ranks_xy = ranks_xy[indices_sort_xy]

    ranks_xy_x = ranks_xy.copy()
    ranks_xy_x[sorted_distrib_indicator==1] = 0
    ranks_xy_x[sorted_distrib_indicator==0] += 1
    ranks_xy_x_cum = _insert_constant_values_int(ranks_xy_x, 0, sorted_distrib_indicator)

    ranks_xy_y = ranks_xy.copy()
    ranks_xy_y[sorted_distrib_indicator==0] = 0
    ranks_xy_y[sorted_distrib_indicator==1] += 1
    ranks_xy_y_cum = _insert_constant_values_int(ranks_xy_y, 1, sorted_distrib_indicator)

    diff_ranks = ranks_xy_x_cum - ranks_xy_y_cum

    return ranks_xy, diff_ranks

@njit(cache=True, fastmath=True)
def partial_ot_1d(x, y, max_iter):
    """Main routine for the partial OT problem in 1D.
    
    Does:
    
    1. Preprocessing of the distribs (sorted & co)
    2. Precomputations (ranks, cumulative sums)
    3. Extraction of groups
    4. Generate and return solution

    Note that the indices in `indices_x` and `indices_y` are ordered wrt their order of
    appearance in the solution such that `indices_x[:10]` (resp y) is the set of indices
    from x (resp. y) for the partial problem of size 10.

    Arguments
    ---------
    x : np.ndarray of shape (n, )
        First distrib to be considered (weights are considered uniform)
    y : np.ndarray of shape (m, )
        Second distrib to be considered (weights are considered uniform)
    max_iter : int
        Number of iterations of the algorithm, which is equal to the number of pairs
        in the returned solution.

    Returns
    -------
    indices_x : np.ndarray of shape (min(n, m, max_iter), )
        Indices of elements from the x distribution to be included in the partial solutions
        Order of appearance in this array indicates order of inclusion in the solution
    indices_y : np.ndarray of shape (min(n, m, max_iter), )
        Indices of elements from the x distribution to be included in the partial solutions
        Order of appearance in this array indicates order of inclusion in the solution
    list_marginal_costs : list of length min(n, m, max_iter)
        List of marginal costs associated to the intermediate partial problems
        `np.cumsum(list_marginal_costs)` gives the corresponding total costs for intermediate partial problems

    Examples
    --------
    >>> x = np.array([5., -2., 4.])
    >>> y = np.array([-1., 1., 3.])
    >>> partial_ot_1d(x, y, max_iter=2)
    (array([1, 2]), array([0, 2]), [1.0, 1.0])
    """
    # Sort distribs and keep track of their original indices
    indices_sort_x, indices_sort_y, indices_sort_xy, sorted_distrib_indicator = preprocess(x, y)

    # Precompute useful quantities
    diff_cum_sum = compute_cumulative_sum_differences(x[indices_sort_x], 
                                                      y[indices_sort_y], 
                                                      indices_sort_xy,
                                                      sorted_distrib_indicator)
    ranks_xy, diff_ranks = compute_rank_differences(indices_sort_xy, sorted_distrib_indicator)

    # Compute costs for "groups"
    costs, pack_costs_cumsum = compute_costs(diff_cum_sum, diff_ranks, sorted_distrib_indicator)

    # Generate solution from sorted costs
    sol_indices_x_sorted, sol_indices_y_sorted, sol_costs = generate_solution_using_marginal_costs(costs, 
                                                                                                   ranks_xy, 
                                                                                                   pack_costs_cumsum, 
                                                                                                   max_iter, 
                                                                                                   sorted_distrib_indicator)

    # Convert back into indices in original `x` and `y` distribs
    indices_x = indices_sort_x[sol_indices_x_sorted]
    indices_y = indices_sort_y[sol_indices_y_sorted]
    marginal_costs = sol_costs
    return indices_x, indices_y, marginal_costs



def partial_ot_1d_elbow(x, y, return_all_solutions=False):
    """Main routine for the partial OT problem in 1D.
    
    Does:
    
    1. Preprocessing of the distribs (sorted & co)
    2. Precomputations (ranks, cumulative sums)
    3. Extraction of groups
    4. Generate and return solution
    
    Here, the elbow method is used to compute the optimal partial problem size.
    If no elbow is found on the cumulative cost series, the full solution is returned.

    Note that the indices in `indices_x` and `indices_y` are ordered wrt their order of
    appearance in the solution such that `indices_x[:10]` (resp y) is the set of indices
    from x (resp. y) for the partial problem of size 10.

    Arguments
    ---------
    x : np.ndarray of shape (n, )
        First distrib to be considered (weights are considered uniform)
    y : np.ndarray of shape (m, )
        Second distrib to be considered (weights are considered uniform)
    return_all_solutions : bool
        Whether all solutions should be returned (eg. to visualize the elbow) beyond
        the elbow

    Returns
    -------
    indices_x : np.ndarray of shape (min(n, m, max_iter), )
        Indices of elements from the x distribution to be included in the partial solutions
        Order of appearance in this array indicates order of inclusion in the solution
    indices_y : np.ndarray of shape (min(n, m, max_iter), )
        Indices of elements from the x distribution to be included in the partial solutions
        Order of appearance in this array indicates order of inclusion in the solution
    list_marginal_costs : list of length min(n, m, max_iter)
        List of marginal costs associated to the intermediate partial problems
        `np.cumsum(list_marginal_costs)` gives the corresponding total costs for intermediate partial problems
    elbow_index : int
        Elbow index

    Examples
    --------
    >>> x = np.array([5., -2., 4.])
    >>> y = np.array([-1., 1., 3.])
    >>> partial_ot_1d_elbow(x, y)
    (array([1, 2, 0]), array([0, 2, 1]), [1.0, 1.0, 4.0])
    """
    n = min(len(x), len(y))
    indices_x, indices_y, marginal_costs = partial_ot_1d(x, y, max_iter=n)
    kneedle = KneeLocator(x=np.arange(len(marginal_costs)), 
                          y=np.cumsum(marginal_costs), 
                          S=1.0, 
                          curve="convex", 
                          direction="increasing")
    if kneedle.elbow is None:
        # No elbow has been detected
        idx_elbow = n
    else:
        idx_elbow = int(kneedle.elbow)
    if return_all_solutions:
        return (indices_x, 
                indices_y, 
                marginal_costs,
                idx_elbow + 1)
    else:
        return (indices_x[:idx_elbow + 1], 
                indices_y[:idx_elbow + 1], 
                marginal_costs[:idx_elbow + 1],
                idx_elbow + 1)


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(30, )
    y = np.random.rand(40, )
    indices_x, indices_y, marginal_costs, elbow = partial_ot_1d_elbow(x, y)
    print(len(indices_x), len(indices_y))
    print(indices_x)
    print(indices_y)